import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path

from model import FocalLoss

def train_model(rally_predictor, train_loader, val_loader, num_epochs=50, start_epoch=24):
    """
    Training the model using GPU
    """

    # Clear GPU memory before starting
    torch.cuda.empty_cache()
    print("Cleared the GPU memory cache before starting the training")

    # Get device from model
    device = next(rally_predictor.parameters()).device
    
    # Set up loss functions with class weights
    action_weights = train_loader.dataset.action_weights
    action_weights_tensor = torch.tensor([action_weights[action] for action in sorted(action_weights.keys())], 
                                        dtype=torch.float32).to(device)
    
    group_weights = train_loader.dataset.group_weights
    group_weights_tensor = torch.tensor([group_weights[group] for group in sorted(group_weights.keys())], 
                                       dtype=torch.float32).to(device)
    
    person_criterion = FocalLoss(alpha=action_weights_tensor, gamma=1.0)
    group_criterion = nn.CrossEntropyLoss(weight=group_weights_tensor)
    rally_ranking_criterion = nn.MarginRankingLoss(margin=0.2)
    rally_heuristic_criterion = nn.MSELoss()
    
    # Initial learning rate setup
    initial_lr = 0.0001
    
    # Define optimizer with weight decay
    optimizer = optim.AdamW(rally_predictor.parameters(), lr=initial_lr, betas=(0.9, 0.999), weight_decay=5e-3)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=12, T_mult=1, eta_min=5e-7)
    
    scaler = amp.GradScaler('cuda')
    
    persistent_dir = Path("output/checkpoints")
    persistent_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize best_val_group_acc and history for moving average

    # Set to the best validation accuracy from epoch 20 (only coz we did multiple kaggle sessions of training)
    
    best_val_group_acc = 0.4534

    # Store validation group accuracies for moving average
    val_group_acc_history = []
    
    # Gradient accumulation settings
    accumulation_steps = 2
    effective_batch_size = 16 * accumulation_steps
    print(f"Using gradient accumulation: {accumulation_steps} steps, effective batch size = {effective_batch_size}")
    
    # Gradient clipping threshold
    max_grad_norm = 1.0

    # Load checkpoint if continuing training
    checkpoint_path = Path(f"output/checkpoints/checkpoint_epoch_{start_epoch-1}_train_alexnet.pth")
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        state_dict = checkpoint['rally_predictor_state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            if isinstance(rally_predictor, nn.DataParallel) and key.startswith("module."):
                new_key = key
                new_state_dict[new_key] = value
            elif not isinstance(rally_predictor, nn.DataParallel) and key.startswith("module."):
                new_key = key[len("module."):]
                print(f"Stripping 'module.' prefix: {key} -> {new_key}")
                new_state_dict[new_key] = value
            elif isinstance(rally_predictor, nn.DataParallel) and not key.startswith("module."):
                new_key = "module." + key
                print(f"Adding 'module.' prefix: {key} -> {new_key}")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        if isinstance(rally_predictor, nn.DataParallel):
            print("Loading state_dict into rally_predictor.module with strict=False")
            rally_predictor.module.load_state_dict(new_state_dict, strict=False)
        else:
            print("Loading state_dict into rally_predictor with strict=False")
            rally_predictor.load_state_dict(new_state_dict, strict=False)
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"Loaded scheduler state from checkpoint")
            except Exception as e:
                print(f"Couldn't load scheduler state, advancing scheduler manually: {e}")
                for _ in range(start_epoch - 1):
                    scheduler.step()
        else:
            print(f"No scheduler state in checkpoint, advancing scheduler manually")
            for _ in range(start_epoch - 1):
                scheduler.step()
                
        # Set best_val_group_acc based on previous best from epoch 20 validation
        print(f"Using best validation accuracy: {best_val_group_acc}")
                
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Starting from scratch.")
        
        # Advance scheduler
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"Advanced scheduler to epoch {start_epoch}")
    
    print("Training...")
    for epoch in range(start_epoch, num_epochs + 1):
        # Log current learning rate from scheduler
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{num_epochs}: Starting training loop... (Learning Rate: {current_lr:.6f})")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"GPU {i} memory cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        
        rally_predictor.train()
        running_person_loss = 0.0
        running_group_loss = 0.0
        running_rally_ranking_loss = 0.0
        running_rally_heuristic_loss = 0.0
        person_correct = 0
        group_correct = 0
        total_person = 0
        total_group = 0
        total_rally_samples = 0
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            images, bboxes, person_actions, group_activity = batch
            images = images.to(device)
            bboxes = bboxes.to(device)
            person_actions = person_actions.to(device)
            group_activity = group_activity.to(device)
            
            with amp.autocast('cuda'):
                person_features, action_logits, group_logits, rally_probs, heuristic_probs = rally_predictor(images, bboxes)
                
                action_logits = action_logits.view(-1, action_logits.size(-1))
                person_actions = person_actions.view(-1)
                person_loss = person_criterion(action_logits, person_actions)
                
                group_loss = group_criterion(group_logits, group_activity)
                
                left_probs = rally_probs[:, 0]
                right_probs = rally_probs[:, 1]
                left_team_activities = torch.tensor([0, 2, 4, 6], device=device)
                right_team_activities = torch.tensor([1, 3, 5, 7], device=device)
                target = torch.zeros_like(left_probs)
                target[torch.isin(group_activity, left_team_activities)] = 1
                target[torch.isin(group_activity, right_team_activities)] = -1
                rally_ranking_loss = rally_ranking_criterion(left_probs, right_probs, target)
                
                rally_heuristic_loss = rally_heuristic_criterion(rally_probs, heuristic_probs)
                
                total_loss = 2.5 * person_loss + 2.5 * group_loss + 1.5 * rally_ranking_loss + 1.0 * rally_heuristic_loss

                # Scale the loss for accumulation
                total_loss = total_loss / accumulation_steps 
            
            scaler.scale(total_loss).backward()
            
            # Perform optimizer step after accumulation_steps or at the last batch
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):

                # Unscale optimizer gradients for clipping
                scaler.unscale_(optimizer)
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(rally_predictor.parameters(), max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Accumulate metrics
            running_person_loss += person_loss.item() * accumulation_steps
            running_group_loss += group_loss.item() * accumulation_steps
            running_rally_ranking_loss += rally_ranking_loss.item() * accumulation_steps
            running_rally_heuristic_loss += rally_heuristic_loss.item() * accumulation_steps
            
            _, person_pred = torch.max(action_logits, 1)
            person_correct += (person_pred == person_actions).sum().item()
            total_person += person_actions.size(0)
            
            _, group_pred = torch.max(group_logits, 1)
            group_correct += (group_pred == group_activity).sum().item()
            total_group += group_activity.size(0)
            
            total_rally_samples += images.size(0)
            
            # Log the progress once every 50 batches for monitoring
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}: "
                      f"Person Loss: {person_loss.item():.4f}, Group Loss: {group_loss.item():.4f}")
        
        # Step scheduler at the end of each epoch
        scheduler.step()
        next_lr = optimizer.param_groups[0]['lr']
        print(f"Scheduler stepped: next epoch learning rate will be {next_lr:.6f}")
        
        avg_person_loss = running_person_loss / len(train_loader)
        avg_group_loss = running_group_loss / len(train_loader)
        avg_rally_ranking_loss = running_rally_ranking_loss / len(train_loader)
        avg_rally_heuristic_loss = running_rally_heuristic_loss / len(train_loader)
        person_acc = person_correct / total_person
        group_acc = group_correct / total_group
        
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Train Person Loss: {avg_person_loss:.4f}, Acc: {person_acc:.4f}")
        print(f"Train Group Loss: {avg_group_loss:.4f}, Acc: {group_acc:.4f}")
        print(f"Train Rally Ranking Loss: {avg_rally_ranking_loss:.4f}")
        print(f"Train Rally Heuristic Loss: {avg_rally_heuristic_loss:.4f}")
        
        # Save checkpoint after training phase (every epoch)
        checkpoint = {
            'epoch': epoch,
            'rally_predictor_state_dict': rally_predictor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'person_loss': avg_person_loss,
            'group_loss': avg_group_loss,
            'rally_ranking_loss': avg_rally_ranking_loss,
            'rally_heuristic_loss': avg_rally_heuristic_loss,
            'person_acc': person_acc,
            'group_acc': group_acc,
        }
        train_checkpoint_path = persistent_dir / f"checkpoint_epoch_{epoch}_train_alexnet.pth"
        torch.save(checkpoint, train_checkpoint_path)
        print(f"Saved checkpoint for epoch {epoch} (after training) at {train_checkpoint_path}")
        
        # Perform validation every 5 epochs or on the last epoch
        if epoch % 5 == 0 or epoch == num_epochs:  
            rally_predictor.eval()
            val_person_loss = 0.0
            val_group_loss = 0.0
            val_rally_heuristic_loss = 0.0
            val_rally_ranking_loss = 0.0
            val_person_correct = 0
            val_group_correct = 0
            val_total_person = 0
            val_total_rally_samples = 0
            val_total_group = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    images, bboxes, person_actions, group_activity = batch
                    images = images.to(device)
                    bboxes = bboxes.to(device)
                    person_actions = person_actions.to(device)
                    group_activity = group_activity.to(device)
                    
                    with amp.autocast('cuda'):
                        person_features, action_logits, group_logits, rally_probs, heuristic_probs = rally_predictor(images, bboxes)
                        
                        action_logits = action_logits.view(-1, action_logits.size(-1))
                        person_actions = person_actions.view(-1)
                        person_loss = person_criterion(action_logits, person_actions)
                        group_loss = group_criterion(group_logits, group_activity)
                        
                        left_probs = rally_probs[:, 0]
                        right_probs = rally_probs[:, 1]
                        target = torch.zeros_like(left_probs)
                        target[torch.isin(group_activity, left_team_activities)] = 1
                        target[torch.isin(group_activity, right_team_activities)] = -1
                        rally_ranking_loss = rally_ranking_criterion(left_probs, right_probs, target)
                        rally_heuristic_loss = rally_heuristic_criterion(rally_probs, heuristic_probs)
                    
                    val_person_loss += person_loss.item()
                    val_group_loss += group_loss.item()
                    val_rally_ranking_loss += rally_ranking_loss.item()
                    val_rally_heuristic_loss += rally_heuristic_loss.item()
                    
                    _, person_pred = torch.max(action_logits, 1)
                    val_person_correct += (person_pred == person_actions).sum().item()
                    val_total_person += person_actions.size(0)
                    
                    _, group_pred = torch.max(group_logits, 1)
                    val_group_correct += (group_pred == group_activity).sum().item()
                    val_total_group += group_activity.size(0)
                    
                    val_total_rally_samples += images.size(0)
            
            avg_val_person_loss = val_person_loss / len(val_loader)
            avg_val_group_loss = val_group_loss / len(val_loader)
            avg_val_rally_ranking_loss = val_rally_ranking_loss / len(val_loader)
            avg_val_rally_heuristic_loss = val_rally_heuristic_loss / len(val_loader)
            val_person_acc = val_person_correct / val_total_person
            val_group_acc = val_group_correct / val_total_group
            
            # Compute moving average of validation group accuracy
            val_group_acc_history.append(val_group_acc)

            # Keep only the last 3 validation points
            if len(val_group_acc_history) > 3:
                val_group_acc_history.pop(0)
            moving_avg_val_group_acc = sum(val_group_acc_history) / len(val_group_acc_history)
            
            # Log the validation metrics
            print(f"Val Person Loss: {avg_val_person_loss:.4f}, Acc: {val_person_acc:.4f}")
            print(f"Val Group Loss: {avg_val_group_loss:.4f}, Acc: {val_group_acc:.4f}")
            print(f"Val Rally Ranking Loss: {avg_val_rally_ranking_loss:.4f}")
            print(f"Val Rally Heuristic Loss: {avg_val_rally_heuristic_loss:.4f}")
            print(f"Moving Average Val Group Accuracy (last {len(val_group_acc_history)} validation points): {moving_avg_val_group_acc:.4f}")
            
            # Save checkpoint after validation phase
            checkpoint = {
                'epoch': epoch,
                'rally_predictor_state_dict': rally_predictor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'person_loss': avg_person_loss,
                'group_loss': avg_group_loss,
                'rally_ranking_loss': avg_rally_ranking_loss,
                'rally_heuristic_loss': avg_rally_heuristic_loss,
                'person_acc': person_acc,
                'group_acc': group_acc,
                'val_person_acc': val_person_acc,
                'val_group_acc': val_group_acc,
                'moving_avg_val_group_acc': moving_avg_val_group_acc
            }
            val_checkpoint_path = persistent_dir / f"checkpoint_epoch_{epoch}_val_alexnet.pth"
            torch.save(checkpoint, val_checkpoint_path)
            print(f"Saved checkpoint for epoch {epoch} (after validation) at {val_checkpoint_path}")
            
            # Save best model based on validation group accuracy
            if val_group_acc > best_val_group_acc:
                best_val_group_acc = val_group_acc
                best_model_path = persistent_dir / "best_model_alexnet.pth"
                torch.save({
                    'epoch': epoch,
                    'rally_predictor_state_dict': rally_predictor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'person_loss': avg_person_loss,
                    'group_loss': avg_group_loss,
                    'rally_ranking_loss': avg_rally_ranking_loss,
                    'rally_heuristic_loss': avg_rally_heuristic_loss,
                    'person_acc': person_acc,
                    'group_acc': group_acc,
                    'val_person_acc': val_person_acc,
                    'val_group_acc': val_group_acc,
                }, best_model_path)

                print(f"Saved best model at epoch {epoch} with Val Group Accuracy: {best_val_group_acc:.4f}")
    
    # Save final model
    final_model_path = persistent_dir / "final_model_alexnet.pth"
    torch.save({
        'rally_predictor_state_dict': rally_predictor.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'person_acc': person_acc,
        'group_acc': group_acc,
    }, final_model_path)
    print(f"Saved final model at {final_model_path}")
