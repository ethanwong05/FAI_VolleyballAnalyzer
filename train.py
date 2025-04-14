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
        # If no checkpoint but still starting from later epoch, advance scheduler
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"Advanced scheduler to epoch {start_epoch}")
    
    
