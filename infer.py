import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import os
from model import RallyPredictor
from dataset import VolleyballDataset, test_transform, TEST_VIDEOS
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_predictions(model_path, dataset, sample_indices=None, num_samples=10, output_dir="output/frames"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    rally_predictor = RallyPredictor().to(device)
    model = rally_predictor
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(rally_predictor)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['rally_predictor_state_dict']
    if list(state_dict.keys())[0].startswith('module.') and not isinstance(model, nn.DataParallel):
        print("Stripping 'module.' prefix from checkpoint keys")
        new_state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    model.eval()

    # Get class mappings
    action_classes = {v: k for k, v in dataset.dataset.action_classes.items()}
    group_classes = {v: k for k, v in dataset.dataset.group_classes.items()}
    action_colors = {
        'waiting': (255, 255, 0),     # yellow
        'setting': (0, 255, 0),       # green
        'digging': (255, 255, 0),     # cyan 
        'falling': (0, 165, 255),     # orange  
        'spiking': (0, 0, 255),       # red  
        'blocking': (128, 0, 128),    # purple  
        'jumping': (203, 192, 255),   # pink  
        'moving': (255, 0, 0),        # blue  
        'standing': (128, 128, 128)   # gray  
    }

    # Get the underlying dataset from the DataLoader
    volleyball_dataset = dataset.dataset

    # Select samples
    if sample_indices is None:
        # Get random samples
        import random
        total_samples = len(volleyball_dataset)
        sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))

    # Limit to num_samples
    sample_indices = sample_indices[:num_samples]
    samples = [volleyball_dataset[idx] for idx in sample_indices if idx < len(volleyball_dataset)]

    # Process each sample
    for idx, sample in enumerate(samples):
        print(f"\n===== Processing Sample {idx+1}/{len(samples)} =====")
        images, bboxes, person_actions, group_activity = sample
        
        # Add batch dimension since dataset returns single samples
        images = images.unsqueeze(0).to(device)
        bboxes = bboxes.unsqueeze(0).to(device)
        person_actions = person_actions.unsqueeze(0)
        group_activity = torch.tensor([group_activity])  # Wrap in tensor
        
        # Run inference
        with torch.no_grad():
            _, action_logits, group_logits, rally_probs, _ = model(images, bboxes)
        
        # Get predictions
        _, action_preds = torch.max(action_logits, dim=2)
        _, group_pred = torch.max(group_logits, dim=1)
        
        # Get middle frame for visualization
        middle_idx = images.size(1) // 2
        
        # Process the sample
        gt_group = group_classes[group_activity[0].item()]
        pred_group = group_classes[group_pred[0].item()]
        gt_actions = [action_classes[a.item()] for a in person_actions[0]]
        pred_actions = [action_classes[a.item()] for a in action_preds[0]]
        left_prob, right_prob = rally_probs[0, 0].item(), rally_probs[0, 1].item()
        
        print(f"BATCH 0 RESULTS:")
        print(f"Group Activity - Ground Truth: {gt_group} | Prediction: {pred_group}")
        print(f"Rally Probabilities - Left Team: {left_prob:.2f}, Right Team: {right_prob:.2f}")
        print("\nPlayer Action Comparison:")
        correct_actions = sum(1 for gt, pred in zip(gt_actions, pred_actions) if gt == pred)
        total_actions = len(gt_actions)
        action_accuracy = correct_actions / total_actions * 100
        for p, (gt, pred) in enumerate(zip(gt_actions, pred_actions)):
            match = "✓" if gt == pred else "✗"
            print(f"Player {p}: {gt:10} → {pred:10} {match}")
        print(f"\nAction Accuracy: {correct_actions}/{total_actions} ({action_accuracy:.1f}%)")
        
        # Get the middle frame and denormalize
        middle_frame = images[0, middle_idx].cpu().permute(1, 2, 0).numpy()
        middle_frame = middle_frame * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        middle_frame = np.clip(middle_frame, 0, 1)
        
        # Convert to BGR for OpenCV (matplotlib uses RGB, OpenCV uses BGR)
        middle_frame_bgr = (middle_frame * 255).astype(np.uint8)
        middle_frame_bgr = cv2.cvtColor(middle_frame_bgr, cv2.COLOR_RGB2BGR)
        
        # Create a new image with predictions
        boxes = bboxes[0].cpu().numpy()
        for p, (box, pred_action) in enumerate(zip(boxes, pred_actions)):
            x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)
            if w == 0 or h == 0:
                continue
            
            # Draw bounding box
            color = action_colors[pred_action]
            cv2.rectangle(middle_frame_bgr, (x, y), (x+w, y+h), color, 2)
            
            # Add player label
            label = f"{p}: {pred_action}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(middle_frame_bgr, (x, y-text_h-4), (x+text_w, y), color, -1)
            cv2.putText(middle_frame_bgr, label, (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add title with prediction information
        title = f"Ground Truth: {gt_group} | Prediction: {pred_group}"
        cv2.putText(middle_frame_bgr, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add rally probabilities
        rally_text = f"Rally Probs - Left Team: {left_prob:.2f}, Right Team: {right_prob:.2f}"
        cv2.putText(middle_frame_bgr, rally_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Save the image
        output_path = os.path.join(output_dir, f"sample_{idx+1}_of_{len(samples)}.jpg")
        cv2.imwrite(output_path, middle_frame_bgr)
        print(f"Saved output to {output_path}")

if __name__ == "__main__":
    model_path = "output/checkpoint_epoch_60_train_alexnet.pth"
    output_dir = "output/prediction_frames"
    
    # Define dataset
    data_dir = Path("dataset")
    videos_dir = data_dir / "videos"
    annotations_dir = data_dir / "volleyball_tracking_annotation"
    test_dataset = VolleyballDataset(video_ids=TEST_VIDEOS, videos_dir=videos_dir, 
                                   annotations_dir=annotations_dir, transform=test_transform)
    
    # No. of samples to process
    num_samples = 10
    
    sample_indices = None  # Set to None for random samples
    
    # sample_indices = [5, 10, 15, 17, 21, 25, 30, 35, 40, 45]
    
    visualize_predictions(
        model_path=model_path, 
        dataset=test_dataset, 
        sample_indices=sample_indices, 
        num_samples=num_samples,
        output_dir=output_dir
    )
