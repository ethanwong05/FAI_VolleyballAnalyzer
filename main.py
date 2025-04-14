import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser(description="Volleyball Activity Recognition")
    parser.add_argument("--mode", type=str, default="train",
                      choices=["train", "evaluate", "visualize"],
                      help="Mode: train, evaluate, or visualize")
    parser.add_argument("--data_dir", type=str, default="dataset",
                      help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="output",
                      help="Output directory")
    parser.add_argument("--model_path", type=str, default=None,
                      help="Path to model checkpoint (for evaluate and visualize modes)")
    parser.add_argument("--num_epochs", type=int, default=60,
                      help="Number of epochs to train")
    parser.add_argument("--start_epoch", type=int, default=1,
                      help="Starting epoch")
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size")
    parser.add_argument("--num_samples", type=int, default=3,
                      help="Number of samples to visualize")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / "checkpoints", exist_ok=True)
    
    # Import modules
    from dataset import VolleyballDataset, transform, test_transform, TRAIN_VIDEOS, VAL_VIDEOS, TEST_VIDEOS
    from model import RallyPredictor
    from train import train_model
    from evaluate import evaluate_model
    from infer import visualize_predictions
    import torch.utils.data as data
    
    # Create datasets and dataloaders
    data_dir = Path(args.data_dir)
    videos_dir = data_dir / "videos"
    annotations_dir = data_dir / "volleyball_tracking_annotation"
    
    if args.mode == "train" or args.mode == "evaluate":
        train_dataset = VolleyballDataset(TRAIN_VIDEOS, videos_dir, annotations_dir, transform=transform)
        val_dataset = VolleyballDataset(VAL_VIDEOS, videos_dir, annotations_dir, transform=test_transform)
        test_dataset = VolleyballDataset(TEST_VIDEOS, videos_dir, annotations_dir, transform=test_transform)
        
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                     num_workers=0, pin_memory=True)
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                   num_workers=0, pin_memory=True)
        test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                    num_workers=0, pin_memory=True)
        
        print(f"Training dataset size: {len(train_dataset)} clips")
        print(f"Validation dataset size: {len(val_dataset)} clips")
        print(f"Test dataset size: {len(test_dataset)} clips")
    
    if args.mode == "train":
        # Initialize model
        rally_predictor = RallyPredictor().to(device)
        
        # Use DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            rally_predictor = nn.DataParallel(rally_predictor)
        
        # Train model
        train_model(rally_predictor, train_loader, val_loader, num_epochs=args.num_epochs, 
                   start_epoch=args.start_epoch)
        
    elif args.mode == "evaluate":
        # Determine model path
        if args.model_path is None:
            model_path = output_dir / "checkpoints" / "best_model_alexnet.pth"
            if not model_path.exists():
                model_path = output_dir / "checkpoints" / "final_model_alexnet.pth"
        else:
            model_path = Path(args.model_path)
        
        if not model_path.exists():
            print(f"Error: Model file {model_path} does not exist.")
            return
        
        print(f"Evaluating model: {model_path}")
        evaluate_model(model_path, test_loader, device)
        
    elif args.mode == "visualize":
        # Determine model path
        if args.model_path is None:
            model_path = output_dir / "checkpoints" / "best_model_alexnet.pth"
            if not model_path.exists():
                model_path = output_dir / "checkpoints" / "final_model_alexnet.pth"
        else:
            model_path = Path(args.model_path)
        
        if not model_path.exists():
            print(f"Error: Model file {model_path} does not exist.")
            return
        
        # Create test dataset and loader
        test_dataset = VolleyballDataset(TEST_VIDEOS, videos_dir, annotations_dir, transform=test_transform)
        test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        
        vis_output_dir = output_dir / "visualizations"
        os.makedirs(vis_output_dir, exist_ok=True)
        
        print(f"Visualizing model: {model_path}")
        visualize_predictions(model_path, test_loader, num_samples=args.num_samples, 
                            output_dir=str(vis_output_dir))

if __name__ == "__main__":
    main()
