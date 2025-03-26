import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms

class VolleyballDataset(Dataset):
    def __init__(self, csv_file, video_ids=None, transform=None):
        self.data = pd.read_csv(csv_file)
        if video_ids is not None:
            self.data = self.data[self.data['frame_path'].str.contains('|'.join([f'videos_full/{vid}/' for vid in video_ids]))]
        
        self.transform = transform
        self.action_encoder = LabelEncoder()
        self.actions = self.action_encoder.fit_transform(self.data['action'])
        self.data['action_encoded'] = self.actions
        
        print("Precomputing action indices...")
        self.action_to_indices = {}
        for idx, action in enumerate(self.actions):
            action_str = self.action_encoder.inverse_transform([action])[0]
            self.action_to_indices.setdefault(action_str, []).append(idx)
        print("Action indices computed.")
        
        # Enhanced augmentation for minority classes
        self.minority_actions = ['spiking', 'blocking', 'falling', 'jumping', 'digging', 'moving', 'setting', 'waiting']
        self.minority_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor()
        ])
        
        # Reduced augmentation for majority class
        self.base_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(5),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['frame_path']
        img = cv2.imread(img_path)
        
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        x, y, w, h = map(int, [row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']])
        
        # Ensure bounding box is within image
        h_img, w_img, _ = img.shape
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))
        
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (64, 64))
        
        action = row['action']
        if action in self.minority_actions:
            roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            roi = self.minority_transform(roi)
        else:
            roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            roi = self.base_transform(roi)
        
        action = torch.tensor(row['action_encoded'], dtype=torch.long)
        outcome = torch.tensor(row['outcome'], dtype=torch.float32)
        rally_id = row['rally_id']
        activity = row['activity']
        
        return roi, action, outcome, rally_id, img_path, (x, y, w, h), activity

if __name__ == '__main__':
    train_video_ids = [str(i) for i in range(54)]
    test_video_ids = ['54']
    
    train_dataset = VolleyballDataset("/Users/shiven/Downloads/VolleyballRallyAnalyzer/data/annotations.csv", video_ids=train_video_ids)
    test_dataset = VolleyballDataset("/Users/shiven/Downloads/VolleyballRallyAnalyzer/data/annotations.csv", video_ids=test_video_ids)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Action classes: {train_dataset.action_encoder.classes_}")
    
    roi, action, outcome, rally_id, img_path, bbox, activity = train_dataset[0]
    print(f"Sample ROI shape: {roi.shape}")
    print(f"Action: {train_dataset.action_encoder.inverse_transform([action.item()])}")
    print(f"Outcome: {outcome.item()}")
    print(f"Rally ID: {rally_id}")
    print(f"Image path: {img_path}")
    print(f"Bbox: {bbox}")
    print(f"Activity: {activity}")
