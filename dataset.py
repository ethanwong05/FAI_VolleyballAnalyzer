import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path

# Transform definitions
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define constants
TRAIN_VIDEOS = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31", "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]
VAL_VIDEOS = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]
TEST_VIDEOS = ["4", "5", "9", "11", "14", "20", "21", "25", "29", "34", "35", "37", "43", "44", "45", "47"]
HIGH_RES_VIDEOS = ["2", "37", "38", "39", "40", "41", "44", "45"]

# Dataset class
class VolleyballDataset(Dataset):
    def __init__(self, video_ids, videos_dir, annotations_dir, sequence_length=10, transform=None):
        self.videos_dir = videos_dir
        self.annotations_dir = annotations_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.high_res_videos = HIGH_RES_VIDEOS
        
        self.action_classes = {
            'waiting': 0, 'setting': 1, 'digging': 2, 'falling': 3, 'spiking': 4,
            'blocking': 5, 'jumping': 6, 'moving': 7, 'standing': 8
        }
        self.group_classes = {
            'l-pass': 0, 'r-pass': 1, 'l_set': 2, 'r_set': 3,
            'l-spike': 4, 'r_spike': 5, 'l_winpoint': 6, 'r_winpoint': 7
        }
        
        self.action_counts = {action: 0 for action in self.action_classes.keys()}
        self.group_counts = {group: 0 for group in self.group_classes.keys()}
        
        self.data = []
        
        for video_id in video_ids:
            video_path = videos_dir / video_id
            annot_path = videos_dir / video_id / "annotations.txt"
            if not video_path.exists() or not annot_path.exists():
                print(f"Skipping video {video_id}: Path or annotation file missing")
                continue
            
            print(f"Processing video {video_id}...")
            with open(annot_path, 'r') as f:
                lines = f.readlines()
                for line_num, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 2:
                        print(f"Video {video_id}, Line {line_num}: Malformed line, skipping: {line.strip()}")
                        continue
                    
                    frame_id = parts[0].split('.')[0]
                    group_activity = parts[1]
                    if group_activity not in self.group_classes:
                        print(f"Video {video_id}, Frame {frame_id}: Invalid group activity, skipping")
                        continue
                    
                    try:
                        frame_num = int(frame_id)
                    except ValueError:
                        print(f"Video {video_id}, Frame {frame_id}: Invalid frame ID, skipping")
                        continue
                    # Use 10 frames: 5 before and 5 after the keyframe
                    frame_ids = list(range(frame_num - 5, frame_num + 5))
                    
                    players = []
                    for i in range(2, len(parts), 5):
                        if i + 4 >= len(parts):
                            print(f"Video {video_id}, Frame {frame_id}: Incomplete player data, skipping frame")
                            break
                        try:
                            bbox = list(map(int, parts[i:i+4]))
                            action = parts[i+4]
                        except ValueError:
                            print(f"Video {video_id}, Frame {frame_id}: Invalid bounding box, skipping frame")
                            break
                        if action not in self.action_classes:
                            print(f"Video {video_id}, Frame {frame_id}: Invalid action, skipping frame")
                            break
                        players.append((action, bbox))
                        self.action_counts[action] += 1
                    self.group_counts[group_activity] += 1
                    
                    if len(players) == 0:
                        print(f"Video {video_id}, Frame {frame_id}: No valid players parsed, skipping")
                        continue
                    
                    if len(players) < 12:
                        while len(players) < 12:
                            players.append(('standing', [0, 0, 0, 0]))
                            self.action_counts['standing'] += 1
                    elif len(players) > 12:
                        players = players[:12]
                    
                    self.data.append({
                        'video_id': video_id,
                        'frame_id': frame_id,
                        'frame_ids': frame_ids,
                        'group_activity': group_activity,
                        'players': players
                    })
        
        # Compute adjusted class weights for person actions
        total_actions = sum(self.action_counts.values())
        self.action_weights = {}
        for action, count in self.action_counts.items():
            if count == 0:
                self.action_weights[action] = 1.0
            else:
                weight = total_actions / (len(self.action_classes) * count)
                if action in ['spiking', 'blocking', 'digging', 'setting', 'moving', 'jumping', 'falling']:
                    weight *= 1.5
                self.action_weights[action] = weight
        print("Action counts:", self.action_counts)
        print("Adjusted Action weights:", self.action_weights)
        
        # Compute adjusted class weights for group activities
        total_groups = sum(self.group_counts.values())
        self.group_weights = {}
        for group, count in self.group_counts.items():
            if count == 0:
                self.group_weights[group] = 1.0
            else:
                weight = total_groups / (len(self.group_classes) * count)
                # Increase weight for rare classes (example- count < 100)
                if count < 100:  # Adjust threshold based on actual counts after inspecting group_counts
                    weight *= 2  # Double the weight for rare classes
                self.group_weights[group] = weight
        print("Group counts:", self.group_counts)
        print("Adjusted Group weights:", self.group_weights)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        video_id = item['video_id']
        frame_id = item['frame_id']
        target_frame_num = int(frame_id)
        frame_ids = list(range(target_frame_num - 5, target_frame_num + 5))
        group_activity = self.group_classes[item['group_activity']]
        players = item['players']
        
        images = []
        for fid in frame_ids:
            img_path = self.videos_dir / video_id / frame_id / f"{fid}.jpg"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            if video_id in self.high_res_videos:
                img = cv2.resize(img, (1280, 720))
            
            if self.transform:
                img = self.transform(img)
            else:
                img = test_transform(img)
            
            images.append(img)
        
        images = torch.stack(images)
        bboxes = torch.tensor([player[1] for player in players], dtype=torch.float32)
        actions = torch.tensor([self.action_classes[player[0]] for player in players], dtype=torch.long)
        
        return images, bboxes, actions, group_activity
