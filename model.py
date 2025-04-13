import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Focal Loss class
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# PersonActionClassifier
class PersonActionClassifier(nn.Module):
    def __init__(self, feature_dim=4096, hidden_dim=512, num_actions=9, dropout_rate=0.6):
        super(PersonActionClassifier, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # AlexNet backbone
        self.backbone = models.alexnet(pretrained=True)
        self.backbone.classifier = nn.Sequential(*list(self.backbone.classifier.children())[:-1])
        
        # Freeze early layers
        for name, param in self.backbone.named_parameters():
            if any(layer in name for layer in ["features.5", "features.6", "features.7", "features.8", "features.9", "features.10", "features.11", "features.12"]):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # LSTM for temporal modeling
        lstm_input_size = feature_dim + 4  # Features + normalized bbox coordinates
        print(f"Initializing Person LSTM with input_size={lstm_input_size}")
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.5)
        
        # Batch normalization and dropout
        self.bn = nn.BatchNorm1d(12)  # 12 players
        self.dropout = nn.Dropout(dropout_rate)
        
        # Action classifier
        self.fc = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, images, bboxes):
        # images: [B, T, C, H, W], bboxes: [B, 12, 4]
        B, T, C, H, W = images.shape
        num_players = bboxes.shape[1]
        
        # Normalize bounding box coordinates
        bboxes_normalized = bboxes.clone()
        bboxes_normalized[:, :, 0] /= 1280  # x / width
        bboxes_normalized[:, :, 1] /= 720   # y / height
        bboxes_normalized[:, :, 2] /= 1280  # w / width
        bboxes_normalized[:, :, 3] /= 720   # h / height
        
        # Process images
        images_flat = images.view(B * T, C, H, W)
        features = self.backbone(images_flat).view(B, T, -1)  # [B, T, 4096]
        
        # Expand features to all players and join with bbox coordinates
        features_expanded = features.unsqueeze(1).repeat(1, num_players, 1, 1)  # [B, 12, T, 4096]
        bboxes_expanded = bboxes_normalized.unsqueeze(2).repeat(1, 1, T, 1)  # [B, 12, T, 4]
        
        # Combine features and bbox for LSTM input
        player_input = torch.cat([features_expanded, bboxes_expanded], dim=-1)  # [B, 12, T, 4100]
        player_input = player_input.view(B * num_players, T, -1)  # [B*12, T, 4100]
        
        # LSTM processes each player's sequence
        lstm_out, _ = self.lstm(player_input)  # [B*12, T, hidden_dim]
        lstm_out = lstm_out.view(B, num_players, T, -1)  # [B, 12, T, 512]
        
        # Hybrid approach - combine last timestep with mean pooling
        last_timestep = lstm_out[:, :, -1, :]  # [B, 12, 512]
        mean_features = lstm_out.mean(dim=2)  # [B, 12, 512]
        hybrid_features = (last_timestep + mean_features) / 2  # Simple averaging
        
        # Apply batch normalization and dropout
        person_features = self.bn(hybrid_features)
        person_features = self.dropout(person_features)
        
        # Action classification
        action_logits = self.fc(person_features.reshape(B * num_players, -1))  # [B*12, num_actions]
        action_logits = action_logits.view(B, num_players, -1)  # [B, 12, num_actions]
        
        return lstm_out, action_logits

# SelfAttention class
class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.feature_dim = feature_dim
        
    def forward(self, x):
        # x: [B, 12, feature_dim]
        scale = torch.sqrt(torch.FloatTensor([self.feature_dim])).to(x.device)
        
        Q = self.query(x)  # [B, 12, feature_dim]
        K = self.key(x)    # [B, 12, feature_dim]
        V = self.value(x)  # [B, 12, feature_dim]
        
        # Attention scores
        energy = torch.matmul(Q, K.permute(0, 2, 1)) / scale  # [B, 12, 12]
        attention = F.softmax(energy, dim=-1)  # [B, 12, 12]
        
        # Apply attention
        x = torch.matmul(attention, V)  # [B, 12, feature_dim]
        
        return x

# GroupActivityClassifier
class GroupActivityClassifier(nn.Module):
    def __init__(self, person_dim=512, hidden_dim=512, num_groups=8, dropout_rate=0.6):  
        super(GroupActivityClassifier, self).__init__()
        self.person_dim = person_dim
        self.hidden_dim = hidden_dim
        
        # Self-attention for player relationships
        self.attention = SelfAttention(person_dim)
        
        # Group-level LSTM
        print(f"Initializing Group LSTM with input_size={person_dim}")
        self.lstm = nn.LSTM(input_size=person_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        
        # Batch normalization and dropout
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Group activity classifier
        self.fc = nn.Linear(hidden_dim, num_groups)
    
    def forward(self, person_features):
        # person_features: [B, 12, T, person_dim]
        B, num_players, T, D = person_features.shape
        
        # Take the last timestep for attention and LSTM
        last_timestep_features = person_features[:, :, -1, :]  # [B, 12, person_dim]
        
        # Apply self-attention
        attended_features = self.attention(last_timestep_features)  # [B, 12, person_dim]
        
        # Group-level LSTM processes the 12 players as a sequence
        lstm_out, _ = self.lstm(attended_features)  # [B, 12, hidden_dim]
        
        # Take the last timestep and apply batch normalization
        last_timestep = lstm_out[:, -1, :]  # [B, hidden_dim]
        last_timestep = self.bn(last_timestep)  # [B, hidden_dim]
        
        # Apply dropout and final classifier
        out = self.dropout(last_timestep)  # [B, hidden_dim]
        group_logits = self.fc(out)  # [B, num_groups]
        
        return group_logits

# STGCN class
class STGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STGCN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x, A):
        # x: [B, N, C], A: [B, N, N] adjacency matrix
        B, N, C = x.shape
        x = x.unsqueeze(-1)  # [B, N, C, 1]
        x = x.permute(0, 2, 1, 3)  # [B, C, N, 1]
        
        # Graph convolution
        out = []
        for i in range(B):
            # Normalize adjacency matrix
            A_norm = A[i] / (A[i].sum(dim=1, keepdim=True) + 1e-6)
            xi = x[i]  # [C, N, 1]
            x_graph = torch.matmul(xi.squeeze(-1), A_norm).unsqueeze(-1)  # [C, N, 1]
            out.append(x_graph)
        
        out = torch.stack(out)  # [B, C, N, 1]
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        out = out.permute(0, 2, 1, 3).squeeze(-1)  # [B, N, C]
        
        return out


# GraphFeatureExtractor
class GraphFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=512):
        super(GraphFeatureExtractor, self).__init__()
        self.gcn1 = STGCN(feature_dim, hidden_dim)
        self.gcn2 = STGCN(hidden_dim, hidden_dim)
        # Reuse tensors
        self.A = None
        self.centers = None
        
    def forward(self, person_features, bboxes):
        # Extract features for graph processing
        B, N, T, D = person_features.shape
        features_last = person_features[:, :, -1, :]  # [B, 12, feature_dim]
        
        # Create adjacency matrix
        if self.A is None or self.A.shape[0] != B:
            self.A = torch.zeros(B, N, N, device=features_last.device)
            self.centers = torch.zeros(B, N, 2, device=bboxes.device)
        else:
            self.A.zero_()  # Clear previous values
        
        # Compute centers of bounding boxes
        self.centers[:, :, 0] = bboxes[:, :, 0] + bboxes[:, :, 2] / 2  # x center
        self.centers[:, :, 1] = bboxes[:, :, 1] + bboxes[:, :, 3] / 2  # y center
        
        # Create edges based on proximity
        for b in range(B):
            for i in range(N):
                for j in range(N):
                    if i != j:
                        dist = torch.norm(self.centers[b, i] - self.centers[b, j])
                        # threshold 150 for broader player connections
                        if dist < 150:  
                            self.A[b, i, j] = 1.0
        
        # Ensure self-loops
        self.A = self.A + torch.eye(N, device=self.A.device).unsqueeze(0).repeat(B, 1, 1)
        
        # Apply GCN layers with error handling
        try:
            x = self.gcn1(features_last, self.A)
            x = self.gcn2(x, self.A)
            enhanced_features = x
            
            # Update features in the temporal sequence
            enhanced_features_temporal = person_features.clone()
            enhanced_features_temporal[:, :, -1, :] = enhanced_features
            
            return enhanced_features_temporal
            
        except Exception as e:
            print(f"Warning: Graph feature extraction failed: {e}, falling back to original features")
            return person_features  # Fallback if graph enhancement fails


# RallyPredictor
class RallyPredictor(nn.Module):
    def __init__(self, person_classifier=None, group_classifier=None, hidden_dim=512):
        super(RallyPredictor, self).__init__()
        
        if person_classifier is None:
            self.person_classifier = PersonActionClassifier(feature_dim=4096, hidden_dim=hidden_dim, 
                                                          num_actions=9, dropout_rate=0.7)
        else:
            self.person_classifier = person_classifier
            
        if group_classifier is None:
            self.group_classifier = GroupActivityClassifier(person_dim=hidden_dim, hidden_dim=hidden_dim, 
                                                         num_groups=8, dropout_rate=0.7)
        else:
            self.group_classifier = group_classifier
            
        self.graph_feature_extractor = GraphFeatureExtractor(hidden_dim, hidden_dim)
        
        # Action weights for rally prediction heuristic
        self.action_weights = {
            'spiking': 1.5,
            'setting': 1.0,
            'blocking': 0.75,
            'digging': 0.75,
            'standing': 0.0,
            'moving': 0.0,
            'waiting': 0.0,
            'jumping': 0.0,
            'falling': 0.0
        }
        
        # Will be initialized during forward pass
        self.action_classes = None
        
        # Neural network for rally prediction
        self.rally_fc = nn.Sequential(
            nn.Linear(hidden_dim * 12, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def compute_heuristic_probs(self, action_logits, bboxes):
        B = action_logits.shape[0]
        left_scores = []
        right_scores = []
        
        action_probs = torch.softmax(action_logits, dim=-1)
        action_preds = torch.argmax(action_probs, dim=-1)
        
        for b in range(B):
            left_action_score = 0.0
            right_action_score = 0.0
            
            # Left team (players 0-5)
            for p in range(6):
                action_idx = action_preds[b, p].item()
                action = self.action_classes[action_idx]
                left_action_score += self.action_weights[action]
            
            # Right team (players 6-11)
            for p in range(6, 12):
                action_idx = action_preds[b, p].item()
                action = self.action_classes[action_idx]
                right_action_score += self.action_weights[action]
            
            # Compute position scores
            left_x = bboxes[b, :6, 0].mean().item()
            right_x = bboxes[b, 6:, 0].mean().item()
            net_position = 1280 / 2
            left_position_score = 0.5 if abs(left_x - net_position) < abs(right_x - net_position) else 0.0
            right_position_score = 0.5 if abs(right_x - net_position) < abs(left_x - net_position) else 0.0
            
            left_total = left_action_score + left_position_score
            right_total = right_action_score + right_position_score
            
            total = left_total + right_total
            if total == 0:
                left_prob = 0.5
                right_prob = 0.5
            else:
                left_prob = left_total / total
                right_prob = right_total / total
            
            left_scores.append(left_prob)
            right_scores.append(right_prob)
        
        return torch.stack([torch.tensor(left_scores), torch.tensor(right_scores)], dim=1).to(action_logits.device)

    def forward(self, images, bboxes):
        # Initialize action_classes mapping from input data
        if self.action_classes is None:
            self.action_classes = {
                0: 'waiting', 1: 'setting', 2: 'digging', 3: 'falling', 4: 'spiking',
                5: 'blocking', 6: 'jumping', 7: 'moving', 8: 'standing'
            }
            
        # Get person features and action logits
        person_features, action_logits = self.person_classifier(images, bboxes)
    
        # Use graph to enhance features with spatial relationships
        enhanced_features = self.graph_feature_extractor(person_features, bboxes)
    
        # Use both last timestep and mean pooled features
        last_timestep = enhanced_features[:, :, -1, :]  # [B, 12, 512]
        mean_pooled = enhanced_features.mean(dim=2)  # [B, 12, 512]
        combined_features = (last_timestep + mean_pooled) / 2  # Simple averaging
    
        # Group classifier
        group_logits = self.group_classifier(enhanced_features)
    
        # Rally prediction heuristic
        heuristic_probs = self.compute_heuristic_probs(action_logits, bboxes)
    
        # Neural network for rally prediction
        B = person_features.shape[0]
        last_timestep_features = person_features[:, :, -1, :].reshape(B, -1)  # [B, 12*512]
        rally_logits = self.rally_fc(last_timestep_features)
        rally_probs = torch.softmax(rally_logits, dim=-1)
    
        return person_features, action_logits, group_logits, rally_probs, heuristic_probs
        
