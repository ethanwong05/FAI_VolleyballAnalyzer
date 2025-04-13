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
