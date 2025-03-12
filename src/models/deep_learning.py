import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Query, Key, Value projections
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        value = self.value(x).view(batch_size, -1, H * W)
        
        # Attention map
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Output
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class MultiTaskMedicalNet(nn.Module):
    def __init__(self, num_classes_classification=2, num_classes_segmentation=1):
        super(MultiTaskMedicalNet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Attention mechanism
        self.attention = AttentionBlock(256)
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes_classification)
        )
        
        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes_segmentation, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Attention
        x = self.attention(x)
        
        # Task-specific outputs
        classification_output = self.classification_head(x)
        segmentation_output = self.segmentation_head(x)
        
        return classification_output, segmentation_output

class FeatureFusion(nn.Module):
    def __init__(self, img_features, tabular_features):
        super(FeatureFusion, self).__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(img_features + tabular_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
    def forward(self, img_features, tabular_features):
        combined = torch.cat((img_features, tabular_features), dim=1)
        return self.fusion(combined)

def create_model(num_classes_classification=2, num_classes_segmentation=1):
    """Factory function to create a new model instance"""
    model = MultiTaskMedicalNet(
        num_classes_classification=num_classes_classification,
        num_classes_segmentation=num_classes_segmentation
    )
    return model

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim // 8)
        self.key = nn.Linear(hidden_dim, hidden_dim // 8)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()
        
        # Query, Key, Value projections
        query = self.query(x)  # (batch_size, seq_len, hidden_dim//8)
        key = self.key(x)      # (batch_size, seq_len, hidden_dim//8)
        value = self.value(x)  # (batch_size, seq_len, hidden_dim)
        
        # Attention scores
        attention = torch.bmm(query, key.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        attention = F.softmax(attention, dim=-1)
        
        # Output
        out = torch.bmm(attention, value)  # (batch_size, seq_len, hidden_dim)
        
        return self.gamma * out + x

class LSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        return self.dropout(x)

class MultiTaskMarketNet(nn.Module):
    def __init__(self, input_dim, seq_length, num_classes_direction=3, num_classes_pattern=5):
        super(MultiTaskMarketNet, self).__init__()
        
        self.hidden_dim = 128
        
        # Price data encoder
        self.price_encoder = nn.Sequential(
            LSTMBlock(input_dim, self.hidden_dim),
            LSTMBlock(self.hidden_dim, self.hidden_dim)
        )
        
        # Technical indicator encoder
        self.technical_encoder = nn.Sequential(
            LSTMBlock(input_dim * 2, self.hidden_dim),  # More features for technical indicators
            LSTMBlock(self.hidden_dim, self.hidden_dim)
        )
        
        # Attention mechanisms
        self.price_attention = TemporalAttention(self.hidden_dim)
        self.technical_attention = TemporalAttention(self.hidden_dim)
        
        # Price direction prediction head
        self.direction_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, num_classes_direction)  # Up, Down, Sideways
        )
        
        # Pattern recognition head
        self.pattern_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, num_classes_pattern)  # Different chart patterns
        )
        
        # Volatility prediction head (regression)
        self.volatility_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()  # Normalize volatility to [0,1]
        )

    def forward(self, price_data, technical_data):
        # Process price data
        price_features = self.price_encoder(price_data)
        price_features = self.price_attention(price_features)
        
        # Process technical indicators
        tech_features = self.technical_encoder(technical_data)
        tech_features = self.technical_attention(tech_features)
        
        # Combine features
        combined_features = torch.cat(
            (price_features[:, -1, :], tech_features[:, -1, :]), 
            dim=1
        )
        
        # Task-specific outputs
        direction_pred = self.direction_head(combined_features)
        pattern_pred = self.pattern_head(combined_features)
        volatility_pred = self.volatility_head(combined_features)
        
        return direction_pred, pattern_pred, volatility_pred

class MarketFeatureFusion(nn.Module):
    def __init__(self, price_features, technical_features, fundamental_features):
        super(MarketFeatureFusion, self).__init__()
        
        total_features = price_features + technical_features + fundamental_features
        
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
    def forward(self, price_features, technical_features, fundamental_features):
        combined = torch.cat((price_features, technical_features, fundamental_features), dim=1)
        return self.fusion(combined)

def create_market_model(input_dim=5, seq_length=60, num_classes_direction=3, num_classes_pattern=5):
    """Factory function to create a new market model instance"""
    model = MultiTaskMarketNet(
        input_dim=input_dim,
        seq_length=seq_length,
        num_classes_direction=num_classes_direction,
        num_classes_pattern=num_classes_pattern
    )
    return model 