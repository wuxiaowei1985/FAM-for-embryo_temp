import torch
import torch.nn as nn
from model.utils.shared_encoder import SharedEncoder
from model.utils.classifier import ClassificationHead

class MeanFusion(nn.Module):
    # 平均融合七个焦平面
    # Input:(B, F, D)
    # Output:(B, D)
    def __init__(self):
        super().__init__()
    def forward(self, features):
        return torch.mean(features, dim=1)

class MeanModel(nn.Module):
    def __init__(self, pretrained=True, num_classes=16):
        super().__init__()
        self.encoder = SharedEncoder(pretrained=pretrained)
        self.fusion = MeanFusion()
        self.head = ClassificationHead(in_features=self.encoder.feature_dim, hidden_features=256, num_classes=num_classes)

    def forward(self, batch):
        images = batch["images"]
        features = self.encoder(images)
        fused = self.fusion(features)
        logits = self.head(fused)
        return logits