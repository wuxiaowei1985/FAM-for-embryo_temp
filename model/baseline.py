import torch
import torch.nn as nn
from model.utils.resnet18 import ResNet18Backbone
from model.utils.classifier import ClassificationHead

class BaselineModel(nn.Module):
    # 仅使用F0
    # Input: batch["images"] : (B, 7, 1, 224, 224)
    # Output: logits : (B, num_classes)
    def __init__(self, pretrained=True, num_classes=16):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained, in_channels=1)
        self.head = ClassificationHead(in_features=self.backbone.feature_dim, hidden_features=256, num_classes=num_classes, dropout=0.5)
        # F0 对应的焦平面索引
        self.focus_index = 3

    def forward(self, batch, return_feature=False):
        images = batch["images"]
        # (B,1,H,W)
        f0 = images[:, self.focus_index]
        feature = self.backbone(f0)
        logits = self.head(feature)
        if return_feature:
            return logits, feature
        return logits

if __name__ == "__main__":
    model = BaselineModel()
    batch = {"images": torch.randn(8, 7, 1, 224, 224)}
    pred = model(batch)
    print(pred.shape)