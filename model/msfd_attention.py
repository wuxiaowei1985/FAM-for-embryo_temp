import torch.nn as nn
from model.utils.shared_encoder import SharedEncoder
from model.utils.MSFD_Attention import MSFDAttention
from model.utils.classifier import ClassificationHead
from configs import config as cfg

class MSFDAttentionModel(nn.Module):
    def __init__(self, pretrained=True, num_classes=16, dropout=0.4):
        super().__init__()
        self.encoder = SharedEncoder(pretrained=pretrained)
        self.fusion = MSFDAttention(feature_dim=self.encoder.feature_dim, depth=cfg.NUM_LAYERS, dropout=dropout)
        self.head = ClassificationHead(in_features=self.encoder.feature_dim, hidden_features=256, num_classes=num_classes, dropout=dropout)
    def forward(self, batch, return_dict=False):
        images = batch["images"]
        features = self.encoder(images)
        fused, attention = self.fusion(features)
        logits = self.head(fused)
        if return_dict:
            return {"logits": logits, "attention": attention, "feature": fused}
        return logits