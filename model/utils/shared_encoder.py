import torch
import torch.nn as nn
from model.utils.resnet18 import ResNet18Backbone

class SharedEncoder(nn.Module):
    # Input:(B,7,1,H,W)
    # Output: (B,7,512)
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = ResNet18Backbone(pretrained=pretrained, in_channels=1)
        self.feature_dim = self.encoder.feature_dim

    def forward(self, images):
        B, F, C, H, W = images.shape
        # (B*F,1,H,W)
        images = images.reshape(B * F, C, H,W)
        features = self.encoder(images)
        # (B,F,512)
        features = features.reshape(B, F, self.feature_dim)
        return features


if "__main__" == __name__:
    import torch
    from model.utils.shared_encoder import SharedEncoder
    model = SharedEncoder()
    x = torch.randn(4,7,1,224,224)
    feature = model(x)
    print(feature.shape)