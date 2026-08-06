import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Backbone(nn.Module):
    # Input:(B, 1, H, W)
    # Output:(B, 512)
    def __init__(self,pretrained = True,in_channels = 1,):
        super().__init__()
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = resnet18(weights=weights)
        if in_channels == 1:
            old_weight = model.conv1.weight.data
            model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                model.conv1.weight.data = old_weight.mean(dim=1, keepdim=True)
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
        )
        self.feature_dim = 512

    def forward(self, x):
        # x : (B,1,224,224)
        # Returns (B,512)
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

if __name__ == "__main__":
    model = ResNet18Backbone(pretrained=True)
    x = torch.randn(4, 1, 224, 224)
    y = model(x)
    num_params = sum(p.numel()for p in model.parameters())
    print(y.shape)
    print(f"{num_params / 1e6:.2f} M")