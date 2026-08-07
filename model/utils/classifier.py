import torch.nn as nn
import torch

class ClassificationHead(nn.Module):
    # Input: (B, feature_dim)
    # Output: (B, num_classes)
    def __init__(self, in_features = 512, hidden_features = 256, num_classes = 16, dropout = 0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            # 第一层：512 -> 256
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # 新增第二层：256 -> 256（深度增加，提高非线性拟合能力）
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # 输出层：256 -> num_classes
            nn.Linear(hidden_features, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

if __name__ == "__main__":
    head = ClassificationHead()
    x = torch.randn(4,512)
    y = head(x)
    print(y.shape)