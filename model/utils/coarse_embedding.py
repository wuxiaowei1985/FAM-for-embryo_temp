import torch
import torch.nn as nn

class CoarseEmbedding(nn.Module):
    """
    将三个粗粒度发育阶段的概率
    转换成连续的 coarse semantic embedding。
    Input:
        coarse_probs: [B, 3]
    Output:
        embedding: [B, feature_dim]
    """
    def __init__(self, num_classes=3, feature_dim=512):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.proj = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, coarse_probs):
        # [B,3] @ [3,512]
        embedding = torch.matmul(coarse_probs, self.embedding)
        embedding = self.proj(embedding)
        embedding = self.norm(embedding)
        return embedding