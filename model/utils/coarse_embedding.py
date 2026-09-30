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
    def __init__(self, num_classes=3, num_tokens=448, feature_dim=512):
        super().__init__()
        self.token_embedding = nn.Parameter(torch.randn(num_tokens, feature_dim))
        self.class_embedding = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.proj = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, coarse_probs):
        # [B,3] @ [3,512]
        embedding = torch.matmul(coarse_probs, self.class_embedding)
        embedding = self.token_embedding.unsqueeze(0) + embedding.unsqueeze(1)
        embedding = self.proj(embedding)
        embedding = self.norm(embedding)
        return embedding

if __name__ == "__main__":
    model = CoarseEmbedding(feature_dim=512)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params / 1e6:.3f} M")