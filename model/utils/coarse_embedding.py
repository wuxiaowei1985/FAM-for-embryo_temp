import torch
import torch.nn as nn

class CoarseEmbedding(nn.Module):
    """
    Input:
        coarse_probs: [B,3]
    3D semantic embedding:
        [B,512,7,7,7]
    Output:
        [B,343,512]
    """
    def __init__(self, num_classes=3, feature_dim=512, num_focus=7, query_h=7, query_w=7, dropout=0.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_focus = num_focus
        self.query_h = query_h
        self.query_w = query_w
        self.num_tokens = num_focus * query_h * query_w
        self.class_embedding = nn.Parameter(torch.empty(num_classes, feature_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, feature_dim, num_focus, query_h, query_w))
        self.proj = nn.Conv3d(feature_dim, feature_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.class_embedding, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, coarse_probs):
        if coarse_probs.ndim != 2 or coarse_probs.size(1) != self.class_embedding.size(0):
            raise ValueError(f"Expected [B,{self.class_embedding.size(0)}], got {tuple(coarse_probs.shape)}")
        B = coarse_probs.size(0)
        semantic = torch.matmul(coarse_probs, self.class_embedding).view(B, self.feature_dim, 1, 1, 1)
        embedding = semantic + self.pos_embed
        embedding = self.proj(embedding)
        embedding = embedding.flatten(2).transpose(1, 2)
        embedding = self.norm(embedding)
        return self.dropout(embedding)

if __name__ == "__main__":
    model = CoarseEmbedding()
    output = model(torch.softmax(torch.randn(2, 3), dim=1))
    print("embedding:", output.shape)