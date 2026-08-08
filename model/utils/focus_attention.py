import torch
import torch.nn as nn

class FocusAttentionBlock(nn.Module):
    # 一个 Focus Transformer Encoder Block
    def __init__(self, feature_dim=512, num_heads=8, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_dim)
        self.gamma1 = nn.Parameter(torch.ones(feature_dim))
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.gamma2 = nn.Parameter(torch.ones(feature_dim))
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        attn_out, _ = self.attn( x, x, x, need_weights=False)
        x = identity + self.gamma1 * attn_out
        identity = x
        x = self.norm2(x)
        x = identity + self.gamma2 * self.ffn(x)
        return x

class FocusAttentionEncoder(nn.Module):
    def __init__(self, depth=4, feature_dim=512, num_heads=8, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList([FocusAttentionBlock(feature_dim=feature_dim, num_heads=num_heads, dropout=dropout) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class FocusAttention(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, depth=4, dropout=0.2):
        super().__init__()
        self.focus_embedding = nn.Parameter(torch.randn(7, feature_dim))
        self.encoder = FocusAttentionEncoder(depth=depth, feature_dim=feature_dim, num_heads=num_heads, dropout=dropout)
        self.final_norm = nn.LayerNorm(feature_dim)
        self.score = nn.Linear(feature_dim, 1)

    def forward(self, x):
        x = x + self.focus_embedding.unsqueeze(0)
        x = self.encoder(x)
        x = self.final_norm(x)
        score = self.score(x).squeeze(-1)
        weight = torch.softmax(score, dim=1)
        fused = torch.sum(x * weight.unsqueeze(-1), dim=1)
        return fused, weight

if __name__ == "__main__":
    model = FocusAttention()
    x = torch.randn(8, 7, 512)
    feature, weight = model(x)
    print(feature.shape)
    print(weight.shape)
    print(weight.sum(dim=1))