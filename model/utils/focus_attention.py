import torch
import torch.nn as nn

class FocusAttentionBlock(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, dropout=0.2, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_dim)
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.hidden_dim = int(feature_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, feature_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x

class FocusAttention(nn.Module):
    """
    Input: [B,343,512]
    3D positional embedding: [1,512,7,7,7]
    Output sequence: [B,343,512]
    Fused feature: [B,512]
    Focus weights: [B,7]
    """
    def __init__(self, feature_dim=512, num_heads=8, depth=2, num_focus=7, query_h=7, query_w=7, dropout=0.2):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.feature_dim = feature_dim
        self.num_focus = num_focus
        self.query_h = query_h
        self.query_w = query_w
        self.num_tokens = num_focus * query_h * query_w
        self.pos_embed = nn.Parameter(torch.zeros(1, feature_dim, num_focus, query_h, query_w))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.encoder = nn.ModuleList([
            FocusAttentionBlock(feature_dim=feature_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(feature_dim)
        self.score = nn.Linear(feature_dim, 1)

    def _add_3d_position_embedding(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.num_tokens, self.feature_dim):
            raise ValueError(f"Expected [B,{self.num_tokens},{self.feature_dim}], got {tuple(x.shape)}")
        B = x.size(0)
        position = self.pos_embed.flatten(2).transpose(1, 2)
        return x + position.expand(B, -1, -1)

    def encode(self, x):
        x = self._add_3d_position_embedding(x)
        for layer in self.encoder:
            x = layer(x)
        return self.final_norm(x)

    def fuse(self, x):
        token_weight = torch.softmax(self.score(x).squeeze(-1), dim=1)
        fused = torch.sum(x * token_weight.unsqueeze(-1), dim=1)
        focus_weight = token_weight.reshape(x.size(0), self.num_focus, self.query_h * self.query_w).sum(dim=-1)
        return fused, token_weight, focus_weight

    def forward(self, x, return_sequence=False):
        sequence = self.encode(x)
        fused, token_weight, focus_weight = self.fuse(sequence)
        if return_sequence:
            return sequence, fused, token_weight, focus_weight
        return fused, token_weight, focus_weight

if __name__ == "__main__":
    model = FocusAttention()
    x = torch.randn(2, 343, 512)
    with torch.no_grad():
        sequence, fused, token_weight, focus_weight = model(x, return_sequence=True)
    print("sequence:", sequence.shape)
    print("fused:", fused.shape)
    print("token_weight:", token_weight.shape)
    print("focus_weight:", focus_weight.shape)