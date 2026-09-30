import torch
import torch.nn as nn

class FocusAttentionBlock(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_dim)
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        identity = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn( x_norm, x_norm, x_norm, need_weights=False)
        x = identity + attn_out
        identity = x
        x = identity + self.ffn(self.norm2(x))
        return x

class FocusAttention(nn.Module):
    """
    Input: [B,L,D]
    Current: [B,448,512]
    Output sequence: [B,448,512]
    Coarse pooled feature: [B,512]
    """
    def __init__(self,feature_dim=512, num_heads=8, depth=2, num_focus=7, patches_per_focus=64, dropout=0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_focus = num_focus
        self.patches_per_focus = patches_per_focus
        self.num_tokens = (num_focus * patches_per_focus)
        self.focus_embedding = nn.Parameter(torch.randn(num_focus * patches_per_focus, feature_dim))
        self.encoder = nn.ModuleList([
            FocusAttentionBlock(feature_dim=feature_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(feature_dim)
        self.score = nn.Linear(feature_dim, 1)

    def _add_focus_embedding(self, x):
        B, L, D = x.shape
        if L != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} tokens, "f"got {L}")
        if D != self.feature_dim:
            raise ValueError(f"Expected feature dimension "f"{self.feature_dim}, got {D}")
        embedding = self.focus_embedding.unsqueeze(0)
        return x + embedding

    def encode(self, x):
        x = self._add_focus_embedding(x)
        for layer in self.encoder:
            x = layer(x)
        return self.final_norm(x)

    def fuse(self, x):
        score = self.score(x).squeeze(-1)
        # [B,L]
        token_weight = torch.softmax(score, dim=1)
        # [B,D]
        fused = torch.sum(x * token_weight.unsqueeze(-1), dim=1)
        # [B,7]
        focus_weight = token_weight.view(x.size(0), self.num_focus, self.patches_per_focus).sum(dim=-1)
        return fused, token_weight, focus_weight

    def forward(self, x, return_sequence=False):
        sequence = self.encode(x)
        fused, token_weight, focus_weight = self.fuse(sequence)
        if return_sequence:
            return sequence, fused, token_weight, focus_weight
        return fused, token_weight, focus_weight

if __name__ == "__main__":
    model = FocusAttention(depth=2)
    x = torch.randn(2, 7 * 8 * 8, 512)
    with torch.no_grad():
        sequence, fused, token_weight, focus_weight = model(x, return_sequence=True)
    print("sequence:", sequence.shape)
    print("fused:", fused.shape)
    print("token_weight:", token_weight.shape)
    print("focus_weight:", focus_weight.shape)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: "f"{num_params / 1e6:.3f} M")