import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleDeformableAttention3D(nn.Module):
    """
    Query:
        [B, N, D]
    Value levels:
        p2: [B, D, F, H2, W2]
        p3: [B, D, F, H3, W3]
        p4: [B, D, F, H4, W4]
    """
    def __init__(self, feature_dim=256, num_heads=8, num_levels=3, num_points=4, dropout=0.1):
        super().__init__()
        assert feature_dim % num_heads == 0
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = feature_dim // num_heads
        self.sampling_offsets = nn.Linear(feature_dim, num_heads * num_levels * num_points * 3)
        self.attention_weights = nn.Linear(feature_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.ModuleList([
            nn.Conv3d(feature_dim, feature_dim, 1, bias=False)
            for _ in range(num_levels)
        ])
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.sampling_offsets.weight)
        nn.init.zeros_(self.sampling_offsets.bias)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)

    def _sample_level(self, value, locations):
        """
        value: [B,H,Dh,F,Hs,Ws]
        locations: [B,H,N,P,3] in normalized [0,1] z/y/x order
        returns: [B,H,N,P,Dh]
        """
        B, heads, Dh, Fz, Hy, Wx = value.shape
        _, _, N, P, _ = locations.shape
        value = value.view(B * heads, Dh, Fz, Hy, Wx)
        grid = locations[..., [2, 1, 0]] * 2.0 - 1.0
        grid = grid.reshape(B * heads, N * P, 1, 1, 3)
        sampled = F.grid_sample(value, grid, mode="bilinear", padding_mode="border", align_corners=True)
        sampled = sampled.view(B, heads, Dh, N, P)
        return sampled.permute(0, 1, 3, 4, 2).contiguous()

    def forward(self, query, pyramid, reference_points):
        B, N, D = query.shape
        if len(pyramid) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} feature levels, got {len(pyramid)}")
        offsets = self.sampling_offsets(query)
        offsets = offsets.view(B, N, self.num_heads, self.num_levels, self.num_points, 3).permute(0, 2, 1, 3, 4, 5).contiguous()
        weights = self.attention_weights(query)
        weights = weights.view(B, N, self.num_heads, self.num_levels * self.num_points).permute(0, 2, 1, 3).contiguous()
        weights = torch.softmax(weights, dim=-1)
        weights = weights.view(B, self.num_heads, N, self.num_levels, self.num_points)
        ref = reference_points.to(dtype=query.dtype, device=query.device)
        ref = ref.view(1, 1, N, 1, 3)
        output = query.new_zeros(B, self.num_heads, N, self.head_dim)
        for level, feature in enumerate(pyramid):
            # feature: [B,D,F,H,W]
            value = self.value_proj[level](feature)
            _, _, Fz, Hy, Wx = value.shape
            value = value.view(B, self.num_heads, self.head_dim, Fz, Hy, Wx)
            scale = query.new_tensor([
                1.0 / max(Fz, 1),
                1.0 / max(Hy, 1),
                1.0 / max(Wx, 1),
            ])
            level_offset = torch.tanh(offsets[:, :, :, level]) * (2.0 * scale)
            locations = (ref + level_offset).clamp(0.0, 1.0)
            sampled = self._sample_level(value, locations)
            weight = weights[:, :, :, level].unsqueeze(-1)
            output = output + (sampled * weight).sum(dim=3)
        output = output.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        return self.dropout(self.output_proj(output))

class Deformable3DBlock(nn.Module):
    def __init__(self, feature_dim=256, num_heads=8, num_levels=3, num_points=4, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_dim)
        self.attn = MultiScaleDeformableAttention3D(feature_dim=feature_dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points, dropout=dropout)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )
        self.gamma1 = nn.Parameter(torch.ones(feature_dim))
        self.gamma2 = nn.Parameter(torch.ones(feature_dim))

    def forward(self, query, pyramid, reference_points):
        query = query + self.gamma1 * self.attn(self.norm1(query), pyramid, reference_points)
        query = query + self.gamma2 * self.ffn(self.norm2(query))
        return query

class MSFDAttention(nn.Module):
    """
    Input:
        query_tokens: [B, 448, 256]
        pyramid: [p2, p3, p4], each [B,256,7,H,W]
    Output:
        fused: [B,256]
        focus_weight: [B,7]
    """
    def __init__(self, feature_dim=256, num_heads=8, depth=2, num_levels=3, num_points=4, dropout=0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.depth = depth
        self.num_focus = 7
        self.query_h = 8
        self.query_w = 8
        self.focus_embedding = nn.Parameter(torch.zeros(self.num_focus, feature_dim))
        self.encoder = nn.ModuleList([
            Deformable3DBlock(
                feature_dim=feature_dim,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(feature_dim)
        self.score = nn.Linear(feature_dim, 1)
        reference_points = self._build_reference_points()
        self.register_buffer("reference_points", reference_points, persistent=False)
        nn.init.trunc_normal_(self.focus_embedding, std=0.02)

    def _build_reference_points(self):
        points = []
        for f in range(self.num_focus):
            z = (f + 0.5) / self.num_focus
            for y in range(self.query_h):
                yy = (y + 0.5) / self.query_h
                for x in range(self.query_w):
                    xx = (x + 0.5) / self.query_w
                    points.append((z, yy, xx))
        return torch.tensor(points, dtype=torch.float32)

    def forward(self, query_tokens, pyramid):
        if query_tokens.ndim != 3:
            raise ValueError(f"Expected query_tokens [B,N,D], got {tuple(query_tokens.shape)}")
        B, N, D = query_tokens.shape
        expected_n = self.num_focus * self.query_h * self.query_w
        if N != expected_n:
            raise ValueError(f"Expected {expected_n} query tokens, got {N}")
        x = query_tokens + self.focus_embedding.repeat_interleave(self.query_h * self.query_w, dim=0).unsqueeze(0)
        for block in self.encoder:
            x = block(x, pyramid, self.reference_points)
        x = self.final_norm(x)
        x_focus = x.view(B, self.num_focus, self.query_h * self.query_w, D).mean(dim=2)
        score = self.score(x_focus).squeeze(-1)
        weight = torch.softmax(score, dim=1)
        fused = torch.sum(x_focus * weight.unsqueeze(-1), dim=1)
        return fused, weight

if __name__ == "__main__":
    model = MSFDAttention()
    x = torch.randn(2, 7 * 8 * 8, 256)
    pyramid = [
        torch.randn(2, 256, 7, 63, 63),
        torch.randn(2, 256, 7, 32, 32),
        torch.randn(2, 256, 7, 16, 16),
    ]
    with torch.no_grad():
        y, w = model(x, pyramid)
    print(y.shape, w.shape, w.sum(dim=1))