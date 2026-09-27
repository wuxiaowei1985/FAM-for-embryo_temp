import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleDeformableAttention3D(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, num_levels=3, num_points=4, dropout=0.2):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = (feature_dim // num_heads)
        # ΔF, ΔH, ΔW
        self.sampling_offsets = nn.Linear(feature_dim, num_heads * num_levels * num_points * 3)
        self.attention_weights = nn.Linear(feature_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.ModuleList([
            nn.Conv3d(feature_dim, feature_dim, kernel_size=1, bias=False)
            for _ in range(num_levels)
        ])
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.sampling_offsets.weight)
        nn.init.zeros_(self.sampling_offsets.bias)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)

    @staticmethod
    def _sample_level(value, locations):
        """
        value: [B,H,Dh,F,H,W]
        locations: [B,H,N,P,3]
        coordinate order: F,Y,X
        """
        B, heads, Dh, Fz, Hy, Wx = value.shape
        _, _, N, P, _ = locations.shape
        value = value.reshape(B * heads, Dh, Fz, Hy, Wx)
        # grid_sample: 5D input -> grid order X,Y,Z
        grid = locations[..., [2, 1, 0]]
        grid = (grid * 2.0 - 1.0)
        grid = grid.reshape(B * heads, N * P, 1, 1, 3)
        sampled = F.grid_sample(value, grid, mode="bilinear", padding_mode="border", align_corners=False)
        sampled = sampled.reshape(B, heads, Dh, N, P)
        return sampled.permute(0, 1, 3, 4, 2).contiguous()

    def forward(self, query, pyramid, reference_points, return_attention=False):
        B, N, D = query.shape
        if D != self.feature_dim:
            raise ValueError(f"Expected D={self.feature_dim}, "f"got {D}")
        if len(pyramid) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} levels, "f"got {len(pyramid)}")
        if reference_points.shape != (N, 3):
            raise ValueError(
                f"Expected reference_points "f"[{N},3], got "f"{tuple(reference_points.shape)}")
        # =========================================================
        # Learnable 3D offsets
        # =========================================================
        offsets = self.sampling_offsets(query)
        offsets = offsets.view(B, N, self.num_heads, self.num_levels, self.num_points, 3)
        offsets = offsets.permute(0, 2, 1, 3, 4, 5).contiguous()
        # =========================================================
        # Attention weights
        # =========================================================
        weights = self.attention_weights(query)
        weights = weights.view(B, N, self.num_heads, self.num_levels * self.num_points)
        weights = weights.permute(0, 2, 1, 3).contiguous()
        weights = torch.softmax(weights, dim=-1)
        weights = weights.view(B, self.num_heads, N, self.num_levels, self.num_points)
        # [N,3]
        ref = reference_points.to(device=query.device, dtype=query.dtype)
        ref = ref.view(1, 1, N, 1, 3)
        output = query.new_zeros(B, self.num_heads, N, self.head_dim)
        attention_maps = []
        # =========================================================
        # Multi-scale sampling
        # =========================================================
        for level, feature in enumerate(pyramid):
            if feature.ndim != 5:
                raise ValueError(f"Pyramid level {level} "f"must be [B,D,F,H,W]")
            value = self.value_proj[level](feature)
            _, _, Fz, Hy, Wx = value.shape
            value = value.view(B, self.num_heads, self.head_dim, Fz, Hy, Wx)
            # One voxel step in normalized coordinates.
            step = query.new_tensor([1.0 / max(Fz, 1), 1.0 / max(Hy, 1), 1.0 / max(Wx, 1)])
            level_offset = (torch.tanh(offsets[:, :, :, level]) * (2.0 * step))
            locations = (ref + level_offset).clamp(0.0, 1.0)
            sampled = self._sample_level(value, locations)
            weight = weights[:, :, :, level].unsqueeze(-1)
            output = output + (sampled * weight).sum(dim=3)
            if return_attention:
                attention_maps.append(weights[:, :, :, level])
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(B, N, D)
        output = self.output_proj(output)
        output = self.dropout(output)
        if return_attention:
            return output, attention_maps
        return output

class Deformable3DBlock(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, num_levels=3, num_points=4, dropout=0.2):
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

    def forward(self, query, pyramid, reference_points, return_attention=False):
        if return_attention:
            attn_out, attention = self.attn(self.norm1(query), pyramid, reference_points, return_attention=True)
        else:
            attn_out = self.attn(self.norm1(query), pyramid, reference_points, return_attention=False)
            attention = None
        query = query + attn_out
        query = query + self.ffn(self.norm2(query))
        return query, attention

class MSFDAttention(nn.Module):
    """
    Input:
        query: [B,448,512]
        pyramid:
            p2 [B,512,7,63,63]
            p3 [B,512,7,32,32]
            p4 [B,512,7,16,16]
    Output:
        refined_tokens: [B,448,512]
    """
    def __init__(self, feature_dim=512, num_heads=8, depth=2, num_levels=3, num_points=4, num_focus=7, query_h=8, query_w=8, dropout=0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.depth = depth
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_focus = num_focus
        self.query_h = query_h
        self.query_w = query_w
        self.num_tokens = (num_focus * query_h * query_w)
        self.encoder = nn.ModuleList([
            Deformable3DBlock(feature_dim=feature_dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points, dropout=dropout)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(feature_dim)
        self.score = nn.Linear(feature_dim, 1)
        self.register_buffer("reference_points", self._build_reference_points(), persistent=False)

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

    def forward(self, query_tokens, pyramid, return_attention=False):
        if query_tokens.ndim != 3:
            raise ValueError("Expected query ""[B,N,D]")
        B, N, D = query_tokens.shape
        if N != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} "f"tokens, got {N}")
        if D != self.feature_dim:
            raise ValueError(f"Expected D={self.feature_dim}, "f"got {D}")
        x = query_tokens
        attention_maps = []
        for block in self.encoder:
            x, attention = block(x, pyramid, self.reference_points, return_attention=return_attention)
            if return_attention:
                attention_maps.append(attention)
        x = self.final_norm(x)
        score = self.score(x).squeeze(-1)
        token_weight = torch.softmax(score, dim=1)
        fused = torch.sum(x * token_weight.unsqueeze(-1), dim=1)
        if return_attention:
            return fused, attention_maps
        return fused, None

if __name__ == "__main__":
    model = MSFDAttention(feature_dim=512)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: "f"{num_params / 1e6:.3f} M")