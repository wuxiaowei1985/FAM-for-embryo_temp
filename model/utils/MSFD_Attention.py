import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleFocusDeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Multi-Head Attention
    for 1D focus-plane sequence.
    输入:
        x: [B, F, D]
           B = batch size
           F = focus planes, 当前为 7
           D = feature dimension, 当前为 512
    输出:
        out: [B, F, D]
    这里将 7 个焦平面视为沿 Z 轴排列的 1D reference points，
    并构造多个 focus scales。
    Scale 1:
        7 个焦平面
    Scale 2:
        4 个粗粒度焦平面
    Scale 3:
        2 个更粗粒度焦平面
    """
    def __init__(self, feature_dim=512, num_heads=8, num_levels=3, num_points=4, dropout=0.1):
        super().__init__()
        assert feature_dim % num_heads == 0, \
            "feature_dim 必须能够被 num_heads 整除"
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = feature_dim // num_heads
        # ---------------------------------------------------------
        # Query / Value projection
        # ---------------------------------------------------------
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        # ---------------------------------------------------------
        # 预测每一个 query 的 sampling offsets
        # 每个 query:
        #   num_heads
        #   × num_levels
        #   × num_points
        # 每个 sampling point 只有一个 1D offset
        # ---------------------------------------------------------
        self.sampling_offsets = nn.Linear(feature_dim, num_heads * num_levels * num_points)
        # ---------------------------------------------------------
        # 预测 attention weights
        # ---------------------------------------------------------
        self.attention_weights = nn.Linear(feature_dim, num_heads * num_levels * num_points)
        # ---------------------------------------------------------
        # LayerNorm
        # ---------------------------------------------------------
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        # ---------------------------------------------------------
        # 初始化
        # offsets 初始为 0
        # 让模型刚开始训练时先接近普通 attention，
        # 避免一开始 sampling 位置完全随机。
        # ---------------------------------------------------------
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

    def _build_multi_scale_features(self, x):
        """
        构造多尺度焦平面特征。
        输入:
            x: [B, N, D]
            B = batch size
            N = focus planes
            D = feature dimension
        返回:
            multi_scale:
                [
                    [B, 7, 512],
                    [B, 4, 512],
                    [B, 2, 512]
                ]
        """
        B, num_focus, D = x.shape
        # ---------------------------------------------------------
        # Scale 1
        # 原始焦平面特征
        # [B, 7, D]
        # ---------------------------------------------------------
        scale1 = x
        # ---------------------------------------------------------
        # Scale 2
        # 7 → 4
        # 在 focus dimension 上进行自适应平均池化
        # [B,7,D]
        #     ↓ transpose
        # [B,D,7]
        #     ↓ adaptive_avg_pool1d
        # [B,D,4]
        #     ↓ transpose
        # [B,4,D]
        # ---------------------------------------------------------
        scale2 = F.adaptive_avg_pool1d(x.transpose(1, 2), output_size=4).transpose(1, 2)
        # ---------------------------------------------------------
        # Scale 3
        # 7 → 2
        # [B,7,D]
        #     ↓
        # [B,D,7]
        #     ↓
        # [B,D,2]
        #     ↓
        # [B,2,D]
        # ---------------------------------------------------------
        scale3 = F.adaptive_avg_pool1d(x.transpose(1, 2), output_size=2).transpose(1, 2)
        return [scale1, scale2, scale3]

    def _sample_1d(self, value, positions):
        """
        对 1D focus sequence 进行可变形采样。
        value:
            [B, H, L, D]
        positions:
            [B, H, Q, P]
        返回:
            sampled:
                [B, H, Q, P, D]
        """
        B, H, L, D = value.shape
        _, _, Q, P = positions.shape
        # ---------------------------------------------------------
        # grid_sample 需要二维输入。
        # 将 focus dimension L 看成 W，
        # 高度维度设成 1。
        # [B,H,L,D]
        # → [B*H, D, 1, L]
        # ---------------------------------------------------------
        value = value.permute(0, 1, 3, 2).contiguous()
        value = value.view( B * H, D, 1, L)
        # ---------------------------------------------------------
        # positions:
        # [0, L-1]
        # 转换到:
        # [-1, 1]
        # ---------------------------------------------------------
        normalized = positions / max(L - 1, 1)
        normalized = normalized * 2.0 - 1.0
        # grid:
        # [B*H, Q, P, 2]
        # y 坐标固定为 0
        # x 坐标使用 sampling position
        # ---------------------------------------------------------
        y = torch.zeros_like(normalized)
        grid = torch.stack([normalized, y], dim=-1)
        grid = grid.view(B * H, Q, P, 2)
        sampled = F.grid_sample(value, grid, mode="bilinear", padding_mode="border", align_corners=True)
        # ---------------------------------------------------------
        # sampled:
        # [B*H, D, Q, P] → [B,H,Q,P,D]
        # ---------------------------------------------------------
        sampled = sampled.view( B, H, D, Q, P)
        sampled = sampled.permute(0, 1, 3, 4, 2).contiguous()
        return sampled

    def forward(self, x):
        """
        Args:
            x:
                [B, N, D]
            B = batch size
            N = number of focus planes
            D = feature dimension
        Returns:
            output:
                [B, N, D]
        """
        B, num_focus, D = x.shape
        # ---------------------------------------------------------
        # Pre-normalization
        # ---------------------------------------------------------
        identity = x
        x = self.norm(x)
        # ---------------------------------------------------------
        # Query
        # ---------------------------------------------------------
        query = self.q_proj(x)
        query = query.view(B, num_focus, self.num_heads, self.head_dim)
        query = query.permute(0, 2, 1, 3).contiguous()
        # ---------------------------------------------------------
        # Value
        # ---------------------------------------------------------
        value = self.v_proj(x)
        # ---------------------------------------------------------
        # Multi-scale features
        # ---------------------------------------------------------
        multi_scale = self._build_multi_scale_features(value)
        # ---------------------------------------------------------
        # Sampling offsets
        # [B,N,D]→[B,N,H,L,P]
        # ---------------------------------------------------------
        offsets = self.sampling_offsets(x)
        offsets = offsets.view(B, num_focus, self.num_heads, self.num_levels, self.num_points)
        offsets = offsets.permute(0, 2, 1, 3, 4).contiguous()
        # ---------------------------------------------------------
        # Attention weights
        # ---------------------------------------------------------
        attention = self.attention_weights(x)
        attention = attention.view(B, num_focus, self.num_heads, self.num_levels, self.num_points)
        attention = attention.permute(0, 2, 1, 3, 4).contiguous()
        # ---------------------------------------------------------
        # Softmax over:
        # level × sampling point
        # ---------------------------------------------------------
        attention = attention.view( B, self.num_heads, num_focus, self.num_levels * self.num_points)
        attention = torch.softmax(attention, dim=-1)
        attention = attention.view(B, self.num_heads, num_focus, self.num_levels, self.num_points)
        # ---------------------------------------------------------
        # Reference positions
        # ---------------------------------------------------------
        reference_positions = []
        for level, feature in enumerate(multi_scale):
            L = feature.shape[1]
            if num_focus == 1:
                ref = torch.zeros(B, self.num_heads, num_focus, device=x.device, dtype=x.dtype)
            else:
                base = torch.arange(num_focus, device=x.device, dtype=x.dtype)
                # [0, N-1]
                base = base / (num_focus - 1)
                # 映射到当前 level
                # [0, L-1]
                base = base * (L - 1)
                ref = base.view(1, 1, num_focus).expand(B, self.num_heads, num_focus)
            reference_positions.append(ref)
        # ---------------------------------------------------------
        # Output
        # ---------------------------------------------------------
        output = torch.zeros(B, self.num_heads, num_focus, self.head_dim, device=x.device, dtype=x.dtype)
        # ---------------------------------------------------------
        # Multi-scale deformable sampling
        # ---------------------------------------------------------
        for level in range(self.num_levels):
            feature = multi_scale[level]
            L = feature.shape[1]
            reference = reference_positions[level]
            # -----------------------------------------------------
            # 当前 level 的 offset
            # [B,H,N,P]
            # -----------------------------------------------------
            offset = offsets[:, :, :, level, :]
            # 限制 sampling offset 范围
            offset = torch.tanh(offset)
            # 根据不同 scale 调整 offset
            offset = (offset * max(L - 1, 1) / max(num_focus - 1, 1))
            # -----------------------------------------------------
            # Sampling locations
            # [B,H,N,P]
            # -----------------------------------------------------
            sampling_locations = (reference.unsqueeze(-1) + offset)
            # -----------------------------------------------------
            # 当前 level 的 value
            # [B,L,D]→[B,H,L,D_head]
            # -----------------------------------------------------
            value_level = feature.view(B, L, self.num_heads, self.head_dim)
            value_level = value_level.permute(0, 2, 1, 3).contiguous()
            # -----------------------------------------------------
            # Deformable sampling
            # [B,H,N,P,D]
            # -----------------------------------------------------
            sampled = self._sample_1d( value_level, sampling_locations)
            # -----------------------------------------------------
            # Attention weight
            # [B,H,N,P]
            # -----------------------------------------------------
            weight = attention[:, :, :, level, :]
            weight = weight.unsqueeze(-1)
            # -----------------------------------------------------
            # Weighted aggregation
            # [B,H,N,P,D]→[B,H,N,D]
            # -----------------------------------------------------
            output = output + (sampled * weight).sum(dim=3)
        # ---------------------------------------------------------
        # Merge heads
        # [B,H,N,D_head]→[B,N,D]
        # ---------------------------------------------------------
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(B, num_focus, D)
        # --------------------------------------------------------
        # Output projection
        # ---------------------------------------------------------
        output = self.out_proj(output)
        output = self.dropout(output)
        # ---------------------------------------------------------
        # Residual
        # ---------------------------------------------------------
        output = identity + output
        return output

class MultiScaleFocusDeformableAttentionBlock(nn.Module):
    """
    Multi-Scale Deformable Focus Attention Block
    """
    def __init__(self, feature_dim=512, num_heads=8, num_levels=3, num_points=4, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.attn = MultiScaleFocusDeformableAttention(feature_dim=feature_dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )
        # LayerScale
        self.gamma1 = nn.Parameter(torch.ones(feature_dim))
        self.gamma2 = nn.Parameter(torch.ones(feature_dim))
    def forward(self, x):
        # ---------------------------------------------------------
        # Deformable Attention
        # ---------------------------------------------------------
        attn_out = self.attn(self.norm(x))
        x = x + self.gamma1 * attn_out
        # ---------------------------------------------------------
        # FFN
        # ---------------------------------------------------------
        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + self.gamma2 * ffn_out
        return x

class MultiScaleFocusDeformableAttentionEncoder(nn.Module):
    """
    Stack multiple deformable attention blocks.
    """
    def __init__(self, depth=4, feature_dim=512, num_heads=8, num_levels=3, num_points=4, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList([MultiScaleFocusDeformableAttentionBlock(feature_dim=feature_dim,  num_heads=num_heads,  num_levels=num_levels,  num_points=num_points,  dropout=dropout) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MSFDAttention(nn.Module):
    """
    Multi-Scale Deformable Focus Attention
    Input:
        [B,7,512]
    Output:
        fused:
            [B,512]
        weight:
            [B,7]
    """
    def __init__(self, feature_dim=512, num_heads=8, depth=4, num_levels=3, num_points=4, dropout=0.2):
        super().__init__()
        # ---------------------------------------------------------
        # 七个焦平面的可学习位置编码
        # ---------------------------------------------------------
        self.focus_embedding = nn.Parameter(torch.randn(7, feature_dim))
        # ---------------------------------------------------------
        # Deformable Attention Encoder
        # ---------------------------------------------------------
        self.encoder = MultiScaleFocusDeformableAttentionEncoder(depth=depth, feature_dim=feature_dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points, dropout=dropout)
        self.final_norm = nn.LayerNorm(feature_dim)
        # ---------------------------------------------------------
        # Focus importance score
        # ---------------------------------------------------------
        self.score = nn.Linear(feature_dim, 1)

    def forward(self, x):
        """
        x:
            [B,7,512]
        """
        # ---------------------------------------------------------
        # 添加焦平面位置编码
        # ---------------------------------------------------------
        x = (x + self.focus_embedding.unsqueeze(0))
        # ---------------------------------------------------------
        # Multi-scale deformable attention
        # --------------------------------------------------------
        x = self.encoder(x)
        # ---------------------------------------------------------
        # Final normalization
        # ---------------------------------------------------------
        x = self.final_norm(x)
        # ---------------------------------------------------------
        # 每个焦平面的重要性
        # ---------------------------------------------------------
        score = self.score(x).squeeze(-1)
        weight = torch.softmax(score, dim=1)
        # ---------------------------------------------------------
        # Focus fusion
        # ---------------------------------------------------------
        fused = torch.sum(x * weight.unsqueeze(-1), dim=1)
        return fused, weight

if __name__ == "__main__":
    model = MSFDAttention(feature_dim=512, num_heads=8, depth=2, num_levels=3, num_points=4, dropout=0.2)
    x = torch.randn(8, 7, 512)
    feature, weight = model(x)
    print("Input:", x.shape)
    print("Feature:", feature.shape)
    print("Weight:", weight.shape)
    print("Weight sum:", weight.sum(dim=1))
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params / 1e6:.3f} M")