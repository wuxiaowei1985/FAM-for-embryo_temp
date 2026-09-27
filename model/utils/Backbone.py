import torch
import torch.nn as nn

class ConvBNAct3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=None):
        super().__init__()
        if padding is None:
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            else:
                padding = tuple(k // 2 for k in kernel_size)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )
    def forward(self, x):
        return self.block(x)

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1, 1)):
        super().__init__()
        self.conv1 = ConvBNAct3D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        if in_channels != out_channels or stride != (1, 1, 1):
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        x = self.act(x)
        return x

class Focal3DBackbone(nn.Module):
    """
    Input:
        [B, F, 1, 500, 500]
    Outputs:
        p2:
            [B, 512, 7, 63, 63]
        p3:
            [B, 512, 7, 32, 32]
        p4:
            [B, 512, 7, 16, 16]
        tokens:
            [B, 448, 512]
    """
    def __init__(self, num_focus=7, embed_dim=512):
        super().__init__()
        self.num_focus = num_focus
        self.embed_dim = embed_dim
        self.query_h = 8
        self.query_w = 8
        # =========================================================
        # Stem 500 -> 125
        # F: 7 -> 7
        # =========================================================
        self.stem = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 4, 4), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.GELU()
        )
        # =========================================================
        # Stage 1
        # 125 -> 63
        # =========================================================
        self.stage1 = nn.Sequential(
            ResidualBlock3D(64, 128, stride=(1, 2, 2)),
            ResidualBlock3D(128, 128)
        )
        self.p2_projection = nn.Conv3d(128, embed_dim, kernel_size=1, bias=False)
        # =========================================================
        # Stage 2
        # 63 -> 32
        # =========================================================
        self.stage2 = nn.Sequential(
            ResidualBlock3D(128, 256, stride=(1, 2, 2)),
            ResidualBlock3D(256, 256)
        )
        self.p3_projection = nn.Conv3d(256, embed_dim, kernel_size=1, bias=False)
        # =========================================================
        # Stage 3
        # 32 -> 16
        # =========================================================
        self.stage3 = nn.Sequential(
            ResidualBlock3D(256, 512, stride=(1, 2, 2)),
            ResidualBlock3D(512, 512)
        )
        self.p4_projection = nn.Conv3d(512, embed_dim, kernel_size=1, bias=False)
        # =========================================================
        # Patch Embedding
        # P4: [B,512,7,16,16] -> [B,512,7,8,8]
        # 只切 H/W，不切 focal dimension
        # =========================================================
        self.patch_embed = nn.Conv3d(embed_dim, embed_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected [B,F,C,H,W], got {tuple(x.shape)}")
        B, F, C, H, W = x.shape
        if F != self.num_focus:
            raise ValueError(f"Expected {self.num_focus} focal planes, got {F}")
        if C != 1:
            raise ValueError(f"Expected grayscale C=1, got {C}")
        if (H, W) != (500, 500):
            raise ValueError(f"Expected native 500x500 input, got {(H, W)}")
        # =========================================================
        # [B,F,C,H,W] -> [B,C,F,H,W]
        # =========================================================
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # =========================================================
        # Stem
        # =========================================================
        x = self.stem(x)
        # [B,64,7,125,125]
        # =========================================================
        # Stage 1
        # =========================================================
        x1 = self.stage1(x)
        # [B,128,7,63,63]
        p2 = self.p2_projection(x1)
        # [B,512,7,63,63]
        # =========================================================
        # Stage 2
        # =========================================================
        x2 = self.stage2(x1)
        # [B,256,7,32,32]
        p3 = self.p3_projection(x2)
        # [B,512,7,32,32]
        # =========================================================
        # Stage 3
        # =========================================================
        x3 = self.stage3(x2)
        # [B,512,7,16,16]
        p4 = self.p4_projection(x3)
        # [B,512,7,16,16]
        # =========================================================
        # Patch embedding
        # =========================================================
        patch = self.patch_embed(p4)
        # [B,512,7,8,8]
        # =========================================================
        # Flatten
        # 7 × 8 × 8 = 448
        # =========================================================
        tokens = patch.flatten(2).transpose(1, 2)
        # [B,448,512]
        tokens = self.patch_norm(tokens)
        expected_tokens = (self.num_focus * self.query_h * self.query_w)
        if tokens.shape[1] != expected_tokens:
            raise RuntimeError(f"Expected {expected_tokens} tokens, "f"got {tokens.shape[1]}")
        return {
            "p2": p2,
            "p3": p3,
            "p4": p4,
            "tokens": tokens,
        }

if __name__ == "__main__":
    model = Focal3DBackbone(num_focus=7, embed_dim=512)
    x = torch.randn(1, 7, 1, 500, 500)
    with torch.no_grad():
        output = model(x)
    for key, value in output.items():
        print(f"{key:10s}: "f"{tuple(value.shape)}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: "f"{num_params / 1e6:.3f} M")