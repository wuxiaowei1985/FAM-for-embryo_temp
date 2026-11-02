import torch
import torch.nn as nn

class ConvBNAct3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)
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
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
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
        return self.act(x + identity)

class Focal3DBackbone(nn.Module):
    """
    Input: [B,7,1,224,224]
    Spatial resolution: 224 -> 112 -> 56 -> 28 -> 14 -> 7
    Outputs:
        p2: [B,512,7,56,56]
        p3: [B,512,7,28,28]
        p4: [B,512,7,14,14]
        tokens: [B,343,512]
    """
    def __init__(self, num_focus=7, embed_dim=512, image_size=224):
        super().__init__()
        if image_size != 224:
            raise ValueError("Focal3DBackbone requires image_size=224.")
        if num_focus != 7:
            raise ValueError("Focal3DBackbone requires exactly 7 focal planes.")
        self.num_focus = num_focus
        self.embed_dim = embed_dim
        self.image_size = 224
        self.query_h = 7
        self.query_w = 7
        self.stem = nn.Sequential(
            ConvBNAct3D(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        )
        self.stage1 = nn.Sequential(
            ResidualBlock3D(64, 128, stride=(1, 2, 2)),
            ResidualBlock3D(128, 128)
        )
        self.p2_projection = nn.Conv3d(128, embed_dim, kernel_size=1, bias=False)
        self.stage2 = nn.Sequential(
            ResidualBlock3D(128, 256, stride=(1, 2, 2)),
            ResidualBlock3D(256, 256)
        )
        self.p3_projection = nn.Conv3d(256, embed_dim, kernel_size=1, bias=False)
        self.stage3 = nn.Sequential(
            ResidualBlock3D(256, 512, stride=(1, 2, 2)),
            ResidualBlock3D(512, 512)
        )
        self.p4_projection = nn.Conv3d(512, embed_dim, kernel_size=1, bias=False)
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
        if (F, C, H, W) != (7, 1, 224, 224):
            raise ValueError(f"Expected [B,7,1,224,224], got {tuple(x.shape)}")
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.stem(x)                                      # [B,64,7,112,112]
        x1 = self.stage1(x)                                   # [B,128,7,56,56]
        p2 = self.p2_projection(x1)
        x2 = self.stage2(x1)                                  # [B,256,7,28,28]
        p3 = self.p3_projection(x2)
        x3 = self.stage3(x2)                                  # [B,512,7,14,14]
        p4 = self.p4_projection(x3)
        patch = self.patch_embed(p4)                          # [B,512,7,7,7]
        tokens = self.patch_norm(patch.flatten(2).transpose(1, 2))
        if tokens.shape[1:] != (343, self.embed_dim):
            raise RuntimeError(f"Unexpected token shape: {tuple(tokens.shape)}")
        return {"p2": p2, "p3": p3, "p4": p4, "tokens": tokens}

if __name__ == "__main__":
    model = Focal3DBackbone()
    x = torch.randn(1, 7, 1, 224, 224)
    with torch.no_grad():
        output = model(x)
    for key, value in output.items():
        print(f"{key:10s}: {tuple(value.shape)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")
