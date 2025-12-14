import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, res_scale: float = 0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Basic residual block with residual scaling for stability."""
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = residual + out * self.res_scale
        return out


class ResidualSRModel(nn.Module):
    def __init__(self, scale_factor: int = 4, num_channels: int = 3, num_features: int = 32, num_res_blocks: int = 8, res_scale: float = 0.1):
        super().__init__()
        self.scale_factor = scale_factor

        self.conv_in = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([
                ResidualBlock(num_features, num_features, res_scale=res_scale)
                for _ in range(num_res_blocks)
            ])
        self.conv_mid = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv_upsample = nn.Conv2d(num_features, num_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shallow residual network with pixel shuffle upsampling."""
        x_feat = self.conv_in(x)
        out = x_feat
        for block in self.res_blocks:
            out = block(out)
        out = self.conv_mid(out)
        out = out + x_feat
        out = self.conv_upsample(out)
        out = self.pixel_shuffle(out)
        return out
