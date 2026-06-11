from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .common import BaseSaliencyUNet, DoubleConv, SingleConv


class UpBlockV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        compressed_in_channels: int,
        compressed_skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, compressed_in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(compressed_in_channels),
            nn.ReLU(inplace=True),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, compressed_skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(compressed_skip_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = DoubleConv(compressed_in_channels + compressed_skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.skip_proj(skip)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class SaliencyUNetV2(BaseSaliencyUNet):
    architecture_name = "v2"

    def __init__(
        self,
        encoder_pretrained: bool = False,
        base_channels: int = 32,
        width_mult: float = 1.0,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if encoder_pretrained else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        bridge_channels = max(32, int(round(256 * width_mult / 8.0)) * 8)
        up4_channels = max(32, int(round(128 * width_mult / 8.0)) * 8)
        up3_channels = max(16, int(round(64 * width_mult / 8.0)) * 8)
        up2_channels = max(16, int(round(32 * width_mult / 8.0)) * 8)
        up1_channels = max(16, int(round(base_channels * width_mult / 8.0)) * 8)

        self.bridge = SingleConv(512, bridge_channels)
        self.up4 = UpBlockV2(bridge_channels, 256, bridge_channels, up4_channels, up4_channels)
        self.up3 = UpBlockV2(up4_channels, 128, up4_channels, up3_channels, up3_channels)
        self.up2 = UpBlockV2(up3_channels, 64, up3_channels, up2_channels, up2_channels)
        self.up1 = UpBlockV2(up2_channels, 64, up2_channels, up1_channels, up1_channels)
        self.head = nn.Sequential(
            SingleConv(up1_channels, up1_channels),
            nn.Conv2d(up1_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        bridge = self.bridge(x4)
        y = self.up4(bridge, x3)
        y = self.up3(y, x2)
        y = self.up2(y, x1)
        y = self.up1(y, x0)
        logits = self.head(y)
        return nn.functional.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
