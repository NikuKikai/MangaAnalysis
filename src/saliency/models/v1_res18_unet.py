from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .common import BaseSaliencyUNet, DoubleConv


class UpBlockV1(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SaliencyUNetV1(BaseSaliencyUNet):
    architecture_name = "v1"

    def __init__(self, encoder_pretrained: bool = False, base_channels: int = 64) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if encoder_pretrained else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.bridge = DoubleConv(512, 512)
        self.up4 = UpBlockV1(512, 256, 256)
        self.up3 = UpBlockV1(256, 128, 128)
        self.up2 = UpBlockV1(128, 64, 64)
        self.up1 = UpBlockV1(64, 64, base_channels)
        self.head = nn.Sequential(
            DoubleConv(base_channels, base_channels),
            nn.Conv2d(base_channels, 1, kernel_size=1),
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
