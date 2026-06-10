from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SingleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaseSaliencyUNet(nn.Module):
    def freeze_encoder_stages(self, freeze_stages: int) -> None:
        stages = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        for stage in stages[: max(0, freeze_stages + 1)]:
            for parameter in stage.parameters():
                parameter.requires_grad = False
            for module in stage.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

    def train(self, mode: bool = True) -> "BaseSaliencyUNet":
        super().train(mode)
        stages = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        for stage in stages:
            if any(parameter.requires_grad for parameter in stage.parameters()):
                continue
            stage.eval()
        return self
