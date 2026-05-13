"""
Orientation model with alternate backbones (ResNet-50, EfficientNet).

Targets improved orientation estimation, especially roll (worst with ResNet-18).
Uses orientation_loss and orientation_metrics from orientation_model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
from typing import Tuple

from orientation_model import orientation_loss, orientation_metrics

def _weights(enum_name: str, pretrained: bool):
    if not pretrained:
        return None
    try:
        return getattr(torchvision.models, enum_name).IMAGENET1K_V1
    except AttributeError:
        return "DEFAULT"


class OrientationNetAlt(nn.Module):
    """
    Orientation-only regression with ResNet-50 or EfficientNet-B4 backbone.

    Stronger backbones for better feature extraction (roll, pitch, yaw).
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet50":
            w = _weights("ResNet50_Weights", pretrained)
            self.backbone = torchvision.models.resnet50(weights=w)
            self.backbone.fc = nn.Linear(2048, 4)
        elif backbone == "efficientnet_b4":
            w = _weights("EfficientNet_B4_Weights", pretrained)
            self.backbone = torchvision.models.efficientnet_b4(weights=w)
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(1792, 4),
            )
        elif backbone == "efficientnet_b0":
            w = _weights("EfficientNet_B0_Weights", pretrained)
            self.backbone = torchvision.models.efficientnet_b0(weights=w)
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1280, 4),
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Use resnet50, efficientnet_b4, efficientnet_b0")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
