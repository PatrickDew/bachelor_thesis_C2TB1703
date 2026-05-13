"""
Orientation-only model for spacecraft pose estimation.

Specialized model that predicts only quaternion (q1,q2,q3,q4) from images.
Used when splitting the pose task: position from baseline PoseNet, orientation from this model.

Architecture: ResNet backbone + Linear(512 → 4) for quaternion.
Loss: Quaternion geodesic (1 - |<q_pred, q_gt>|).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
from typing import Tuple


class OrientationNet(nn.Module):
    """
    Orientation-only regression: ResNet backbone + FC head for quaternion (4 values).

    Same backbone as PoseNet but output head predicts only [q1, q2, q3, q4].
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
    ):
        super().__init__()
        def _weights(enum_name, pretrained_flag):
            if not pretrained_flag:
                return None
            try:
                return getattr(torchvision.models, enum_name).IMAGENET1K_V1
            except AttributeError:
                return "DEFAULT"

        if backbone == "resnet18":
            w = _weights("ResNet18_Weights", pretrained)
            self.backbone = torchvision.models.resnet18(weights=w)
            feat_dim = 512
        elif backbone == "resnet34":
            w = _weights("ResNet34_Weights", pretrained)
            self.backbone = torchvision.models.resnet34(weights=w)
            feat_dim = 512
        elif backbone == "resnet50":
            w = _weights("ResNet50_Weights", pretrained)
            self.backbone = torchvision.models.resnet50(weights=w)
            feat_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone.fc = nn.Linear(feat_dim, 4)  # quaternion only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def orientation_loss(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """
    Quaternion geodesic loss: 1 - |<q_pred, q_gt>|.

    Args:
        pred: (B, 4) predicted quaternion [q1,q2,q3,q4]
        target: (B, 4) ground truth quaternion

    Returns:
        loss, dict with quat_loss
    """
    pred_quat = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    dot = (pred_quat * target).sum(dim=1).abs().clamp(0, 1)
    quat_loss = (1 - dot).mean()
    return quat_loss, {"quat_loss": quat_loss.item()}


def orientation_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    ESA-style orientation metrics: mean and median error in degrees.

    Args:
        pred: (N, 4) predicted quaternions
        target: (N, 4) ground truth

    Returns:
        dict with ori_error_deg, ori_error_median_deg
    """
    pred_quat = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    dot = (pred_quat * target).sum(dim=1).abs().clamp(0, 1)
    ori_err_rad = 2 * torch.acos(dot)
    ori_err_deg = torch.rad2deg(ori_err_rad)
    return {
        "ori_error_deg": ori_err_deg.mean().item(),
        "ori_error_median_deg": ori_err_deg.median().item(),
    }
