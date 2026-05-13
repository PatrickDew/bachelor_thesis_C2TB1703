"""
Pose estimation model for 6D spacecraft pose regression.

Architecture: ResNet backbone + FC head for (x, y, z, q1, q2, q3, q4).
Loss: Position L2 + quaternion geodesic (1 - |<q_pred, q_gt>|).
Reference: ESA Pose Estimation Challenge, UrsoNet (arXiv:1907.04298).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
from typing import Tuple

# Pose columns (URSO format)
POSE_COLS = ["x", "y", "z", "q1", "q2", "q3", "q4"]


class PoseNet(nn.Module):
    """6D pose regression: ResNet backbone + FC head."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        num_outputs: int = 7,
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

        self.backbone.fc = nn.Linear(feat_dim, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def pose_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 1.0,
    quat_weight: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Combined pose loss: position L2 + quaternion geodesic.

    Args:
        pred: (B, 7) predicted [x, y, z, q1, q2, q3, q4]
        target: (B, 7) ground truth
        pos_weight: weight for position loss
        quat_weight: weight for orientation loss

    Returns:
        total_loss, dict of component losses
    """
    pred_pos = pred[:, :3]
    pred_quat = pred[:, 3:7]
    target_pos = target[:, :3]
    target_quat = target[:, 3:7]

    # Normalize predicted quaternion
    pred_quat = pred_quat / (pred_quat.norm(dim=1, keepdim=True) + 1e-8)

    # Position: L2
    pos_loss = ((pred_pos - target_pos) ** 2).sum(dim=1).mean()
    pos_loss_m = pos_loss.sqrt().item()

    # Quaternion: geodesic distance, 1 - |<q_pred, q_gt>|
    # Handle q and -q equivalence: use abs of dot product
    dot = (pred_quat * target_quat).sum(dim=1)
    dot = torch.clamp(dot.abs(), 0, 1)
    quat_loss = (1 - dot).mean()

    total = pos_weight * pos_loss + quat_weight * quat_loss

    return total, {
        "pos_loss": pos_loss.item(),
        "pos_loss_m": pos_loss_m,
        "quat_loss": quat_loss.item(),
    }


def pose_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """
    ESA-style metrics: position error (m), orientation error (deg).

    Args:
        pred: (N, 7) predicted poses
        target: (N, 7) ground truth

    Returns:
        dict with pos_error_m, ori_error_deg, etc.
    """
    pred_pos = pred[:, :3]
    pred_quat = pred[:, 3:7]
    target_pos = target[:, :3]
    target_quat = target[:, 3:7]

    pred_quat = pred_quat / (pred_quat.norm(dim=1, keepdim=True) + 1e-8)

    # Position error (m)
    pos_err = ((pred_pos - target_pos) ** 2).sum(dim=1).sqrt()
    pos_error_m = pos_err.mean().item()
    pos_error_median_m = pos_err.median().item()

    # Orientation error (deg): 2 * arccos(|<q_pred, q_gt>|)
    dot = (pred_quat * target_quat).sum(dim=1).abs().clamp(0, 1)
    ori_err_rad = 2 * torch.acos(dot)
    ori_err_deg = torch.rad2deg(ori_err_rad)
    ori_error_deg = ori_err_deg.mean().item()
    ori_error_median_deg = ori_err_deg.median().item()

    return {
        "pos_error_m": pos_error_m,
        "pos_error_median_m": pos_error_median_m,
        "ori_error_deg": ori_error_deg,
        "ori_error_median_deg": ori_error_median_deg,
    }
