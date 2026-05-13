"""
Combined pose inference: PoseNet (position) + OrientationNet (orientation).

Use this for Isaac Sim or any application that needs full 6D pose from a single image.
Loads both models and returns [x, y, z, q1, q2, q3, q4].
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import numpy as np
from torchvision import transforms

from src.pose_model import PoseNet
from src.orientation_model import OrientationNet
from src.orientation_model_alt import OrientationNetAlt


def default_image_transform(img_size: int = 224):
    """Default preprocessing for URSO/Isaac Sim images (H, W, 3) uint8."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class SplitPosePredictor:
    """
    Combines PoseNet (position) and OrientationNet (orientation) for full 6D pose.

    Usage:
        predictor = SplitPosePredictor(
            pos_model_path="models/pose_net_baseline.pt",
            ori_model_path="models/orientation_net.pt",
        )
        pose = predictor.predict(image)  # (7,) numpy [x,y,z,q1,q2,q3,q4]
    """

    def __init__(
        self,
        pos_model_path: Union[str, Path],
        ori_model_path: Union[str, Path],
        backbone: str = "resnet18",
        device: str = None,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        # Position model (PoseNet) - we use only first 3 outputs
        self.pos_model = PoseNet(backbone=backbone, pretrained=False)
        ckpt = torch.load(pos_model_path, map_location=self.device)
        self.pos_model.load_state_dict(ckpt["model"])
        self.pos_model = self.pos_model.to(self.device).eval()

        # Orientation model (auto-detect: OrientationNetAlt for resnet50/efficientnet)
        ckpt = torch.load(ori_model_path, map_location=self.device)
        ori_backbone = ckpt.get("backbone", backbone)
        if ori_backbone in ("resnet50", "efficientnet_b4", "efficientnet_b0"):
            self.ori_model = OrientationNetAlt(backbone=ori_backbone, pretrained=False)
        else:
            self.ori_model = OrientationNet(backbone=ori_backbone, pretrained=False)
        self.ori_model.load_state_dict(ckpt["model"])
        self.ori_model = self.ori_model.to(self.device).eval()

        self._transform = default_image_transform()

    def set_transform(self, transform):
        """Set image preprocessing (e.g. Resize, ToTensor, Normalize)."""
        self._transform = transform

    def predict(self, image: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Predict full 6D pose from image.

        Args:
            image: (H, W, 3) numpy or (1, 3, H, W) tensor, already preprocessed if tensor

        Returns:
            (7,) numpy [x, y, z, q1, q2, q3, q4]
        """
        if isinstance(image, np.ndarray):
            if self._transform is None:
                raise ValueError("Provide transform or pass preprocessed tensor")
            image = self._transform(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            pos_out = self.pos_model(image)
            pos = pos_out[:, :3].cpu().numpy()

            ori_out = self.ori_model(image)
            ori = ori_out / (ori_out.norm(dim=1, keepdim=True) + 1e-8)
            ori = ori.cpu().numpy()

        pose = np.concatenate([pos, ori], axis=1)
        return pose.squeeze(0)
