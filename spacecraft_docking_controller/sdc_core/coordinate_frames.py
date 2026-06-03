"""
Coordinate frame transforms for vision-based docking in Isaac Sim.

Camera / URSO optical frame (pose estimation output):
  X = right, Y = down, Z = forward (depth toward target)

Isaac Sim chaser body / world command frame (force application):
  X = forward, Y = left, Z = up
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class FrameRemapConfig:
    """Per-axis sign flips after camera -> Isaac remap."""
    invert_x: float = 1.0
    invert_y: float = 1.0
    invert_z: float = 1.0


def camera_to_isaac_vector(
    cam_vec: np.ndarray,
    config: FrameRemapConfig | None = None,
) -> np.ndarray:
    """
    Map a 3-vector from camera optical frame to Isaac command frame.

    Position, velocity, acceleration, and force vectors use the same rotation
    (reflections only — no translation offset in relative pose control).
    """
    cfg = config or FrameRemapConfig()
    cam = np.asarray(cam_vec, dtype=float).reshape(3)
    return np.array([
        cam[2] * cfg.invert_x,   # forward  <- camera Z (depth)
        -cam[0] * cfg.invert_y,  # left     <- -camera X (right)
        -cam[1] * cfg.invert_z,  # up       <- -camera Y (down)
    ])


def isaac_to_camera_vector(
    isaac_vec: np.ndarray,
    config: FrameRemapConfig | None = None,
) -> np.ndarray:
    """Inverse of camera_to_isaac_vector (same config)."""
    cfg = config or FrameRemapConfig()
    isaac = np.asarray(isaac_vec, dtype=float).reshape(3)
    return np.array([
        -isaac[1] / cfg.invert_y if cfg.invert_y != 0 else 0.0,
        -isaac[2] / cfg.invert_z if cfg.invert_z != 0 else 0.0,
        isaac[0] / cfg.invert_x if cfg.invert_x != 0 else 0.0,
    ])


def closing_velocity(pos_error: np.ndarray, velocity: np.ndarray) -> float:
    """
    Signed closing rate along line-of-sight (negative = approaching target).

    pos_error should be target - current (points toward target).
    """
    pos_error = np.asarray(pos_error, dtype=float).reshape(3)
    velocity = np.asarray(velocity, dtype=float).reshape(3)
    dist = np.linalg.norm(pos_error)
    if dist < 1e-6:
        return 0.0
    los = pos_error / dist
    return float(np.dot(velocity, los))
