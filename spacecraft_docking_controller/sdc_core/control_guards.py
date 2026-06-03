"""
Safety guards for vision-based docking control.
"""

from __future__ import annotations

import numpy as np
from collections import deque


def ensure_closing_direction(
    control: np.ndarray,
    pos_error: np.ndarray,
    velocity: np.ndarray,
    range_rate_threshold: float = 0.005,
) -> np.ndarray:
    """
    If moving away from target (positive closing rate), remove radial
    acceleration that increases range and add braking.
    """
    control = np.asarray(control, dtype=float).reshape(3)
    pos_error = np.asarray(pos_error, dtype=float).reshape(3)
    velocity = np.asarray(velocity, dtype=float).reshape(3)

    dist = float(np.linalg.norm(pos_error))
    if dist < 1e-6:
        return control

    los = pos_error / dist
    closing_rate = float(np.dot(velocity, los))

    if closing_rate <= range_rate_threshold:
        return control

    radial = float(np.dot(control, los))
    if radial > 0.0:
        control = control - radial * los

    control = control - 0.5 * closing_rate * los
    return control


def ensure_control_toward_target(
    control: np.ndarray,
    pos_error: np.ndarray,
) -> np.ndarray:
    """Flip control if P-action would push away from target (wrong frame sign)."""
    control = np.asarray(control, dtype=float).reshape(3)
    pos_error = np.asarray(pos_error, dtype=float).reshape(3)
    if float(np.dot(control, pos_error)) < 0.0:
        control = -control
    return control


class RangeMonotonicGuard:
    """Detect sustained range increase and scale down / brake."""

    def __init__(self, window: int = 15, worsen_tol_m: float = 0.03):
        self.window = window
        self.worsen_tol_m = worsen_tol_m
        self._history: deque = deque(maxlen=window)

    def reset(self):
        self._history.clear()

    def update(self, range_m: float) -> float:
        """Return gain scale in [0, 1] — lower when diverging."""
        self._history.append(range_m)
        if len(self._history) < 5:
            return 1.0
        recent_min = min(self._history)
        if range_m > recent_min + self.worsen_tol_m:
            return 0.15
        return 1.0
