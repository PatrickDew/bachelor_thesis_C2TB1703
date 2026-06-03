"""
Multi-stage pose filtering for noisy vision-based docking.

Pipeline: outlier reject -> EMA -> per-axis rate limit
"""

from __future__ import annotations

import numpy as np


class VisionPoseFilter:
    """Smooth vision pose and reject single-frame jumps."""

    def __init__(
        self,
        ema_alpha: float = 0.08,
        max_step_m: float = 0.08,
        max_rate_m_s: float = 0.05,
    ):
        self.ema_alpha = float(np.clip(ema_alpha, 0.01, 1.0))
        self.max_step_m = max_step_m
        self.max_rate_m_s = max_rate_m_s
        self._state: np.ndarray | None = None
        self._last_time: float | None = None

    def reset(self):
        self._state = None
        self._last_time = None

    def update(self, raw: np.ndarray, timestamp: float) -> np.ndarray:
        raw = np.asarray(raw, dtype=float).reshape(3)

        if self._state is None:
            self._state = raw.copy()
            self._last_time = timestamp
            return self._state.copy()

        step = raw - self._state
        step_norm = float(np.linalg.norm(step))
        if step_norm > self.max_step_m:
            raw = self._state + step * (self.max_step_m / step_norm)

        dt = max(timestamp - (self._last_time or timestamp), 1e-3)
        self._last_time = timestamp
        max_delta = self.max_rate_m_s * dt
        delta = raw - self._state
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > max_delta:
            raw = self._state + delta * (max_delta / delta_norm)

        self._state = (
            self.ema_alpha * raw + (1.0 - self.ema_alpha) * self._state
        )
        return self._state.copy()
