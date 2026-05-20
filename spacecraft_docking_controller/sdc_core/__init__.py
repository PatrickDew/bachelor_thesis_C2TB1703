"""
Spacecraft Docking Controller Core Library (sdc_core)

A research-grade library for spacecraft proximity operations and 
autonomous rendezvous and docking (AR&D) in NVIDIA Isaac Sim.

Key components:
- Clohessy-Wiltshire relative orbital dynamics
- PID Controller (baseline)
- LQR Controller (optimal linear)
- MPC Controller (model predictive control)

Author: Pakawat Nutthanithipat
License: MIT
"""

__version__ = "1.0.0"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "ClohessyWiltshireDynamics":
        from .dynamics import ClohessyWiltshireDynamics
        return ClohessyWiltshireDynamics
    elif name == "SpacecraftDynamics":
        from .dynamics import SpacecraftDynamics
        return SpacecraftDynamics
    elif name == "PIDController":
        from .controllers import PIDController
        return PIDController
    elif name == "LQRController":
        from .controllers import LQRController
        return LQRController
    elif name == "MPCController":
        from .controllers import MPCController
        return MPCController
    elif name == "StateEstimator":
        from .state_estimator import StateEstimator
        return StateEstimator
    elif name == "TrajectoryGenerator":
        from .trajectory_generator import TrajectoryGenerator
        return TrajectoryGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ClohessyWiltshireDynamics",
    "SpacecraftDynamics", 
    "PIDController",
    "LQRController",
    "MPCController",
    "StateEstimator",
    "TrajectoryGenerator",
]

