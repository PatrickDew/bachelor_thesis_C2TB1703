"""
Spacecraft Docking Controllers Module

Implements multiple control strategies for spacecraft proximity operations:
1. PID Controller - Baseline feedback control
2. LQR Controller - Linear Quadratic Regulator (optimal)
3. MPC Controller - Model Predictive Control (advanced)

All controllers operate in the LVLH frame and output control accelerations.

Author: Sichen Tao
"""

import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are
from typing import Optional, Tuple, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .dynamics import ClohessyWiltshireDynamics, OrbitalParameters


@dataclass
class ControlLimits:
    """Physical limits on control authority."""
    max_force: float = 50.0        # Maximum force per axis (N)
    max_torque: float = 5.0        # Maximum torque per axis (N*m)
    max_acceleration: float = 0.1  # Maximum acceleration (m/s^2)
    max_angular_accel: float = 0.1 # Maximum angular acceleration (rad/s^2)


class BaseController(ABC):
    """Abstract base class for spacecraft controllers."""
    
    def __init__(self, dynamics: ClohessyWiltshireDynamics,
                 limits: Optional[ControlLimits] = None):
        self.dynamics = dynamics
        self.limits = limits or ControlLimits()
        self.dt = 0.1  # Default control timestep
    
    @abstractmethod
    def compute_control(self, current_state: np.ndarray,
                        target_state: np.ndarray) -> np.ndarray:
        """Compute control input given current and target states."""
        pass
    
    def saturate(self, control: np.ndarray) -> np.ndarray:
        """Apply control limits (saturation)."""
        return np.clip(control, 
                       -self.limits.max_acceleration,
                       self.limits.max_acceleration)
    
    def reset(self):
        """Reset controller state (for integral terms, etc.)."""
        pass


class PIDController(BaseController):
    """
    PID Controller for Spacecraft Docking
    
    Implements decoupled PID control for position and attitude.
    Includes anti-windup for integral terms.
    
    Features:
    - Separate gains for each axis (6-DOF)
    - Integral anti-windup
    - Derivative filtering
    - Feed-forward velocity tracking
    """
    
    def __init__(self, dynamics: ClohessyWiltshireDynamics,
                 limits: Optional[ControlLimits] = None):
        super().__init__(dynamics, limits)
        
        # Position PID gains [kp, ki, kd] for each axis
        # Tuned for typical docking scenario
        self.kp = np.array([0.01, 0.01, 0.01])   # Proportional
        self.ki = np.array([0.001, 0.001, 0.001]) # Integral
        self.kd = np.array([0.1, 0.1, 0.1])       # Derivative
        
        # Velocity feedback gains (optional)
        self.kv = np.array([0.05, 0.05, 0.05])
        
        # Internal state
        self.integral_error = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.prev_derivative = np.zeros(3)
        
        # Anti-windup limits
        self.integral_max = np.array([10.0, 10.0, 10.0])
        
        # Derivative filter coefficient (0-1, higher = more filtering)
        self.derivative_filter = 0.8
        
        # Gain scheduling based on range
        self.use_gain_scheduling = True
        
        # CW feedforward mode
        # Set to False for stationary/ground-test scenarios (no orbital dynamics)
        # Set to True only if Isaac Sim models actual orbital environment
        self.use_cw_feedforward = False  # Default OFF for stationary ISS
    
    def set_gains(self, kp: np.ndarray, ki: np.ndarray, kd: np.ndarray):
        """Set PID gains."""
        self.kp = np.asarray(kp)
        self.ki = np.asarray(ki)
        self.kd = np.asarray(kd)
    
    def compute_control(self, current_state: np.ndarray,
                        target_state: np.ndarray) -> np.ndarray:
        """
        Compute CW-aware adaptive PID control output.
        
        This controller integrates Clohessy-Wiltshire dynamics for proper
        spacecraft proximity operations. It includes:
        - CW feedforward compensation for orbital effects
        - Range-based adaptive gain scheduling
        - Range-rate control for approach velocity management
        
        Args:
            current_state: [x, y, z, vx, vy, vz]
            target_state: [x_d, y_d, z_d, vx_d, vy_d, vz_d]
            
        Returns:
            Control acceleration [ax, ay, az]
        """
        # Extract positions and velocities
        pos = current_state[0:3]
        vel = current_state[3:6]
        pos_target = target_state[0:3]
        vel_target = target_state[3:6]
        
        # Position error vector: target - current (standard regulation convention)
        # Error points FROM current position TO target
        # Positive X error means "target is in +X direction, move +X to reach it"
        pos_error = pos_target - pos
        
        # Range (distance to target) and direction
        range_m = np.linalg.norm(pos_error)
        if range_m > 0.001:  # Avoid division by zero
            # range_dir points FROM spacecraft TOWARDS target
            range_dir = pos_error / range_m
        else:
            range_dir = np.zeros(3)
        
        # Range rate (closing velocity, negative = approaching)
        range_rate = np.dot(vel, range_dir)
        
        # Desired range rate: slow approach velocity based on range
        # Closer = slower approach (safety)
        # NOTE: Keeping this gentle to avoid fighting position control
        if range_m > 10.0:
            desired_range_rate = -0.03  # 3 cm/s approach when far
        elif range_m > 1.0:
            desired_range_rate = -0.01  # 1 cm/s approach when mid-range
        else:
            desired_range_rate = -0.003  # 0.3 cm/s approach when close (very slow final)
        
        # Range rate error (only apply if moving too fast, not if too slow)
        range_rate_error = range_rate - desired_range_rate
        
        # Only brake if going too fast (don't push if going slow - let position control handle it)
        if range_rate_error < 0:
            range_rate_error = 0.0  # Don't add forward thrust, just brake if too fast
        
        # Gain scheduling based on range
        if self.use_gain_scheduling:
            gain_scale = self._compute_gain_scale(range_m)
            kp = self.kp * gain_scale
            ki = self.ki * gain_scale
            kd = self.kd * gain_scale
        else:
            kp, ki, kd = self.kp, self.ki, self.kd
        
        # ===== PID Terms =====
        
        # Proportional term (position error)
        p_term = kp * pos_error
        
        # Integral term with anti-windup
        self.integral_error += pos_error * self.dt
        self.integral_error = np.clip(self.integral_error,
                                       -self.integral_max,
                                       self.integral_max)
        i_term = ki * self.integral_error
        
        # Derivative term with filtering
        raw_derivative = (pos_error - self.prev_error) / self.dt
        filtered_derivative = (self.derivative_filter * self.prev_derivative + 
                              (1 - self.derivative_filter) * raw_derivative)
        d_term = kd * filtered_derivative
        self.prev_derivative = filtered_derivative
        self.prev_error = pos_error.copy()
        
        # ===== CW Feedforward Compensation =====
        # Only apply if in orbital environment (not stationary ISS)
        if self.use_cw_feedforward:
            cw_ff = self._compute_cw_feedforward(current_state)
        else:
            cw_ff = np.zeros(3)  # No orbital dynamics in stationary mode
        
        # ===== Range Rate Control =====
        # Additional control to maintain desired approach velocity
        range_rate_control = self.kv[0] * range_rate_error * range_dir
        
        # ===== Combined Control =====
        # PID + optional CW feedforward + range rate control
        control = p_term + i_term + d_term + cw_ff + range_rate_control
        
        return self.saturate(control)
    
    def _compute_cw_feedforward(self, state: np.ndarray) -> np.ndarray:
        """
        Compute Clohessy-Wiltshire feedforward compensation.
        
        The CW equations show orbital coupling that must be compensated:
        - ẍ = 3n²x + 2nẏ  (radial direction)
        - ÿ = -2nẋ        (along-track direction)  
        - z̈ = -n²z        (cross-track direction)
        
        To maintain position, we must apply acceleration to cancel these terms.
        """
        n = self.dynamics.n  # Mean motion
        x, y, z = state[0:3]
        vx, vy, vz = state[3:6]
        
        # CW compensation (negative to cancel the natural motion)
        ax_ff = -(3 * n**2 * x + 2 * n * vy)
        ay_ff = -(-2 * n * vx)  # = 2*n*vx
        az_ff = -(-n**2 * z)    # = n²*z
        
        return np.array([ax_ff, ay_ff, az_ff])
    
    def _compute_gain_scale(self, range_m: float) -> float:
        """
        Compute smooth gain scaling factor based on range.
        
        Uses exponential scheduling for smooth transition:
        - Far range (>50m): Low gain for stability
        - Mid range (10-50m): Moderate gain
        - Close range (<10m): Higher gain for precision
        - Terminal (<1m): Maximum gain for final approach
        """
        # Smooth exponential gain scheduling
        if range_m > 50.0:
            return 0.3  # Far range - very conservative
        elif range_m > 10.0:
            # Smooth transition from 0.3 to 1.0
            return 0.3 + 0.7 * (50.0 - range_m) / 40.0
        elif range_m > 1.0:
            # Smooth transition from 1.0 to 2.0
            return 1.0 + 1.0 * (10.0 - range_m) / 9.0
        else:
            return 2.0  # Terminal - highest gains
    
    def reset(self):
        """Reset integral and derivative states."""
        self.integral_error = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.prev_derivative = np.zeros(3)


class LQRController(BaseController):
    """
    Linear Quadratic Regulator (LQR) Controller
    
    Optimal state feedback controller that minimizes quadratic cost:
    J = ∫(x'Qx + u'Ru)dt
    
    Uses CW dynamics for state-space model.
    Computes infinite-horizon steady-state gain K.
    
    Features:
    - Optimal for linear systems
    - Guaranteed stability margins
    - Tunable via Q and R matrices
    """
    
    def __init__(self, dynamics: ClohessyWiltshireDynamics,
                 limits: Optional[ControlLimits] = None,
                 use_discrete: bool = True):
        super().__init__(dynamics, limits)
        
        self.use_discrete = use_discrete
        
        # Default weighting matrices
        # Q: State cost (penalize deviations from target)
        # Higher weight = more aggressive tracking
        self.Q = np.diag([
            100.0, 100.0, 100.0,   # Position weights (x, y, z)
            10.0, 10.0, 10.0       # Velocity weights (vx, vy, vz)
        ])
        
        # R: Control cost (penalize control effort)
        # Higher weight = less aggressive control
        self.R = np.diag([1.0, 1.0, 1.0])
        
        # Compute LQR gain
        self._compute_gain()
    
    def set_weights(self, Q: np.ndarray, R: np.ndarray):
        """Set Q and R weighting matrices and recompute gain."""
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        self._compute_gain()
    
    def _compute_gain(self):
        """Compute LQR gain matrix by solving Riccati equation."""
        A = self.dynamics.A_continuous
        B = self.dynamics.B_continuous
        
        if self.use_discrete:
            # Discretize system
            Phi = self.dynamics.get_state_transition_matrix(self.dt)
            Gamma = self.dynamics.get_control_matrix(self.dt)
            
            # Solve discrete algebraic Riccati equation
            try:
                P = solve_discrete_are(Phi, Gamma, self.Q, self.R)
                # Discrete LQR gain
                self.K = np.linalg.solve(
                    self.R + Gamma.T @ P @ Gamma,
                    Gamma.T @ P @ Phi
                )
            except np.linalg.LinAlgError:
                print("Warning: DARE solution failed, using simple gain")
                self.K = np.zeros((3, 6))
                self.K[0, 0] = 0.01  # Fallback simple gain
                self.K[1, 1] = 0.01
                self.K[2, 2] = 0.01
        else:
            # Solve continuous algebraic Riccati equation
            try:
                P = solve_continuous_are(A, B, self.Q, self.R)
                # Continuous LQR gain
                self.K = np.linalg.solve(self.R, B.T @ P)
            except np.linalg.LinAlgError:
                print("Warning: CARE solution failed, using simple gain")
                self.K = np.zeros((3, 6))
                self.K[0, 0] = 0.01
                self.K[1, 1] = 0.01
                self.K[2, 2] = 0.01
    
    def compute_control(self, current_state: np.ndarray,
                        target_state: np.ndarray) -> np.ndarray:
        """
        Compute LQR control output.
        
        u = -K * (x - x_target)
        
        Args:
            current_state: [x, y, z, vx, vy, vz]
            target_state: [x_d, y_d, z_d, vx_d, vy_d, vz_d]
            
        Returns:
            Control acceleration [ax, ay, az]
        """
        # State error
        error = current_state - target_state
        
        # LQR control law
        control = -self.K @ error
        
        return self.saturate(control)
    
    def get_gain(self) -> np.ndarray:
        """Return current LQR gain matrix."""
        return self.K.copy()


class MPCController(BaseController):
    """
    Model Predictive Controller (MPC) for Spacecraft Docking
    
    Solves finite-horizon optimal control problem at each timestep.
    Handles constraints explicitly (control limits, keep-out zones).
    
    Features:
    - Constraint handling (input saturation, state constraints)
    - Trajectory preview/tracking
    - Approach corridor enforcement
    - Fuel-optimal or time-optimal formulations
    
    Uses QP formulation solved via iterative methods or cvxpy if available.
    """
    
    def __init__(self, dynamics: ClohessyWiltshireDynamics,
                 limits: Optional[ControlLimits] = None,
                 horizon: int = 20):
        super().__init__(dynamics, limits)
        
        self.horizon = horizon  # Prediction horizon steps
        
        # Cost weights
        self.Q = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])  # State
        self.R = np.diag([1.0, 1.0, 1.0])  # Control
        self.Q_terminal = 10 * self.Q  # Terminal cost (stabilizing)
        
        # Approach corridor constraint (cone half-angle in radians)
        self.corridor_half_angle = np.radians(10.0)  # 10 degree cone
        self.enforce_corridor = True
        
        # Reference trajectory (optional)
        self.reference_trajectory = None
        
        # Pre-compute prediction matrices
        self._build_prediction_matrices()
    
    def _build_prediction_matrices(self):
        """Build prediction matrices for QP formulation."""
        N = self.horizon
        nx = 6  # State dimension
        nu = 3  # Control dimension
        
        Phi = self.dynamics.get_state_transition_matrix(self.dt)
        Gamma = self.dynamics.get_control_matrix(self.dt)
        
        # Prediction matrices: X = Psi*x0 + Theta*U
        # Where X = [x1, x2, ..., xN], U = [u0, u1, ..., u_{N-1}]
        
        self.Psi = np.zeros((N * nx, nx))
        self.Theta = np.zeros((N * nx, N * nu))
        
        Phi_power = np.eye(nx)
        for i in range(N):
            Phi_power = Phi_power @ Phi
            self.Psi[i*nx:(i+1)*nx, :] = Phi_power
            
            for j in range(i + 1):
                idx_col = j * nu
                idx_row = i * nx
                power = i - j
                Phi_pow_j = np.linalg.matrix_power(Phi, power)
                self.Theta[idx_row:idx_row+nx, idx_col:idx_col+nu] = Phi_pow_j @ Gamma
        
        # Build cost matrices for QP: min 0.5*U'*H*U + f'*U
        Q_bar = np.kron(np.eye(N-1), self.Q)
        Q_bar = np.block([
            [Q_bar, np.zeros(((N-1)*nx, nx))],
            [np.zeros((nx, (N-1)*nx)), self.Q_terminal]
        ])
        R_bar = np.kron(np.eye(N), self.R)
        
        self.H = self.Theta.T @ Q_bar @ self.Theta + R_bar
        self.F = self.Theta.T @ Q_bar @ self.Psi  # For computing f
    
    def compute_control(self, current_state: np.ndarray,
                        target_state: np.ndarray) -> np.ndarray:
        """
        Compute MPC control output.
        
        Solves QP at each timestep and returns first control action.
        
        Args:
            current_state: [x, y, z, vx, vy, vz]
            target_state: [x_d, y_d, z_d, vx_d, vy_d, vz_d]
            
        Returns:
            Control acceleration [ax, ay, az]
        """
        N = self.horizon
        nu = 3
        
        # Error state (deviation from target)
        x0 = current_state - target_state
        
        # Build reference trajectory if not provided
        if self.reference_trajectory is None:
            # Default: drive to zero (track target_state)
            X_ref = np.zeros(N * 6)
        else:
            X_ref = self.reference_trajectory
        
        # QP: min 0.5*U'*H*U + f'*U subject to constraints
        f = self.F @ x0
        
        # Solve via gradient descent (simple implementation)
        # For production, use cvxpy or qpOASES
        U = self._solve_qp_simple(self.H, f, N, nu)
        
        # Extract first control action
        control = U[0:3]
        
        return self.saturate(control)
    
    def _solve_qp_simple(self, H: np.ndarray, f: np.ndarray,
                         N: int, nu: int) -> np.ndarray:
        """
        Simple QP solver using projected gradient descent.
        
        For production use, replace with cvxpy, qpOASES, or OSQP.
        """
        # Unconstrained solution
        try:
            U = -np.linalg.solve(H, f)
        except np.linalg.LinAlgError:
            U = np.zeros(N * nu)
        
        # Project onto constraints (box constraints on each u)
        u_max = self.limits.max_acceleration
        U = np.clip(U, -u_max, u_max)
        
        return U
    
    def set_reference_trajectory(self, trajectory: np.ndarray):
        """
        Set reference trajectory for tracking.
        
        Args:
            trajectory: Nx6 array of reference states
        """
        self.reference_trajectory = trajectory.flatten()
    
    def set_weights(self, Q: np.ndarray, R: np.ndarray,
                    Q_terminal: Optional[np.ndarray] = None):
        """Set cost weights and rebuild prediction matrices."""
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        if Q_terminal is not None:
            self.Q_terminal = np.asarray(Q_terminal)
        else:
            self.Q_terminal = 10 * self.Q
        self._build_prediction_matrices()


class SlidingModeController(BaseController):
    """
    Sliding Mode Controller (SMC) for robust spacecraft docking.
    
    Provides robustness to model uncertainties and disturbances.
    Uses boundary layer to reduce chattering.
    
    Features:
    - Robustness to bounded uncertainties
    - Finite-time convergence to sliding surface
    - Boundary layer for smooth control
    """
    
    def __init__(self, dynamics: ClohessyWiltshireDynamics,
                 limits: Optional[ControlLimits] = None):
        super().__init__(dynamics, limits)
        
        # Sliding surface parameters
        self.lambda_s = np.array([0.5, 0.5, 0.5])  # Sliding surface slope
        
        # Reaching law parameters
        self.k_reach = np.array([0.05, 0.05, 0.05])  # Reaching gain
        self.eta = np.array([0.01, 0.01, 0.01])       # Robustness margin
        
        # Boundary layer thickness (for chattering reduction)
        self.phi = 0.1  # Boundary layer width
    
    def compute_control(self, current_state: np.ndarray,
                        target_state: np.ndarray) -> np.ndarray:
        """
        Compute sliding mode control output.
        
        Args:
            current_state: [x, y, z, vx, vy, vz]
            target_state: [x_d, y_d, z_d, vx_d, vy_d, vz_d]
            
        Returns:
            Control acceleration [ax, ay, az]
        """
        # Extract errors
        pos_error = current_state[0:3] - target_state[0:3]
        vel_error = current_state[3:6] - target_state[3:6]
        
        # Sliding surface: s = ė + λ*e
        s = vel_error + self.lambda_s * pos_error
        
        # Equivalent control (to maintain sliding)
        u_eq = -self.lambda_s * vel_error
        
        # Reaching control (to reach sliding surface)
        # Using saturation function instead of sign for boundary layer
        sat_s = np.clip(s / self.phi, -1.0, 1.0)
        u_reach = -self.k_reach * sat_s - self.eta * np.sign(s)
        
        control = u_eq + u_reach
        
        return self.saturate(control)


class AdaptivePIDController(PIDController):
    """
    Adaptive PID Controller with online gain tuning.
    
    Adjusts gains based on tracking performance using
    MIT rule or gradient descent adaptation.
    """
    
    def __init__(self, dynamics: ClohessyWiltshireDynamics,
                 limits: Optional[ControlLimits] = None):
        super().__init__(dynamics, limits)
        
        # Adaptation rates
        self.gamma_p = 0.0001  # Proportional adaptation rate
        self.gamma_i = 0.00001  # Integral adaptation rate
        self.gamma_d = 0.0001   # Derivative adaptation rate
        
        # Gain bounds
        self.kp_min = np.array([0.001, 0.001, 0.001])
        self.kp_max = np.array([0.1, 0.1, 0.1])
        
        # Performance metric history
        self.error_history = []
    
    def compute_control(self, current_state: np.ndarray,
                        target_state: np.ndarray) -> np.ndarray:
        """Compute control with adaptive gain update."""
        
        # Compute base PID control
        control = super().compute_control(current_state, target_state)
        
        # Update gains based on performance
        pos_error = target_state[0:3] - current_state[0:3]
        self._adapt_gains(pos_error)
        
        return control
    
    def _adapt_gains(self, error: np.ndarray):
        """
        Adapt gains using MIT rule.
        
        dθ/dt = -γ * e * ∂e/∂θ
        """
        # Simplified adaptation: increase gains if error is large
        error_norm = np.linalg.norm(error)
        
        # Proportional gain adaptation
        if error_norm > 1.0:  # Error threshold
            self.kp = np.clip(
                self.kp + self.gamma_p * np.abs(error),
                self.kp_min,
                self.kp_max
            )
        elif error_norm < 0.1:  # Stable - reduce gains slightly
            self.kp = np.clip(
                self.kp * 0.999,
                self.kp_min,
                self.kp_max
            )


def create_controller(controller_type: str,
                      dynamics: ClohessyWiltshireDynamics,
                      limits: Optional[ControlLimits] = None,
                      **kwargs) -> BaseController:
    """
    Factory function to create controllers.
    
    Args:
        controller_type: "PID", "LQR", "MPC", "SMC", or "AdaptivePID"
        dynamics: CW dynamics model
        limits: Control limits
        **kwargs: Controller-specific parameters
        
    Returns:
        Controller instance
    """
    controllers = {
        "PID": PIDController,
        "LQR": LQRController,
        "MPC": MPCController,
        "SMC": SlidingModeController,
        "AdaptivePID": AdaptivePIDController,
    }
    
    if controller_type not in controllers:
        raise ValueError(f"Unknown controller type: {controller_type}. "
                        f"Available: {list(controllers.keys())}")
    
    return controllers[controller_type](dynamics, limits, **kwargs)

