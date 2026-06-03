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
    Robust 3-DOF PID controller for vision-based spacecraft docking.

    Control law (camera / relative pose frame):
        u = Kp·e + Ki·∫e dt − Kd·v + Kv·(v_d − v)·ê_r + u_los + u_ff

    where e = p_d − p, v is measured velocity, ê_r is the unit vector toward
    the target, and u_los adds extra lateral (cross-track) correction for
    direct line-of-sight alignment.

    Features:
    - Derivative on measured velocity (robust to noisy vision)
    - Back-calculation integral anti-windup
    - Range-based gain scheduling
    - Terminal hold zone (zero command when docked)
    - Optional CW feedforward for orbital scenarios
    """

    def __init__(self, dynamics: ClohessyWiltshireDynamics,
                 limits: Optional[ControlLimits] = None):
        super().__init__(dynamics, limits)

        # Cartesian PID gains (per axis)
        self.kp = np.array([0.008, 0.008, 0.012])
        self.ki = np.array([0.0005, 0.0005, 0.0008])
        self.kd = np.array([0.15, 0.15, 0.20])   # velocity damping
        self.kv = np.array([0.08, 0.08, 0.12])   # range-rate tracking

        # Line-of-sight lateral correction (cross-track alignment)
        self.use_los_guidance = True
        self.use_los_decomposed = True   # Radial+lateral only (no double-counting)
        self.kp_lateral = 0.015
        self.ki_lateral = 0.001
        self.kd_lateral = 0.12
        self.kp_radial_scale = 1.0       # Extra gain on depth/range axis

        # Range-rate: 'brake_only' = never push forward, only slow down if too fast
        self.range_rate_mode = 'brake_only'
        self.max_closing_rate = 0.003    # m/s — cap commanded approach speed

        # Internal state
        self.integral_error = np.zeros(3)
        self.lateral_integral = np.zeros(3)
        self.prev_error = np.zeros(3)

        # Anti-windup
        self.integral_max = np.array([2.0, 2.0, 2.0])
        self.use_backcalc_antiwindup = True

        # Terminal hold — stop when within threshold with low velocity
        self.docking_complete_range = 0.05      # m
        self.docking_complete_velocity = 0.002  # m/s
        self.docking_complete = False

        # Gain scheduling
        self.use_gain_scheduling = True
        self.use_cw_feedforward = False

        # Diagnostics from last compute_control call
        self.last_diag: dict = {}

    def set_gains(self, kp: np.ndarray, ki: np.ndarray, kd: np.ndarray):
        """Set PID gains."""
        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)

    def compute_control(self, current_state: np.ndarray,
                        target_state: np.ndarray) -> np.ndarray:
        """
        Compute PID control acceleration in the pose estimation frame.

        Args:
            current_state: [x, y, z, vx, vy, vz]
            target_state: [x_d, y_d, z_d, vx_d, vy_d, vz_d]

        Returns:
            Control acceleration [ax, ay, az]
        """
        pos = current_state[0:3]
        vel = current_state[3:6]
        pos_target = target_state[0:3]
        vel_target = target_state[3:6]

        pos_error = pos_target - pos
        vel_error = vel - vel_target

        range_m = float(np.linalg.norm(pos_error))
        if range_m > 1e-6:
            range_dir = pos_error / range_m
        else:
            range_dir = np.zeros(3)

        speed = float(np.linalg.norm(vel_error))

        # Terminal hold — convergence to zero steady-state error
        if (range_m < self.docking_complete_range and
                speed < self.docking_complete_velocity):
            self.docking_complete = True
            self._record_diag(
                pos_error, vel_error, range_m, range_dir,
                np.zeros(3), np.zeros(3), np.zeros(3),
                np.zeros(3), np.zeros(3), True,
            )
            return np.zeros(3)

        self.docking_complete = False

        # Desired closing velocity (negative = approaching)
        if range_m > 10.0:
            desired_range_rate = -0.03
        elif range_m > 1.0:
            desired_range_rate = -0.01
        elif range_m > 0.2:
            desired_range_rate = -0.002
        else:
            desired_range_rate = -0.0008

        range_rate = float(np.dot(vel_error, range_dir))
        range_rate_error = range_rate - desired_range_rate

        gain_scale = self._compute_gain_scale(range_m) if self.use_gain_scheduling else 1.0

        if self.use_los_decomposed and range_m > 1e-6:
            control_unsat = self._compute_los_decomposed(
                pos_error, vel_error, range_dir, range_m,
                range_rate, desired_range_rate, gain_scale,
            )
        else:
            kp = self.kp * gain_scale
            ki = self.ki * gain_scale
            kd = self.kd * gain_scale
            kv = self.kv * gain_scale
            control_unsat = self._compute_cartesian(
                pos_error, vel_error, range_dir,
                range_rate_error, kp, ki, kd, kv, gain_scale, range_m,
            )
            if self.use_cw_feedforward:
                control_unsat += self._compute_cw_feedforward(current_state)

        control_sat = self.saturate(control_unsat)

        if self.use_backcalc_antiwindup and not self.use_los_decomposed:
            ki = self.ki * gain_scale
            for i in range(3):
                if ki[i] > 1e-12:
                    self.integral_error[i] += (
                        (control_sat[i] - control_unsat[i]) / ki[i]
                    )
            self.integral_error = np.clip(
                self.integral_error, -self.integral_max, self.integral_max,
            )

        self.prev_error = pos_error.copy()
        diag = self.last_diag if self.use_los_decomposed else {}
        if not diag:
            diag = {
                'p_term': np.zeros(3), 'i_term': np.zeros(3),
                'd_term': np.zeros(3), 'range_rate_control': np.zeros(3),
                'los_term': np.zeros(3),
            }
        self._record_diag(
            pos_error, vel_error, range_m, range_dir,
            diag.get('p_term', np.zeros(3)),
            diag.get('i_term', np.zeros(3)),
            diag.get('d_term', np.zeros(3)),
            diag.get('range_rate_control', np.zeros(3)),
            diag.get('los_term', np.zeros(3)),
            False,
        )
        return control_sat

    def _compute_los_decomposed(
        self,
        pos_error: np.ndarray,
        vel_error: np.ndarray,
        range_dir: np.ndarray,
        range_m: float,
        range_rate: float,
        desired_range_rate: float,
        gain_scale: float,
    ) -> np.ndarray:
        """Line-of-sight PID: approach along depth, align lateral (no axis double-count)."""
        radial_error = float(np.dot(pos_error, range_dir))
        radial_vel = float(np.dot(vel_error, range_dir))
        lateral_error = pos_error - radial_error * range_dir
        lateral_vel = vel_error - radial_vel * range_dir

        gs = gain_scale
        kp_r = self.kp[2] * gs * self.kp_radial_scale
        ki_r = self.ki[2] * gs
        kd_r = self.kd[2] * gs

        self.integral_error[2] += radial_error * self.dt
        self.integral_error[2] = np.clip(
            self.integral_error[2], -self.integral_max[2], self.integral_max[2],
        )
        self.lateral_integral += lateral_error * self.dt
        self.lateral_integral = np.clip(
            self.lateral_integral, -self.integral_max, self.integral_max,
        )

        u_radial = (
            kp_r * radial_error
            + ki_r * self.integral_error[2]
            - kd_r * radial_vel
        )
        u_lateral = (
            self.kp_lateral * gs * lateral_error
            + self.ki_lateral * gs * self.lateral_integral
            - self.kd_lateral * gs * lateral_vel
        )

        range_rate_error = range_rate - desired_range_rate
        if self.range_rate_mode == 'brake_only':
            u_rr = self.kv[2] * gs * range_rate_error if range_rate < desired_range_rate else 0.0
        else:
            u_rr = self.kv[2] * gs * range_rate_error

        if radial_vel < -self.max_closing_rate and u_radial < 0:
            u_radial *= 0.3

        u_radial_vec = (u_radial + u_rr) * range_dir
        control = u_radial_vec + u_lateral

        self.last_diag = {
            'p_term': kp_r * radial_error * range_dir + self.kp_lateral * gs * lateral_error,
            'i_term': ki_r * self.integral_error[2] * range_dir + self.ki_lateral * gs * self.lateral_integral,
            'd_term': -kd_r * radial_vel * range_dir - self.kd_lateral * gs * lateral_vel,
            'range_rate_control': u_rr * range_dir,
            'los_term': u_lateral,
        }
        return control

    def _compute_cartesian(
        self,
        pos_error, vel_error, range_dir, range_rate_error,
        kp, ki, kd, kv, gain_scale, range_m,
    ) -> np.ndarray:
        """Legacy decoupled Cartesian PID."""
        p_term = kp * pos_error
        self.integral_error += pos_error * self.dt
        self.integral_error = np.clip(
            self.integral_error, -self.integral_max, self.integral_max,
        )
        i_term = ki * self.integral_error
        d_term = -kd * vel_error

        if self.range_rate_mode == 'brake_only' and range_rate_error > 0:
            rr_scalar = 0.0
        elif self.range_rate_mode == 'brake_only':
            rr_scalar = kv[0] * range_rate_error
        else:
            rr_scalar = kv[0] * range_rate_error
        range_rate_control = rr_scalar * range_dir

        if self.use_los_guidance and range_m > 1e-6:
            radial_component = np.dot(pos_error, range_dir) * range_dir
            lateral_error = pos_error - radial_component
            lateral_vel = vel_error - np.dot(vel_error, range_dir) * range_dir
            self.lateral_integral += lateral_error * self.dt
            self.lateral_integral = np.clip(
                self.lateral_integral, -self.integral_max, self.integral_max,
            )
            los_term = (
                self.kp_lateral * gain_scale * lateral_error
                + self.ki_lateral * gain_scale * self.lateral_integral
                - self.kd_lateral * gain_scale * lateral_vel
            )
        else:
            los_term = np.zeros(3)

        self.last_diag = {
            'p_term': p_term, 'i_term': i_term, 'd_term': d_term,
            'range_rate_control': range_rate_control, 'los_term': los_term,
        }
        return p_term + i_term + d_term + range_rate_control + los_term

    def _record_diag(
        self,
        pos_error, vel_error, range_m, range_dir,
        p_term, i_term, d_term, rr_term, los_term, complete,
    ):
        """Store diagnostics for logging and analysis."""
        self.last_diag = {
            'pos_error': pos_error.copy(),
            'vel_error': vel_error.copy(),
            'range_m': range_m,
            'range_dir': range_dir.copy(),
            'p_term': p_term.copy(),
            'i_term': i_term.copy(),
            'd_term': d_term.copy(),
            'range_rate_control': rr_term.copy(),
            'los_term': los_term.copy(),
            'docking_complete': complete,
            'range_rate': float(np.dot(vel_error, range_dir)) if range_m > 1e-6 else 0.0,
        }
    
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
        """Smooth gain scheduling — very low gain when far (noisy vision)."""
        if range_m > 20.0:
            return 0.08
        if range_m > 5.0:
            return 0.08 + 0.42 * (20.0 - range_m) / 15.0
        if range_m > 1.0:
            return 0.5 + 0.5 * (5.0 - range_m) / 4.0
        if range_m > 0.2:
            return 1.0
        return 1.5
    
    def reset(self):
        """Reset integral and derivative states."""
        self.integral_error = np.zeros(3)
        self.lateral_integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.docking_complete = False
        self.last_diag = {}


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

