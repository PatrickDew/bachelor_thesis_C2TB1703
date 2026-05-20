"""
Spacecraft Relative Motion Dynamics Module

Implements the Clohessy-Wiltshire (Hill's) equations for linearized relative
orbital motion in the LVLH (Local Vertical Local Horizontal) frame.

Reference:
- Clohessy, W. H., & Wiltshire, R. S. (1960). "Terminal Guidance System for 
  Satellite Rendezvous." Journal of the Aerospace Sciences.
- Fehse, W. (2003). "Automated Rendezvous and Docking of Spacecraft."
  Cambridge Aerospace Series.

The LVLH frame is centered on the target spacecraft with:
- x (radial): positive away from Earth center
- y (along-track): positive in velocity direction
- z (cross-track): completes right-handed system

Author: Sichen Tao
"""

import numpy as np
from scipy.linalg import expm
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class OrbitalParameters:
    """Orbital parameters for the target spacecraft."""
    semi_major_axis: float = 6.778e6  # ISS orbit ~400km altitude (m)
    eccentricity: float = 0.0         # Circular orbit assumption
    inclination: float = 51.6         # ISS inclination (deg)
    mu: float = 3.986004418e14        # Earth gravitational parameter (m^3/s^2)
    
    @property
    def mean_motion(self) -> float:
        """Calculate mean motion n = sqrt(mu/a^3) (rad/s)."""
        return np.sqrt(self.mu / self.semi_major_axis**3)
    
    @property
    def orbital_period(self) -> float:
        """Orbital period T = 2*pi/n (seconds)."""
        return 2.0 * np.pi / self.mean_motion


class ClohessyWiltshireDynamics:
    """
    Clohessy-Wiltshire (CW) equations for linearized relative orbital motion.
    
    The CW equations describe the motion of a chaser spacecraft relative to 
    a target in a circular reference orbit. Valid for small relative distances
    compared to orbital radius.
    
    State vector: [x, y, z, vx, vy, vz]^T (6 states)
    Control input: [Fx, Fy, Fz]^T / mass (accelerations in m/s^2)
    
    Continuous-time dynamics: ẋ = Ax + Bu
    Discrete-time dynamics: x_{k+1} = Φx_k + Γu_k
    """
    
    def __init__(self, orbital_params: Optional[OrbitalParameters] = None,
                 spacecraft_mass: float = 500.0):
        """
        Initialize CW dynamics model.
        
        Args:
            orbital_params: Orbital parameters (defaults to ISS-like orbit)
            spacecraft_mass: Chaser spacecraft mass in kg
        """
        self.orbital_params = orbital_params or OrbitalParameters()
        self.mass = spacecraft_mass
        self.n = self.orbital_params.mean_motion  # Mean motion (rad/s)
        
        # Pre-compute matrices
        self._compute_state_matrices()
    
    def _compute_state_matrices(self):
        """Compute continuous-time A and B matrices for CW equations."""
        n = self.n
        
        # Continuous-time state matrix A (6x6)
        # CW equations:
        # ẍ = 3n²x + 2nẏ + ax
        # ÿ = -2nẋ + ay
        # z̈ = -n²z + az
        self.A_continuous = np.array([
            [0,      0,   0,   1,    0,   0],
            [0,      0,   0,   0,    1,   0],
            [0,      0,   0,   0,    0,   1],
            [3*n**2, 0,   0,   0,    2*n, 0],
            [0,      0,   0,  -2*n,  0,   0],
            [0,      0, -n**2, 0,    0,   0]
        ])
        
        # Continuous-time control matrix B (6x3)
        self.B_continuous = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    
    def get_state_transition_matrix(self, dt: float) -> np.ndarray:
        """
        Compute closed-form state transition matrix Φ(t).
        
        For CW equations, the analytical solution exists:
        Φ(t) = exp(A*t) has closed form.
        
        Args:
            dt: Time interval in seconds
            
        Returns:
            6x6 state transition matrix
        """
        n = self.n
        nt = n * dt
        c = np.cos(nt)
        s = np.sin(nt)
        
        # Analytical state transition matrix (Fehse, 2003)
        Phi = np.array([
            [4 - 3*c,      0,  0,      s/n,        2*(1-c)/n,    0],
            [6*(s - nt),   1,  0,     -2*(1-c)/n,  (4*s - 3*nt)/n, 0],
            [0,            0,  c,      0,          0,            s/n],
            [3*n*s,        0,  0,      c,          2*s,          0],
            [-6*n*(1-c),   0,  0,     -2*s,        4*c - 3,      0],
            [0,            0, -n*s,   0,          0,            c]
        ])
        
        return Phi
    
    def get_control_matrix(self, dt: float) -> np.ndarray:
        """
        Compute discrete-time control input matrix Γ.
        
        Γ = ∫₀^dt Φ(τ)B dτ
        
        Args:
            dt: Time interval in seconds
            
        Returns:
            6x3 control input matrix
        """
        n = self.n
        nt = n * dt
        c = np.cos(nt)
        s = np.sin(nt)
        
        # Analytical control influence matrix
        Gamma = np.array([
            [(1-c)/n**2,         2*(nt-s)/n**2,        0],
            [-2*(nt-s)/n**2,     (4*(1-c) - 1.5*n*dt**2)/n**2, 0],
            [0,                  0,                     (1-c)/n**2],
            [s/n,                2*(1-c)/n,            0],
            [-2*(1-c)/n,         (4*s - 3*dt)/n,       0],
            [0,                  0,                     s/n]
        ])
        
        return Gamma
    
    def propagate(self, state: np.ndarray, control: np.ndarray, 
                  dt: float) -> np.ndarray:
        """
        Propagate state forward in time with control input.
        
        Args:
            state: Current state [x, y, z, vx, vy, vz]
            control: Control accelerations [ax, ay, az] (m/s^2)
            dt: Time step (seconds)
            
        Returns:
            Next state
        """
        Phi = self.get_state_transition_matrix(dt)
        Gamma = self.get_control_matrix(dt)
        
        return Phi @ state + Gamma @ control
    
    def compute_cw_trajectory(self, initial_state: np.ndarray, 
                              time_span: float, dt: float = 1.0,
                              control_profile: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute trajectory over specified time span.
        
        Args:
            initial_state: Initial state vector [x, y, z, vx, vy, vz]
            time_span: Total simulation time (seconds)
            dt: Time step (seconds)
            control_profile: Optional Nx3 control inputs, defaults to zero
            
        Returns:
            times: Array of time points
            states: Nx6 array of states at each time
        """
        n_steps = int(time_span / dt)
        times = np.linspace(0, time_span, n_steps + 1)
        states = np.zeros((n_steps + 1, 6))
        states[0] = initial_state
        
        if control_profile is None:
            control_profile = np.zeros((n_steps, 3))
        
        for i in range(n_steps):
            control = control_profile[i] if i < len(control_profile) else np.zeros(3)
            states[i + 1] = self.propagate(states[i], control, dt)
        
        return times, states
    
    def compute_v_bar_approach(self, range_m: float, approach_velocity: float,
                               lateral_offset: float = 0.0) -> np.ndarray:
        """
        Compute state for V-bar (velocity vector) approach.
        
        V-bar approach is along the orbital velocity direction (y-axis).
        Standard for ISS docking.
        
        Args:
            range_m: Distance along V-bar (y-axis)
            approach_velocity: Closing velocity (negative for approach)
            lateral_offset: Offset in x-direction
            
        Returns:
            State vector for V-bar approach
        """
        return np.array([
            lateral_offset,      # x: radial offset
            range_m,             # y: along-track (V-bar)
            0.0,                 # z: cross-track
            0.0,                 # vx
            approach_velocity,   # vy: approach velocity
            0.0                  # vz
        ])
    
    def compute_r_bar_approach(self, range_m: float, approach_velocity: float,
                               along_track_offset: float = 0.0) -> np.ndarray:
        """
        Compute state for R-bar (radial) approach.
        
        R-bar approach is along the Earth radial direction (x-axis).
        Used for some docking scenarios.
        
        Args:
            range_m: Distance along R-bar (x-axis)
            approach_velocity: Closing velocity (negative for approach)
            along_track_offset: Offset in y-direction
            
        Returns:
            State vector for R-bar approach
        """
        return np.array([
            range_m,             # x: radial (R-bar)
            along_track_offset,  # y: along-track offset
            0.0,                 # z
            approach_velocity,   # vx: approach velocity
            0.0,                 # vy
            0.0                  # vz
        ])
    
    def is_on_natural_motion_circumnavigation(self, state: np.ndarray,
                                               tolerance: float = 0.01) -> bool:
        """
        Check if state lies on a natural motion circumnavigation (NMC) trajectory.
        
        NMC trajectories are periodic, closed orbits around the target that
        require no control input (free-drift safety corridor).
        
        Args:
            state: State vector
            tolerance: Relative tolerance for checking
            
        Returns:
            True if on NMC trajectory
        """
        x, y, z, vx, vy, vz = state
        n = self.n
        
        # NMC conditions (Fehse, 2003):
        # vy = -2*n*x (for in-plane)
        # Out-of-plane is independent oscillation
        expected_vy = -2 * n * x
        
        if np.abs(y) < tolerance:
            return np.abs(vy - expected_vy) < tolerance * np.abs(expected_vy + 1e-10)
        return False


class SpacecraftDynamics:
    """
    Full 6-DOF spacecraft dynamics including attitude.
    
    Combines CW translational dynamics with rigid body rotational dynamics.
    Uses quaternion for attitude representation.
    """
    
    def __init__(self, mass: float = 500.0, 
                 inertia: Optional[np.ndarray] = None,
                 orbital_params: Optional[OrbitalParameters] = None):
        """
        Initialize 6-DOF spacecraft dynamics.
        
        Args:
            mass: Spacecraft mass (kg)
            inertia: 3x3 inertia tensor (kg*m^2), defaults to symmetric
            orbital_params: Orbital parameters
        """
        self.mass = mass
        
        # Default inertia tensor (typical small spacecraft)
        if inertia is None:
            self.inertia = np.diag([50.0, 50.0, 30.0])  # kg*m^2
        else:
            self.inertia = inertia
        
        self.inertia_inv = np.linalg.inv(self.inertia)
        
        # CW dynamics for translation
        self.cw = ClohessyWiltshireDynamics(orbital_params, mass)
    
    def quaternion_derivative(self, q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Compute quaternion derivative from angular velocity.
        
        q̇ = 0.5 * Ω(ω) * q
        
        Args:
            q: Quaternion [qw, qx, qy, qz]
            omega: Angular velocity [wx, wy, wz] (rad/s)
            
        Returns:
            Quaternion derivative
        """
        qw, qx, qy, qz = q
        wx, wy, wz = omega
        
        # Quaternion multiplication matrix
        Omega = 0.5 * np.array([
            [0,   -wx, -wy, -wz],
            [wx,   0,   wz, -wy],
            [wy, -wz,   0,   wx],
            [wz,  wy, -wx,   0]
        ])
        
        return Omega @ q
    
    def euler_equation(self, omega: np.ndarray, torque: np.ndarray) -> np.ndarray:
        """
        Euler's equation for rigid body rotation.
        
        I*ω̇ = τ - ω × (I*ω)
        
        Args:
            omega: Angular velocity (rad/s)
            torque: Applied torque (N*m)
            
        Returns:
            Angular acceleration (rad/s^2)
        """
        omega_dot = self.inertia_inv @ (torque - np.cross(omega, self.inertia @ omega))
        return omega_dot
    
    def propagate_6dof(self, state: np.ndarray, force: np.ndarray,
                       torque: np.ndarray, dt: float) -> np.ndarray:
        """
        Propagate full 6-DOF state with RK4 integration.
        
        State: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz] (13 elements)
        
        Args:
            state: 13-element state vector
            force: Applied force in body frame (N)
            torque: Applied torque in body frame (N*m)
            dt: Time step (seconds)
            
        Returns:
            Updated state
        """
        def derivatives(s, f, tau):
            pos = s[0:3]
            vel = s[3:6]
            quat = s[6:10]
            omega = s[10:13]
            
            # Translational (CW dynamics)
            accel = self.cw.A_continuous[3:6, :] @ np.concatenate([pos, vel]) + f / self.mass
            
            # Rotational
            quat_dot = self.quaternion_derivative(quat, omega)
            omega_dot = self.euler_equation(omega, tau)
            
            return np.concatenate([vel, accel, quat_dot, omega_dot])
        
        # RK4 integration
        k1 = derivatives(state, force, torque)
        k2 = derivatives(state + 0.5*dt*k1, force, torque)
        k3 = derivatives(state + 0.5*dt*k2, force, torque)
        k4 = derivatives(state + dt*k3, force, torque)
        
        new_state = state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Normalize quaternion
        new_state[6:10] /= np.linalg.norm(new_state[6:10])
        
        return new_state


def compute_delta_v_for_transfer(cw: ClohessyWiltshireDynamics,
                                  start_state: np.ndarray,
                                  target_position: np.ndarray,
                                  transfer_time: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute two-impulse transfer delta-V using CW targeting.
    
    Lambert-like problem for relative motion.
    
    Args:
        cw: CW dynamics model
        start_state: Initial state [x, y, z, vx, vy, vz]
        target_position: Target position [x, y, z]
        transfer_time: Time of flight (seconds)
        
    Returns:
        delta_v1: Initial impulse (m/s)
        delta_v2: Final impulse for station-keeping (m/s)
    """
    Phi = cw.get_state_transition_matrix(transfer_time)
    
    # Partition state transition matrix
    Phi_rr = Phi[0:3, 0:3]  # Position -> Position
    Phi_rv = Phi[0:3, 3:6]  # Velocity -> Position
    Phi_vr = Phi[3:6, 0:3]  # Position -> Velocity
    Phi_vv = Phi[3:6, 3:6]  # Velocity -> Velocity
    
    r0 = start_state[0:3]
    v0 = start_state[3:6]
    rf = target_position
    
    # Required initial velocity for transfer
    v0_required = np.linalg.solve(Phi_rv, rf - Phi_rr @ r0)
    
    # Initial delta-V
    delta_v1 = v0_required - v0
    
    # Final velocity after transfer
    vf = Phi_vr @ r0 + Phi_vv @ v0_required
    
    # For station-keeping at target, final delta-V
    delta_v2 = -vf  # Assumes target is stationary in LVLH
    
    return delta_v1, delta_v2

