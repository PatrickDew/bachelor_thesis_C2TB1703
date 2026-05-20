"""
State Estimator Module for Spacecraft Docking

Processes pose estimation from FoundationPose and converts to
spacecraft state in LVLH frame with filtering and prediction.

Features:
- Extended Kalman Filter (EKF) for state estimation
- Coordinate transformation from camera to LVLH frame
- Velocity estimation from pose differentiation
- Outlier rejection and covariance tracking

Author: Sichen Tao
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from scipy.spatial.transform import Rotation


@dataclass 
class EstimatorConfig:
    """Configuration for state estimator."""
    # Process noise (state uncertainty growth)
    q_position: float = 0.001      # Position process noise (m²)
    q_velocity: float = 0.01       # Velocity process noise (m²/s²)
    q_attitude: float = 0.0001     # Attitude process noise (rad²)
    q_angular_vel: float = 0.001   # Angular velocity process noise (rad²/s²)
    
    # Measurement noise (sensor uncertainty)
    r_position: float = 0.01       # Position measurement noise (m²)
    r_attitude: float = 0.001      # Attitude measurement noise (rad²)
    
    # Outlier rejection threshold (Mahalanobis distance)
    outlier_threshold: float = 5.0
    
    # Velocity estimation
    velocity_filter_alpha: float = 0.3  # Low-pass filter coefficient


class StateEstimator:
    """
    Extended Kalman Filter for spacecraft state estimation.
    
    Fuses pose estimation with CW dynamics model for smooth,
    accurate state estimates suitable for control.
    
    State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    """
    
    def __init__(self, config: Optional[EstimatorConfig] = None,
                 dynamics=None):
        self.config = config or EstimatorConfig()
        self.dynamics = dynamics
        
        # State dimension
        self.n_states = 13  # 6 translational + 4 quaternion + 3 angular velocity
        
        # Initialize state and covariance
        self.state = np.zeros(self.n_states)
        self.state[6] = 1.0  # Quaternion w = 1 (identity)
        
        # Covariance matrix
        self.P = np.eye(self.n_states) * 10.0  # Large initial uncertainty
        
        # Process noise covariance Q
        self.Q = self._build_process_noise()
        
        # Measurement noise covariance R
        self.R = self._build_measurement_noise()
        
        # Previous measurements for velocity estimation
        self.prev_position = None
        self.prev_attitude = None
        self.prev_time = None
        
        # Filtered velocity
        self.filtered_velocity = np.zeros(3)
        self.filtered_angular_velocity = np.zeros(3)
        
        # Statistics
        self.measurement_count = 0
        self.rejected_count = 0
        self.is_initialized = False
    
    def _build_process_noise(self) -> np.ndarray:
        """Build process noise covariance matrix Q."""
        Q = np.zeros((self.n_states, self.n_states))
        
        # Position process noise
        Q[0:3, 0:3] = np.eye(3) * self.config.q_position
        
        # Velocity process noise
        Q[3:6, 3:6] = np.eye(3) * self.config.q_velocity
        
        # Attitude process noise (quaternion)
        Q[6:10, 6:10] = np.eye(4) * self.config.q_attitude
        
        # Angular velocity process noise
        Q[10:13, 10:13] = np.eye(3) * self.config.q_angular_vel
        
        return Q
    
    def _build_measurement_noise(self) -> np.ndarray:
        """Build measurement noise covariance matrix R."""
        # Measurement: [x, y, z, qw, qx, qy, qz]
        R = np.zeros((7, 7))
        
        # Position measurement noise
        R[0:3, 0:3] = np.eye(3) * self.config.r_position
        
        # Attitude measurement noise
        R[3:7, 3:7] = np.eye(4) * self.config.r_attitude
        
        return R
    
    def predict(self, dt: float, control: Optional[np.ndarray] = None):
        """
        EKF prediction step using dynamics model.
        
        Args:
            dt: Time step (seconds)
            control: Optional control input [ax, ay, az]
        """
        if control is None:
            control = np.zeros(3)
        
        # State prediction using CW dynamics (translational)
        if self.dynamics is not None:
            Phi = self.dynamics.get_state_transition_matrix(dt)
            Gamma = self.dynamics.get_control_matrix(dt)
            
            # Predict translational state
            trans_state = self.state[0:6]
            self.state[0:6] = Phi @ trans_state + Gamma @ control
        else:
            # Simple integration if no dynamics model
            self.state[0:3] += self.state[3:6] * dt + 0.5 * control * dt**2
            self.state[3:6] += control * dt
        
        # Predict attitude (simple integration)
        omega = self.state[10:13]
        quat = self.state[6:10]
        quat_dot = self._quaternion_derivative(quat, omega)
        self.state[6:10] += quat_dot * dt
        self.state[6:10] /= np.linalg.norm(self.state[6:10])  # Normalize
        
        # Covariance prediction
        # Simplified: use constant process noise addition
        # Full implementation would linearize dynamics
        self.P = self.P + self.Q * dt
    
    def update(self, position: np.ndarray, quaternion: np.ndarray,
               timestamp: float, confidence: float = 1.0) -> bool:
        """
        EKF update step with new measurement.
        
        Args:
            position: Measured position [x, y, z] in camera frame
            quaternion: Measured orientation [qw, qx, qy, qz]
            timestamp: Measurement timestamp
            confidence: Pose estimation confidence (0-1)
            
        Returns:
            True if measurement was accepted, False if rejected
        """
        self.measurement_count += 1
        
        # Scale measurement noise by inverse confidence
        R_scaled = self.R / max(confidence, 0.1)
        
        # Compute measurement
        z = np.concatenate([position, quaternion])
        
        # Predicted measurement
        z_pred = np.concatenate([self.state[0:3], self.state[6:10]])
        
        # Innovation (measurement residual)
        y = z - z_pred
        
        # Handle quaternion wraparound
        if y[3] < -0.5:  # qw sign flip
            y[3:7] = -y[3:7]
        
        # Measurement matrix H (identity for direct measurements)
        H = np.zeros((7, self.n_states))
        H[0:3, 0:3] = np.eye(3)  # Position
        H[3:7, 6:10] = np.eye(4)  # Quaternion
        
        # Innovation covariance
        S = H @ self.P @ H.T + R_scaled
        
        # Mahalanobis distance for outlier detection
        try:
            mahal_dist = np.sqrt(y.T @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            mahal_dist = np.linalg.norm(y)
        
        if mahal_dist > self.config.outlier_threshold:
            self.rejected_count += 1
            return False
        
        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros((self.n_states, 7))
        
        # State update
        self.state = self.state + K @ y
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.n_states) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_scaled @ K.T
        
        # Normalize quaternion
        self.state[6:10] /= np.linalg.norm(self.state[6:10])
        
        # Estimate velocity from finite differences
        if self.prev_position is not None and self.prev_time is not None:
            dt = timestamp - self.prev_time
            if dt > 0.001:  # Avoid division by very small dt
                raw_velocity = (position - self.prev_position) / dt
                # Low-pass filter
                alpha = self.config.velocity_filter_alpha
                self.filtered_velocity = (alpha * raw_velocity + 
                                         (1 - alpha) * self.filtered_velocity)
                # Update state velocity if not yet converged
                if not self.is_initialized:
                    self.state[3:6] = self.filtered_velocity
        
        # Estimate angular velocity from quaternion differences
        if self.prev_attitude is not None and self.prev_time is not None:
            dt = timestamp - self.prev_time
            if dt > 0.001:
                raw_angular_vel = self._estimate_angular_velocity(
                    self.prev_attitude, quaternion, dt)
                alpha = self.config.velocity_filter_alpha
                self.filtered_angular_velocity = (alpha * raw_angular_vel + 
                                                 (1 - alpha) * self.filtered_angular_velocity)
                if not self.is_initialized:
                    self.state[10:13] = self.filtered_angular_velocity
        
        # Update previous values
        self.prev_position = position.copy()
        self.prev_attitude = quaternion.copy()
        self.prev_time = timestamp
        
        if self.measurement_count > 10:
            self.is_initialized = True
        
        return True
    
    def _quaternion_derivative(self, q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Compute quaternion derivative from angular velocity."""
        qw, qx, qy, qz = q
        wx, wy, wz = omega
        
        q_dot = 0.5 * np.array([
            -qx*wx - qy*wy - qz*wz,
            qw*wx + qy*wz - qz*wy,
            qw*wy + qz*wx - qx*wz,
            qw*wz + qx*wy - qy*wx
        ])
        
        return q_dot
    
    def _estimate_angular_velocity(self, q1: np.ndarray, q2: np.ndarray,
                                    dt: float) -> np.ndarray:
        """Estimate angular velocity from two quaternions."""
        # Quaternion difference: q_diff = q2 * q1^(-1)
        r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])  # xyzw format
        r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
        
        r_diff = r2 * r1.inv()
        rotvec = r_diff.as_rotvec()
        
        return rotvec / dt
    
    def get_state(self) -> np.ndarray:
        """Return current state estimate."""
        return self.state.copy()
    
    def get_translational_state(self) -> np.ndarray:
        """Return position and velocity [x, y, z, vx, vy, vz]."""
        return self.state[0:6].copy()
    
    def get_position(self) -> np.ndarray:
        """Return position [x, y, z]."""
        return self.state[0:3].copy()
    
    def get_velocity(self) -> np.ndarray:
        """Return velocity [vx, vy, vz]."""
        return self.state[3:6].copy()
    
    def get_attitude(self) -> np.ndarray:
        """Return attitude quaternion [qw, qx, qy, qz]."""
        return self.state[6:10].copy()
    
    def get_angular_velocity(self) -> np.ndarray:
        """Return angular velocity [wx, wy, wz]."""
        return self.state[10:13].copy()
    
    def get_covariance(self) -> np.ndarray:
        """Return state covariance matrix."""
        return self.P.copy()
    
    def get_position_uncertainty(self) -> np.ndarray:
        """Return 1-sigma position uncertainty (m)."""
        return np.sqrt(np.diag(self.P[0:3, 0:3]))
    
    def reset(self, initial_state: Optional[np.ndarray] = None):
        """Reset estimator to initial state."""
        if initial_state is not None:
            self.state = initial_state.copy()
        else:
            self.state = np.zeros(self.n_states)
            self.state[6] = 1.0
        
        self.P = np.eye(self.n_states) * 10.0
        self.prev_position = None
        self.prev_attitude = None
        self.prev_time = None
        self.filtered_velocity = np.zeros(3)
        self.filtered_angular_velocity = np.zeros(3)
        self.measurement_count = 0
        self.rejected_count = 0
        self.is_initialized = False


class CoordinateTransformer:
    """
    Coordinate transformation between frames.
    
    Transforms:
    - Camera frame to body frame
    - Body frame to LVLH frame
    - Inertial to LVLH
    """
    
    def __init__(self, camera_to_body: Optional[np.ndarray] = None):
        """
        Initialize transformer.
        
        Args:
            camera_to_body: 4x4 transform from camera to body frame
        """
        # Default: camera aligned with body -Z axis (looking forward)
        if camera_to_body is None:
            self.camera_to_body = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            self.camera_to_body = camera_to_body
    
    def camera_to_lvlh(self, position_camera: np.ndarray,
                       quaternion_camera: np.ndarray,
                       body_to_lvlh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform pose from camera frame to LVLH frame.
        
        Args:
            position_camera: Position in camera frame
            quaternion_camera: Orientation in camera frame
            body_to_lvlh: Rotation matrix from body to LVLH
            
        Returns:
            position_lvlh, quaternion_lvlh
        """
        # Camera to body
        R_cb = self.camera_to_body[0:3, 0:3]
        t_cb = self.camera_to_body[0:3, 3]
        
        position_body = R_cb @ position_camera + t_cb
        
        # Body to LVLH
        position_lvlh = body_to_lvlh @ position_body
        
        # Quaternion transformation
        r_camera = Rotation.from_quat([quaternion_camera[1], quaternion_camera[2],
                                       quaternion_camera[3], quaternion_camera[0]])
        r_cb = Rotation.from_matrix(R_cb)
        r_bl = Rotation.from_matrix(body_to_lvlh)
        
        r_lvlh = r_bl * r_cb * r_camera
        q_lvlh = r_lvlh.as_quat()  # xyzw
        quaternion_lvlh = np.array([q_lvlh[3], q_lvlh[0], q_lvlh[1], q_lvlh[2]])  # wxyz
        
        return position_lvlh, quaternion_lvlh

