"""
Trajectory Generator for Spacecraft Docking

Generates reference trajectories for different docking phases:
- Far-range approach
- Mid-range station keeping
- Close-range final approach
- Terminal docking alignment

Supports V-bar and R-bar approaches with safe corridor constraints.

Author: Sichen Tao
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from .dynamics import ClohessyWiltshireDynamics, compute_delta_v_for_transfer


class ApproachType(Enum):
    """Docking approach direction."""
    V_BAR = "v_bar"      # Along velocity vector (y-axis)
    R_BAR = "r_bar"      # Along radial (x-axis)
    H_BAR = "h_bar"      # Along angular momentum (z-axis)


class DockingPhase(Enum):
    """Docking mission phases."""
    FAR_RANGE = "far_range"        # > 100m
    MID_RANGE = "mid_range"        # 100m - 10m
    CLOSE_RANGE = "close_range"    # 10m - 1m
    TERMINAL = "terminal"          # < 1m
    STATION_KEEPING = "station_keeping"
    HOLD = "hold"                  # Safe hold position


@dataclass
class DockingWaypoint:
    """Waypoint for docking trajectory."""
    position: np.ndarray       # [x, y, z] in LVLH (m)
    velocity: np.ndarray       # [vx, vy, vz] (m/s)
    time_from_start: float     # Time from trajectory start (s)
    phase: DockingPhase
    hold_time: float = 0.0     # Time to hold at this point (s)


@dataclass
class ApproachCorridor:
    """Safe approach corridor definition."""
    half_angle_deg: float = 10.0    # Corridor half-angle
    min_range: float = 0.5          # Minimum standoff distance (m)
    approach_axis: str = 'y'        # Primary approach axis
    
    @property
    def half_angle_rad(self) -> float:
        return np.radians(self.half_angle_deg)


class TrajectoryGenerator:
    """
    Generates docking trajectories with safety constraints.
    
    Implements standard rendezvous and docking approaches:
    - V-bar hop maneuvers
    - R-bar approaches
    - Forced motion corridors
    """
    
    def __init__(self, dynamics: ClohessyWiltshireDynamics,
                 approach_type: ApproachType = ApproachType.V_BAR):
        self.dynamics = dynamics
        self.approach_type = approach_type
        self.corridor = ApproachCorridor()
        
        # Default velocity constraints (m/s)
        self.max_approach_velocity = {
            DockingPhase.FAR_RANGE: 1.0,
            DockingPhase.MID_RANGE: 0.5,
            DockingPhase.CLOSE_RANGE: 0.1,
            DockingPhase.TERMINAL: 0.03,
        }
        
        # Default waypoint ranges (m)
        self.phase_boundaries = {
            DockingPhase.FAR_RANGE: (100.0, float('inf')),
            DockingPhase.MID_RANGE: (10.0, 100.0),
            DockingPhase.CLOSE_RANGE: (1.0, 10.0),
            DockingPhase.TERMINAL: (0.0, 1.0),
        }
    
    def generate_v_bar_approach(self, start_position: np.ndarray,
                                final_range: float = 0.5,
                                n_waypoints: int = 5) -> List[DockingWaypoint]:
        """
        Generate V-bar approach trajectory.
        
        V-bar is the standard ISS docking approach along the
        orbital velocity vector (LVLH y-axis).
        
        Args:
            start_position: Initial position [x, y, z]
            final_range: Final standoff distance (m)
            n_waypoints: Number of waypoints
            
        Returns:
            List of docking waypoints
        """
        waypoints = []
        
        initial_range = np.abs(start_position[1])  # V-bar distance
        
        # Generate waypoint ranges (logarithmic spacing for smoother approach)
        ranges = np.logspace(np.log10(final_range), np.log10(initial_range), n_waypoints)
        ranges = ranges[::-1]  # Start from far, approach closer
        
        total_time = 0.0
        prev_range = initial_range
        
        for i, range_m in enumerate(ranges):
            # Determine phase
            phase = self._get_phase_for_range(range_m)
            
            # Position (V-bar approach: primarily y-axis)
            position = np.array([0.0, range_m, 0.0])
            
            # Compute velocity based on approach rate
            max_vel = self.max_approach_velocity.get(phase, 0.1)
            
            if i < len(ranges) - 1:
                # Approaching velocity (negative because range decreasing)
                approach_vel = -min(max_vel, (prev_range - range_m) / 60.0)
            else:
                approach_vel = 0.0  # Stop at final waypoint
            
            velocity = np.array([0.0, approach_vel, 0.0])
            
            # Time estimate
            if i > 0:
                delta_range = prev_range - range_m
                transfer_time = max(delta_range / max_vel, 30.0)
                total_time += transfer_time
            
            # Hold time at mid-range for inspection
            hold_time = 60.0 if phase == DockingPhase.MID_RANGE else 0.0
            
            waypoints.append(DockingWaypoint(
                position=position,
                velocity=velocity,
                time_from_start=total_time,
                phase=phase,
                hold_time=hold_time
            ))
            
            prev_range = range_m
        
        return waypoints
    
    def generate_r_bar_approach(self, start_position: np.ndarray,
                                final_range: float = 0.5,
                                n_waypoints: int = 5) -> List[DockingWaypoint]:
        """
        Generate R-bar (radial) approach trajectory.
        
        R-bar approach is along the Earth radial direction.
        Inherently passively safe (natural drift away on abort).
        
        Args:
            start_position: Initial position [x, y, z]
            final_range: Final standoff distance (m)
            n_waypoints: Number of waypoints
            
        Returns:
            List of docking waypoints
        """
        waypoints = []
        
        initial_range = np.abs(start_position[0])  # R-bar distance
        
        ranges = np.logspace(np.log10(final_range), np.log10(initial_range), n_waypoints)
        ranges = ranges[::-1]
        
        total_time = 0.0
        prev_range = initial_range
        
        for i, range_m in enumerate(ranges):
            phase = self._get_phase_for_range(range_m)
            
            # Position (R-bar approach: primarily x-axis)
            position = np.array([range_m, 0.0, 0.0])
            
            max_vel = self.max_approach_velocity.get(phase, 0.1)
            
            if i < len(ranges) - 1:
                approach_vel = -min(max_vel, (prev_range - range_m) / 60.0)
            else:
                approach_vel = 0.0
            
            velocity = np.array([approach_vel, 0.0, 0.0])
            
            if i > 0:
                delta_range = prev_range - range_m
                transfer_time = max(delta_range / max_vel, 30.0)
                total_time += transfer_time
            
            waypoints.append(DockingWaypoint(
                position=position,
                velocity=velocity,
                time_from_start=total_time,
                phase=phase,
                hold_time=30.0 if phase == DockingPhase.MID_RANGE else 0.0
            ))
            
            prev_range = range_m
        
        return waypoints
    
    def generate_hop_maneuver(self, current_state: np.ndarray,
                              target_range: float,
                              hop_time: float = 300.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate V-bar hop maneuver (two-impulse transfer).
        
        Hop maneuvers are fuel-efficient for large range changes.
        
        Args:
            current_state: Current state [x, y, z, vx, vy, vz]
            target_range: Target V-bar range (m)
            hop_time: Time of flight for hop (s)
            
        Returns:
            delta_v1, delta_v2: Impulses for hop maneuver
        """
        target_position = np.array([0.0, target_range, 0.0])
        
        delta_v1, delta_v2 = compute_delta_v_for_transfer(
            self.dynamics, current_state, target_position, hop_time)
        
        return delta_v1, delta_v2
    
    def interpolate_trajectory(self, waypoints: List[DockingWaypoint],
                               dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate waypoints to dense trajectory.
        
        Args:
            waypoints: List of waypoints
            dt: Output sample interval (s)
            
        Returns:
            times: Time array
            states: Nx6 state array [x, y, z, vx, vy, vz]
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")
        
        # Total trajectory time
        total_time = waypoints[-1].time_from_start
        n_samples = int(total_time / dt) + 1
        
        times = np.linspace(0, total_time, n_samples)
        states = np.zeros((n_samples, 6))
        
        # Linear interpolation between waypoints
        # (Could use splines for smoother trajectories)
        for i, t in enumerate(times):
            # Find surrounding waypoints
            wp_before = waypoints[0]
            wp_after = waypoints[-1]
            
            for j in range(len(waypoints) - 1):
                if waypoints[j].time_from_start <= t <= waypoints[j+1].time_from_start:
                    wp_before = waypoints[j]
                    wp_after = waypoints[j+1]
                    break
            
            # Interpolation factor
            dt_segment = wp_after.time_from_start - wp_before.time_from_start
            if dt_segment > 0:
                alpha = (t - wp_before.time_from_start) / dt_segment
            else:
                alpha = 0.0
            
            # Linear interpolation
            states[i, 0:3] = (1 - alpha) * wp_before.position + alpha * wp_after.position
            states[i, 3:6] = (1 - alpha) * wp_before.velocity + alpha * wp_after.velocity
        
        return times, states
    
    def check_corridor_constraint(self, position: np.ndarray) -> Tuple[bool, float]:
        """
        Check if position is within approach corridor.
        
        Args:
            position: [x, y, z] position in LVLH
            
        Returns:
            is_within: True if within corridor
            violation_distance: Distance outside corridor (0 if within)
        """
        # Compute range along approach axis
        if self.approach_type == ApproachType.V_BAR:
            range_along_axis = np.abs(position[1])
            lateral_offset = np.sqrt(position[0]**2 + position[2]**2)
        elif self.approach_type == ApproachType.R_BAR:
            range_along_axis = np.abs(position[0])
            lateral_offset = np.sqrt(position[1]**2 + position[2]**2)
        else:  # H_BAR
            range_along_axis = np.abs(position[2])
            lateral_offset = np.sqrt(position[0]**2 + position[1]**2)
        
        # Corridor width at current range
        max_lateral = range_along_axis * np.tan(self.corridor.half_angle_rad)
        
        is_within = lateral_offset <= max_lateral
        violation_distance = max(0.0, lateral_offset - max_lateral)
        
        return is_within, violation_distance
    
    def generate_station_keeping_position(self, range_m: float,
                                          approach_type: ApproachType = None) -> np.ndarray:
        """
        Generate station keeping position at specified range.
        
        Args:
            range_m: Distance from target
            approach_type: Approach direction
            
        Returns:
            Position vector [x, y, z]
        """
        if approach_type is None:
            approach_type = self.approach_type
        
        if approach_type == ApproachType.V_BAR:
            return np.array([0.0, range_m, 0.0])
        elif approach_type == ApproachType.R_BAR:
            return np.array([range_m, 0.0, 0.0])
        else:  # H_BAR
            return np.array([0.0, 0.0, range_m])
    
    def _get_phase_for_range(self, range_m: float) -> DockingPhase:
        """Determine docking phase based on range."""
        for phase, (min_r, max_r) in self.phase_boundaries.items():
            if min_r <= range_m < max_r:
                return phase
        return DockingPhase.TERMINAL


class SafetyMonitor:
    """
    Monitors trajectory safety constraints.
    
    Checks:
    - Approach corridor violations
    - Velocity limits
    - Range rate limits
    - Keep-out zone violations
    """
    
    def __init__(self, corridor: ApproachCorridor):
        self.corridor = corridor
        
        # Keep-out zones (spherical, centered on target)
        self.keep_out_zones: List[Tuple[np.ndarray, float]] = []
        
        # Range rate limits per phase
        self.max_range_rate = {
            DockingPhase.FAR_RANGE: 1.0,
            DockingPhase.MID_RANGE: 0.5,
            DockingPhase.CLOSE_RANGE: 0.1,
            DockingPhase.TERMINAL: 0.05,
        }
    
    def add_keep_out_zone(self, center: np.ndarray, radius: float):
        """Add a spherical keep-out zone."""
        self.keep_out_zones.append((np.asarray(center), radius))
    
    def check_safety(self, state: np.ndarray, 
                     phase: DockingPhase) -> Tuple[bool, List[str]]:
        """
        Check all safety constraints.
        
        Args:
            state: Current state [x, y, z, vx, vy, vz]
            phase: Current docking phase
            
        Returns:
            is_safe: True if all constraints satisfied
            violations: List of violation descriptions
        """
        violations = []
        
        position = state[0:3]
        velocity = state[3:6]
        
        # Range rate check
        range_m = np.linalg.norm(position)
        if range_m > 0.1:
            range_rate = np.dot(position, velocity) / range_m
            max_rate = self.max_range_rate.get(phase, 0.5)
            if np.abs(range_rate) > max_rate:
                violations.append(
                    f"Range rate {range_rate:.3f} m/s exceeds limit {max_rate:.3f} m/s")
        
        # Corridor check
        # (Simplified - actual implementation would match corridor type)
        lateral_dist = np.sqrt(position[0]**2 + position[2]**2)
        max_lateral = np.abs(position[1]) * np.tan(self.corridor.half_angle_rad)
        if lateral_dist > max_lateral and np.abs(position[1]) > 1.0:
            violations.append(
                f"Outside approach corridor: lateral {lateral_dist:.3f} > {max_lateral:.3f} m")
        
        # Keep-out zone check
        for center, radius in self.keep_out_zones:
            dist = np.linalg.norm(position - center)
            if dist < radius:
                violations.append(
                    f"Inside keep-out zone at {center}, distance {dist:.3f} < {radius:.3f} m")
        
        return len(violations) == 0, violations

