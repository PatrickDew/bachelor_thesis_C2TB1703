# DockingStatus.msg
# Docking mission status and telemetry

std_msgs/Header header

# Mission phase
uint8 PHASE_IDLE = 0
uint8 PHASE_FAR_RANGE = 1        # > 100m, initial approach
uint8 PHASE_MID_RANGE = 2        # 100m - 10m, station keeping
uint8 PHASE_CLOSE_RANGE = 3      # 10m - 1m, final approach
uint8 PHASE_TERMINAL = 4         # < 1m, docking alignment
uint8 PHASE_CONTACT = 5          # Physical contact
uint8 PHASE_CAPTURED = 6         # Soft capture complete
uint8 PHASE_RIGIDIZED = 7        # Hard dock complete
uint8 PHASE_ABORT = 8            # Mission abort
uint8 current_phase

# Range and range rate
float64 range_m                  # Distance to docking port (m)
float64 range_rate_mps          # Approach velocity (m/s, negative = closing)

# Alignment errors
float64 lateral_error_m          # Cross-track error (m)
float64 angular_error_deg        # Attitude alignment error (deg)

# Corridor constraints
bool within_approach_corridor    # Inside safe approach cone
float64 corridor_half_angle_deg  # Current corridor constraint

# Controller status
string active_controller         # "PID", "LQR", "MPC"
float64 control_effort_percent   # 0-100 actuator utilization

# Safety status
bool pose_valid                  # Pose estimation healthy
bool trajectory_valid           # Within planned trajectory bounds
bool fuel_adequate              # Sufficient propellant
float64 delta_v_remaining_mps   # Estimated remaining delta-V

# Timing
float64 time_to_contact_s       # Estimated time to docking (s)
float64 mission_elapsed_time_s  # Time since mission start




