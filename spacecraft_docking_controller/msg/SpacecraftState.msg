# SpacecraftState.msg
# Complete state vector for spacecraft relative motion (LVLH frame)
# Uses Hill/Clohessy-Wiltshire coordinate system

std_msgs/Header header

# Position in LVLH frame (meters)
# x: radial (positive outward from Earth center)
# y: along-track (positive in velocity direction)  
# z: cross-track (completes right-hand system)
float64 x
float64 y
float64 z

# Velocity in LVLH frame (m/s)
float64 vx
float64 vy
float64 vz

# Orientation (quaternion) - chaser relative to target
geometry_msgs/Quaternion orientation

# Angular velocity (rad/s)
float64 wx
float64 wy
float64 wz

# Covariance matrix (6x6 for position/velocity, row-major)
float64[36] covariance

# Timestamps and quality metrics
float64 pose_confidence      # 0-1 confidence from pose estimation
bool is_valid               # Whether state estimate is valid




