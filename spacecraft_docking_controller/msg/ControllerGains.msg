# ControllerGains.msg
# Dynamic controller gain adjustment

std_msgs/Header header

# Controller type
string controller_type    # "PID", "LQR", "MPC"

# PID gains (position)
float64 kp_x
float64 kp_y
float64 kp_z
float64 ki_x
float64 ki_y
float64 ki_z
float64 kd_x
float64 kd_y
float64 kd_z

# PID gains (attitude)
float64 kp_roll
float64 kp_pitch
float64 kp_yaw
float64 ki_roll
float64 ki_pitch
float64 ki_yaw
float64 kd_roll
float64 kd_pitch
float64 kd_yaw

# LQR weights (Q matrix diagonal for position/velocity)
float64[6] q_weights
# LQR weights (R matrix diagonal for control effort)
float64[3] r_weights

# MPC parameters
int32 mpc_horizon           # Prediction horizon
float64 mpc_dt              # MPC timestep
float64[6] terminal_weights # Terminal cost weights




