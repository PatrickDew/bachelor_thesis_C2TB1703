# ControlCommand.msg
# Control commands for spacecraft actuators
# Supports both force/torque and velocity commands

std_msgs/Header header

# Command mode
uint8 MODE_FORCE_TORQUE = 0     # Direct force/torque control
uint8 MODE_VELOCITY = 1          # Velocity setpoint
uint8 MODE_POSITION = 2          # Position setpoint (trajectory tracking)
uint8 control_mode

# Force command in body frame (Newtons)
float64 force_x
float64 force_y
float64 force_z

# Torque command in body frame (Newton-meters)
float64 torque_x
float64 torque_y
float64 torque_z

# Velocity setpoint in LVLH frame (m/s) - when MODE_VELOCITY
float64 vel_x_cmd
float64 vel_y_cmd
float64 vel_z_cmd

# Angular velocity setpoint (rad/s)
float64 omega_x_cmd
float64 omega_y_cmd
float64 omega_z_cmd

# Thruster duty cycles (0-1) for low-level control
float64[12] thruster_commands   # Typical 12-thruster configuration

# Safety flags
bool enable_control            # Master enable
bool emergency_stop           # Emergency stop command
float64 thrust_magnitude      # Total thrust magnitude for verification




