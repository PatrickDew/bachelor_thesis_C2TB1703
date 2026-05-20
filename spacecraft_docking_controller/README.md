# Spacecraft Docking Controller

A research-grade ROS2 package for autonomous spacecraft proximity operations and rendezvous & docking (AR&D) in NVIDIA Isaac Sim.

## Features

### Dynamics Models
- **Clohessy-Wiltshire (Hill's) Equations**: Linearized relative orbital motion dynamics
- **Full 6-DOF Dynamics**: Translational + attitude dynamics with quaternion representation
- Closed-form state transition matrices for efficient propagation

### Control Algorithms
- **PID Controller**: Baseline feedback control with anti-windup and gain scheduling
- **LQR Controller**: Linear Quadratic Regulator for optimal state feedback
- **MPC Controller**: Model Predictive Control with constraint handling
- **Sliding Mode Controller**: Robust control for uncertainty rejection
- **Adaptive PID**: Self-tuning PID with online gain adaptation

### State Estimation
- Extended Kalman Filter (EKF) for sensor fusion
- Outlier rejection and covariance tracking
- Velocity estimation from pose differentiation

### Trajectory Generation
- V-bar and R-bar approach trajectories
- Hop maneuvers (two-impulse transfers)
- Approach corridor enforcement
- Phase-based waypoint generation

### Isaac Sim Integration
- ROS2 bridge for force/torque commands
- Action Graph setup guide
- Camera sensor integration with FoundationPose

## Installation

### Prerequisites
- ROS2 Humble or later
- Python 3.10+
- NumPy, SciPy

### Build
```bash
cd ~/Documents
source /opt/ros/humble/setup.bash

# Build the package
cd spacecraft_docking_controller
colcon build

# Source the workspace
source install/setup.bash
```

## Usage

### Launch Controller
```bash
# With default LQR controller
ros2 launch spacecraft_docking_controller docking_controller.launch.py

# With MPC controller
ros2 launch spacecraft_docking_controller docking_controller.launch.py controller:=MPC

# With custom pose topic
ros2 launch spacecraft_docking_controller docking_controller.launch.py \
    pose_topic:=/my_pose_topic
```

### Control Commands
```bash
# Enable control output
ros2 topic pub /docking/enable std_msgs/Bool "data: true" -1

# Start docking sequence
ros2 topic pub /docking/command std_msgs/String "data: 'start'" -1

# Stop (controlled)
ros2 topic pub /docking/command std_msgs/String "data: 'stop'" -1

# Emergency abort
ros2 topic pub /docking/command std_msgs/String "data: 'abort'" -1

# Hold position
ros2 topic pub /docking/command std_msgs/String "data: 'hold'" -1
```

### Monitor Status
```bash
# View docking status
ros2 topic echo /docking/status

# View state estimate
ros2 topic echo /docking/state

# View control commands
ros2 topic echo /docking/control_wrench
```

## ROS2 Topics

### Subscribed Topics
| Topic | Type | Description |
|-------|------|-------------|
| `/pose_estimation/object_pose` | PoseStamped | Target pose from FoundationPose |
| `/docking/command` | String | High-level commands |
| `/docking/enable` | Bool | Control enable/disable |

### Published Topics
| Topic | Type | Description |
|-------|------|-------------|
| `/docking/control_wrench` | WrenchStamped | Force/torque commands |
| `/docking/control_twist` | Twist | Velocity commands |
| `/docking/status` | String | Docking status |
| `/docking/state` | Float64MultiArray | State estimate |
| `/docking/visualization` | MarkerArray | RViz markers |
| `/spacecraft/force_command` | Float64MultiArray | Isaac Sim forces |

## Configuration

Edit `config/docking_params.yaml` to adjust:

```yaml
docking_controller_node:
  ros__parameters:
    # Controller selection
    controller_type: "LQR"  # PID, LQR, or MPC
    
    # Spacecraft parameters
    spacecraft_mass: 500.0  # kg
    
    # LQR weights
    lqr_q_pos: 100.0
    lqr_q_vel: 10.0
    lqr_r: 1.0
    
    # Approach parameters
    approach_type: "v_bar"
    initial_standoff: 50.0  # m
    final_standoff: 0.5     # m
```

## Package Structure

```
spacecraft_docking_controller/
├── sdc_core/                        # Python core library
│   ├── __init__.py
│   ├── dynamics.py                  # CW & 6-DOF dynamics
│   ├── controllers.py               # PID, LQR, MPC controllers
│   ├── state_estimator.py           # EKF state estimation
│   └── trajectory_generator.py      # Trajectory planning
├── scripts/                         # ROS2 node executables
│   ├── docking_controller_node      # Main integrated controller
│   ├── pid_controller_node          # Standalone PID
│   ├── lqr_controller_node          # Standalone LQR
│   ├── mpc_controller_node          # Standalone MPC
│   └── state_estimator_node         # Standalone estimator
├── msg/                             # Custom messages
│   ├── SpacecraftState.msg
│   ├── ControlCommand.msg
│   ├── DockingStatus.msg
│   └── ControllerGains.msg
├── launch/                          # Launch files
│   ├── docking_controller.launch.py
│   └── full_docking_system.launch.py
├── config/                          # Configuration files
│   ├── docking_params.yaml
│   └── isaac_sim_config.yaml
├── ISAAC_SIM_SETUP.md              # Isaac Sim integration guide
├── CMakeLists.txt
├── package.xml
└── README.md
```

## Isaac Sim Integration

See [ISAAC_SIM_SETUP.md](ISAAC_SIM_SETUP.md) for detailed instructions on:
- Setting up Action Graphs for ROS2 control
- Camera configuration for FoundationPose
- Physics settings for spacecraft simulation
- Complete feedback loop setup

## Theory

### Clohessy-Wiltshire Equations

The CW equations describe linearized relative motion in the LVLH frame:

```
ẍ = 3n²x + 2nẏ + aₓ
ÿ = -2nẋ + aᵧ
z̈ = -n²z + aᵤ
```

Where:
- n = mean motion (rad/s)
- (x, y, z) = position in LVLH frame
- (aₓ, aᵧ, aᵤ) = control accelerations

### LQR Controller

Minimizes quadratic cost:
```
J = ∫(x'Qx + u'Ru)dt
```

Computes optimal gain K by solving the Algebraic Riccati Equation (ARE).

### MPC Controller

Solves finite-horizon optimal control:
```
min Σ(xₖ'Qxₖ + uₖ'Ruₖ) + x_N'P_f x_N
subject to:
  xₖ₊₁ = Φxₖ + Γuₖ
  |uₖ| ≤ u_max
```

## References

1. Clohessy, W. H., & Wiltshire, R. S. (1960). "Terminal Guidance System for Satellite Rendezvous."
2. Fehse, W. (2003). "Automated Rendezvous and Docking of Spacecraft." Cambridge Aerospace Series.
3. Wie, B. (2008). "Space Vehicle Dynamics and Control." AIAA Education Series.

## License

MIT License

## Author

Pakawat Nutthanithipat

