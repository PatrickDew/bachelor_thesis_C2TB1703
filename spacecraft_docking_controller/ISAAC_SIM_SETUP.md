# Isaac Sim Integration Guide

This document explains how to integrate the Spacecraft Docking Controller ROS2 package with NVIDIA Isaac Sim for closed-loop docking simulation.

## Overview

The complete feedback loop works as follows:

```
┌─────────────────┐     Pose      ┌─────────────────────┐
│  FoundationPose │──────────────>│  Docking Controller │
│  (Perception)   │               │  (PID/LQR/MPC)      │
└─────────────────┘               └──────────┬──────────┘
        ▲                                    │
        │ Camera                             │ Force/Torque
        │ Image                              │ Commands
        │                                    ▼
┌───────┴─────────────────────────────────────────────┐
│                    NVIDIA Isaac Sim                  │
│  ┌────────────┐              ┌────────────────────┐ │
│  │  Camera    │              │  Spacecraft Rigid  │ │
│  │  Sensor    │              │  Body (Chaser)     │ │
│  └────────────┘              └────────────────────┘ │
│                                       │             │
│                              Apply Forces/Torques   │
└─────────────────────────────────────────────────────┘
```

## Step 1: Enable ROS2 Bridge in Isaac Sim

1. **Launch Isaac Sim** with ROS2 enabled:
   ```bash
   # Set ROS2 environment
   source /opt/ros/humble/setup.bash
   
   # Launch Isaac Sim
   ./isaac-sim.sh --ros2
   ```

2. **Verify ROS2 connection** by checking topics:
   ```bash
   ros2 topic list
   ```

## Step 2: Set Up Scene in Isaac Sim

### 2.1 Create Spacecraft Objects

1. **Import Chaser Spacecraft**:
   - Go to `File > Import`
   - Load your spacecraft USD/URDF model
   - Set it as a **Rigid Body** with physics enabled

2. **Import Target Spacecraft** (ISS/Station):
   - Import the target object
   - Set as **Static** or **Rigid Body** depending on scenario

3. **Add Camera**:
   - Add a Camera prim to the chaser spacecraft
   - Position it pointing at docking port
   - Enable depth if needed

### 2.2 Physics Configuration

Set appropriate physics parameters:
- **Mass**: Match `spacecraft_mass` parameter (default: 500 kg)
- **Inertia Tensor**: Configure realistic inertia
- **Linear/Angular Damping**: Set low for space environment (≈0)

## Step 3: Create Action Graph for ROS2 Control

The Action Graph receives ROS2 commands and applies forces to the spacecraft.

### 3.1 Create New Action Graph

1. Right-click in Stage > `Create > Visual Scripting > Action Graph`
2. Name it `DockingControlGraph`

### 3.2 Add Required Nodes

Add these nodes to the Action Graph:

```
┌──────────────────────────────────────────────────────────────┐
│                     Action Graph                              │
│                                                              │
│  ┌─────────────────┐                                         │
│  │ On Playback Tick│──────┐                                  │
│  └─────────────────┘      │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────┐                │
│  │ ROS2 Subscribe Float64MultiArray        │                │
│  │ Topic: /spacecraft/force_command        │                │
│  │ Queue: 1                                │                │
│  └──────────────────┬──────────────────────┘                │
│                     │ data                                   │
│                     ▼                                        │
│  ┌─────────────────────────────────────────┐                │
│  │ Array Get (index 0,1,2)                 │                │
│  │ Extract [Fx, Fy, Fz]                    │                │
│  └──────────────────┬──────────────────────┘                │
│                     │                                        │
│                     ▼                                        │
│  ┌─────────────────────────────────────────┐                │
│  │ Apply Force to Rigid Body               │                │
│  │ Target: /World/Chaser/Body              │                │
│  │ Force: [Fx, Fy, Fz]                     │                │
│  └─────────────────────────────────────────┘                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 3.3 Node Configuration Details

**1. On Playback Tick**
- Triggers every simulation step
- Connect `Tick` output to subscriber `Exec In`

**2. ROS2 Context (if not automatic)**
```python
# Properties:
domain_id: 0
```

**3. ROS2 Subscribe Float64MultiArray**
```python
# Properties:
topic_name: "/spacecraft/force_command"
queue_size: 1
```

**4. Apply Force (Isaac Sim Script Node)**

Create a Python Script node with:

```python
import omni.isaac.core.utils.prims as prim_utils
from pxr import UsdPhysics, Gf

def apply_force(force_x, force_y, force_z, prim_path="/World/Chaser"):
    """Apply force to spacecraft rigid body."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    
    if prim.IsValid():
        rigid_body = UsdPhysics.RigidBodyAPI(prim)
        
        # Apply force at center of mass
        # Force is in world coordinates - transform if needed
        force = Gf.Vec3f(force_x, force_y, force_z)
        
        # Use physics force API
        from omni.physx import get_physx_interface
        physx = get_physx_interface()
        physx.apply_force_at_pos(
            prim_path,
            force,
            Gf.Vec3f(0, 0, 0),  # At CoM
            "Force"
        )
```

## Step 4: Camera Setup for Pose Estimation

### 4.1 Create Camera Publisher

Add to Action Graph:

```
┌─────────────────────────────────────────────────────────────┐
│  Camera Publisher Nodes                                      │
│                                                             │
│  ┌───────────────┐      ┌────────────────────────────────┐ │
│  │ Isaac Read    │─────>│ ROS2 Publish Image              │ │
│  │ Camera RGB    │      │ Topic: /camera/image_raw        │ │
│  └───────────────┘      └────────────────────────────────┘ │
│                                                             │
│  ┌───────────────┐      ┌────────────────────────────────┐ │
│  │ Isaac Camera  │─────>│ ROS2 Publish Camera Info        │ │
│  │ Info          │      │ Topic: /camera/camera_info      │ │
│  └───────────────┘      └────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Camera Configuration

```python
# Camera parameters (match FoundationPose settings)
width: 640
height: 480
focal_length: 35.0  # mm
horizontal_aperture: 36.0  # mm
```

## Step 5: Launch Complete System

### Terminal 1: Isaac Sim
```bash
# Start Isaac Sim with ROS2
./isaac-sim.sh
# Load your docking scene
# Press Play
```

### Terminal 2: Vision pose (bachelor_thesis_C2TB1703)

```bash
export VISION_BENCHMARK_ROOT=/path/to/bachelor_thesis_C2TB1703
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 launch vision_benchmark_ros isaac_multitask.launch.py \
    image_topic:=/rgb
```

Publishes `/vision_benchmark/pose_xyz_quat` (`Float64MultiArray`: x,y,z,q1,q2,q3,q4).

**Legacy option:** FoundationPose on `/pose_estimation/object_pose` (`PoseStamped`) — skip the bridge in that case.

### Terminal 3: Pose bridge + docking controller

```bash
source /opt/ros/humble/setup.bash
cd /path/to/spacecraft_docking_controller
colcon build --packages-select spacecraft_docking_controller
source install/setup.bash

# Bridge + controller (recommended with vision_benchmark)
ros2 launch spacecraft_docking_controller isaac_vision_docking.launch.py controller:=PID

# Or bridge only:
# ros2 launch spacecraft_docking_controller pose_bridge.launch.py
# ros2 launch spacecraft_docking_controller docking_controller.launch.py controller:=PID
```

### Terminal 4: Start Docking
```bash
# Enable control
ros2 topic pub /docking/enable std_msgs/Bool "data: true" -1

# Start docking sequence
ros2 topic pub /docking/command std_msgs/String "data: 'start'" -1
```

## Step 6: Monitor and Debug

### View Topics
```bash
# Check pose estimation output
ros2 topic echo /pose_estimation/object_pose

# Check control commands
ros2 topic echo /docking/control_wrench

# Check state estimate
ros2 topic echo /docking/state
```

### RViz Visualization
```bash
rviz2 -d /path/to/docking.rviz
```

Add these displays:
- `/docking/visualization` (MarkerArray)
- `/pose_estimation/object_pose` (PoseStamped)
- TF tree

## Alternative: Isaac Sim Python API

If you prefer Python API over Action Graphs:

```python
# isaac_sim_docking_controller.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import omni
from pxr import UsdPhysics

class IsaacSimController:
    def __init__(self, chaser_prim_path="/World/Chaser"):
        self.prim_path = chaser_prim_path
        
        # Initialize ROS2
        rclpy.init()
        self.node = rclpy.create_node('isaac_force_receiver')
        
        self.subscription = self.node.create_subscription(
            Float64MultiArray,
            '/spacecraft/force_command',
            self.force_callback,
            10
        )
        
        self.latest_force = [0.0, 0.0, 0.0]
    
    def force_callback(self, msg):
        if len(msg.data) >= 3:
            self.latest_force = list(msg.data[:3])
    
    def apply_force(self):
        """Call this in Isaac Sim physics step."""
        from omni.physx import get_physx_interface
        physx = get_physx_interface()
        
        force = carb.Float3(*self.latest_force)
        physx.apply_force_at_pos(
            self.prim_path,
            force,
            carb.Float3(0, 0, 0),
            "Force"
        )
    
    def spin_once(self):
        rclpy.spin_once(self.node, timeout_sec=0.001)
```

## Troubleshooting

### No ROS2 Topics Visible
1. Check `ROS_DOMAIN_ID` matches
2. Verify Isaac Sim ROS2 bridge is enabled
3. Check network/firewall settings

### Forces Not Applied
1. Verify rigid body physics is enabled
2. Check prim path in Action Graph
3. Ensure simulation is playing

### Pose Estimation Issues
1. Verify camera topic publishing
2. Check camera intrinsics match
3. Verify CAD model is loaded

### Controller Not Converging
1. Check gains in config file
2. Verify coordinate frames match
3. Enable debug logging:
   ```bash
   ros2 run spacecraft_docking_controller docking_controller_node \
       --ros-args --log-level debug
   ```

## References

- [Isaac Sim ROS2 Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/ros2_tutorials/)
- [OmniGraph Documentation](https://docs.omniverse.nvidia.com/kit/docs/omni.graph.docs/)
- [PhysX Force API](https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/)




