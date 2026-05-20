"""
Isaac Sim Script Node for Spacecraft Force Control

This script receives force commands from the ROS2 docking controller
and applies them to the spacecraft rigid body.

Setup in Isaac Sim:
1. Create an Action Graph with a Script Node
2. Set the prim path to your spacecraft rigid body
3. Connect to ROS2 subscriber for force commands

Prim path example: /World/CAD_Spacecraft_modified
Script node path: /World/CAD_Spacecraft_modified/Driver/script_node
"""

import numpy as np

# Try to import Isaac Sim modules
try:
    import omni
    import omni.graph.core as og
    from pxr import UsdPhysics, Gf, UsdGeom
    import carb
except ImportError:
    print("Note: Isaac Sim modules not available (running outside Isaac Sim)")


def setup(db: og.Database):
    """Called once when the script node is initialized."""
    state = db.internal_state
    
    # Store the spacecraft prim path - MODIFY THIS TO YOUR SPACECRAFT
    state.spacecraft_prim_path = "/World/CAD_Spacecraft_modified"
    
    # Initialize force storage
    state.force = [0.0, 0.0, 0.0]
    state.torque = [0.0, 0.0, 0.0]
    
    print(f"[ForceController] Initialized for: {state.spacecraft_prim_path}")
    return True


def compute(db: og.Database):
    """Called every simulation step."""
    state = db.internal_state
    
    try:
        # Get force command from input (connected to ROS2 subscriber)
        # The input should be connected to a ROS2 Subscribe Float64MultiArray node
        if hasattr(db.inputs, 'force_command'):
            force_data = db.inputs.force_command
            if force_data is not None and len(force_data) >= 3:
                state.force = [float(force_data[0]), float(force_data[1]), float(force_data[2])]
        
        # Apply force to spacecraft
        apply_force_to_rigid_body(
            state.spacecraft_prim_path,
            state.force
        )
        
    except Exception as e:
        carb.log_warn(f"[ForceController] Error: {e}")
    
    return True


def apply_force_to_rigid_body(prim_path: str, force: list):
    """Apply force to a rigid body at its center of mass."""
    try:
        # Get the physx interface
        from omni.physx import get_physx_interface
        physx = get_physx_interface()
        
        # Apply force at center of mass
        # Force is in world coordinates
        physx.apply_force_at_pos(
            prim_path,
            carb.Float3(force[0], force[1], force[2]),
            carb.Float3(0.0, 0.0, 0.0),  # Position offset (0 = CoM)
            "Force"
        )
        
    except Exception as e:
        carb.log_warn(f"[ForceController] Failed to apply force: {e}")


# Alternative: Standalone version without OmniGraph
class SpacecraftForceController:
    """
    Standalone force controller for use in Isaac Sim extensions or standalone scripts.
    
    Usage:
        controller = SpacecraftForceController("/World/CAD_Spacecraft_modified")
        controller.apply_force([10.0, 0.0, 0.0])  # Apply 10N in X direction
    """
    
    def __init__(self, spacecraft_prim_path: str):
        self.prim_path = spacecraft_prim_path
        self.stage = omni.usd.get_context().get_stage()
        
        # Verify prim exists
        prim = self.stage.GetPrimAtPath(self.prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {self.prim_path}")
        
        # Check if it has rigid body API
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            carb.log_warn(f"Prim {self.prim_path} does not have RigidBodyAPI")
        
        print(f"[SpacecraftForceController] Attached to: {self.prim_path}")
    
    def apply_force(self, force: list, position: list = None):
        """
        Apply force to spacecraft.
        
        Args:
            force: [Fx, Fy, Fz] in Newtons (world frame)
            position: Optional [x, y, z] position offset from CoM
        """
        if position is None:
            position = [0.0, 0.0, 0.0]
        
        try:
            from omni.physx import get_physx_interface
            physx = get_physx_interface()
            
            physx.apply_force_at_pos(
                self.prim_path,
                carb.Float3(force[0], force[1], force[2]),
                carb.Float3(position[0], position[1], position[2]),
                "Force"
            )
        except Exception as e:
            carb.log_error(f"Failed to apply force: {e}")
    
    def apply_torque(self, torque: list):
        """
        Apply torque to spacecraft.
        
        Args:
            torque: [Tx, Ty, Tz] in Newton-meters (body frame)
        """
        try:
            from omni.physx import get_physx_interface
            physx = get_physx_interface()
            
            physx.apply_torque(
                self.prim_path,
                carb.Float3(torque[0], torque[1], torque[2]),
                "Torque"
            )
        except Exception as e:
            carb.log_error(f"Failed to apply torque: {e}")


# ROS2 Integration Helper
def create_ros2_force_subscriber():
    """
    Helper to create ROS2 subscription for force commands.
    Call this from Isaac Sim's Python scripting console or extension.
    """
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    
    class ForceSubscriber(Node):
        def __init__(self, spacecraft_controller):
            super().__init__('isaac_force_subscriber')
            self.controller = spacecraft_controller
            
            self.subscription = self.create_subscription(
                Float64MultiArray,
                '/spacecraft/force_command',
                self.force_callback,
                10
            )
            print("[ROS2] Subscribed to /spacecraft/force_command")
        
        def force_callback(self, msg):
            if len(msg.data) >= 3:
                self.controller.apply_force(list(msg.data[:3]))
    
    return ForceSubscriber



