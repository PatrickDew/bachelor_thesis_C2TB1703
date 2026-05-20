#!/usr/bin/env python3
"""
Launch file for Spacecraft Docking Controller

Launches the complete docking control system including:
- Docking controller node
- State estimator
- Visualization

Usage:
  ros2 launch spacecraft_docking_controller docking_controller.launch.py
  ros2 launch spacecraft_docking_controller docking_controller.launch.py controller:=MPC
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('spacecraft_docking_controller')
    
    # Configuration file path
    config_file = os.path.join(pkg_share, 'config', 'docking_params.yaml')
    
    # Launch arguments
    controller_arg = DeclareLaunchArgument(
        'controller',
        default_value='LQR',
        description='Controller type: PID, LQR, or MPC'
    )
    
    pose_topic_arg = DeclareLaunchArgument(
        'pose_topic',
        default_value='/pose_estimation/object_pose',
        description='Topic for pose estimation input'
    )
    
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Namespace for all nodes'
    )
    
    # Main docking controller node
    docking_controller_node = Node(
        package='spacecraft_docking_controller',
        executable='docking_controller_node',
        name='docking_controller_node',
        output='screen',
        parameters=[
            config_file,
            {
                'controller_type': LaunchConfiguration('controller'),
                'pose_topic': LaunchConfiguration('pose_topic'),
            }
        ],
        remappings=[
            ('/pose_estimation/object_pose', LaunchConfiguration('pose_topic')),
        ]
    )
    
    return LaunchDescription([
        controller_arg,
        pose_topic_arg,
        namespace_arg,
        docking_controller_node,
    ])




