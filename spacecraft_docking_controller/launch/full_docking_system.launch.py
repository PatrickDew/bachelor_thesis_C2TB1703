#!/usr/bin/env python3
"""
Launch file for Full Docking System

Launches everything needed for docking:
- FoundationPose pose estimation
- Docking controller
- Visualization

Usage:
  ros2 launch spacecraft_docking_controller full_docking_system.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directories
    controller_pkg = get_package_share_directory('spacecraft_docking_controller')
    
    # Try to get pose_estimation package (may not exist)
    try:
        pose_pkg = get_package_share_directory('pose_estimation')
        has_pose_estimation = True
    except Exception:
        has_pose_estimation = False
    
    # Configuration
    config_file = os.path.join(controller_pkg, 'config', 'docking_params.yaml')
    
    # Launch arguments
    args = [
        DeclareLaunchArgument('controller', default_value='LQR'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('cad_model_path', default_value=''),
    ]
    
    # Docking controller
    docking_controller = Node(
        package='spacecraft_docking_controller',
        executable='docking_controller_node',
        name='docking_controller_node',
        output='screen',
        parameters=[
            config_file,
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'controller_type': LaunchConfiguration('controller')},
        ]
    )
    
    # State estimator (standalone)
    state_estimator = Node(
        package='spacecraft_docking_controller',
        executable='state_estimator_node',
        name='state_estimator_node',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
    )
    
    nodes = [docking_controller, state_estimator]
    
    return LaunchDescription(args + nodes)




