#!/usr/bin/env python3
"""Launch pose_array_bridge_node (vision_benchmark -> docking controller)."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('spacecraft_docking_controller')
    config_file = os.path.join(pkg_share, 'config', 'pose_bridge_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'input_topic',
            default_value='/vision_benchmark/pose_xyz_quat',
            description='Float64MultiArray pose from isaac_multitask_node',
        ),
        DeclareLaunchArgument(
            'output_topic',
            default_value='/pose_estimation/object_pose',
            description='PoseStamped for docking_controller_node',
        ),
        Node(
            package='spacecraft_docking_controller',
            executable='pose_array_bridge_node',
            name='pose_array_bridge_node',
            output='screen',
            parameters=[
                config_file,
                {
                    'input_topic': LaunchConfiguration('input_topic'),
                    'output_topic': LaunchConfiguration('output_topic'),
                },
            ],
        ),
    ])
