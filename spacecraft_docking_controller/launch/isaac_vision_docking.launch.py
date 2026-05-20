#!/usr/bin/env python3
"""
Launch pose bridge + docking controller for Isaac Sim + bachelor_thesis vision stack.

Does NOT start Isaac Sim or vision_benchmark_ros — run those separately, then:

  ros2 launch spacecraft_docking_controller isaac_vision_docking.launch.py
  ros2 topic pub /docking/enable std_msgs/Bool "{data: true}" --once
  ros2 topic pub /docking/command std_msgs/String "{data: 'start'}" --once
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('spacecraft_docking_controller')
    bridge_config = os.path.join(pkg_share, 'config', 'pose_bridge_params.yaml')
    docking_config = os.path.join(pkg_share, 'config', 'docking_params.yaml')
    vision_pid_config = os.path.join(pkg_share, 'config', 'vision_pid_params.yaml')

    controller_arg = DeclareLaunchArgument(
        'controller',
        default_value='PID',
        description='Controller type (thesis stack uses PID + vision_pid_params.yaml)',
    )
    pose_input_arg = DeclareLaunchArgument(
        'pose_input_topic',
        default_value='/vision_benchmark/pose_xyz_quat',
    )
    pose_output_arg = DeclareLaunchArgument(
        'pose_output_topic',
        default_value='/pose_estimation/object_pose',
    )

    bridge_node = Node(
        package='spacecraft_docking_controller',
        executable='pose_array_bridge_node',
        name='pose_array_bridge_node',
        output='screen',
        parameters=[
            bridge_config,
            {
                'input_topic': LaunchConfiguration('pose_input_topic'),
                'output_topic': LaunchConfiguration('pose_output_topic'),
            },
        ],
    )

    controller_node = Node(
        package='spacecraft_docking_controller',
        executable='docking_controller_node',
        name='docking_controller_node',
        output='screen',
        parameters=[
            docking_config,
            vision_pid_config,
            {
                'controller_type': LaunchConfiguration('controller'),
                'pose_topic': LaunchConfiguration('pose_output_topic'),
            },
        ],
    )

    return LaunchDescription([
        controller_arg,
        pose_input_arg,
        pose_output_arg,
        bridge_node,
        controller_node,
    ])
