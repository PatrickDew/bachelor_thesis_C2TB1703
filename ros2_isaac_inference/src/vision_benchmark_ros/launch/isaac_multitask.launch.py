"""
Launch Isaac Sim / camera subscriber with pose + optional Mask R-CNN.

Set environment VISION_BENCHMARK_ROOT to your vision_benchmark repo root, or pass
vision_benchmark_root as a launch argument. Example:

  export VISION_BENCHMARK_ROOT=/path/to/vision_benchmark
  ros2 launch vision_benchmark_ros isaac_multitask.launch.py \\
    pos_model_path:=/path/to/pose_pos.pt \\
    ori_model_path:=/path/to/pose_ori.pt \\
    pos_backbone:=resnet50 \\
    image_topic:=/rgb \\
    enable_mask_rcnn:=true \\
    mask_rcnn_weights:=/path/to/mask_rcnn.pt \\
    mask_rcnn_num_classes:=2
"""
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _find_vision_benchmark_root() -> str:
    env = os.environ.get("VISION_BENCHMARK_ROOT", "").strip()
    if env:
        return env
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "src" / "pose_inference.py").is_file():
            return str(p)
    return ""


def generate_launch_description() -> LaunchDescription:
    default_root = _find_vision_benchmark_root()

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "vision_benchmark_root",
                default_value=default_root,
                description="Path to vision_benchmark repo (contains src/).",
            ),
            DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument(
                "annotated_image_topic", default_value="/vision_benchmark/annotated_image"
            ),
            DeclareLaunchArgument("pose_text_topic", default_value="/vision_benchmark/pose_text"),
            DeclareLaunchArgument(
                "pose_array_topic", default_value="/vision_benchmark/pose_xyz_quat"
            ),
            DeclareLaunchArgument("enable_pose", default_value="true"),
            DeclareLaunchArgument(
                "pos_model_path",
                default_value=os.path.join(default_root, "models", "pose_net_baseline.pt")
                if default_root
                else "",
                description="Checkpoint for PoseNet (position), e.g. models/pose_net_baseline.pt",
            ),
            DeclareLaunchArgument(
                "ori_model_path",
                default_value=os.path.join(
                    default_root, "models", "orientation_net_efficientnet_b4.pt"
                )
                if default_root
                else "",
                description="Checkpoint for OrientationNet / OrientationNetAlt.",
            ),
            DeclareLaunchArgument("pos_backbone", default_value="resnet50"),
            DeclareLaunchArgument("use_cuda", default_value="true"),
            DeclareLaunchArgument("enable_mask_rcnn", default_value="false"),
            DeclareLaunchArgument("mask_rcnn_weights", default_value=""),
            DeclareLaunchArgument(
                "mask_rcnn_num_classes",
                default_value="91",
                description="Total classes including background (torchvision convention).",
            ),
            DeclareLaunchArgument("mask_rcnn_score_threshold", default_value="0.5"),
            DeclareLaunchArgument("mask_rcnn_input_size", default_value="640"),
            DeclareLaunchArgument(
                "mask_rcnn_class_names",
                default_value="",
                description="Comma-separated names for label ids 1..N (optional).",
            ),
            DeclareLaunchArgument("enable_frcnn", default_value="false"),
            DeclareLaunchArgument(
                "frcnn_weights",
                default_value=os.path.join(
                    default_root, "models", "run_20260407_223752", "frcnn_det_best.pt"
                )
                if default_root
                else "",
            ),
            DeclareLaunchArgument(
                "frcnn_num_classes",
                default_value="13",
                description="Total classes including background (torchvision convention).",
            ),
            DeclareLaunchArgument("frcnn_score_threshold", default_value="0.5"),
            DeclareLaunchArgument("frcnn_class_names", default_value=""),
            Node(
                package="vision_benchmark_ros",
                executable="isaac_multitask_node",
                name="isaac_multitask_node",
                output="screen",
                parameters=[
                    {
                        "vision_benchmark_root": LaunchConfiguration("vision_benchmark_root"),
                        "image_topic": LaunchConfiguration("image_topic"),
                        "annotated_image_topic": LaunchConfiguration("annotated_image_topic"),
                        "pose_text_topic": LaunchConfiguration("pose_text_topic"),
                        "pose_array_topic": LaunchConfiguration("pose_array_topic"),
                        "enable_pose": LaunchConfiguration("enable_pose"),
                        "pos_model_path": LaunchConfiguration("pos_model_path"),
                        "ori_model_path": LaunchConfiguration("ori_model_path"),
                        "pos_backbone": LaunchConfiguration("pos_backbone"),
                        "use_cuda": LaunchConfiguration("use_cuda"),
                        "enable_mask_rcnn": LaunchConfiguration("enable_mask_rcnn"),
                        "mask_rcnn_weights": LaunchConfiguration("mask_rcnn_weights"),
                        "mask_rcnn_num_classes": LaunchConfiguration("mask_rcnn_num_classes"),
                        "mask_rcnn_score_threshold": LaunchConfiguration(
                            "mask_rcnn_score_threshold"
                        ),
                        "mask_rcnn_input_size": LaunchConfiguration("mask_rcnn_input_size"),
                        "mask_rcnn_class_names": LaunchConfiguration("mask_rcnn_class_names"),
                        "enable_frcnn": LaunchConfiguration("enable_frcnn"),
                        "frcnn_weights": LaunchConfiguration("frcnn_weights"),
                        "frcnn_num_classes": LaunchConfiguration("frcnn_num_classes"),
                        "frcnn_score_threshold": LaunchConfiguration("frcnn_score_threshold"),
                        "frcnn_class_names": LaunchConfiguration("frcnn_class_names"),
                    }
                ],
            ),
        ]
    )
