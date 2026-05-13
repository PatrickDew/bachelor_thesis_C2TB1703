# bachelor_thesis_C2TB1703

Spacecraft **detection**, **segmentation**, and **6D pose estimation** with ROS 2 deployment for NVIDIA Isaac Sim.

## Repository layout

```
├── src/                          # PyTorch inference (SplitPosePredictor)
├── models/                       # Trained checkpoints (.pt)
└── ros2_isaac_inference/
    └── src/vision_benchmark_ros/ # ROS 2 node + launch file
```

## Model weights

| File | Role |
|------|------|
| `models/pose_net_baseline.pt` | PoseNet position (ResNet-18) |
| `models/orientation_net.pt` | OrientationNet (ResNet-18) |
| `models/orientation_net_efficientnet_b4.pt` | OrientationNetAlt (EfficientNet-B4) |
| `models/workflow_20260324_204517/pose_resnet50_best.pt` | Full 6D pose (ResNet-50) |
| `models/run_20260407_223752/frcnn_det_best.pt` | Faster R-CNN detection |
| `models/run_20260407_223752/deeplabv3_seg_best.pt` | DeepLabV3 segmentation |

Recommended for the ROS pose node:

- `pos_model_path` → `models/pose_net_baseline.pt`
- `ori_model_path` → `models/orientation_net_efficientnet_b4.pt`

## ROS 2 setup (Isaac Sim)

```bash
export VISION_BENCHMARK_ROOT=/path/to/bachelor_thesis_C2TB1703
cd ~/ros2_ws/src
ln -s /path/to/bachelor_thesis_C2TB1703/ros2_isaac_inference/src/vision_benchmark_ros .
cd ~/ros2_ws
colcon build --packages-select vision_benchmark_ros
source install/setup.bash

ros2 launch vision_benchmark_ros isaac_multitask.launch.py \
  image_topic:=/rgb
```

Default launch paths point at `models/pose_net_baseline.pt` and `models/orientation_net_efficientnet_b4.pt` when `VISION_BENCHMARK_ROOT` is set.

### Topics

| Direction | Topic | Type |
|-----------|-------|------|
| Subscribe | `/camera/image_raw` (configurable) | `sensor_msgs/Image` |
| Publish | `/vision_benchmark/annotated_image` | `sensor_msgs/Image` |
| Publish | `/vision_benchmark/pose_xyz_quat` | `std_msgs/Float64MultiArray` |
| Publish | `/vision_benchmark/pose_text` | `std_msgs/String` |

## Python dependencies

```bash
pip install -r requirements.txt
```

Also need ROS 2 packages: `rclpy`, `sensor_msgs`, `std_msgs`, `cv_bridge`.

## Standalone inference

```python
from src.pose_inference import SplitPosePredictor

predictor = SplitPosePredictor(
    "models/pose_net_baseline.pt",
    "models/orientation_net_efficientnet_b4.pt",
    backbone="resnet18",
)
pose = predictor.predict(image)  # [x, y, z, q1, q2, q3, q4]
```
