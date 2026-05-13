"""
ROS 2 subscriber: camera image → pose (SplitPosePredictor, ResNet50-capable) +
optional Mask R-CNN (detection + instance masks). Publishes annotated image and pose text.

Expects the vision_benchmark repo on PYTHONPATH via vision_benchmark_root (parameter or
VISION_BENCHMARK_ROOT env): repo root must contain src/ (for orientation_model imports).
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterType
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String


def _imgmsg_to_bgr(msg: Image) -> np.ndarray:
    """Convert sensor_msgs/Image to BGR uint8 without cv_bridge (NumPy 2.x safe)."""
    h, w = msg.height, msg.width
    if msg.encoding in ("bgr8", "8UC3"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
        return np.ascontiguousarray(arr)
    if msg.encoding in ("rgb8",):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    raise ValueError(f"unsupported image encoding: {msg.encoding}")


def _bgr_to_imgmsg(img: np.ndarray, encoding: str = "bgr8") -> Image:
    out = Image()
    out.height, out.width = img.shape[:2]
    out.encoding = encoding
    out.is_bigendian = 0
    out.step = img.shape[1] * 3
    out.data = np.ascontiguousarray(img).tobytes()
    return out


def _param_type(value) -> int:
    # Humble: ParameterValue.type_; Jazzy+: ParameterValue.type
    return getattr(value, "type_", getattr(value, "type", 0))


def _read_bool(node: Node, name: str) -> bool:
    v = node.get_parameter(name).get_parameter_value()
    t = _param_type(v)
    if t == ParameterType.PARAMETER_BOOL:
        return bool(v.bool_value)
    if t == ParameterType.PARAMETER_STRING:
        s = v.string_value.strip().lower()
        return s in ("1", "true", "yes", "on")
    if t == ParameterType.PARAMETER_INTEGER:
        return v.integer_value != 0
    return False


def _read_int(node: Node, name: str) -> int:
    v = node.get_parameter(name).get_parameter_value()
    t = _param_type(v)
    if t == ParameterType.PARAMETER_INTEGER:
        return int(v.integer_value)
    if t == ParameterType.PARAMETER_STRING:
        return int(v.string_value.strip())
    return int(v.double_value)


def _read_double(node: Node, name: str) -> float:
    v = node.get_parameter(name).get_parameter_value()
    t = _param_type(v)
    if t == ParameterType.PARAMETER_DOUBLE:
        return float(v.double_value)
    if t == ParameterType.PARAMETER_INTEGER:
        return float(v.integer_value)
    if t == ParameterType.PARAMETER_STRING:
        return float(v.string_value.strip())
    return 0.0


def _ensure_repo_paths(repo_root: Path) -> None:
    r = repo_root.resolve()
    if not (r / "src").is_dir():
        raise FileNotFoundError(f"vision_benchmark_root must contain src/: {r}")
    if str(r) not in sys.path:
        sys.path.insert(0, str(r))
    if str(r / "src") not in sys.path:
        sys.path.insert(0, str(r / "src"))


def _fmt_pose_line(pose: np.ndarray) -> str:
    x, y, z, q1, q2, q3, q4 = pose.tolist()
    return (
        f"x={x:.4f} y={y:.4f} z={z:.4f} | "
        f"q1={q1:.4f} q2={q2:.4f} q3={q3:.4f} q4={q4:.4f}"
    )


def _draw_pose_block(img: np.ndarray, pose: np.ndarray, y0: int = 24) -> None:
    lines = [
        f"x y z: {pose[0]:.4f} {pose[1]:.4f} {pose[2]:.4f}",
        f"q1 q2 q3 q4: {pose[3]:.4f} {pose[4]:.4f} {pose[5]:.4f} {pose[6]:.4f}",
    ]
    y = y0
    for line in lines:
        cv2.putText(
            img,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 22


def _build_mask_rcnn(num_classes: int):
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def _overlay_masks(
    img_bgr: np.ndarray,
    out: dict,
    score_thr: float,
    class_names: list[str] | None,
    rng: random.Random,
) -> None:
    """Draw instance masks and boxes from Mask R-CNN output on img_bgr in-place."""
    boxes = out["boxes"].cpu().numpy()
    labels = out["labels"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    masks = out["masks"].cpu().numpy()

    h, w = img_bgr.shape[:2]
    for i in range(len(scores)):
        if scores[i] < score_thr:
            continue
        lab = int(labels[i])
        if lab == 0:
            continue
        color = (
            rng.randint(40, 255),
            rng.randint(40, 255),
            rng.randint(40, 255),
        )
        m = masks[i, 0]
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = m > 0.5
        overlay = img_bgr.copy()
        overlay[mask_bool] = color
        cv2.addWeighted(overlay, 0.35, img_bgr, 0.65, 0, dst=img_bgr)

        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        name = ""
        if class_names and 0 <= lab - 1 < len(class_names):
            name = class_names[lab - 1]
        cap = f"{lab}:{name} {scores[i]:.2f}" if name else f"id={lab} {scores[i]:.2f}"
        cv2.putText(
            img_bgr,
            cap,
            (x1, max(y1 - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )


class IsaacMultitaskNode(Node):
    def __init__(self) -> None:
        super().__init__("isaac_multitask_node")

        self.declare_parameter("vision_benchmark_root", "")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("annotated_image_topic", "/vision_benchmark/annotated_image")
        self.declare_parameter("pose_text_topic", "/vision_benchmark/pose_text")
        self.declare_parameter("pose_array_topic", "/vision_benchmark/pose_xyz_quat")
        self.declare_parameter("enable_pose", True)
        self.declare_parameter("pos_model_path", "")
        self.declare_parameter("ori_model_path", "")
        self.declare_parameter("pos_backbone", "resnet50")
        self.declare_parameter("use_cuda", True)
        self.declare_parameter("enable_mask_rcnn", False)
        self.declare_parameter("mask_rcnn_weights", "")
        self.declare_parameter("mask_rcnn_num_classes", 91)
        self.declare_parameter("mask_rcnn_score_threshold", 0.5)
        self.declare_parameter("mask_rcnn_input_size", 640)
        self.declare_parameter("mask_rcnn_class_names", "")
        self.declare_parameter("log_pose_each_frame", False)

        self._enable_pose = _read_bool(self, "enable_pose")
        self._enable_mcnn = _read_bool(self, "enable_mask_rcnn")

        root = self.get_parameter("vision_benchmark_root").get_parameter_value().string_value
        if not root.strip():
            root = os.environ.get("VISION_BENCHMARK_ROOT", "").strip()
        if self._enable_pose and not root:
            self.get_logger().fatal(
                "enable_pose requires vision_benchmark_root (or VISION_BENCHMARK_ROOT) "
                "pointing at the repository root (folder containing src/)."
            )
            raise RuntimeError("vision_benchmark_root not set")
        if self._enable_pose:
            _ensure_repo_paths(Path(root))

        self._pose_predictor = None
        if self._enable_pose:
            from src.pose_inference import SplitPosePredictor  # noqa: WPS433

            pos_p = self.get_parameter("pos_model_path").get_parameter_value().string_value
            ori_p = self.get_parameter("ori_model_path").get_parameter_value().string_value
            if not pos_p or not ori_p:
                self.get_logger().fatal("enable_pose requires pos_model_path and ori_model_path.")
                raise RuntimeError("pose paths missing")
            bb = self.get_parameter("pos_backbone").get_parameter_value().string_value
            use_cuda = _read_bool(self, "use_cuda")
            device = "cuda" if use_cuda else "cpu"
            self._pose_predictor = SplitPosePredictor(
                pos_p, ori_p, backbone=bb, device=device
            )
            self.get_logger().info(f"Loaded SplitPosePredictor (pos_backbone={bb}, device={device}).")

        self._mcnn = None
        self._mcnn_device = None
        self._mcnn_thr = _read_double(self, "mask_rcnn_score_threshold")
        self._mcnn_size = _read_int(self, "mask_rcnn_input_size")
        names_csv = self.get_parameter("mask_rcnn_class_names").get_parameter_value().string_value
        self._mcnn_class_names = (
            [s.strip() for s in names_csv.split(",") if s.strip()] if names_csv else None
        )
        self._mask_rng = random.Random(0)

        if self._enable_mcnn:
            import torch

            wpath = self.get_parameter("mask_rcnn_weights").get_parameter_value().string_value
            if not wpath:
                self.get_logger().fatal("enable_mask_rcnn requires mask_rcnn_weights path.")
                raise RuntimeError("mask_rcnn_weights missing")
            ncls = _read_int(self, "mask_rcnn_num_classes")
            use_cuda = _read_bool(self, "use_cuda")
            self._mcnn_device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
            self._mcnn = _build_mask_rcnn(ncls)
            ckpt = torch.load(wpath, map_location=self._mcnn_device)
            state = ckpt.get("model", ckpt)
            inc = self._mcnn.load_state_dict(state, strict=False)
            if inc.missing_keys or inc.unexpected_keys:
                self.get_logger().warn(
                    f"Mask R-CNN load_state_dict strict=False: "
                    f"missing={inc.missing_keys}, unexpected={inc.unexpected_keys}"
                )
            self._mcnn = self._mcnn.to(self._mcnn_device).eval()
            self.get_logger().info(
                f"Loaded Mask R-CNN ({ncls} classes) from {wpath} on {self._mcnn_device}."
            )

        if not self._enable_pose and not self._enable_mcnn:
            self.get_logger().warn(
                "Both enable_pose and enable_mask_rcnn are false; output image mirrors input."
            )

        itopic = self.get_parameter("image_topic").get_parameter_value().string_value
        atopic = self.get_parameter("annotated_image_topic").get_parameter_value().string_value
        ptopic = self.get_parameter("pose_text_topic").get_parameter_value().string_value
        patopic = self.get_parameter("pose_array_topic").get_parameter_value().string_value

        self._pub_img = self.create_publisher(Image, atopic, 10)
        self._pub_pose_text = self.create_publisher(String, ptopic, 10)
        self._pub_pose_arr = self.create_publisher(Float64MultiArray, patopic, 10)
        self.create_subscription(Image, itopic, self._on_image, 10)
        self._log_pose_each = _read_bool(self, "log_pose_each_frame")
        self.get_logger().info(
            f"Subscribed to {itopic}; publishing annotated {atopic}, pose text {ptopic}, array {patopic}."
        )

    def _on_image(self, msg: Image) -> None:
        try:
            cv_img = _imgmsg_to_bgr(msg)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"image decode failed: {e}")
            return

        vis = cv_img.copy()

        if self._pose_predictor is not None:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pose_vec = self._pose_predictor.predict(rgb)
            pose_summary = _fmt_pose_line(pose_vec)
            _draw_pose_block(vis, pose_vec)
            if self._log_pose_each:
                self.get_logger().info(pose_summary)
            else:
                self.get_logger().debug(pose_summary)

            arr = Float64MultiArray()
            arr.data = [float(x) for x in pose_vec.tolist()]
            self._pub_pose_arr.publish(arr)
            st = String()
            st.data = pose_summary
            self._pub_pose_text.publish(st)

        if self._mcnn is not None:
            import torch
            from PIL import Image as PILImage
            from torchvision import transforms

            h0, w0 = cv_img.shape[:2]
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(rgb)
            x = transforms.Compose(
                [
                    transforms.Resize((self._mcnn_size, self._mcnn_size)),
                    transforms.ToTensor(),
                ]
            )(pil).to(self._mcnn_device)
            with torch.no_grad():
                det = self._mcnn([x])[0]
            scale_x = w0 / self._mcnn_size
            scale_y = h0 / self._mcnn_size
            det["boxes"] = det["boxes"].clone()
            det["boxes"][:, [0, 2]] *= scale_x
            det["boxes"][:, [1, 3]] *= scale_y
            if det["masks"].numel() > 0:
                m = det["masks"]
                mup = []
                for i in range(m.shape[0]):
                    mi = m[i : i + 1]
                    mi = torch.nn.functional.interpolate(
                        mi, size=(h0, w0), mode="bilinear", align_corners=False
                    )
                    mup.append(mi)
                det["masks"] = torch.cat(mup, dim=0)
            _overlay_masks(vis, det, self._mcnn_thr, self._mcnn_class_names, self._mask_rng)

        try:
            out_msg = _bgr_to_imgmsg(vis, encoding="bgr8")
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"image encode failed: {e}")
            return
        out_msg.header = msg.header
        self._pub_img.publish(out_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IsaacMultitaskNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
