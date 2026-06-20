"""
RAFT-Stereo disparity node.

Deep-learning stereo backend using the Princeton Vision RAFT-Stereo model
(https://github.com/princeton-vl/RAFT-Stereo). Tuned for an NVIDIA RTX 2060
(6 GB VRAM) via --mixed_precision + --corr_implementation alt and a reduced
iteration count for sub-100 ms latency.

The rectifier (ros2_stereo_rectifier.py) publishes at native camera resolution;
input_width/input_height below downscale to this node's own working resolution
(default 320x240, matching the rectifier's old hardcoded output) and the computed
disparity is rescaled back up to native resolution before publishing.

Drop-in replacement for ros2_hitnet_disparity.py: same input topics, same
32FC1 disparity output, same horizon-based sky mask, same debug overlay.

Setup (one-time):
    git clone https://github.com/princeton-vl/RAFT-Stereo.git   # repo root
    cd RAFT-Stereo && bash download_models.sh

Run (from repo root):
    python3 Nodes/ros2_raftstereo_disparity.py --ros-args \\
        -p restore_ckpt:=RAFT-Stereo/models/raftstereo-middlebury.pth \\
        -p use_sim_time:=true
"""

import os
import sys
import time
import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
from std_msgs.msg import Float64
from cv_bridge import CvBridge

import cv2
import numpy as np
import torch

from stereo_common import (
    HorizonMasker, extract_baseline_fx,
    make_disparity_msg, make_color_msg,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _colorize(disp_float, max_disp=None):
    """JET colormap; zero/invalid pixels stay black.
    max_disp auto-computed from the 95th percentile if not given."""
    if max_disp is None or max_disp <= 0:
        valid = disp_float[disp_float > 0]
        max_disp = float(np.percentile(valid, 95)) if valid.size > 0 else 64.0
    scaled = np.clip(disp_float / max_disp * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    color[disp_float <= 0.0] = 0
    return color


# ── Node ───────────────────────────────────────────────────────────────────────

class RAFTStereoDisparityNode(Node):
    def __init__(self):
        super().__init__('raftstereo_disparity_node')

        self.declare_parameter('left_rect_topic',     '/stereo/left/image_rect')
        self.declare_parameter('right_rect_topic',    '/stereo/right/image_rect')
        self.declare_parameter('rect_info_topic',     '/stereo/camera_info_rect')
        self.declare_parameter('disp_raw_topic',      '/stereo/disparity')
        self.declare_parameter('disp_color_topic',    '/stereo/disparity_color/compressed')
        # RAFT-Stereo repo root (absolute, or relative to the project root)
        self.declare_parameter('raft_repo_path',      'RAFT-Stereo')
        self.declare_parameter('restore_ckpt',        'RAFT-Stereo/models/raftstereo-realtime.pth')
        # Optimization flags for RTX 2060 (6 GB VRAM)
        self.declare_parameter('mixed_precision',     True)   # FP16 via Tensor Cores
        self.declare_parameter('corr_implementation', 'alt')  # memory-efficient alt. corr volume
        self.declare_parameter('valid_iters',         12)     # 8–12 for speed; 32 = full quality
        # Architecture — change these to match the checkpoint (realtime needs different values)
        # Realtime preset: shared_backbone=true, n_downsample=3, n_gru_layers=2, slow_fast_gru=true
        self.declare_parameter('shared_backbone',     True)
        self.declare_parameter('n_downsample',        3)
        self.declare_parameter('n_gru_layers',        2)
        self.declare_parameter('slow_fast_gru',       True)
        # The rectifier publishes at native camera resolution; downscale here to a
        # working resolution for RAFT-Stereo (the dominant cost) and rescale the
        # disparity back up afterward. -1/-1 disables this and runs at native res
        # (likely too slow/VRAM-heavy on a 6 GB card at full camera resolution).
        self.declare_parameter('input_width',         320)
        self.declare_parameter('input_height',        240)
        self.declare_parameter('max_distance',        30.0)
        self.declare_parameter('sky_crop_pct',        0.40)
        self.declare_parameter('horizon_margin_pct',  0.03)
        self.declare_parameter('debug_horizon',       True)

        p = self.get_parameter

        self._bridge      = CvBridge()
        self._model       = None
        self._padder_cls  = None
        self._device      = None
        self._model_ready = False

        self._horizon = HorizonMasker(
            fallback_crop_pct=p('sky_crop_pct').value,
            horizon_margin_pct=p('horizon_margin_pct').value,
        )

        pub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=5)
        vis_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=5)

        self._pub_disp_raw   = self.create_publisher(Image,           p('disp_raw_topic').value,   pub_qos)
        self._pub_disp_color = self.create_publisher(CompressedImage, p('disp_color_topic').value,  vis_qos)
        self._pub_debug      = self.create_publisher(CompressedImage, '/stereo/debug/horizon/compressed', vis_qos)
        self._pub_roll       = self.create_publisher(Float64,         '/stereo/horizon_roll',       vis_qos)

        self._sub_info = self.create_subscription(
            CameraInfo, p('rect_info_topic').value, self._cb_camera_info, pub_qos
        )
        self.get_logger().info('RAFT-Stereo node waiting for rectified CameraInfo…')

        self._sync_qos                  = pub_qos
        self._sub_left = self._sub_right = self._sync = None
        self._last_msg_stamp: float | None = None
        self._last_frame_time: float       = time.monotonic()
        self._build_sync()
        self.create_timer(3.0, self._cb_watchdog)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _resolve_path(self, path: str) -> str:
        """Resolve a path that may be relative to the project root."""
        if os.path.isabs(path):
            return path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, path)

    def _load_model(self):
        repo_dir  = self._resolve_path(self.get_parameter('raft_repo_path').value)
        core_dir  = os.path.join(repo_dir, 'core')
        for d in (repo_dir, core_dir):
            if d not in sys.path:
                sys.path.insert(0, d)

        from raft_stereo import RAFTStereo       # noqa: PLC0415
        from utils.utils import InputPadder      # noqa: PLC0415

        self._padder_cls = InputPadder

        p = self.get_parameter
        args = argparse.Namespace(
            mixed_precision     = p('mixed_precision').value,
            valid_iters         = p('valid_iters').value,
            corr_implementation = p('corr_implementation').value,
            shared_backbone     = p('shared_backbone').value,
            n_downsample        = p('n_downsample').value,
            n_gru_layers        = p('n_gru_layers').value,
            slow_fast_gru       = p('slow_fast_gru').value,
            hidden_dims         = [128] * 3,
            context_dims        = [128] * 3,
            corr_levels         = 4,
            corr_radius         = 4,
            context_norm        = 'batch',
        )

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'RAFT-Stereo device: {self._device}')

        ckpt_path = self._resolve_path(p('restore_ckpt').value)
        state = torch.load(ckpt_path, map_location=self._device, weights_only=False)
        # Checkpoints may be wrapped under a 'model' key and/or DataParallel
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        state = {k.replace('module.', ''): v for k, v in state.items()}

        model = RAFTStereo(args)
        model.load_state_dict(state)
        model = model.to(self._device).eval()

        self._model       = model
        self._valid_iters = args.valid_iters
        self.get_logger().info(f'RAFT-Stereo checkpoint loaded: {ckpt_path}')

    # ── Sync management ───────────────────────────────────────────────────────

    def _build_sync(self):
        p = self.get_parameter
        if self._sub_left is not None:
            self.destroy_subscription(self._sub_left.sub)
        if self._sub_right is not None:
            self.destroy_subscription(self._sub_right.sub)
        self._sub_left  = message_filters.Subscriber(
            self, Image, p('left_rect_topic').value,  qos_profile=self._sync_qos)
        self._sub_right = message_filters.Subscriber(
            self, Image, p('right_rect_topic').value, qos_profile=self._sync_qos)
        self._sync = message_filters.TimeSynchronizer(
            [self._sub_left, self._sub_right], queue_size=10)
        self._sync.registerCallback(self._cb_images)

    def _reset_sync(self, reason: str):
        self.get_logger().warn(f'RAFT-Stereo: resetting sync — {reason}')
        self._build_sync()
        self._last_frame_time = time.monotonic()
        self._last_msg_stamp  = None

    def _cb_watchdog(self):
        elapsed = time.monotonic() - self._last_frame_time
        if elapsed > 3.0:
            self._reset_sync(f'no frame for {elapsed:.1f}s')

    # ── Camera info ───────────────────────────────────────────────────────────

    def _cb_camera_info(self, msg: CameraInfo):
        if self._model_ready:
            return
        baseline, fx = extract_baseline_fx(msg)
        self.get_logger().info(f'Baseline: {baseline:.4f} m  fx: {fx:.2f} px')
        try:
            self._load_model()
            self._model_ready = True
            self.get_logger().info('RAFT-Stereo inference engine online.')
            self.destroy_subscription(self._sub_info)
        except Exception as exc:
            self.get_logger().error(f'Failed to load RAFT-Stereo: {exc}')

    # ── Inference ─────────────────────────────────────────────────────────────

    def _to_tensor(self, img_bgr):
        """BGR uint8 → float32 RGB tensor [1, 3, H, W]."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).float()[None]

    def _infer_disparity(self, left_cv, right_cv) -> np.ndarray:
        """Returns float32 disparity array (pixels, positive), same H×W as input."""
        inp_w = self.get_parameter('input_width').value
        inp_h = self.get_parameter('input_height').value
        orig_h, orig_w = left_cv.shape[:2]

        if inp_w > 0 and inp_h > 0:
            left_cv  = cv2.resize(left_cv,  (inp_w, inp_h), interpolation=cv2.INTER_LINEAR)
            right_cv = cv2.resize(right_cv, (inp_w, inp_h), interpolation=cv2.INTER_LINEAR)

        img1 = self._to_tensor(left_cv).to(self._device)
        img2 = self._to_tensor(right_cv).to(self._device)

        padder       = self._padder_cls(img1.shape, divis_by=32)
        img1, img2   = padder.pad(img1, img2)

        with torch.no_grad():
            _, flow_up = self._model(img1, img2, iters=self._valid_iters, test_mode=True)

        flow_up = padder.unpad(flow_up)
        # RAFT-Stereo returns negative horizontal flow (right-shift convention);
        # negate to get positive disparity in pixels.
        disp_np = (-flow_up[0, 0]).cpu().numpy().astype(np.float32)
        disp_np = np.maximum(disp_np, 0.0)

        if inp_w > 0 and inp_h > 0:
            scale_x = orig_w / inp_w
            disp_np = cv2.resize(disp_np, (orig_w, orig_h),
                                 interpolation=cv2.INTER_LINEAR) * scale_x

        return disp_np

    # ── Main callback ─────────────────────────────────────────────────────────

    def _cb_images(self, left_msg: Image, right_msg: Image):
        if not self._model_ready:
            return

        stamp_sec = left_msg.header.stamp.sec + left_msg.header.stamp.nanosec * 1e-9
        if self._last_msg_stamp is not None and stamp_sec < self._last_msg_stamp - 1.0:
            self._reset_sync(f'time jump {self._last_msg_stamp:.1f}→{stamp_sec:.1f}s')
            return
        self._last_msg_stamp  = stamp_sec
        self._last_frame_time = time.monotonic()

        t0 = time.perf_counter()

        left_cv  = self._bridge.imgmsg_to_cv2(left_msg,  desired_encoding='bgr8')
        right_cv = self._bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

        # 1. Detect horizon + build sky mask
        sky_mask, source = self._horizon.compute_mask(left_cv)

        # 2. Publish roll angle derived from horizon direction
        if self._horizon._cur_raw is not None:
            dx, dy = float(self._horizon._cur_raw[1][0]), float(self._horizon._cur_raw[1][1])
            if dx < 0:
                dx, dy = -dx, -dy
            roll_msg      = Float64()
            roll_msg.data = float(np.arctan2(dy, dx))
            self._pub_roll.publish(roll_msg)

        # 3. Run disparity inference
        disparity_map = self._infer_disparity(left_cv, right_cv)

        # 4. Zero out sky
        disparity_map[sky_mask] = 0.0

        latency = (time.perf_counter() - t0) * 1000

        # 5. Publish raw disparity (32FC1, pixels)
        disp_msg        = self._bridge.cv2_to_imgmsg(disparity_map, encoding='32FC1')
        disp_msg.header = left_msg.header
        disp_msg.step   = disparity_map.shape[1] * 4
        self._pub_disp_raw.publish(disp_msg)

        # 6. Publish colourised preview
        color_img = _colorize(disparity_map)
        color_img[sky_mask] = 0
        color_msg = make_color_msg(color_img, left_msg.header)
        if color_msg:
            self._pub_disp_color.publish(color_msg)

        # 7. Publish horizon debug overlay
        if self.get_parameter('debug_horizon').value:
            dbg = self._horizon.make_debug_image(left_cv, source)
            if dbg is not None:
                dbg_msg = make_color_msg(dbg, left_msg.header)
                if dbg_msg:
                    self._pub_debug.publish(dbg_msg)

        self.get_logger().info(
            f'RAFT-Stereo | [{source}] | {latency:.1f} ms'
        )


def main(args=None):
    rclpy.init(args=args)
    node = RAFTStereoDisparityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
