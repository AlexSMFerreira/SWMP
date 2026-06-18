"""
CUDA Semi-Global Matching (cv2.cuda.StereoSGM) disparity node — GPU.

Drop-in alternative to ros2_hitnet_disparity.py for the disparity-backend
comparison (see DISPARITY_COMPARISON.md). Same inputs, same 32FC1 disparity
output (pixels), same sky mask, same /stereo/debug/horizon overlay, so
ros2_pointcloud_node.py works unchanged.

Defaults are tuned for the airship use case: ~1 m stereo baseline, medium-long
range over the sea. Disparity and depth are related by  d_px = fx_px * B / Z, so
with B = 1 m a num_disparities of 128 searches down to Z_min = fx/128 metres
(e.g. fx≈1500 -> ~12 m). num_disparities must be 64, 128 or 256 for the CUDA SGM.

Requires OpenCV built WITH CUDA support. Checked at startup; if the build has no
CUDA, the node logs a clear fatal error and exits instead of crashing later.
"""

import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
from cv_bridge import CvBridge

import cv2
import time

from stereo_common import (
    HorizonMasker, extract_baseline_fx, to_float_disparity,
    make_disparity_msg, colorize_disparity, make_color_msg,
)


def cuda_available():
    return hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0


class SGMCudaDisparityNode(Node):
    def __init__(self):
        super().__init__('sgm_cuda_disparity_node')

        self.declare_parameter('left_rect_topic',   '/stereo/left/image_rect')
        self.declare_parameter('right_rect_topic',  '/stereo/right/image_rect')
        self.declare_parameter('rect_info_topic',   '/stereo/camera_info_rect')
        self.declare_parameter('disp_raw_topic',    '/stereo/disparity')
        self.declare_parameter('disp_color_topic',  '/stereo/disparity_color/compressed')
        self.declare_parameter('debug_topic',       '/stereo/debug/horizon/compressed')

        # ── Disparity tuning (1 m baseline, medium-long range over water) ──────
        self.declare_parameter('min_disparity',      0)
        self.declare_parameter('num_disparities',    256)  # cuda SGM requires 64, 128 or 256; Z_min = fx/128 m
        self.declare_parameter('block_size',         9)    # only feeds the P1/P2 smoothness terms
        self.declare_parameter('uniqueness_ratio',   10)
        # StereoSGM mode (the named enum is not exposed in the Python binding, so
        # use the int values shared with SGBM): MODE_HH=1 (full, best), MODE_HH4=3 (faster).
        self.declare_parameter('mode',               1)

        self.declare_parameter('sky_crop_pct',       0.40)
        self.declare_parameter('horizon_margin_pct', 0.03)
        self.declare_parameter('debug_horizon',      True)

        p = self.get_parameter
        self._bridge = CvBridge()
        self._min_disp = p('min_disparity').value
        self._num_disp = p('num_disparities').value

        self._masker = HorizonMasker(
            fallback_crop_pct=p('sky_crop_pct').value,
            horizon_margin_pct=p('horizon_margin_pct').value,
        )

        bs = p('block_size').value
        self._matcher = cv2.cuda.createStereoSGM(
            minDisparity=self._min_disp,
            numDisparities=self._num_disp,
            P1=8  * bs * bs,
            P2=32 * bs * bs,
            uniquenessRatio=p('uniqueness_ratio').value,
            mode=p('mode').value,
        )
        # Reused device buffers (allocated lazily on first frame).
        self._gpu_left = cv2.cuda_GpuMat()
        self._gpu_right = cv2.cuda_GpuMat()

        pub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,    history=HistoryPolicy.KEEP_LAST, depth=5)
        vis_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)

        self._pub_disp  = self.create_publisher(Image,           p('disp_raw_topic').value,   pub_qos)
        self._pub_color = self.create_publisher(CompressedImage, p('disp_color_topic').value, vis_qos)
        self._pub_debug = self.create_publisher(CompressedImage, p('debug_topic').value,      vis_qos)
        self._debug_horizon = p('debug_horizon').value

        self._sub_info = self.create_subscription(
            CameraInfo, p('rect_info_topic').value, self._cb_camera_info, pub_qos
        )
        self.get_logger().info(
            f"CUDA StereoSGM Node online ({cv2.cuda.getCudaEnabledDeviceCount()} device(s)). "
            "Waiting for rectified CameraInfo..."
        )

        self._sub_left  = message_filters.Subscriber(self, Image, p('left_rect_topic').value,  qos_profile=pub_qos)
        self._sub_right = message_filters.Subscriber(self, Image, p('right_rect_topic').value, qos_profile=pub_qos)
        self._sync = message_filters.TimeSynchronizer([self._sub_left, self._sub_right], queue_size=10)
        self._sync.registerCallback(self._cb_images)

    def _cb_camera_info(self, msg: CameraInfo):
        baseline, fx = extract_baseline_fx(msg)
        self.get_logger().info(f"Baseline: {baseline:.4f}m | fx: {fx:.2f}px | CUDA SGM ready.")
        self.destroy_subscription(self._sub_info)

    def _cb_images(self, left_msg: Image, right_msg: Image):
        start = time.perf_counter()

        left_bgr  = self._bridge.imgmsg_to_cv2(left_msg,  desired_encoding='bgr8')
        right_bgr = self._bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
        left_gray  = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

        self._gpu_left.upload(left_gray)
        self._gpu_right.upload(right_gray)
        gpu_disp = self._matcher.compute(self._gpu_left, self._gpu_right)
        raw = gpu_disp.download()                            # CV_16S, x16
        disp = to_float_disparity(raw, self._min_disp)

        sky_mask, source = self._masker.compute_mask(left_bgr)
        disp[sky_mask] = 0.0

        latency = (time.perf_counter() - start) * 1000.0

        self._pub_disp.publish(make_disparity_msg(self._bridge, disp, left_msg.header))

        color = colorize_disparity(disp, self._num_disp)
        color[sky_mask] = 0
        color_msg = make_color_msg(color, left_msg.header)
        if color_msg is not None:
            self._pub_color.publish(color_msg)

        if self._debug_horizon:
            debug = self._masker.make_debug_image(left_bgr, source)
            debug_msg = make_color_msg(debug, left_msg.header) if debug is not None else None
            if debug_msg is not None:
                self._pub_debug.publish(debug_msg)

        self.get_logger().info(f"CUDA-SGM disparity | [{source}] | {latency:.1f} ms")


def main(args=None):
    rclpy.init(args=args)
    if not cuda_available():
        print("FATAL: OpenCV was not built with CUDA (cv2.cuda unavailable or no GPU). "
              "Use ros2_sgbm_disparity.py for a CPU semi-global matcher instead.",
              file=sys.stderr)
        rclpy.shutdown()
        sys.exit(1)
    node = SGMCudaDisparityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
