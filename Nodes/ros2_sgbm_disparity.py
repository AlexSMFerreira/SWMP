"""
StereoSGBM (semi-global block-matching) disparity node — CPU.

Drop-in alternative to ros2_hitnet_disparity.py for the disparity-backend
comparison (see DISPARITY_COMPARISON.md). Same inputs, same 32FC1 disparity
output (pixels), same sky mask, same /stereo/debug/horizon overlay, so
ros2_pointcloud_node.py works unchanged. Best CPU quality, slower than plain
StereoBM.

Defaults are tuned for the airship use case: ~1 m stereo baseline, medium-long
range over the sea. Disparity and depth are related by  d_px = fx_px * B / Z, so
with B = 1 m a num_disparities of 128 searches down to Z_min = fx/128 metres
(e.g. fx≈1500 -> ~12 m). SGBM's global smoothness term (P1/P2) makes it the
strongest classical option on the low-texture water surface.
"""

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


class SGBMDisparityNode(Node):
    def __init__(self):
        super().__init__('sgbm_disparity_node')

        self.declare_parameter('left_rect_topic',   '/stereo/left/image_rect')
        self.declare_parameter('right_rect_topic',  '/stereo/right/image_rect')
        self.declare_parameter('rect_info_topic',   '/stereo/camera_info_rect')
        self.declare_parameter('disp_raw_topic',    '/stereo/disparity')
        self.declare_parameter('disp_color_topic',  '/stereo/disparity_color/compressed')
        self.declare_parameter('debug_topic',       '/stereo/debug/horizon/compressed')

        # ── Disparity tuning (1 m baseline, medium-long range over water) ──────
        self.declare_parameter('min_disparity',      0)
        self.declare_parameter('num_disparities',    128)   # divisible by 16; Z_min = fx/128 m
        self.declare_parameter('block_size',         5)     # odd; matched-block window
        self.declare_parameter('uniqueness_ratio',   10)    # reject ambiguous matches on water
        self.declare_parameter('speckle_window_size', 6)  # remove larger noise blobs
        self.declare_parameter('speckle_range',      1)     # SGBM scales this x16 internally
        self.declare_parameter('disp12_max_diff',    1)     # left-right consistency check
        self.declare_parameter('pre_filter_cap',     63)
        self.declare_parameter('mode',               cv2.STEREO_SGBM_MODE_SGBM)
        self.declare_parameter('sky_crop_pct',       0.40)
        self.declare_parameter('horizon_margin_pct', 0.03)
        self.declare_parameter('debug_horizon',      True)

        # ── WLS post-filter (edge-aware smoothing, fills low-confidence gaps) ──
        self.declare_parameter('use_wls_filter',     True)
        self.declare_parameter('wls_lambda',         8000.0)  # regularization: higher = smoother
        self.declare_parameter('wls_sigma',          1.5)     # edge sensitivity: higher = less edge-aware

        p = self.get_parameter
        self._bridge = CvBridge()
        self._min_disp = p('min_disparity').value
        self._num_disp = p('num_disparities').value

        self._masker = HorizonMasker(
            fallback_crop_pct=p('sky_crop_pct').value,
            horizon_margin_pct=p('horizon_margin_pct').value,
        )

        bs = p('block_size').value
        channels = 1   # matching on grayscale
        self._matcher = cv2.StereoSGBM_create(
            minDisparity=self._min_disp,
            numDisparities=self._num_disp,
            blockSize=bs,
            P1=8  * channels * bs * bs,
            P2=32 * channels * bs * bs,
            disp12MaxDiff=p('disp12_max_diff').value,
            uniquenessRatio=p('uniqueness_ratio').value,
            speckleWindowSize=p('speckle_window_size').value,
            speckleRange=p('speckle_range').value,
            preFilterCap=p('pre_filter_cap').value,
            mode=p('mode').value,
        )

        self._use_wls = p('use_wls_filter').value
        if self._use_wls:
            self._right_matcher = cv2.ximgproc.createRightMatcher(self._matcher)
            self._wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self._matcher)
            self._wls_filter.setLambda(p('wls_lambda').value)
            self._wls_filter.setSigmaColor(p('wls_sigma').value)

        pub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,    history=HistoryPolicy.KEEP_LAST, depth=5)
        vis_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)

        self._pub_disp  = self.create_publisher(Image,           p('disp_raw_topic').value,   pub_qos)
        self._pub_color = self.create_publisher(CompressedImage, p('disp_color_topic').value, vis_qos)
        self._pub_debug = self.create_publisher(CompressedImage, p('debug_topic').value,      vis_qos)
        self._debug_horizon = p('debug_horizon').value

        self._sub_info = self.create_subscription(
            CameraInfo, p('rect_info_topic').value, self._cb_camera_info, pub_qos
        )
        self.get_logger().info("StereoSGBM Node waiting for rectified CameraInfo...")

        self._sub_left  = message_filters.Subscriber(self, Image, p('left_rect_topic').value,  qos_profile=pub_qos)
        self._sub_right = message_filters.Subscriber(self, Image, p('right_rect_topic').value, qos_profile=pub_qos)
        self._sync = message_filters.TimeSynchronizer([self._sub_left, self._sub_right], queue_size=10)
        self._sync.registerCallback(self._cb_images)

    def _cb_camera_info(self, msg: CameraInfo):
        baseline, fx = extract_baseline_fx(msg)
        self.get_logger().info(f"Baseline: {baseline:.4f}m | fx: {fx:.2f}px | StereoSGBM online.")
        self.destroy_subscription(self._sub_info)

    def _cb_images(self, left_msg: Image, right_msg: Image):
        start = time.perf_counter()

        left_bgr  = self._bridge.imgmsg_to_cv2(left_msg,  desired_encoding='bgr8')
        right_bgr = self._bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
        left_gray  = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

        raw = self._matcher.compute(left_gray, right_gray)   # CV_16S, x16
        if self._use_wls:
            raw_right = self._right_matcher.compute(right_gray, left_gray)
            raw = self._wls_filter.filter(raw, left_gray, disparity_map_right=raw_right)
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

        self.get_logger().info(f"SGBM disparity | [{source}] | {latency:.1f} ms")


def main(args=None):
    rclpy.init(args=args)
    node = SGBMDisparityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
