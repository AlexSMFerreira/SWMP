"""
StereoBM (classical block-matching) disparity node — CPU.

Drop-in alternative to ros2_hitnet_disparity.py for the disparity-backend
comparison (see DISPARITY_COMPARISON.md). Same inputs, same 32FC1 disparity
output (pixels), same sky mask, same /stereo/debug/horizon overlay, so
ros2_pointcloud_node.py works unchanged. Fastest CPU option, weakest quality on
low-texture water.

Defaults are tuned for the water-surface use case: ~1 m stereo baseline, medium-long
range over the sea. Disparity and depth are related by  d_px = fx_px * B / Z, so
with B = 1 m a num_disparities of 128 searches down to Z_min = fx/128 metres
(e.g. fx≈1500 -> ~12 m), which comfortably covers the expected depth range while
capping the (BM) search cost.

Key tuning decisions for water:
  - texture_threshold=0: water has almost no texture; gating on it kills coverage.
    Speckle filtering + WLS handle spurious matches instead.
  - pre_filter_type=XSOBEL: edge-response filter handles specular/reflective surfaces
    better than NORMALIZED_RESPONSE, which is thrown off by local mean shifts.
  - pre_filter_cap=63: stronger normalisation clamps specular highlights.
  - speckle_range=2: tight connectivity for a smooth surface (only adjacent disparity
    values considered the same blob); 32 would merge unrelated regions.
  - WLS post-filter (use_wls_filter=True): edge-aware gap-filling using the left image's
    edge structure fills the sparse coverage SBM produces on textureless water.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
from std_msgs.msg import Float32
from cv_bridge import CvBridge

import cv2
import time

from stereo_common import (
    HorizonMasker, extract_baseline_fx, to_float_disparity,
    make_disparity_msg, colorize_disparity, make_color_msg,
    downscale_pair, rescale_disparity, photometric_consistency_error,
)


class SBMDisparityNode(Node):
    def __init__(self):
        super().__init__('sbm_disparity_node')

        self.declare_parameter('left_rect_topic',   '/stereo/left/image_rect')
        self.declare_parameter('right_rect_topic',  '/stereo/right/image_rect')
        self.declare_parameter('rect_info_topic',   '/stereo/camera_info_rect')
        self.declare_parameter('disp_raw_topic',    '/stereo/disparity')
        self.declare_parameter('disp_color_topic',  '/stereo/disparity_color/compressed')
        self.declare_parameter('debug_topic',       '/stereo/debug/horizon/compressed')

        # ── Disparity tuning (1 m baseline, medium-long range over water) ──────
        self.declare_parameter('min_disparity',      0)
        self.declare_parameter('num_disparities',    128)   # divisible by 16; Z_min = fx/128 m
        self.declare_parameter('block_size',         21)    # odd; large window for low-texture water
        # texture_threshold=0: don't gate on texture — water has almost none; rely on
        # speckle filtering + WLS to clean up spurious matches instead.
        self.declare_parameter('texture_threshold',  0)
        self.declare_parameter('uniqueness_ratio',   15)
        self.declare_parameter('speckle_window_size', 200)
        # speckle_range=2: tight connectivity — only treat adjacent disparity values as
        # the same blob, appropriate for the smooth water surface.
        self.declare_parameter('speckle_range',      2)
        self.declare_parameter('disp12_max_diff',    1)     # left-right consistency check
        # pre_filter_type: XSOBEL (1) handles specular/reflective surfaces better than
        # NORMALIZED_RESPONSE (0) because it responds to edges rather than local mean.
        self.declare_parameter('pre_filter_type',    cv2.STEREO_BM_PREFILTER_XSOBEL)
        self.declare_parameter('pre_filter_size',    9)     # must be odd, 5..255
        self.declare_parameter('pre_filter_cap',     63)    # stronger normalization for water

        # ── WLS post-filter (edge-aware gap-filling — same as SGBM) ─────────────
        self.declare_parameter('use_wls_filter',     True)
        self.declare_parameter('wls_lambda',         8000.0)
        self.declare_parameter('wls_sigma',          1.5)

        # The rectifier now publishes at native camera resolution; downscale here to
        # a working resolution for the matcher (it's by far the dominant cost, and
        # num_disparities above was tuned for ~320x240) and rescale the disparity
        # back up afterward. -1/-1 disables this and runs StereoBM at native res.
        self.declare_parameter('input_width',        320)
        self.declare_parameter('input_height',       240)

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

        self._matcher = cv2.StereoBM_create(
            numDisparities=self._num_disp,
            blockSize=p('block_size').value,
        )
        self._matcher.setMinDisparity(self._min_disp)
        self._matcher.setTextureThreshold(p('texture_threshold').value)
        self._matcher.setUniquenessRatio(p('uniqueness_ratio').value)
        self._matcher.setSpeckleWindowSize(p('speckle_window_size').value)
        self._matcher.setSpeckleRange(p('speckle_range').value)
        self._matcher.setDisp12MaxDiff(p('disp12_max_diff').value)
        self._matcher.setPreFilterType(p('pre_filter_type').value)
        self._matcher.setPreFilterSize(p('pre_filter_size').value)
        self._matcher.setPreFilterCap(p('pre_filter_cap').value)

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

        # No-reference disparity-quality metric (left-right photometric consistency),
        # for the cross-backend comparison — see stereo_common.photometric_consistency_error.
        self._pub_photo_err = self.create_publisher(Float32, '/stereo/disparity_quality/photo_error', vis_qos)
        self._pub_valid_frac = self.create_publisher(Float32, '/stereo/disparity_quality/valid_fraction', vis_qos)
        self._pub_latency = self.create_publisher(Float32, '/stereo/disparity_quality/latency_ms', vis_qos)

        self._sub_info = self.create_subscription(
            CameraInfo, p('rect_info_topic').value, self._cb_camera_info, pub_qos
        )
        self.get_logger().info("StereoBM Node waiting for rectified CameraInfo...")

        self._sub_left  = message_filters.Subscriber(self, Image, p('left_rect_topic').value,  qos_profile=pub_qos)
        self._sub_right = message_filters.Subscriber(self, Image, p('right_rect_topic').value, qos_profile=pub_qos)
        self._sync = message_filters.TimeSynchronizer([self._sub_left, self._sub_right], queue_size=10)
        self._sync.registerCallback(self._cb_images)

    def _cb_camera_info(self, msg: CameraInfo):
        baseline, fx = extract_baseline_fx(msg)
        self.get_logger().info(f"Baseline: {baseline:.4f}m | fx: {fx:.2f}px | StereoBM online.")
        self.destroy_subscription(self._sub_info)

    def _cb_images(self, left_msg: Image, right_msg: Image):
        start = time.perf_counter()

        left_bgr  = self._bridge.imgmsg_to_cv2(left_msg,  desired_encoding='bgr8')
        right_bgr = self._bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
        full_w, full_h = left_bgr.shape[1], left_bgr.shape[0]

        inp_w = self.get_parameter('input_width').value
        inp_h = self.get_parameter('input_height').value
        proc_left, proc_right = downscale_pair(left_bgr, right_bgr, inp_w, inp_h)
        scale_x = full_w / proc_left.shape[1]

        left_gray  = cv2.cvtColor(proc_left,  cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(proc_right, cv2.COLOR_BGR2GRAY)

        raw = self._matcher.compute(left_gray, right_gray)   # CV_16S, x16
        if self._use_wls:
            raw_right = self._right_matcher.compute(right_gray, left_gray)
            raw = self._wls_filter.filter(raw, left_gray, disparity_map_right=raw_right)
        disp = to_float_disparity(raw, self._min_disp)
        disp = rescale_disparity(disp, (full_w, full_h))

        sky_mask, source = self._masker.compute_mask(left_bgr)
        disp[sky_mask] = 0.0

        full_left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        full_right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
        photo_err, valid_frac = photometric_consistency_error(full_left_gray, full_right_gray, disp)
        self._pub_photo_err.publish(Float32(data=photo_err))
        self._pub_valid_frac.publish(Float32(data=valid_frac))

        latency = (time.perf_counter() - start) * 1000.0
        self._pub_latency.publish(Float32(data=latency))

        self._pub_disp.publish(make_disparity_msg(self._bridge, disp, left_msg.header))

        color = colorize_disparity(disp, self._num_disp * scale_x)
        color[sky_mask] = 0
        color_msg = make_color_msg(color, left_msg.header)
        if color_msg is not None:
            self._pub_color.publish(color_msg)

        if self._debug_horizon:
            debug = self._masker.make_debug_image(left_bgr, source)
            debug_msg = make_color_msg(debug, left_msg.header) if debug is not None else None
            if debug_msg is not None:
                self._pub_debug.publish(debug_msg)

        self.get_logger().info(
            f"SBM disparity | [{source}] | {latency:.1f} ms | "
            f"photo_err={photo_err:.2f} valid={valid_frac:.2%}"
            + (" [WLS]" if self._use_wls else "")
        )


def main(args=None):
    rclpy.init(args=args)
    node = SBMDisparityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
