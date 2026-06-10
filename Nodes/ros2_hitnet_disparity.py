import os
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
from cv_bridge import CvBridge

import cv2
import numpy as np
import time

# The `hitnet` package and `models/` live at the repo root, while this node lives
# in Nodes/. Add the repo root to the path so `import hitnet` resolves regardless
# of where the script sits (run the pipeline from the repo root so the relative
# `models/...` default also resolves).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hitnet import HitNet, ModelType, CameraConfig


class HitNetDisparityNode(Node):
    def __init__(self):
        super().__init__('hitnet_disparity_node')

        self.declare_parameter('left_rect_topic',    '/stereo/left/image_rect')
        self.declare_parameter('right_rect_topic',   '/stereo/right/image_rect')
        self.declare_parameter('rect_info_topic',    '/stereo/camera_info_rect')
        self.declare_parameter('disp_raw_topic',     '/stereo/disparity')
        self.declare_parameter('disp_color_topic',   '/stereo/disparity_color/compressed')
        self.declare_parameter('model_path',         'models/eth3d/saved_model_240x320/model_float32.onnx')
        self.declare_parameter('max_distance',       200.0)
        self.declare_parameter('sky_crop_pct',       0.40)   # Fallback if Hough fails
        self.declare_parameter('horizon_margin_pct', 0.03)   # Downward margin below detected horizon
        self.declare_parameter('debug_horizon',      True)

        p = self.get_parameter

        self._bridge = CvBridge()
        self.depth_estimator = None
        self.model_ready = False
        self.fallback_crop_pct = p('sky_crop_pct').value

        # Last known good horizon: tuple(mean: np.ndarray, direction: np.ndarray)
        # Already has the margin nudge baked in, ready to use directly as a mask.
        # The raw (pre-nudge) horizon is stored separately for debug drawing.
        self._last_horizon_masked = None   # nudged — used for masking
        self._last_horizon_raw    = None   # pre-nudge — used for yellow debug line

        pub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,  history=HistoryPolicy.KEEP_LAST, depth=5)
        vis_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)

        self._pub_disp_raw   = self.create_publisher(Image,           p('disp_raw_topic').value,  pub_qos)
        self._pub_disp_color = self.create_publisher(CompressedImage, p('disp_color_topic').value, vis_qos)
        self._pub_debug      = self.create_publisher(CompressedImage, '/stereo/debug/horizon/compressed', vis_qos)

        self._sub_info = self.create_subscription(
            CameraInfo, p('rect_info_topic').value, self._cb_camera_info, pub_qos
        )
        self.get_logger().info("HITNet Node waiting for rectified CameraInfo...")

        self._sub_left  = message_filters.Subscriber(self, Image, p('left_rect_topic').value,  qos_profile=pub_qos)
        self._sub_right = message_filters.Subscriber(self, Image, p('right_rect_topic').value, qos_profile=pub_qos)
        self._sync = message_filters.TimeSynchronizer([self._sub_left, self._sub_right], queue_size=10)
        self._sync.registerCallback(self._cb_images)

    # ── Camera info ───────────────────────────────────────────────────────────

    def _cb_camera_info(self, msg: CameraInfo):
        if self.model_ready:
            return

        fx_rect = msg.p[0]
        p2_flattened = [float(v) for v in msg.distortion_model.split(';')]
        baseline = abs(-p2_flattened[3] / fx_rect)

        self.get_logger().info(f"Extracted Baseline: {baseline:.4f}m | fx: {fx_rect:.2f}px")

        camera_config = CameraConfig(baseline=baseline, f=fx_rect)
        self.depth_estimator = HitNet(
            self.get_parameter('model_path').value,
            ModelType.eth3d,
            camera_config,
            self.get_parameter('max_distance').value,
        )
        self.model_ready = True
        self.get_logger().info("HITNet Inference Engine Online.")
        self.destroy_subscription(self._sub_info)

    # ── Horizon detection ─────────────────────────────────────────────────────

    def _find_horizon_raw(self, img_bgr):
        """
        Returns (mean, direction) of the horizon line, or None if not found.
        This is the RAW detection — no margin nudge applied yet.
        """
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 50, 150)

        min_line_len = img_bgr.shape[1] * 0.2
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100,
            minLineLength=min_line_len, maxLineGap=50,
        )
        if lines is None:
            return None

        candidates = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            if angle < 20.0 or angle > 160.0:
                candidates.append((x1, y1, x2, y2))

        if not candidates:
            return None

        pts = np.array([[x1, y1] for x1, y1, _, _ in candidates] +
                       [[x2, y2] for _, _, x2, y2 in candidates], dtype=np.float32)
        mean = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - mean)
        direction = vt[0]

        return mean, direction

    def _apply_margin(self, horizon_raw, img_h):
        """Nudges the horizon centroid downward by horizon_margin_pct of image height."""
        mean, direction = horizon_raw
        nudge  = img_h * self.get_parameter('horizon_margin_pct').value
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        if normal[1] > 0:
            normal = -normal  # Ensure normal points toward sky (decreasing y)
        nudged_mean = mean + nudge * (-normal)
        return nudged_mean, direction

    def _make_sky_mask(self, shape, horizon_masked):
        """Boolean mask — True = sky (above the nudged horizon line)."""
        h, w = shape[:2]
        mean, direction = horizon_masked

        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        if normal[1] > 0:
            normal = -normal

        xs, ys = np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
        xg, yg = np.meshgrid(xs, ys)
        dot = normal[0] * (xg - mean[0]) + normal[1] * (yg - mean[1])
        return dot > 0

    # ── Debug drawing ─────────────────────────────────────────────────────────

    def _horizon_endpoints(self, mean, direction, w):
        """Returns (pt_left, pt_right) where the horizon line crosses x=0 and x=w-1."""
        if abs(direction[0]) > 1e-6:
            t_left  = (0       - mean[0]) / direction[0]
            t_right = (w - 1   - mean[0]) / direction[0]
        else:
            t_left = t_right = 0.0
        pt_left  = (0,     int(mean[1] + t_left  * direction[1]))
        pt_right = (w - 1, int(mean[1] + t_right * direction[1]))
        return pt_left, pt_right

    def _publish_debug(self, left_cv, horizon_raw, horizon_masked, source, left_header):
        debug_img = left_cv.copy()
        h, w = debug_img.shape[:2]

        # Yellow — raw Hough detection (where the horizon actually is)
        pt_l, pt_r = self._horizon_endpoints(horizon_raw[0], horizon_raw[1], w)
        cv2.line(debug_img, pt_l, pt_r, (0, 255, 255), 1, cv2.LINE_AA)

        # Red — nudged mask boundary (what actually gets zeroed out)
        pt_l_n, pt_r_n = self._horizon_endpoints(horizon_masked[0], horizon_masked[1], w)
        cv2.line(debug_img, pt_l_n, pt_r_n, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(debug_img, f'horizon [{source}]', (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        success, buf = cv2.imencode('.jpg', debug_img)
        if success:
            msg = CompressedImage()
            msg.header = left_header
            msg.format = 'jpeg'
            msg.data   = buf.tobytes()
            self._pub_debug.publish(msg)

    # ── Main callback ─────────────────────────────────────────────────────────

    def _cb_images(self, left_msg: Image, right_msg: Image):
        if not self.model_ready:
            return

        start_time = time.perf_counter()

        left_cv  = self._bridge.imgmsg_to_cv2(left_msg,  desired_encoding='bgr8')
        right_cv = self._bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
        h = left_cv.shape[0]

        # 1. Detect horizon
        horizon_raw = self._find_horizon_raw(left_cv)

        if horizon_raw is not None:
            horizon_masked = self._apply_margin(horizon_raw, h)
            self._last_horizon_raw    = horizon_raw
            self._last_horizon_masked = horizon_masked
            source = 'hough'
        elif self._last_horizon_masked is not None:
            horizon_raw    = self._last_horizon_raw
            horizon_masked = self._last_horizon_masked
            source = 'cached'
        else:
            # Cold-start fallback: horizontal line at fallback_crop_pct
            w   = left_cv.shape[1]
            row = int(h * self.fallback_crop_pct)
            horizon_raw    = (np.array([w / 2, row], dtype=np.float32),
                              np.array([1.0, 0.0],   dtype=np.float32))
            horizon_masked = self._apply_margin(horizon_raw, h)
            source = 'fallback'

        # 2. Build sky mask from the nudged horizon
        sky_mask = self._make_sky_mask(left_cv.shape, horizon_masked)

        # 3. Run disparity inference
        disparity_map = self.depth_estimator(left_cv, right_cv)

        # 4. Zero out sky
        disparity_map[sky_mask] = 0.0

        latency = (time.perf_counter() - start_time) * 1000

        # 5. Publish raw disparity
        disp_msg = self._bridge.cv2_to_imgmsg(disparity_map, encoding='32FC1')
        disp_msg.header    = left_msg.header
        disp_msg.step      = disparity_map.shape[1] * 4
        self._pub_disp_raw.publish(disp_msg)

        # 6. Publish colourised disparity
        color_disp = self.depth_estimator.draw_disparity()
        color_disp[sky_mask] = 0
        success, buf = cv2.imencode('.jpg', color_disp)
        if success:
            color_msg        = CompressedImage()
            color_msg.header = left_msg.header
            color_msg.format = 'jpeg'
            color_msg.data   = buf.tobytes()
            self._pub_disp_color.publish(color_msg)

        # 7. Publish debug overlay
        if self.get_parameter('debug_horizon').value:
            self._publish_debug(left_cv, horizon_raw, horizon_masked, source, left_msg.header)

        self.get_logger().info(
            f"Disparity | [{source}] "
            f"mean=({horizon_masked[0][0]:.0f},{horizon_masked[0][1]:.0f}) "
            f"dir=({horizon_masked[1][0]:.2f},{horizon_masked[1][1]:.2f}) | "
            f"{latency:.1f} ms"
        )


def main(args=None):
    rclpy.init(args=args)
    node = HitNetDisparityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()