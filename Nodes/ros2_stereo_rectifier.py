import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import message_filters
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from cv_bridge import CvBridge


def camera_info_to_K_D(info: CameraInfo):
    """Dynamically extracts raw camera intrinsics and distortion from the bag."""
    K = np.array(info.k, dtype=np.float64).reshape(3, 3)
    D = np.array(info.d, dtype=np.float64)
    return K, D


def build_rect_info(P1, P2, Q, wh) -> CameraInfo:
    """Constructs a standard-compliant ROS 2 CameraInfo message."""
    msg = CameraInfo()
    msg.header.frame_id = 'camera_left_rect'
    msg.width  = wh[0]
    msg.height = wh[1]
    msg.k = P1[:3, :3].flatten().tolist()
    msg.p = P1.flatten().tolist()
    msg.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
    msg.d = Q.flatten().tolist()                        # Q matrix for PointCloud downstream
    msg.distortion_model = ';'.join(f'{v:.10f}' for v in P2.flatten())  # Stash P2
    return msg


class RectifyNode(Node):
    def __init__(self):
        super().__init__('rectify_node')

        # ── PARAMETERS ────────────────────────────────────────────────────────
        self.declare_parameter('left_image_topic',  '/airship/camera/left/image_color/compressed')
        self.declare_parameter('right_image_topic', '/airship/camera/right/image_color/compressed')
        self.declare_parameter('left_info_topic',   '/airship/camera/left/camera_info')
        self.declare_parameter('right_info_topic',  '/airship/camera/right/camera_info')
        self.declare_parameter('left_rect_topic',   '/stereo/left/image_rect')
        self.declare_parameter('right_rect_topic',  '/stereo/right/image_rect')
        self.declare_parameter('rect_info_topic',   '/stereo/camera_info_rect')
        self.declare_parameter('sync_slop',     0.05)
        # Watchdog: if no frame arrives within this many seconds, reset the sync
        self.declare_parameter('watchdog_timeout', 3.0)
        # Rectified output resolution = native camera resolution * output_scale (e.g.
        # 0.25 = quarter). This is the *only* downscale baked into the pipeline
        # upstream of every disparity backend — each backend still independently
        # controls its own further working resolution on top of whatever this
        # publishes (see ros2_waft_disparity.py's input_width/input_height/
        # scale_factor), and every consumer reads the actual resolution from
        # CameraInfo/image dimensions rather than assuming a fixed size, so this is
        # safe to change without touching any other node. 1.0 reproduces the old
        # native-resolution-always behaviour. Done via stereoRectify's newImageSize +
        # matching initUndistortRectifyMap size, so undistort+rectify+downscale
        # happens in a single remap pass (no separate full-res-then-resize step).
        self.declare_parameter('output_scale', 0.25)

        p = self.get_parameter
        self._native_wh  = None   # raw camera (K/D) resolution, set once CameraInfo arrives
        self._output_wh  = None   # actual published resolution = native * output_scale
        self._sync_slop  = p('sync_slop').value
        self._watchdog_t = p('watchdog_timeout').value

        self._bridge = CvBridge()
        self._maps_ok = False
        self._map_lx = self._map_ly = None
        self._map_rx = self._map_ry = None
        self._rect_info_msg: CameraInfo | None = None

        # Timestamp of the last successfully processed stereo pair
        self._last_frame_time: float = time.monotonic()
        # Timestamp of the last incoming message (either camera), used to detect time jumps
        self._last_msg_stamp: float | None = None

        # ── STEREO EXTRINSICS ─────────────────────────────────────────────────
        self.R_stereo = np.array([
            [ 0.99998433,  0.00309469,  0.00466459],
            [-0.00307396,  0.9999854,  -0.0044443 ],
            [-0.00467827,  0.00442989,  0.99997924],
        ])
        self.T_stereo = np.array([-1.00029476e+00, -1.10997479e-04, 8.05032395e-03])

        # ── QoS ───────────────────────────────────────────────────────────────
        self._pub_qos  = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,   history=HistoryPolicy.KEEP_LAST, depth=10)
        self._info_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)

        # ── PUBLISHERS ────────────────────────────────────────────────────────
        self._pub_left  = self.create_publisher(Image,      p('left_rect_topic').value,  self._pub_qos)
        self._pub_right = self.create_publisher(Image,      p('right_rect_topic').value, self._pub_qos)
        self._pub_info  = self.create_publisher(CameraInfo, p('rect_info_topic').value,  self._pub_qos)

        # ── CAMERA INFO SUBSCRIBERS (one-shot) ────────────────────────────────
        self._info_left  = None
        self._info_right = None
        self._sub_info_l = self.create_subscription(
            CameraInfo, p('left_info_topic').value,  self._cb_info_left,  self._info_qos)
        self._sub_info_r = self.create_subscription(
            CameraInfo, p('right_info_topic').value, self._cb_info_right, self._info_qos)

        # ── IMAGE SYNC (created via helper so it can be rebuilt) ──────────────
        self._sub_left_filter  = None
        self._sub_right_filter = None
        self._sync = None
        self._build_sync()

        # ── WATCHDOG TIMER ────────────────────────────────────────────────────
        self._watchdog = self.create_timer(self._watchdog_t, self._cb_watchdog)

        self.get_logger().info(
            f'Rectify Node ready. output_scale={p("output_scale").value}. '
            f'Watchdog: {self._watchdog_t}s. Waiting for CameraInfo...'
        )

    # ── Sync management ───────────────────────────────────────────────────────

    def _build_sync(self):
        """Creates (or recreates) the ApproximateTimeSynchronizer and its subscribers."""
        p = self.get_parameter

        # Destroy old subscribers if they exist (releases queue state)
        if self._sub_left_filter is not None:
            self.destroy_subscription(self._sub_left_filter.sub)
        if self._sub_right_filter is not None:
            self.destroy_subscription(self._sub_right_filter.sub)

        self._sub_left_filter  = message_filters.Subscriber(
            self, CompressedImage, p('left_image_topic').value,  qos_profile=self._info_qos)
        self._sub_right_filter = message_filters.Subscriber(
            self, CompressedImage, p('right_image_topic').value, qos_profile=self._info_qos)

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._sub_left_filter, self._sub_right_filter],
            queue_size=10,
            slop=self._sync_slop,
        )
        self._sync.registerCallback(self._cb_images)

    def _reset_sync(self, reason: str):
        """Tears down and recreates the synchronizer to flush stale queue state."""
        self.get_logger().warn(f'Resetting sync: {reason}')
        self._build_sync()
        self._last_frame_time = time.monotonic()
        self._last_msg_stamp  = None

    # ── Watchdog ──────────────────────────────────────────────────────────────

    def _cb_watchdog(self):
        """Fires every watchdog_timeout seconds. Resets sync if no frame arrived."""
        now = time.monotonic()
        elapsed = now - self._last_frame_time
        if elapsed > self._watchdog_t:
            self._reset_sync(f'no frame for {elapsed:.1f}s')

    # ── Camera info (one-shot) ────────────────────────────────────────────────

    def _cb_info_left(self, msg: CameraInfo):
        if self._info_left is None:
            self._info_left = msg
            self._try_build_maps()

    def _cb_info_right(self, msg: CameraInfo):
        if self._info_right is None:
            self._info_right = msg
            self._try_build_maps()

    def _try_build_maps(self):
        if self._maps_ok or self._info_left is None or self._info_right is None:
            return

        il, ir = self._info_left, self._info_right
        W_raw, H_raw = il.width, il.height
        self._native_wh = (W_raw, H_raw)

        scale = self.get_parameter('output_scale').value
        W_out = max(1, int(round(W_raw * scale)))
        H_out = max(1, int(round(H_raw * scale)))
        self._output_wh = (W_out, H_out)

        self.get_logger().info(
            f'Intrinsics: {W_raw}x{H_raw} native, rectifying to {W_out}x{H_out} '
            f'(output_scale={scale})'
        )

        K_l, D_l = camera_info_to_K_D(il)
        K_r, D_r = camera_info_to_K_D(ir)

        # newImageSize=(W_out, H_out): P1/P2/Q come out scaled for the smaller output
        # directly, and initUndistortRectifyMap below is built with that same size, so
        # undistort+rectify+downscale happens in one remap pass per image instead of
        # rectifying at native res and resizing separately afterward.
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K_l, D_l, K_r, D_r,
            (W_raw, H_raw), self.R_stereo, self.T_stereo,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
            newImageSize=(W_out, H_out),
        )

        self._map_lx, self._map_ly = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, (W_out, H_out), cv2.CV_32FC1)
        self._map_rx, self._map_ry = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, (W_out, H_out), cv2.CV_32FC1)

        self._rect_info_msg = build_rect_info(P1, P2, Q, (W_out, H_out))
        self._maps_ok = True

        self.get_logger().info(f'Rectification maps ready. P1_fx={P1[0,0]:.2f}')

        self.destroy_subscription(self._sub_info_l)
        self.destroy_subscription(self._sub_info_r)

    # ── Main image callback ───────────────────────────────────────────────────

    def _cb_images(self, left_msg: CompressedImage, right_msg: CompressedImage):
        if not self._maps_ok:
            return

        stamp_sec = left_msg.header.stamp.sec + left_msg.header.stamp.nanosec * 1e-9

        # Detect backward time jump (bag loop) and reset sync to flush stale queue
        if self._last_msg_stamp is not None and stamp_sec < self._last_msg_stamp - 1.0:
            self._reset_sync(f'time jump {self._last_msg_stamp:.1f}→{stamp_sec:.1f}s')
            return  # Drop this frame; next callback will use the fresh sync
        self._last_msg_stamp  = stamp_sec
        self._last_frame_time = time.monotonic()

        stamp = left_msg.header.stamp

        left_raw  = self._bridge.compressed_imgmsg_to_cv2(left_msg,  desired_encoding='bgr8')
        right_raw = self._bridge.compressed_imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

        left_rect  = cv2.remap(left_raw,  self._map_lx, self._map_ly, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_raw, self._map_rx, self._map_ry, cv2.INTER_LINEAR)

        for img, pub in ((left_rect, self._pub_left), (right_rect, self._pub_right)):
            ros_img = self._bridge.cv2_to_imgmsg(img, encoding='bgr8')
            ros_img.header.stamp    = stamp
            ros_img.header.frame_id = 'camera_left_rect'
            pub.publish(ros_img)

        self._rect_info_msg.header.stamp = stamp
        self._pub_info.publish(self._rect_info_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RectifyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()