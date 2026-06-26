import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from cv_bridge import CvBridge

import cv2
import numpy as np
import time

from stereo_common import rescale_disparity

class PointCloudNode(Node):
    def __init__(self):
        super().__init__('pointcloud_node')

        # ── PARAMETERS ────────────────────────────────────────────────────────
        self.declare_parameter('left_rect_topic',  '/stereo/left/image_rect')
        self.declare_parameter('disp_raw_topic',   '/stereo/disparity')
        self.declare_parameter('rect_info_topic',  '/stereo/camera_info_rect')
        self.declare_parameter('pointcloud_topic', '/stereo/points')

        self.declare_parameter('max_depth',        100.0)  # Beyond ~45 m a ±0.5 px error → ±3+ m depth uncertainty
        self.declare_parameter('min_depth',        0.0)  # Below ~2.45 m disparity saturates / hits baseline limit
        self.declare_parameter('downsample_factor', 3)    # 1 = Full, 2 = Half, 3 = Third (Boosts FPS)

        # The rectifier publishes left/right at native camera resolution (e.g.
        # 2464x2056). Most disparity backends rescale their output back up to that
        # same native resolution before publishing — but a backend may instead choose
        # to publish /stereo/disparity at its own smaller working resolution (see
        # ros2_waft_disparity.py, which does this to avoid the native-res upscale +
        # ~20 MB Image publish once its own inference is already running on a small
        # image). _cb_images below detects a left/disparity size mismatch and resizes
        # the left (colour) image down to match the disparity's actual resolution
        # before reprojecting — cheaper than resizing disparity values back up, and
        # the only correctness requirement (per stereo_common.rescale_disparity) is
        # that left and disparity have the same shape, not any particular resolution.
        #
        # input_width/input_height below downscale *further* on top of whatever
        # resolution was just settled on — only if that would shrink things further,
        # never to upscale back toward native — since reprojectImageTo3D + the
        # boolean mask/RGB indexing below cost roughly one unit of work per pixel
        # (full native res was observed to fall behind the bag's frame rate badly
        # enough to build up a memory backlog). Scales the Q matrix's pixel terms
        # (focal length/principal point) by the same ratio so reprojected XYZ stays in
        # correct metres; -1/-1 disables this extra step. downsample_factor still
        # applies afterward, on top, on the resulting point count.
        self.declare_parameter('input_width',  320)
        self.declare_parameter('input_height', 240)

        p = self.get_parameter

        self._bridge = CvBridge()
        self.Q = None
        self._native_wh = None   # (width, height) from CameraInfo, for scaling Q
        self.max_depth = p('max_depth').value
        self.min_depth = p('min_depth').value
        self.ds = p('downsample_factor').value

        # ── PUBLISHERS ────────────────────────────────────────────────────────
        # With Zenoh middleware, we can safely use standard RELIABLE QoS
        # without worrying about 5MB UDP fragmentation limits.
        pc_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
        self._pub_pc = self.create_publisher(PointCloud2, p('pointcloud_topic').value, pc_qos)

        # ── SUBSCRIBERS ───────────────────────────────────────────────────────
        sub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
        
        # Subscribe to the CameraInfo exactly once to extract the Q matrix
        self._sub_info = self.create_subscription(
            CameraInfo, p('rect_info_topic').value, self._cb_camera_info, sub_qos
        )
        self.get_logger().info("PointCloud Node waiting for Q-Matrix metadata...")

        self._sync_qos = sub_qos
        self._sub_left = self._sub_disp = self._sync = None
        self._last_msg_stamp: float | None = None
        self._last_frame_time: float = time.monotonic()
        self._build_sync()
        self.create_timer(3.0, self._cb_watchdog)

    def _build_sync(self):
        p = self.get_parameter
        if self._sub_left is not None:
            self.destroy_subscription(self._sub_left.sub)
        if self._sub_disp is not None:
            self.destroy_subscription(self._sub_disp.sub)
        self._sub_left = message_filters.Subscriber(self, Image, p('left_rect_topic').value,  qos_profile=self._sync_qos)
        self._sub_disp = message_filters.Subscriber(self, Image, p('disp_raw_topic').value,   qos_profile=self._sync_qos)
        self._sync = message_filters.TimeSynchronizer([self._sub_left, self._sub_disp], queue_size=10)
        self._sync.registerCallback(self._cb_images)

    def _reset_sync(self, reason: str):
        self.get_logger().warn(f'PointCloud: resetting sync — {reason}')
        self._build_sync()
        self._last_frame_time = time.monotonic()
        self._last_msg_stamp = None

    def _cb_watchdog(self):
        elapsed = time.monotonic() - self._last_frame_time
        if elapsed > 3.0:
            self._reset_sync(f'no frame for {elapsed:.1f}s')

    def _cb_camera_info(self, msg: CameraInfo):
        """Extracts the 4x4 Disparity-to-Depth Q Matrix packed by the Rectifier Node."""
        if self.Q is not None:
            return

        self.Q = np.array(msg.d, dtype=np.float32).reshape(4, 4)
        self._native_wh = (msg.width, msg.height)
        self.get_logger().info(f"Q-Matrix received ({msg.width}x{msg.height})! 3D Reprojection Engine Online.")
        
        # We only need this matrix once, so we disconnect the subscriber to save CPU
        #self.destroy_subscription(self._sub_info)

    def _cb_images(self, left_msg: Image, disp_msg: Image):
        if self.Q is None:
            return

        stamp_sec = left_msg.header.stamp.sec + left_msg.header.stamp.nanosec * 1e-9
        if self._last_msg_stamp is not None and stamp_sec < self._last_msg_stamp - 1.0:
            self._reset_sync(f'time jump {self._last_msg_stamp:.1f}→{stamp_sec:.1f}s')
            return
        self._last_msg_stamp = stamp_sec
        self._last_frame_time = time.monotonic()

        start_time = time.perf_counter()

        # 1. Decode ROS 2 Images into OpenCV NumPy arrays
        left_img = self._bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
        disparity = self._bridge.imgmsg_to_cv2(disp_msg, desired_encoding='32FC1')

        # 1b. left_img and disparity may already differ in resolution — a disparity
        # backend can publish at its own working resolution instead of native (see
        # ros2_waft_disparity.py) — so resize the left (colour) image to match
        # disparity's actual shape rather than assuming they're equal. Then optionally
        # downscale *further* (input_width/input_height) before the expensive
        # reprojection/masking step below (cost scales with pixel count) — only if
        # that's actually smaller than what we already have, never to upscale back
        # toward native. Scale Q's pixel-dependent terms (focal length, principal
        # point — row/col indices 0,1,2,3 in column 3) by the same ratio so the
        # reprojected X/Y/Z stay in correct metres; Q[3,2] (-1/baseline) is a physical
        # length, not a pixel quantity, and must NOT be scaled.
        left_wh = left_img.shape[1::-1]
        disp_wh = disparity.shape[1::-1]
        if disp_wh != left_wh:
            left_img = cv2.resize(left_img, disp_wh, interpolation=cv2.INTER_LINEAR)
        target_w, target_h = disp_wh

        inp_w = self.get_parameter('input_width').value
        inp_h = self.get_parameter('input_height').value
        if inp_w > 0 and inp_h > 0 and inp_w * inp_h < target_w * target_h:
            left_img = cv2.resize(left_img, (inp_w, inp_h), interpolation=cv2.INTER_LINEAR)
            disparity = rescale_disparity(disparity, (inp_w, inp_h))
            target_w, target_h = inp_w, inp_h

        native_w = self._native_wh[0]
        if target_w == native_w:
            Q = self.Q
        else:
            scale = target_w / native_w
            Q = self.Q.copy()
            Q[0, 3] *= scale
            Q[1, 3] *= scale
            Q[2, 3] *= scale
            Q[3, 3] *= scale

        # 2. Project into 3D Space (X, Y, Z in meters)
        points_3D = cv2.reprojectImageTo3D(disparity, Q)

        # 3. Filter limits based on physically valid bounds
        depth = points_3D[:, :, 2]
        mask = (disparity > 0.0) & (depth > self.min_depth) & (depth < self.max_depth)

        valid_points = points_3D[mask]
        valid_colors = left_img[mask]

        if len(valid_points) == 0:
            return

        # 4. Downsample array to maintain high FPS performance
        if self.ds > 1:
            valid_points = valid_points[::self.ds]
            valid_colors = valid_colors[::self.ds]

        # 5. Super-fast RGB byte packing for ROS 2 PointCloud2 standards
        b = valid_colors[:, 0].astype(np.uint32)
        g = valid_colors[:, 1].astype(np.uint32)
        rgb_r = valid_colors[:, 2].astype(np.uint32)
        rgba = (rgb_r << 16) | (g << 8) | b

        # 8. Construct structured array matching PointCloud2 memory layout (X, Y, Z, RGB)
        cloud_data = np.zeros(len(valid_points), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32)
        ])
        
        # Convert OpenCV Optical Frame to Standard ROS Frame
        cloud_data['x'] = valid_points[:, 2]   # Z (Depth) becomes X (Forward)
        cloud_data['y'] = -valid_points[:, 0]  # X (Right) becomes negative Y (Left)
        cloud_data['z'] = -valid_points[:, 1]  # Y (Down) becomes negative Z (Up)
        cloud_data['rgb'] = rgba

        # 9. Construct final ROS 2 PointCloud2 Message
        msg = PointCloud2()
        msg.header = left_msg.header
        msg.height = 1
        msg.width = len(valid_points)
        msg.is_dense = True
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.data = cloud_data.tobytes()

        # Define data structure layout mappings
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        # Publish the cloud
        self._pub_pc.publish(msg)

        latency = (time.perf_counter() - start_time) * 1000
        valid_depth = depth[mask]
        if self.ds > 1:
            valid_depth = valid_depth[::self.ds]
        self.get_logger().info(
            f"Published Cloud: {len(valid_points)} points | Compute: {latency:.1f}ms | "
            f"Depth min={valid_depth.min():.2f}m max={valid_depth.max():.2f}m mean={valid_depth.mean():.2f}m"
        )

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()