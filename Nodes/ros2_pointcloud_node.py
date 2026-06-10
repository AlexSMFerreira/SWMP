import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Float64
from cv_bridge import CvBridge

import cv2
import numpy as np
import time

class PointCloudNode(Node):
    def __init__(self):
        super().__init__('pointcloud_node')

        # ── PARAMETERS ────────────────────────────────────────────────────────
        self.declare_parameter('left_rect_topic',  '/stereo/left/image_rect')
        self.declare_parameter('disp_raw_topic',   '/stereo/disparity')
        self.declare_parameter('rect_info_topic',  '/stereo/camera_info_rect')
        self.declare_parameter('pointcloud_topic', '/stereo/points')
        
        self.declare_parameter('max_depth',        200.0) # Cutoff distance (removes sky/noise)
        self.declare_parameter('min_depth',        0.1)   # Minimum valid distance
        self.declare_parameter('downsample_factor', 3)    # 1 = Full, 2 = Half, 3 = Third (Boosts FPS)

        p = self.get_parameter

        self._bridge = CvBridge()
        self.Q = None
        self.max_depth = p('max_depth').value
        self.min_depth = p('min_depth').value
        self.ds = p('downsample_factor').value
        self._roll_rad: float = 0.0

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
        self.create_subscription(Float64, '/stereo/horizon_roll', self._cb_roll, sub_qos)
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

    def _cb_roll(self, msg: Float64):
        self._roll_rad = msg.data

    def _cb_camera_info(self, msg: CameraInfo):
        """Extracts the 4x4 Disparity-to-Depth Q Matrix packed by the Rectifier Node."""
        if self.Q is not None:
            return

        self.Q = np.array(msg.d, dtype=np.float32).reshape(4, 4)
        self.get_logger().info("Q-Matrix received! 3D Reprojection Engine Online.")
        
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

        # 2. Project natively into 3D Space (X, Y, Z in meters)
        points_3D = cv2.reprojectImageTo3D(disparity, self.Q)

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

        # 5. Cancel drone roll: rotate around camera Z (forward) by -roll_rad.
        # valid_points columns are (X_cv=right, Y_cv=down, Z_cv=forward).
        r = self._roll_rad
        if abs(r) > 1e-4:
            c, s = np.cos(r), np.sin(r)
            x_cv = valid_points[:, 0].copy()
            y_cv = valid_points[:, 1].copy()
            valid_points[:, 0] = c * x_cv + s * y_cv
            valid_points[:, 1] = -s * x_cv + c * y_cv

        # 6. Super-fast RGB byte packing for ROS 2 PointCloud2 standards
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
        self.get_logger().info(f"Published Cloud: {len(valid_points)} points | Compute: {latency:.1f}ms")

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