#!/usr/bin/env python3
"""
AltimeterPublisherNode — converts raw altimeter messages to sensor_msgs/Range for RViz
and broadcasts static TF frames for each altimeter relative to base_link.

── Inputs (types confirmed from bag 2026_LEIXOES_LOGS/airship_20260528_115149) ──────────
  /airship/left/altimeter/height       geometry_msgs/PoseStamped   AGL height in position.y (m)
  /airship/right/altimeter/height      geometry_msgs/PoseStamped   AGL height in position.y (m)
  /lightware_altimeter/left/altimeter  geometry_msgs/PointStamped  slant range in point.z (m);
                                                                    -1.0 means "no return"

── Outputs ──────────────────────────────────────────────────────────────────────────────
  /altimeter/left/range        sensor_msgs/Range   left downward laser  (frame: altimeter_left)
  /altimeter/right/range       sensor_msgs/Range   right downward laser (frame: altimeter_right)
  /altimeter/lightware/range   sensor_msgs/Range   Lightware slant range (frame: altimeter_lightware)

── TF frames (static, children of base_link, +Z pointing downward) ─────────────────────
  altimeter_left        y = -altimeter_baseline_y / 2  (placeholder — measure physically)
  altimeter_right       y = +altimeter_baseline_y / 2  (placeholder — measure physically)
  altimeter_lightware   at origin                       (placeholder — measure physically)

In RViz: add a Range display, set the topic, set Fixed Frame to "map". The cone shows the
measured distance along the sensor's downward axis, moving with base_link via the TF tree.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, PointStamped, TransformStamped
from sensor_msgs.msg import Range
from tf2_ros import StaticTransformBroadcaster


def _downward_transform(parent: str, child: str, x: float, y: float, z: float) -> TransformStamped:
    """Return a TransformStamped that places `child` at (x,y,z) relative to `parent`
    with +Z pointing in the -Z direction of the parent (i.e. downward for FLU base_link).
    Rotation: 180° around X axis → quaternion (x=1, y=0, z=0, w=0)."""
    t = TransformStamped()
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.translation.z = z
    t.transform.rotation.x = 1.0
    t.transform.rotation.y = 0.0
    t.transform.rotation.z = 0.0
    t.transform.rotation.w = 0.0
    return t


class AltimeterPublisherNode(Node):
    def __init__(self):
        super().__init__('altimeter_publisher_node')

        # Lateral distance (m) between left and right altimeters — needs physical measurement.
        self.declare_parameter('altimeter_baseline_y', 1.0)
        # X offset of altimeters from base_link origin (forward positive).
        self.declare_parameter('altimeter_offset_x', 0.0)
        # Z offset of altimeters from base_link origin.
        self.declare_parameter('altimeter_offset_z', 0.0)
        # X/Y/Z offset for the Lightware unit.
        self.declare_parameter('lightware_offset_x', 0.0)
        self.declare_parameter('lightware_offset_y', 0.0)
        self.declare_parameter('lightware_offset_z', 0.0)

        p = self.get_parameter
        baseline_y = p('altimeter_baseline_y').value
        off_x = p('altimeter_offset_x').value
        off_z = p('altimeter_offset_z').value
        lw_x = p('lightware_offset_x').value
        lw_y = p('lightware_offset_y').value
        lw_z = p('lightware_offset_z').value

        # Static TF: one broadcaster, three frames.
        self._static_tf = StaticTransformBroadcaster(self)
        self._static_tf.sendTransform([
            _downward_transform('base_link', 'altimeter_left',
                                off_x, -baseline_y / 2.0, off_z),
            _downward_transform('base_link', 'altimeter_right',
                                off_x, +baseline_y / 2.0, off_z),
            _downward_transform('base_link', 'altimeter_lightware',
                                lw_x, lw_y, lw_z),
        ])

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=5,
        )

        self._pub_left = self.create_publisher(Range, '/altimeter/left/range', qos)
        self._pub_right = self.create_publisher(Range, '/altimeter/right/range', qos)
        self._pub_lw = self.create_publisher(Range, '/altimeter/lightware/range', qos)

        self.create_subscription(PoseStamped,
                                 '/airship/left/altimeter/height', self._cb_left, qos)
        self.create_subscription(PoseStamped,
                                 '/airship/right/altimeter/height', self._cb_right, qos)
        self.create_subscription(PointStamped,
                                 '/lightware_altimeter/left/altimeter', self._cb_lw, qos)

        self.get_logger().info(
            f'AltimeterPublisher ready. '
            f'baseline_y={baseline_y} m (placeholder — measure physically). '
            f'TF frames: altimeter_left, altimeter_right, altimeter_lightware → base_link.'
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _base_range(frame_id: str, stamp) -> Range:
        msg = Range()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.radiation_type = Range.INFRARED
        msg.field_of_view = 0.01          # ~0.6° half-angle — laser beam
        msg.min_range = 0.1
        msg.max_range = 300.0
        return msg

    # ── callbacks ────────────────────────────────────────────────────────────

    def _cb_left(self, msg: PoseStamped):
        r = self._base_range('altimeter_left', msg.header.stamp)
        r.range = float(msg.pose.position.y)
        self._pub_left.publish(r)

    def _cb_right(self, msg: PoseStamped):
        r = self._base_range('altimeter_right', msg.header.stamp)
        r.range = float(msg.pose.position.y)
        self._pub_right.publish(r)

    def _cb_lw(self, msg: PointStamped):
        r = self._base_range('altimeter_lightware', msg.header.stamp)
        raw = msg.point.z
        # -1.0 signals "no return" from the Lightware unit.
        r.range = float('inf') if raw < 0.0 else float(raw)
        self._pub_lw.publish(r)


def main(args=None):
    rclpy.init(args=args)
    node = AltimeterPublisherNode()
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
