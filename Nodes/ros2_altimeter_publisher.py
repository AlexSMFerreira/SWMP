#!/usr/bin/env python3
"""
AltimeterPublisherNode — converts raw altimeter messages to sensor_msgs/Range for RViz
and broadcasts static TF frames for each altimeter relative to base_link.

── Inputs (types confirmed from bag 2026_LEIXOES_LOGS/airship_20260528_115149) ──────────
  /airship/left/altimeter/height       geometry_msgs/PoseStamped   AGL height (metres) is
                                                                    position.x. position.y is
                                                                    a magnitude/quality value,
                                                                    not height (confirmed by
                                                                    José Carlos Fernandes,
                                                                    INESC TEC, and by position.x
                                                                    matching the Lightware
                                                                    unit's range almost exactly
                                                                    on the same bag — see
                                                                    CLAUDE.md). Supersedes the
                                                                    earlier position.y*0.01
                                                                    "centimetres" reading, which
                                                                    was reading the wrong field.
  /airship/right/altimeter/height      geometry_msgs/PoseStamped   same as left.
  /lightware_altimeter/left/altimeter  geometry_msgs/PointStamped  slant range in point.z (m);
                                                                    -1.0 means "no return"

── Outputs ──────────────────────────────────────────────────────────────────────────────
  /altimeter/left/range        sensor_msgs/Range   left downward laser  (frame: altimeter_left)
  /altimeter/right/range       sensor_msgs/Range   right downward laser (frame: altimeter_right)
  /altimeter/lightware/range   sensor_msgs/Range   Lightware slant range (frame: altimeter_lightware)

── TF frames (static, children of base_link, sensor +X axis pointing downward — that is
   the axis sensor_msgs/Range measures along, per REP 117) ──────────────────────────────
  Translations are real, measured offsets from the rig CAD
  (urdf_estrutura_ondas/urdf/estrutura_ondas.urdf.xacro, links alt_left/alt_right/alt_lidar
  — alt_lidar is the Lightware unit), not placeholders. The rotation (+90° about Y) is kept
  as separately verified via tf2_echo — the xacro's own rpy for these links is a mesh-display
  orientation, not the sensor's measurement axis.
  altimeter_left        (0.077987,  0.46953, -0.10252)
  altimeter_right       (0.077131, -0.43967, -0.10242)   baseline_y ≈ 0.9092 m measured
  altimeter_lightware   (0.099346, -0.20444, -0.076184)

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
    with the sensor's +X axis pointing in the -Z direction of the parent (downward for
    FLU base_link). sensor_msgs/Range measures along the frame's +X axis, not +Z (REP
    117) — so the range cone needs the *X* axis rotated down, not Z.
    Rotation: +90° about Y → quaternion (x=0, y=sin45°, z=0, w=cos45°)."""
    t = TransformStamped()
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.translation.z = z
    t.transform.rotation.x = 0.0
    t.transform.rotation.y = math.sin(math.radians(45.0))
    t.transform.rotation.z = 0.0
    t.transform.rotation.w = math.cos(math.radians(45.0))
    return t


class AltimeterPublisherNode(Node):
    def __init__(self):
        super().__init__('altimeter_publisher_node')

        # Real measured offsets (m) from base_link, taken from the physical rig CAD
        # (urdf_estrutura_ondas/urdf/estrutura_ondas.urdf.xacro, links alt_left/alt_right/
        # alt_lidar — alt_lidar is the Lightware unit). Translation only: the rig's mesh-
        # display rpy for these links does not represent the sensor's measurement axis, so
        # the rotation below (REP 117, +90° about Y) is kept as separately verified via
        # tf2_echo, not taken from the xacro.
        self.declare_parameter('altimeter_left_offset', [0.077987, 0.46953, -0.10252])
        self.declare_parameter('altimeter_right_offset', [0.077131, -0.43967, -0.10242])
        self.declare_parameter('lightware_offset', [0.099346, -0.20444, -0.076184])

        p = self.get_parameter
        left_x, left_y, left_z = p('altimeter_left_offset').value
        right_x, right_y, right_z = p('altimeter_right_offset').value
        lw_x, lw_y, lw_z = p('lightware_offset').value

        # Static TF: one broadcaster, three frames.
        self._static_tf = StaticTransformBroadcaster(self)
        self._static_tf.sendTransform([
            _downward_transform('base_link', 'altimeter_left', left_x, left_y, left_z),
            _downward_transform('base_link', 'altimeter_right', right_x, right_y, right_z),
            _downward_transform('base_link', 'altimeter_lightware', lw_x, lw_y, lw_z),
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
            f'baseline_y={left_y - right_y:.4f} m (measured, from urdf_estrutura_ondas). '
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
        r.range = float(msg.pose.position.x)   # AGL in metres; position.y is a magnitude value
        self._pub_left.publish(r)

    def _cb_right(self, msg: PoseStamped):
        r = self._base_range('altimeter_right', msg.header.stamp)
        r.range = float(msg.pose.position.x)   # AGL in metres; position.y is a magnitude value
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
