#!/usr/bin/env python3
"""
PoseValidationNode — Phase 4 altitude/attitude cross-check for the pose solution.

Runs alongside the PoseBroadcaster and continuously compares the navigation solution
against the independent altimeters, printing a rolling summary. Use it to catch a lever
arm vertical error, a GPS-altitude datum offset, or an attitude bias before trusting the
georeferenced cloud.

── Inputs (types confirmed from bag 2026_LEIXOES_LOGS/airship_20260528_115149) ──────────
  /nav                              nav_msgs/Odometry      altitude in position.z (m, geodetic),
                                                           orientation quaternion (body->ENU)
  /airship/left/altimeter/height    geometry_msgs/PoseStamped   AGL height in position.y (m)
  /airship/right/altimeter/height   geometry_msgs/PoseStamped   AGL height in position.y (m)
  /lightware_altimeter/left/altimeter  geometry_msgs/PointStamped  slant range in point.z (m);
                                                           -1.0 means "no return" (common over water)

── Checks ───────────────────────────────────────────────────────────────────────────
  1. Altitude residual: (tilt-corrected AGL) - nav_alt should be ~CONSTANT. The constant
     is the local sea-surface height below the GPS datum (geoid + tide). A drifting or
     attitude-correlated residual flags a pose problem.
  2. Roll cross-check: with two laterally-separated altimeters,
        roll_est = atan2(h_right - h_left, altimeter_baseline_y)
     compared against the roll extracted from the nav quaternion.

Both altimeters are noisy over water, so everything is reported as rolling mean ± std.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PointStamped


def quat_to_roll_pitch_yaw(x, y, z, w):
    """Quaternion -> (roll, pitch, yaw) in radians (ZYX), for diagnostics only."""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class PoseValidationNode(Node):
    def __init__(self):
        super().__init__('pose_validation_node')

        self.declare_parameter('nav_topic', '/nav')
        self.declare_parameter('alt_left_topic', '/airship/left/altimeter/height')
        self.declare_parameter('alt_right_topic', '/airship/right/altimeter/height')
        self.declare_parameter('lightware_topic', '/lightware_altimeter/left/altimeter')
        # Lateral spacing between the two airship altimeters (m). MEASURE THIS — placeholder.
        self.declare_parameter('altimeter_baseline_y', 1.0)
        self.declare_parameter('report_period_s', 5.0)

        p = self.get_parameter
        self._baseline_y = p('altimeter_baseline_y').value

        # Latest readings.
        self._nav_alt = None
        self._roll = self._pitch = self._yaw = None
        self._h_left = None
        self._h_right = None

        # Accumulators reset each report window.
        self._res = []          # tilt-corrected AGL (left) - nav_alt
        self._roll_nav = []     # nav roll (deg)
        self._roll_alt = []     # altimeter-derived roll (deg)
        self._lw_valid = 0
        self._lw_total = 0

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                         durability=DurabilityPolicy.VOLATILE, depth=10)
        self.create_subscription(Odometry, p('nav_topic').value, self._cb_nav, qos)
        self.create_subscription(PoseStamped, p('alt_left_topic').value, self._cb_left, qos)
        self.create_subscription(PoseStamped, p('alt_right_topic').value, self._cb_right, qos)
        self.create_subscription(PointStamped, p('lightware_topic').value, self._cb_lw, qos)

        self.create_timer(p('report_period_s').value, self._report)
        self.get_logger().info(
            f'PoseValidation ready. baseline_y={self._baseline_y} m. '
            f'Reporting every {p("report_period_s").value}s.'
        )

    def _cb_nav(self, msg: Odometry):
        self._nav_alt = msg.pose.pose.position.z
        o = msg.pose.pose.orientation
        self._roll, self._pitch, self._yaw = quat_to_roll_pitch_yaw(o.x, o.y, o.z, o.w)

    def _cb_left(self, msg: PoseStamped):
        self._h_left = msg.pose.position.y
        self._accumulate()

    def _cb_right(self, msg: PoseStamped):
        self._h_right = msg.pose.position.y
        self._accumulate()

    def _cb_lw(self, msg: PointStamped):
        self._lw_total += 1
        if msg.point.z > 0.0:
            self._lw_valid += 1

    def _accumulate(self):
        if self._nav_alt is None or self._h_left is None:
            return
        # Tilt-correct the AGL height to a true vertical drop.
        vert = self._h_left * math.cos(self._pitch) * math.cos(self._roll)
        self._res.append(vert - self._nav_alt)
        self._roll_nav.append(math.degrees(self._roll))
        if self._h_right is not None and self._baseline_y > 1e-6:
            roll_est = math.atan2(self._h_right - self._h_left, self._baseline_y)
            self._roll_alt.append(math.degrees(roll_est))

    def _report(self):
        if not self._res:
            self.get_logger().info('… waiting for synchronized nav + altimeter data …')
            return
        res = np.array(self._res)
        rn = np.array(self._roll_nav)
        line = (f'[{len(res):3d} samples]  '
                f'AGL-navAlt residual: {res.mean():+7.2f} ± {res.std():.2f} m  |  '
                f'nav roll: {rn.mean():+5.1f}° ± {rn.std():.1f}')
        if self._roll_alt:
            ra = np.array(self._roll_alt)
            line += f'  |  altimeter roll: {ra.mean():+5.1f}° ± {ra.std():.1f}'
        if self._lw_total:
            line += f'  |  lightware valid: {self._lw_valid}/{self._lw_total}'
        self.get_logger().info(line)
        # Reset window.
        self._res.clear(); self._roll_nav.clear(); self._roll_alt.clear()
        self._lw_valid = self._lw_total = 0


def main(args=None):
    rclpy.init(args=args)
    node = PoseValidationNode()
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
