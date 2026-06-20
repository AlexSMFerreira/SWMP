#!/usr/bin/env python3
"""
PoseValidationNode — Phase 4 altitude/attitude cross-check for the pose solution.

Runs alongside the PoseBroadcaster and continuously compares the navigation solution
against the independent altimeters AND the calibrated nav's own water-level estimate,
printing a rolling summary. Use it to catch a lever arm vertical error, a GPS-altitude
datum offset, or an attitude bias before trusting the georeferenced cloud.

── Inputs (calibrated nav recording, see CLAUDE.md) ────────────────────────────────────
  /episea/nav/lla                   nav_msgs/Odometry           altitude in position.z
                                                                 (m, geodetic, WGS84)
  /episea/nav/euler                 geometry_msgs/Vector3Stamped  roll/pitch/yaw, DEGREES,
                                                                 NED convention — read
                                                                 directly, no quaternion
                                                                 decode needed
  /episea/nav/water_level           geometry_msgs/PointStamped  geodetic altitude (point.z)
                                                                 of the detected water
                                                                 surface below the vehicle
                                                                 — independent ground truth
  /airship/left/altimeter/height    geometry_msgs/PoseStamped   AGL height (metres) is
                                                                 position.x — see
                                                                 ros2_altimeter_publisher.py
                                                                 and CLAUDE.md. position.y is
                                                                 a magnitude/quality value, not
                                                                 height.
  /airship/right/altimeter/height   geometry_msgs/PoseStamped   same as left.
  /lightware_altimeter/left/altimeter  geometry_msgs/PointStamped  slant range in point.z (m);
                                                           -1.0 means "no return" (common over water)

── Checks ───────────────────────────────────────────────────────────────────────────
  1. Altitude residual: (tilt-corrected altimeter AGL) - (nav_alt - water_level_alt)
     should be ~CONSTANT (ideally ~0). Both sides are now the same physical quantity —
     local height above the water surface — so the residual isolates a lever-arm/sensor
     bias; it does NOT include the ~56 m WGS84/geoid offset (that cancels out via the
     water_level term, see CLAUDE.md). A drifting or attitude-correlated residual flags
     a pose problem.
  2. Roll cross-check: with two laterally-separated altimeters,
        roll_est = atan2(h_right - h_left, altimeter_baseline_y)
     compared against the roll reported by /episea/nav/euler.
  3. Water-level AGL: nav_alt - water_level_z is the nav filter's own independent
     estimate of local AGL height, used directly in check #1's reference term.

Both altimeters are noisy over water, so everything is reported as rolling mean ± std.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped


class PoseValidationNode(Node):
    def __init__(self):
        super().__init__('pose_validation_node')

        self.declare_parameter('nav_topic', '/episea/nav/lla')
        self.declare_parameter('euler_topic', '/episea/nav/euler')
        self.declare_parameter('water_level_topic', '/episea/nav/water_level')
        self.declare_parameter('alt_left_topic', '/airship/left/altimeter/height')
        self.declare_parameter('alt_right_topic', '/airship/right/altimeter/height')
        self.declare_parameter('lightware_topic', '/lightware_altimeter/left/altimeter')
        # Lateral spacing between the two airship altimeters (m). Measured from the rig CAD
        # (urdf_estrutura_ondas/urdf/estrutura_ondas.urdf.xacro): |y_alt_left - y_alt_right|
        # = |0.46953 - (-0.43967)|.
        self.declare_parameter('altimeter_baseline_y', 0.9092)
        self.declare_parameter('report_period_s', 5.0)

        p = self.get_parameter
        self._baseline_y = p('altimeter_baseline_y').value

        # Latest readings.
        self._nav_alt = None
        self._roll = self._pitch = self._yaw = None     # radians, from /episea/nav/euler
        self._water_level_alt = None                     # geodetic alt (m) of the water surface
        self._h_left = None
        self._h_right = None

        # Accumulators reset each report window.
        self._res = []          # tilt-corrected AGL (left) - nav_alt
        self._roll_nav = []     # nav roll (deg)
        self._roll_alt = []     # altimeter-derived roll (deg)
        self._wl_res = []       # nav_alt - water_level_alt (independent AGL estimate)
        self._lw_valid = 0
        self._lw_total = 0

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                         durability=DurabilityPolicy.VOLATILE, depth=10)
        self.create_subscription(Odometry, p('nav_topic').value, self._cb_nav, qos)
        self.create_subscription(Vector3Stamped, p('euler_topic').value, self._cb_euler, qos)
        self.create_subscription(PointStamped, p('water_level_topic').value, self._cb_water_level, qos)
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
        if self._water_level_alt is not None:
            self._wl_res.append(self._nav_alt - self._water_level_alt)

    def _cb_euler(self, msg: Vector3Stamped):
        # /episea/nav/euler is already roll/pitch/yaw in degrees, NED convention.
        self._roll = math.radians(msg.vector.x)
        self._pitch = math.radians(msg.vector.y)
        self._yaw = math.radians(msg.vector.z)

    def _cb_water_level(self, msg: PointStamped):
        self._water_level_alt = msg.point.z

    def _cb_left(self, msg: PoseStamped):
        self._h_left = msg.pose.position.x   # AGL in metres; position.y is a magnitude value
        self._accumulate()

    def _cb_right(self, msg: PoseStamped):
        self._h_right = msg.pose.position.x  # AGL in metres; position.y is a magnitude value
        self._accumulate()

    def _cb_lw(self, msg: PointStamped):
        self._lw_total += 1
        if msg.point.z > 0.0:
            self._lw_valid += 1

    def _accumulate(self):
        if self._nav_alt is None or self._h_left is None or self._roll is None:
            return
        if self._water_level_alt is None:
            return  # need nav_alt - water_level_alt as the comparable-scale AGL reference
        # Tilt-correct the AGL height to a true vertical drop.
        vert = self._h_left * math.cos(self._pitch) * math.cos(self._roll)
        # Compare against the nav filter's own local AGL (nav_alt - water_level_alt), NOT
        # raw nav_alt: the altimeters read a sub-metre height above the local water
        # surface (see CLAUDE.md "Altimeter publisher fixes"), while nav_alt is a full
        # ellipsoidal altitude (~56 m here, dominated by the WGS84/geoid offset) — the two
        # are not directly comparable.
        true_agl = self._nav_alt - self._water_level_alt
        self._res.append(vert - true_agl)
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
        if self._wl_res:
            wl = np.array(self._wl_res)
            line += f'  |  water-level AGL: {wl.mean():+7.2f} ± {wl.std():.2f} m'
        if self._lw_total:
            line += f'  |  lightware valid: {self._lw_valid}/{self._lw_total}'
        self.get_logger().info(line)
        # Reset window.
        self._res.clear(); self._roll_nav.clear(); self._roll_alt.clear(); self._wl_res.clear()
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
