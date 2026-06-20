#!/usr/bin/env python3
"""
PoseBroadcasterNode — live `map -> base_link` transform from the calibrated nav solution.

Replaces the fixed identity `static_transform_publisher map camera_left_rect` so that
point clouds are placed in a real ENU world frame instead of stacking on top of each
other as if the airship never moved.

── Nav source: /episea/nav/lla (see CLAUDE.md) ────────────────────────────────────────
  The old thin `/nav` topic (all-zero covariance, no velocity) has been replaced by the
  calibrated nav recording's `/episea/nav/lla` — same `nav_msgs/Odometry` field layout
  (so the LLA->ENU pipeline below is unchanged) but a real EKF/INS output with non-zero
  pose+twist covariance:
  /episea/nav/lla : nav_msgs/Odometry @ ~24 Hz, RELIABLE/VOLATILE
         header.frame_id        = 'WGS84'
         pose.pose.position      = (latitude_deg, longitude_deg, altitude_m)  ← GEODETIC
         pose.pose.orientation   = body->NED quaternion (aerospace FRD body), (x, y, z, w)
  The orientation convention (body->NED) was re-verified for this source by converting a
  same-instant /episea/nav/euler roll/pitch/yaw sample to a quaternion and matching it
  (within ~3°, consistent with the ~1.2 s sample gap) against /episea/nav/lla's
  orientation field — same convention as the original /nav, so the NED->ENU conversion
  below still applies unchanged. The IMU topics (/umix/imu, /sbg/imu, /guidenav/imu_lsa,
  /episea/nav/bias_imu) carry only accel/gyro/mag — NO orientation — so /episea/nav/lla
  remains the single attitude source.

Two conversions happen here:
  * Position: geodetic LLA -> local ENU tangent plane anchored at the first fix (datum),
    via pymap3d.geodetic2enu.
  * Orientation: body->NED (FRD) -> body->ENU (FLU), so ROS/RViz show correct heading.
    The NED convention was verified against the dual-GNSS /baseline vector: only the
    NED->ENU transform makes the antenna baseline constant in body frame (+1.33 m along
    body-X / forward) while keeping body-Z pointing up. See NEXT_STEPS_POSE.md Phase 0.

Wave height then falls straight out as the Z component of any transformed point.
Transform timestamps are copied from the nav message header — never the wall/clock — so
pose and sensor data stay aligned during bag playback.

── Orientation source ─────────────────────────────────────────────────────────────────
  orientation_source = 'velocity' (default): the published heading points along the
      DIRECTION OF TRAVEL ("head of the trajectory"), derived from the ENU position track
      and smoothed over `heading_window_m` metres so the slow/wiggly drift does not jitter.
      This is what an operator usually wants to see in RViz.
  orientation_source = 'nav': the true INS body attitude (NED->ENU). Use this when the
      transform must reflect the physical body pose — e.g. for georeferencing the camera
      point cloud, which is rigidly attached to the body, NOT to the velocity vector.
"""

import math
from collections import deque

import pymap3d
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_ros import TransformBroadcaster


def _stamp_from_sec(sec: float) -> Time:
    t = Time()
    t.sec = int(sec)
    t.nanosec = int(round((sec - int(sec)) * 1e9))
    return t

# ── NED->ENU frame rotation ────────────────────────────────────────────────────────────
# q_enu = q_NED_ENU * q_nav * q_FRD_FLU
# q_NED_ENU = 180° about (1,1,0)/sqrt2; q_FRD_FLU = 180° about X.
# Verified against the dual-GNSS /baseline vector (NEXT_STEPS_POSE.md Phase 0/4).
_R_NED_ENU = Rotation.from_quat([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])
_R_FRD_FLU = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])


def ned_quat_to_enu(x: float, y: float, z: float, w: float):
    """Convert a body->NED (FRD) quaternion to a body->ENU (FLU) quaternion."""
    q = _R_NED_ENU * Rotation.from_quat([x, y, z, w]) * _R_FRD_FLU
    return tuple(q.as_quat())  # (x, y, z, w)


def yaw_quat(deg: float):
    """Quaternion (x, y, z, w) for a rotation of `deg` degrees about world +Z (up)."""
    return tuple(Rotation.from_euler('z', deg, degrees=True).as_quat())


class PoseBroadcasterNode(Node):
    def __init__(self):
        super().__init__('pose_broadcaster_node')

        # ── PARAMETERS ──────────────────────────────────────────────────────────────
        self.declare_parameter('nav_topic', '/episea/nav/lla')
        self.declare_parameter('pose_topic', '/airship/pose_enu')
        self.declare_parameter('path_topic', '/airship/path_enu')
        # Max poses kept in the Path (ring buffer); 0 = unbounded. At ~10 Hz, 600 ≈ 1 min.
        self.declare_parameter('path_max_poses', 100)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        # Orientation source: 'velocity' (heading = direction of travel, default) or
        # 'nav' (true INS body attitude — needed for cloud georeferencing).
        self.declare_parameter('orientation_source', 'velocity')
        # Smoothing distance (m) for the velocity heading: the heading is taken from the
        # chord between the current position and the most recent past position at least
        # this far back. The heading only re-aims once the airship has actually translated
        # this far, so a larger value rejects the ~2 m drift wiggle (smoother) at the cost
        # of lag. Smaller = more responsive / jittier. Tune live in RViz to taste.
        self.declare_parameter('heading_window_m', 4.0)
        # Orientation reference frame of /nav (only used when orientation_source='nav'):
        # 'ned' (verified — applies NED->ENU conversion) or 'enu' (pass through).
        self.declare_parameter('nav_orientation', 'ned')
        # Constant heading trim about world-Z (deg), applied when orientation_source='nav'.
        self.declare_parameter('yaw_offset_deg', 0.0)
        # Optional fixed datum (lat, lon, alt). Leave NaN to auto-anchor on first fix.
        self.declare_parameter('datum_lat', float('nan'))
        self.declare_parameter('datum_lon', float('nan'))
        self.declare_parameter('datum_alt', float('nan'))

        p = self.get_parameter
        self._map_frame = p('map_frame').value
        self._base_frame = p('base_frame').value
        self._use_velocity = (p('orientation_source').value.lower() == 'velocity')
        self._heading_window = p('heading_window_m').value
        self._convert_ned = (p('nav_orientation').value.lower() == 'ned')
        self._yaw_offset_deg = p('yaw_offset_deg').value
        self._q_yaw = yaw_quat(self._yaw_offset_deg) if self._yaw_offset_deg else None

        # Recent ENU positions for the velocity-heading estimate, and the last good heading
        # quaternion (held while the airship is momentarily stationary).
        self._enu_hist = deque(maxlen=600)
        self._last_heading_q = None
        self._last_stamp = None             # raw nav header stamp (s), for jump detection
        self._last_published_stamp = None   # unwrapped/re-based stamp (s) actually published
        self._stamp_offset = 0.0            # cumulative re-base offset (s), see _cb_nav

        # ── ENU DATUM ───────────────────────────────────────────────────────────────
        # Stored as (lat_deg, lon_deg, alt_m); passed directly to pymap3d.geodetic2enu.
        self._datum = None
        d_lat = p('datum_lat').value
        d_lon = p('datum_lon').value
        d_alt = p('datum_alt').value
        if not (math.isnan(d_lat) or math.isnan(d_lon) or math.isnan(d_alt)):
            self._set_datum(d_lat, d_lon, d_alt)
            self.get_logger().info(
                f'ENU datum fixed from parameters: '
                f'lat={d_lat:.7f} lon={d_lon:.7f} alt={d_alt:.2f}'
            )

        # ── TF BROADCASTER ──────────────────────────────────────────────────────────
        self._tf_broadcaster = TransformBroadcaster(self)

        # ── POSE PUBLISHER ────────────────────────────────────────────────────────────
        # A PoseStamped in the map frame so RViz can draw a clean position+orientation
        # arrow (the raw /nav cannot be shown directly — its position is in LLA degrees).
        self._pub_pose = self.create_publisher(PoseStamped, p('pose_topic').value, 10)

        # ── PATH PUBLISHER ────────────────────────────────────────────────────────────
        # Accumulates the ENU poses into a nav_msgs/Path so RViz draws the airship track.
        self._path_max = p('path_max_poses').value
        self._path = Path()
        self._path.header.frame_id = self._map_frame
        self._pub_path = self.create_publisher(Path, p('path_topic').value, 10)

        # ── SUBSCRIBER ──────────────────────────────────────────────────────────────
        # /episea/nav/lla is offered RELIABLE / KEEP_LAST(10) / VOLATILE — match it exactly.
        nav_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.create_subscription(Odometry, p('nav_topic').value, self._cb_nav, nav_qos)

        src = ('velocity (direction of travel, '
               f'{self._heading_window:.1f} m window)' if self._use_velocity
               else 'nav (INS body attitude)')
        self.get_logger().info(
            f"PoseBroadcaster ready. Subscribing to {p('nav_topic').value}, "
            f"publishing {self._map_frame} -> {self._base_frame} on /tf. "
            f"orientation_source={src}."
        )
        self._logged_first = False

    # ── ENU helpers ───────────────────────────────────────────────────────────────────

    def _set_datum(self, lat_deg: float, lon_deg: float, alt_m: float):
        """Anchor the local ENU tangent plane at the given geodetic point."""
        self._datum = (lat_deg, lon_deg, alt_m)

    def _lla_to_enu(self, lat_deg: float, lon_deg: float, alt_m: float):
        """Geodetic -> local ENU metres relative to the datum."""
        lat0, lon0, alt0 = self._datum
        return pymap3d.geodetic2enu(lat_deg, lon_deg, alt_m, lat0, lon0, alt0)

    def _velocity_heading_quat(self, east, north, up):
        """Level quaternion (x,y,z,w) pointing along the recent direction of travel.

        Uses the chord from the most recent past position at least `heading_window_m`
        away to the current position. Returns the last good heading while stationary, or
        None until enough motion has accumulated.
        """
        self._enu_hist.append((east, north, up))
        # Walk back to the newest sample that is at least one window away.
        for e0, n0, _ in reversed(self._enu_hist):
            de, dn = east - e0, north - n0
            if de * de + dn * dn >= self._heading_window * self._heading_window:
                yaw = math.atan2(dn, de)
                h = yaw * 0.5
                self._last_heading_q = (0.0, 0.0, math.sin(h), math.cos(h))
                break
        return self._last_heading_q

    # ── Main callback ───────────────────────────────────────────────────────────────

    def _cb_nav(self, msg: Odometry):
        # Detect a backward time jump (bag --loop restart): the nav bag's own header
        # stamp resets to its start every loop (it isn't --start-offset-corrected like
        # the camera bag, see CLAUDE.md), so the raw stamp is non-monotonic across loops.
        # Publishing that raw, now-lower stamp on /tf would be permanently fatal for any
        # downstream listener: tf2's BufferCore rejects (TF_OLD_DATA) any transform for a
        # frame whose stamp is lower than the highest one it has already stored for that
        # frame, with no expiry — so after the first loop, map -> base_link (and anything
        # chained off it, e.g. point clouds and the rig meshes in RViz's "map" Fixed
        # Frame) would silently stop updating for the rest of the session. Fix: re-base
        # ("unwrap") published stamps with a cumulative offset so they keep increasing
        # seamlessly across every loop boundary, while still tracking the raw stamp 1:1
        # within a loop. Verified empirically: TF_OLD_DATA for base_link was absent on a
        # fresh start, then appeared continuously from the first loop wrap onward without
        # this fix.
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_stamp is not None and stamp_sec < self._last_stamp - 1.0:
            self._stamp_offset += (self._last_published_stamp - stamp_sec) + 1e-3
            self._enu_hist.clear()
        self._last_stamp = stamp_sec
        published_stamp_sec = stamp_sec + self._stamp_offset
        self._last_published_stamp = published_stamp_sec
        published_stamp = _stamp_from_sec(published_stamp_sec)

        lat = msg.pose.pose.position.x      # /episea/nav/lla packs geodetic lat in position.x
        lon = msg.pose.pose.position.y      # lon in position.y
        alt = msg.pose.pose.position.z      # altitude (m) in position.z

        # Anchor the ENU origin on the first valid fix.
        if self._datum is None:
            self._set_datum(lat, lon, alt)
            self.get_logger().info(
                f'ENU datum auto-anchored on first fix: '
                f'lat={lat:.7f} lon={lon:.7f} alt={alt:.2f}'
            )

        east, north, up = self._lla_to_enu(lat, lon, alt)

        # Convert the body->NED (FRD) attitude to body->ENU (FLU) for ROS/RViz. Verified
        # against the /baseline antenna vector (see NEXT_STEPS_POSE.md Phase 0/4); set
        # nav_orientation:=enu to disable if a future bag reports ENU directly.
        if self._use_velocity:
            # Heading = direction of travel ("head of the trajectory"), varies over time.
            q = self._velocity_heading_quat(east, north, up)
            if q is None:
                # Not enough motion yet — fall back to the INS attitude so the arrow is
                # not stuck flat at the origin until the first window of travel.
                o = msg.pose.pose.orientation
                q = ned_quat_to_enu(o.x, o.y, o.z, o.w)
            qx, qy, qz, qw = q
        else:
            # True INS body attitude (NED->ENU), with optional constant yaw trim.
            o = msg.pose.pose.orientation
            if self._convert_ned:
                qx, qy, qz, qw = ned_quat_to_enu(o.x, o.y, o.z, o.w)
            else:
                qx, qy, qz, qw = o.x, o.y, o.z, o.w
            if self._q_yaw is not None:
                qx, qy, qz, qw = (
                    Rotation.from_quat(self._q_yaw) * Rotation.from_quat([qx, qy, qz, qw])
                ).as_quat()

        t = TransformStamped()
        # Stamp from the (loop-unwrapped) nav header so TF lookups line up with sensor
        # timestamps and stay strictly increasing across bag --loop restarts — see the
        # jump-detection comment in _cb_nav above.
        t.header.stamp = published_stamp
        t.header.frame_id = self._map_frame
        t.child_frame_id = self._base_frame
        t.transform.translation.x = east
        t.transform.translation.y = north
        t.transform.translation.z = up
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self._tf_broadcaster.sendTransform(t)

        # Same pose as a PoseStamped in the map frame for direct RViz display.
        pose = PoseStamped()
        pose.header.stamp = published_stamp
        pose.header.frame_id = self._map_frame
        pose.pose.position.x = east
        pose.pose.position.y = north
        pose.pose.position.z = up
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        self._pub_pose.publish(pose)

        # Append to the trajectory and republish the Path (ENU track in the map frame).
        self._path.poses.append(pose)
        if self._path_max > 0 and len(self._path.poses) > self._path_max:
            self._path.poses = self._path.poses[-self._path_max:]
        self._path.header.stamp = published_stamp
        self._pub_path.publish(self._path)

        #if not self._logged_first:
        #    self._logged_first = True
        self.get_logger().info(
            f'First transform published: ENU=({east:.2f}, {north:.2f}, {up:.2f}) m'
        )


def main(args=None):
    rclpy.init(args=args)
    node = PoseBroadcasterNode()
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
