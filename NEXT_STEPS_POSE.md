# Next Steps — Airship Pose Estimation & Point Cloud Georeferencing

## Context

The current pipeline publishes `PointCloud2` in the `camera_left_rect` optical frame.
A **fixed identity** static transform `map → camera_left_rect` is broadcast manually
(`init_pipeline.txt`), so all clouds are stacked on top of each other as if the airship
never moves. This must be replaced with a live dynamic transform driven by the navigation
solution before any scientifically valid wave height reconstruction is possible.

---

## Phase 0 — Mandatory Topic Inspection (do before writing any code)

Play the bag and run every command below. Record the **exact message type** and a
**sample echo** for each topic. The node design in Phase 2 depends entirely on these
findings — do not skip or guess.

```bash
ros2 bag play <bag> --loop

# For each topic below, run both commands:
#   ros2 topic info <topic> --verbose
#   ros2 topic echo <topic> --once

ros2 topic info /nav --verbose
ros2 topic echo /nav --once

ros2 topic info /enu --verbose
ros2 topic echo /enu --once

ros2 topic info /lla --verbose
ros2 topic echo /lla --once

ros2 topic info /sbg/imu --verbose
ros2 topic echo /sbg/imu --once

ros2 topic info /umix/imu --verbose
ros2 topic echo /umix/imu --once

ros2 topic info /guidenav/imu_lsa --verbose
ros2 topic echo /guidenav/imu_lsa --once

ros2 topic info /vel_enu --verbose
ros2 topic echo /vel_enu --once

ros2 topic info /airship/left/altimeter/height --verbose
ros2 topic echo /airship/left/altimeter/height --once

ros2 topic info /airship/right/altimeter/height --verbose
ros2 topic echo /airship/right/altimeter/height --once

ros2 topic info /lightware_altimeter/left/altimeter --verbose
ros2 topic echo /lightware_altimeter/left/altimeter --once
```

### What to look for

| Topic | Expected type | Critical fields |
|---|---|---|
| `/nav` | `nav_msgs/Odometry` or custom SBG type | `pose.pose` (position + quaternion), `twist.twist` (velocity) |
| `/enu` | `geometry_msgs/PointStamped` or `PoseStamped` | XYZ position in ENU |
| `/lla` | `sensor_msgs/NavSatFix` | latitude, longitude, altitude |
| `/sbg/imu` | `sensor_msgs/Imu` or `sbg_driver/SbgImuData` | linear_acceleration, angular_velocity |
| `/umix/imu` | `sensor_msgs/Imu` | orientation quaternion (check if covariance is nonzero) |
| `/guidenav/imu_lsa` | `sensor_msgs/Imu` or custom | orientation, angular_velocity |
| `/vel_enu` | `geometry_msgs/TwistStamped` or `Vector3Stamped` | velocity in ENU |
| `/lightware_altimeter/left/altimeter` | `sensor_msgs/Range` | `range` (metres, AGL) |
| `/airship/left/altimeter/height` | `std_msgs/Float32` or `sensor_msgs/Range` | height (metres) |

**Key decision point:** if `/nav` already contains a full 6-DOF pose with orientation
quaternion (position + rotation), that is the primary source and the rest are validation
inputs. If it contains only position, orientation must come from `/umix/imu` or
`/guidenav/imu_lsa`.

> The presence of `/sbg/imu` strongly suggests an SBG Ellipse INS. SBG Ellipse series
> devices run an onboard EKF that fuses GPS + IMU and output a complete navigation
> solution — position, velocity, and orientation. The `/nav` topic is likely this output.
> Verify by checking if the orientation quaternion covariance is tight (< 1e-4 diagonal).

---

## Phase 1 — TF Tree Design

### Target tree

```
map  (ENU world frame, fixed origin)
 └── base_link  (airship body / IMU frame)
       └── camera_left_rect  (left camera optical frame, static)
             └── camera_right_rect  (known from stereo calibration R_stereo, T_stereo)
```

### Transforms required

| Transform | Type | Source |
|---|---|---|
| `map → base_link` | Dynamic | PoseBroadcasterNode (Phase 2) |
| `base_link → camera_left_rect` | Static | Lever arm calibration (Phase 3) |
| `camera_left_rect → camera_right_rect` | Static | Already embedded in stereo calibration |

The current `static_transform_publisher 0 0 0 0 0 0 map camera_left_rect` must be
**removed** once the dynamic broadcaster is running.

### Why ENU as the map frame?

ENU is a local tangent plane centered on the first GPS fix (or a survey datum). It is
metric (metres), right-handed, and aligned with gravity — which makes wave height
simply the Z component of any point after transformation. Using LLA or ECEF would
require non-linear projections at every point.

---

## Phase 2 — PoseBroadcasterNode

Once Phase 0 is complete and the message types are confirmed, implement a node
`ros2_pose_broadcaster.py` that:

1. Subscribes to the best available full-pose source (priority: `/nav` → `/umix/imu`
   combined with `/enu` → `/guidenav/imu_lsa` combined with `/lla`)
2. Converts the incoming position to ENU if it is not already (LLA→ENU via
   `pyproj` or manual haversine; record the datum from the first message)
3. Publishes a `geometry_msgs/TransformStamped` on `/tf` for `map → base_link`
4. Timestamps the transform from the navigation message header — **never use
   `rclpy.clock.now()`** for this, or sensor and pose will desync in bag playback

### Pseudocode skeleton (do not implement until Phase 0 is complete)

```python
# Only fill in after confirming message types from Phase 0

class PoseBroadcasterNode(Node):
    def __init__(self):
        self._tf_broadcaster = TransformBroadcaster(self)
        self._datum_enu = None          # (lat0, lon0, alt0) — set on first message
        self.create_subscription(<TYPE>, '/nav', self._cb_nav, qos)

    def _cb_nav(self, msg):
        # 1. Extract position → convert to ENU if needed
        # 2. Extract orientation quaternion
        # 3. Build TransformStamped: map → base_link
        # 4. self._tf_broadcaster.sendTransform(t)
```

### Scientific note on orientation representation

Use quaternions throughout — never convert to Euler angles internally. Euler angles
have gimbal lock and discontinuities at ±90° pitch, which will corrupt the transform
at steep descent/ascent angles. Only convert to Euler for logging/debugging.

---

## Phase 3 — Lever Arm Calibration

The GPS antenna and the left camera optical center are physically separated on the
airship. This offset (the *lever arm*) must be measured and encoded as a static
transform `base_link → camera_left_rect`.

**Why this matters quantitatively:** if the lever arm has a 0.5 m vertical offset and
the airship rolls 10°, the camera is displaced ~0.087 m horizontally relative to the
GPS fix. Over a sea surface at 30 m altitude this introduces a ~0.29° pointing error,
which maps to ~15 cm of apparent wave height error — comparable to the expected wave
amplitudes in sheltered coastal areas.

### To calibrate

1. Measure the physical XYZ offset (in body frame) from IMU/GPS antenna phase center
   to the left camera optical center.
2. Measure the angular offset (boresight alignment) between the camera Z-axis and the
   body frame X-axis (forward).
3. Encode as a static transform in a URDF or a static broadcaster launch file.

Until a formal calibration is done, estimate from physical measurements and set the
covariance accordingly. **Do not assume the lever arm is zero.**

---

## Phase 4 — Altitude Validation with Laser Altimeter

The Lightware laser altimeter (`/lightware_altimeter/left/altimeter`) measures slant
range to the surface directly below (or near-nadir). This is independent of GPS
vertical accuracy and provides a ground-truth check.

### Validation procedure

After the PoseBroadcaster is running:

1. Take the camera altitude from the TF tree: Z component of `map → camera_left_rect`
2. Compare against the Lightware range reading (corrected for camera tilt using
   the pitch/roll from the nav solution: `h = r · cos(pitch) · cos(roll)`)
3. Plot the residual time series. A bias indicates a lever arm vertical error or a
   GPS altitude offset. A correlated residual indicates the pose solution has attitude
   error.

The dual altimeters (`/airship/left/altimeter/height` and
`/airship/right/altimeter/height`) mounted at known positions on the airship can also
independently constrain roll angle: `roll_est ≈ atan2(h_right - h_left, baseline_y)`.
Compare this with the nav solution roll to cross-validate.

---

## Phase 5 — Georeferenced Point Cloud

Once the TF tree is live, the `PointCloudNode` needs a one-line update: the
`msg.header.frame_id` must be changed from `camera_left_rect` to `map` (or the cloud
must be transformed before publishing). RViz and downstream processors (wave spectrum
analysis, surface fitting) all expect world-frame coordinates.

**Do not transform every point manually in Python.** Use `pcl_ros` or the standard
`tf2_sensor_msgs` `do_transform_cloud()` utility which handles the full transform
chain lookup from the TF tree. This ensures the timestamp-matched transform is used
for each cloud.

### Scientific validation of the georeferenced cloud

Before using the cloud for wave analysis:

1. **Static test**: fly the airship stationary over a known flat surface. The cloud
   Z-spread should equal the combined stereo depth noise, not grow over time (which
   would indicate a drifting pose).
2. **Altimeter consistency**: mean Z of cloud points near nadir should match
   `camera_altitude - lightware_range`.
3. **Repeatability**: on a bag loop, clouds from the same position should overlap
   within the stated GPS accuracy.

---

## Summary of Immediate Actions

| # | Action | Blocker |
|---|---|---|
| 1 | Run all Phase 0 inspection commands while bag is playing | None — do this first |
| 2 | Record exact message types and sample fields for all nav topics | Phase 0 |
| 3 | Decide primary pose source based on Phase 0 findings | Phase 0 |
| 4 | Measure physical lever arm (GPS antenna → camera) | Physical access to airship |
| 5 | Implement PoseBroadcasterNode | Phase 0 + Phase 3 |
| 6 | Replace static TF in `init_pipeline.txt` with dynamic broadcaster | Phase 5 |
| 7 | Validate with altimeter residuals | Phase 5 |
| 8 | Update `PointCloudNode` frame_id and add TF transform | Phase 5 + Phase 3 |
