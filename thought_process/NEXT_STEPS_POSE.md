# Next Steps — Airship Pose Estimation & Point Cloud Georeferencing

## Context

The current pipeline publishes `PointCloud2` in the `camera_left_rect` optical frame.
A **fixed identity** static transform `map → camera_left_rect` is broadcast manually
(`init_pipeline.txt`), so all clouds are stacked on top of each other as if the airship
never moves. This must be replaced with a live dynamic transform driven by the navigation
solution before any scientifically valid wave height reconstruction is possible.

---

## Phase 0 — COMPLETED (bag `2026_LEIXOES_LOGS/airship_20260528_115149`)

Topic inspection was run live against the playing bag. **Actual** message types and
sample fields are recorded below. Several differ from the original guesses — the design
in Phase 2 follows the *measured* types, not the table further down.

### Measured findings

| Topic | **Actual type** | Key fields (sample) |
|---|---|---|
| `/nav` | `nav_msgs/Odometry` | `header.frame_id = world_lla`; `pose.pose.position = (lat°, lon°, alt m)` **GEODETIC, not metric**; `pose.pose.orientation` = full quaternion (xyzw); ~10 Hz, RELIABLE |
| `/enu` | `lsa_sensor_msgs/Gnss` (custom) | `data = [East, North, Up]` metres, `frame_id = world_enu`, far-from-origin datum (~8.7 km E) |
| `/lla` | `lsa_sensor_msgs/Gnss` (custom) | `data = [lat°, lon°, alt m]` — matches `/nav` position |
| `/baseline` | `lsa_sensor_msgs/Gnss` (custom) | `data = [E, N, U]` dual-GNSS antenna baseline vector (m), RTK-fixed — used as the motion-independent heading reference for the orientation convention check |
| `/umix/imu` | `lsa_sensor_msgs/Imu` (custom) | linear_acceleration, angular_velocity, magnetic_field, temperature — **NO orientation quaternion** |
| `/sbg/imu` | `lsa_sensor_msgs/Imu` (custom) | same shape — no orientation |
| `/guidenav/imu_lsa` | `lsa_sensor_msgs/Imu` (custom) | same shape — no orientation |
| `/vel_enu` | `lsa_sensor_msgs/Gnss` | **0 messages in this bag** — unusable |
| `/lightware_altimeter/left/altimeter` | `geometry_msgs/PointStamped` | range in `point.z` (saw `-1.0` = no return), `frame_id = left_altimeter` |
| `/airship/left/altimeter/height` | `geometry_msgs/PoseStamped` | height in `position.y` (saw `72.1`), `frame_id = left_altimeter_link` |
| `/airship/right/altimeter/height` | `geometry_msgs/PoseStamped` | same shape as left |

### Decisions locked in

1. **Primary (and only) pose source is `/nav`.** It carries a full 6-DOF solution:
   geodetic position + orientation quaternion at ~10 Hz.
2. **The IMU topics cannot supply orientation** — they are raw accel/gyro/mag with no
   onboard attitude quaternion. The original fallback plan (use `/umix/imu` for
   orientation) is **not viable** without running our own fusion filter. `/nav` is the
   single attitude source.
3. **`/nav` position must be converted LLA → ENU.** It is geodetic degrees, not metres.
4. **`pyproj` and `tf_transformations` are NOT installed** on this machine. LLA→ENU is
   done manually via WGS84 ECEF→ENU (pure numpy). The orientation quaternion is passed
   through unchanged (no quaternion library needed).
5. ENU datum = the **first received `/nav` fix**, so the cloud sits near the origin
   (cleaner for RViz than the `/enu` topic's far-away survey datum).
6. **Orientation convention RESOLVED: `/nav`'s quaternion is body→NED (aerospace FRD).**
   It is converted to body→ENU (FLU) before publishing. *(A first body-Z-up check was
   inconclusive — body-Z reads +0.995 for BOTH ENU+FLU and NED+FRD, so it could not tell
   them apart.)* The decisive test used the dual-GNSS **`/baseline`** vector (an
   ENU heading reference that works even while hovering): the body-frame baseline is only
   **constant** under the **NED→ENU** transform — `(+1.33, 0, 0) m`, `|std| = 0.006` over
   800 samples (vs `|std| ≈ 0.32` for pass-through / ±90° / 180°). That `(+1.33, 0, 0)`
   means the two GNSS antennas sit 1.33 m apart along body-X (forward), confirming the
   forward axis. The full transform is

   ```
   q_ENU(FLU) = q_NED→ENU ⊗ q_nav ⊗ q_FRD→FLU
       q_NED→ENU = 180° about (1,1,0)/√2   (swap E/N, flip U)
       q_FRD→FLU = 180° about X            (flip Y/Z: down→up, right→left)
   ```

   **Final heading calibration (against actual travel):** aggregating chord directions
   (≥3 m displacement) over a full bag loop, the airship's motion is coherent and mostly
   **eastward** (path extent E 13.2 m vs N 2.4 m, coherence **R = 1.0** over 340 chords).
   The NED→ENU nose heading matched that travel direction to **−4.3° (median −5°)** — i.e.
   the frame conversion alone is essentially correct. So the optional heading trim
   **`yaw_offset_deg` defaults to 0**. Toggle the frame conversion with `nav_orientation:=enu`.

   *Dead end recorded:* the dual-GNSS `/baseline` made the body relationship constant
   (so it confirmed the NED→ENU **frame**), but it is NOT aligned with the nose, so it was
   the wrong reference for the heading magnitude — travel direction is the right one. A
   transient +90° `yaw_offset` guess (from eyeballing a locally-wiggly RViz path) was wrong
   and has been reverted. Instantaneous speed is low (max 0.76 m/s, a slow eastward drift),
   so judge alignment against the **net/overall** path direction, not local wiggles.

---

## Phase 0 (original instructions — kept for reference)

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

## Phase 2 — PoseBroadcasterNode ✅ IMPLEMENTED (`ros2_pose_broadcaster.py`)

Implemented and smoke-tested live against the bag. The node:

1. Subscribes to `/nav` only (the sole 6-DOF source — see Phase 0 decision 1/2),
   matching the offered QoS (RELIABLE / KEEP_LAST(10) / VOLATILE, ~10 Hz).
2. Converts the geodetic position (lat°, lon°, alt m) to a local **ENU** tangent plane
   via WGS84 ECEF→ENU (pure numpy, no `pyproj`). Datum = first received fix; a fixed
   datum can be forced with the `datum_lat/lon/alt` parameters.
3. Broadcasts `geometry_msgs/TransformStamped` on `/tf` for `map → base_link`, and also
   republishes a `geometry_msgs/PoseStamped` on `/airship/pose_enu` (frame `map`) so RViz
   can draw a position+orientation arrow (raw `/nav` can't be shown — its position is in
   LLA degrees), plus a `nav_msgs/Path` on `/airship/path_enu` accumulating the ENU
   trajectory (capped via the `path_max_poses` parameter).
4. Stamps both messages from `msg.header.stamp` (the nav header) — never the wall clock —
   so TF lookups stay aligned with the camera timestamps during bag playback.

**Orientation source (`orientation_source`, default `velocity`):** the published heading
points along the **direction of travel** ("head of the trajectory"), derived from the ENU
position track and smoothed over `heading_window_m` (default 4 m — the heading only re-aims
once the airship has translated that far, which rejects the ~2 m drift wiggle). This is what
the operator wanted in RViz: the arrow turns to follow the path over time. Set
`orientation_source:=nav` to publish the true INS body attitude instead (NED→ENU, optional
`yaw_offset_deg` trim) — required for georeferencing the camera cloud, which is rigidly
attached to the body, not the velocity vector. Verified: the published heading equals the
trajectory tangent (mean −0.4°, std 10° at a matching window). On bag `--loop` the heading
history is reset at the time-jump so it is not computed across the position discontinuity.

### Scientific note on orientation representation

The quaternion is carried through as-is — never converted to Euler internally (gimbal
lock / ±90° pitch discontinuities). Euler is used only for logging in the Phase 4
validation node.

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

## Phase 4 — Altitude/Attitude Validation ✅ IMPLEMENTED (`ros2_pose_validation.py`)

A diagnostic node compares the nav solution against the independent altimeters and prints
a rolling summary every 5 s. Run it alongside the broadcaster:

```bash
python3 ros2_pose_validation.py --ros-args -p use_sim_time:=true \
  -p altimeter_baseline_y:=<measured_metres>
```

It computes:

1. **Altitude residual** = (tilt-corrected airship-altimeter AGL) − nav altitude. Should
   be a ~constant offset (local sea surface below the GPS datum). Tilt correction uses
   `h · cos(pitch) · cos(roll)` from the nav quaternion.
2. **Roll cross-check** = `atan2(h_right − h_left, altimeter_baseline_y)` vs nav roll.
3. **Lightware validity** fraction (`point.z > 0`).

### Observed results on the bag (and caveats — these gate trusting the cloud)

- **Residual ≈ +10 m, std ≈ 12 m.** The +10 m mean is the expected GPS-datum/geoid
  offset (AGL > geodetic altitude here). The **large std is the airship altimeters being
  very noisy over water** (single readings swing 50→85 m), not a pose error.
- **nav roll ≈ 0°** — consistent with the level-platform finding in Phase 0.
- **Altimeter-derived roll is unusable** with the placeholder `baseline_y = 1 m` plus the
  altimeter noise → **the real lateral baseline must be measured (Phase 3)** before this
  check means anything.
- **Lightware returns valid only ~50 %** of the time over water (laser scatters off the
  surface) — treat it as opportunistic, not a continuous reference.

**Conclusion:** altitude residual is a stable bias (no drift, no attitude correlation) —
the pose solution is sound for visualisation. Tight quantitative validation needs the
measured altimeter baseline and ideally a flat-water / known-surface segment.

---

## Phase 5 — Georeferenced Point Cloud

> **The "view live position + orientation in RViz" objective is already met** once the
> broadcaster + static lever-arm transform are running: set the RViz Fixed Frame to `map`
> and add a **Pose** display on `/airship/pose_enu`, a **TF** display (the `base_link`
> triad), and a **PointCloud2** on `/stereo/points`. RViz looks up the TF chain
> `map → base_link → camera_left_rect` and transforms the cloud automatically — **no
> change to `PointCloudNode` is required for visualisation.**

The cloud node still publishes in `camera_left_rect`; that is correct and RViz handles the
rest. A frame change is only needed if **downstream processors that read the raw topic**
(wave spectrum analysis, surface fitting) want world-frame points without doing their own
TF lookup. In that case **do not transform points manually in Python** — use
`tf2_sensor_msgs.do_transform_cloud()` (or `pcl_ros`) with the timestamp-matched transform
from the TF tree. This is deferred until a downstream consumer actually needs it, and it
is blocked on the Phase 3 lever arm (an identity lever arm would georeference the cloud to
the wrong place).

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

## Status Summary

| # | Action | Status |
|---|---|---|
| 1 | Phase 0 topic inspection (types + sample fields) | ✅ Done — see Phase 0 table |
| 2 | Decide primary pose source | ✅ Done — `/nav` (only 6-DOF source) |
| 3 | Confirm orientation convention | ✅ Done — body→ENU, Z-up (measured) |
| 4 | Implement `ros2_pose_broadcaster.py` (`map → base_link` + `/airship/pose_enu`) | ✅ Done & live-tested |
| 5 | Replace static `map → camera_left_rect` with broadcaster + `base_link → camera_left_rect` in `init_pipeline.txt` | ✅ Done |
| 6 | **View live position + orientation in RViz** (Pose + TF + cloud, Fixed Frame = `map`) | ✅ **Objective met** |
| 7 | Implement altitude/attitude validation (`ros2_pose_validation.py`) | ✅ Done — residual is a stable bias |
| 8 | Measure physical lever arm (GPS antenna → camera) + altimeter baseline | ⛔ Needs physical access to airship |
| 9 | Encode measured lever arm in `base_link → camera_left_rect` (replace identity) | ⛔ Blocked on #8 |
| 10 | Georeference raw cloud topic via `do_transform_cloud()` | ⏸ Deferred until a downstream consumer needs it (blocked on #9) |

### Remaining work is physical / downstream only

Everything implementable from the data is done; the live pose (position **and**
orientation) is viewable in RViz now. The open items (#8–#10) require **physical
measurements on the airship** (lever arm, altimeter baseline) or a **downstream consumer**
of world-frame points — they cannot be completed from the bag alone.
