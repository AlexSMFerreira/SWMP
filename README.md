# SWMP — Stereo Wave Monitoring Pipeline

A ROS 2 pipeline for real-time ocean surface reconstruction from an airship-mounted stereo camera pair. Raw compressed stereo images are rectified, fed through a HITNet neural stereo depth estimator, and projected into a coloured 3D point cloud — all running live over a Zenoh middleware transport.

```
[Stereo Bag / Live Camera]
        │
        ▼
  RectifyNode          ← undistorts + rectifies both cameras, emits CameraInfo
        │
        ▼
  HitNetDisparityNode  ← HITNet ONNX inference, sky masking, disparity map
        │
        ▼
  PointCloudNode       ← reprojects disparity → PointCloud2 (RGB + XYZ)
```

---

## Requirements

### System

- ROS 2 (Humble or later)
- Python 3.10+
- `rmw_zenoh_cpp` middleware
- `cv_bridge`, `message_filters` (standard ROS 2 packages)
- `rviz2`, `tf2_ros` (ship with ROS 2 — used by the pose broadcaster / visualisation)

The pose broadcaster and validation nodes use only standard ROS 2 packages
(`rclpy`, `tf2_ros`, `nav_msgs`, `geometry_msgs`, `sensor_msgs`) plus `numpy` — no extra
geodesy library is required; the LLA→ENU conversion is built in (WGS84 ECEF).

### Python

```
pip install -r requirements.txt
```

`requirements.txt` installs: `numpy`, `opencv-python`, `imread-from-url`, `onnx`, `onnxruntime`.

> For GPU inference install `onnxruntime-gpu` instead of `onnxruntime`.

---

## HITNet Models

Models are **not included** in the repository. Download the ONNX exports from the [onnx-hitnet-stereo-depth-estimation](https://github.com/ibaiGorordo/ONNX-HitNet-Stereo-Depth-estimation) repository and place them under `models/`.

The expected layout (matching the node defaults) is:

```
models/
  eth3d/
    saved_model_240x320/
      model_float32.onnx   ← default used by HitNetDisparityNode
    saved_model_480x640/
      model_float32.onnx
  middlebury_d400/
    saved_model_480x640/
      model_float32.onnx
  flyingthings_finalpass_xl/
    saved_model_480x640/
      model_float32.onnx
```

The `eth3d` model is recommended for outdoor/long-range scenes. `middlebury` and `flyingthings` models expect RGB input; `eth3d` operates on grayscale.

---

## Running the Pipeline

### 1. Start Zenoh middleware

```bash
ros2 run rmw_zenoh_cpp rmw_zenohd
```

### 2. Play a ROS 2 bag (or stream from live cameras)

```bash
ros2 bag play <path_to_bag> --loop
```

### 3. Start the rectifier node

```bash
ros2 run <your_package> ros2_stereo_rectifier
```

Subscribes to `/airship/camera/left/image_color/compressed` and `/airship/camera/right/image_color/compressed` plus their `camera_info` topics. Publishes rectified images and a packed `CameraInfo` (containing the Q matrix and P2) on `/stereo/`.

### 4. Start the HITNet disparity node

```bash
ros2 run <your_package> ros2_hitnet_disparity
```

Waits for `CameraInfo` to extract focal length and baseline, then runs HITNet inference on each synchronised stereo pair. Sky regions are detected via Hough line transform and zeroed out in the disparity map.

Key parameters (override with `--ros-args -p <param>:=<value>`):

| Parameter | Default | Description |
|---|---|---|
| `model_path` | `models/eth3d/saved_model_240x320/model_float32.onnx` | Path to ONNX model |
| `max_distance` | `200.0` | Depth clip (metres) |
| `sky_crop_pct` | `0.40` | Fallback sky crop if Hough fails |
| `horizon_margin_pct` | `0.03` | Margin below detected horizon |
| `debug_horizon` | `true` | Publish horizon overlay image |

### 5. Start the point cloud node

```bash
ros2 run <your_package> ros2_pointcloud_node
```

Reprojects the disparity map into a coloured `PointCloud2` using the Q matrix from the rectifier. Publishes on `/stereo/points`.

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `max_depth` | `200.0` | Maximum valid depth (metres) |
| `min_depth` | `0.1` | Minimum valid depth (metres) |
| `downsample_factor` | `3` | Point cloud thinning factor (1 = full density) |

### 6. Broadcast the live pose (TF tree)

The airship pose is driven from the `/nav` navigation solution so clouds land in a real
ENU world frame instead of stacking at the origin. Start the broadcaster and a static
lever-arm transform:

```bash
# Dynamic map → base_link from /nav (geodetic LLA → local ENU; orientation is body→ENU)
python3 Nodes/ros2_pose_broadcaster.py --ros-args -p use_sim_time:=true

# Static base_link → camera_left_rect (lever arm — identity placeholder until calibrated)
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link camera_left_rect
```

The TF tree is `map → base_link → camera_left_rect`. The broadcaster also republishes the
pose as `geometry_msgs/PoseStamped` on `/airship/pose_enu` (frame `map`) for direct
display. See `NEXT_STEPS_POSE.md` for the lever-arm calibration (Phase 3) and altimeter
validation (Phase 4) follow-ups.

### 7. Visualise position and orientation in RViz

```bash
ros2 run rviz2 rviz2 --ros-args -p use_sim_time:=true
```

Set the **Fixed Frame** to `map`, then add:

- **Pose** display on `/airship/pose_enu` — an arrow showing live position and **direction
  of travel** (heading derived from the trajectory; tune `heading_window_m`, or set
  `orientation_source:=nav` for true INS body attitude instead).
- **Path** display on `/airship/path_enu` — the accumulated ENU trajectory (track) of the
  airship.
- **TF** display — the `base_link` axis triad (red=forward/X, green=left/Y, blue=up/Z),
  confirming orientation in real time.
- **PointCloud2** display on `/stereo/points` — the cloud, transformed into `map`
  automatically through the TF tree.

### 8. (Optional) Validate altitude/attitude

```bash
python3 Nodes/ros2_pose_validation.py --ros-args -p use_sim_time:=true \
  -p altimeter_baseline_y:=<measured_metres>
```

Prints a rolling comparison of the nav altitude against the airship laser altimeters
(residual, nav roll vs altimeter-derived roll, lightware validity). See Phase 4 in
`NEXT_STEPS_POSE.md`.

Add a `PointCloud2` display pointing at `/stereo/points` and a `CompressedImage` display for `/stereo/disparity_color/compressed`.

---

## Published Topics

| Topic | Type | Description |
|---|---|---|
| `/stereo/left/image_rect` | `sensor_msgs/Image` | Rectified left image |
| `/stereo/right/image_rect` | `sensor_msgs/Image` | Rectified right image |
| `/stereo/camera_info_rect` | `sensor_msgs/CameraInfo` | Rectified camera info + Q matrix |
| `/stereo/disparity` | `sensor_msgs/Image` (32FC1) | Raw disparity map |
| `/stereo/disparity_color/compressed` | `sensor_msgs/CompressedImage` | Colourised disparity (JPEG) |
| `/stereo/debug/horizon/compressed` | `sensor_msgs/CompressedImage` | Horizon detection overlay |
| `/stereo/points` | `sensor_msgs/PointCloud2` | Coloured 3D point cloud |
