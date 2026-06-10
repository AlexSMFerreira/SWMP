# SWMP
## A Multi-Sensor Sea Wave Modelling and Prediction Framework
Part of the SEAWINGS/AIRSHIP projects, which develop aerial platforms for maritime operation. The goal is real-time perception, reconstruction and prediction of ocean wave surfaces from an airship, using stereo cameras and laser altimeters.

The full pipeline covers:
- **Stereo reconstruction** — rectification, HITNet neural disparity estimation, 3D point cloud generation
- **Pose & georeferencing** — GPS/INS-driven ENU frame so clouds accumulate in world space
- **Multisensor fusion** — combining stereo depth with altimeter measurements
- **Wave analysis** — extraction of wave parameters (amplitude, direction, frequency) and temporal/predictive modelling

Current implementation covers stereo reconstruction and georeferenced visualisation.

```
[Stereo bag / live cameras]
        │
        ▼
  RectifyNode          ← undistort + rectify, emit CameraInfo + Q matrix
        │
        ▼
  HitNetDisparityNode  ← HITNet ONNX inference, sky masking
        │
        ▼
  PointCloudNode       ← reproject disparity → PointCloud2 (RGB + XYZ)
        │
        ▼
  PoseBroadcasterNode  ← /nav LLA → ENU TF (map → base_link)
```

---

## Requirements

- ROS 2 Humble, Python 3.10+, `rmw_zenoh_cpp`, `cv_bridge`, `rviz2`, `tf2_ros`
- Python: `pip install -r requirements.txt` (numpy, opencv-python, onnx, onnxruntime)
- HITNet ONNX model at `models/eth3d/saved_model_240x320/model_float32.onnx` — download from [onnx-hitnet-stereo-depth-estimation](https://github.com/ibaiGorordo/ONNX-HitNet-Stereo-Depth-estimation)

---

## Running

```bash
./start_pipeline.sh [path_to_bag]
```

Launches everything (Zenoh, bag, pose broadcaster, static TFs, RViz, rectifier, HITNet, point cloud) in a `tmux` session named `swmp`. Default bag: `/media/alex/External/2026_LEIXOES_LOGS/airship_20260528_115149`.

```
tmux attach -t swmp          # attach
Ctrl-b w                     # pick window
Ctrl-b d                     # detach (keep running)
tmux kill-session -t swmp    # stop everything
```

In RViz set Fixed Frame to `map` and add: Pose `/airship/pose_enu`, Path `/airship/path_enu`, PointCloud2 `/stereo/points`, CompressedImage `/stereo/disparity_color/compressed`.

---

## Published Topics

| Topic | Type | Description |
|---|---|---|
| `/stereo/left/image_rect` | `sensor_msgs/Image` | Rectified left image |
| `/stereo/right/image_rect` | `sensor_msgs/Image` | Rectified right image |
| `/stereo/camera_info_rect` | `sensor_msgs/CameraInfo` | Rectified camera info + Q matrix |
| `/stereo/disparity` | `sensor_msgs/Image` (32FC1) | Raw disparity map |
| `/stereo/disparity_color/compressed` | `sensor_msgs/CompressedImage` | Colourised disparity |
| `/stereo/debug/horizon/compressed` | `sensor_msgs/CompressedImage` | Horizon detection overlay |
| `/stereo/points` | `sensor_msgs/PointCloud2` | Georeferenced coloured point cloud |
| `/airship/pose_enu` | `geometry_msgs/PoseStamped` | Live airship pose in ENU frame |
| `/airship/path_enu` | `nav_msgs/Path` | Accumulated ENU trajectory |
