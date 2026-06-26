#!/usr/bin/env bash
# Starts the full SWMP pipeline in a tmux session.
# Usage:  ./start_pipeline.sh [bag_path]
# Default bag: /media/alex/External/2026_LEIXOES_LOGS/airship_20260528_115912

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAG="${1:-/media/alex/External/2026_LEIXOES_LOGS/airship_20260528_115912}"
SESSION="swmp"
# Bag playback rate (both bags MUST use the same value or they desync). Override with
# SWMP_RATE=0.5 to play slower — helps when the bag drive can't sustain real-time reads
# (rectifier "no frame" sync resets / point-cloud feed freezing under disk starvation).
RATE="${SWMP_RATE:-1.0}"

# Calibrated-nav companion bag (see CLAUDE.md): same timestamp, lives under ros2_nav/,
# suffixed _nav. ros2_pose_broadcaster.py / ros2_pose_validation.py now read /episea/nav/*
# from this bag, not from the camera bag's old thin /nav — both bags must play together.
NAV_BAG="$(dirname "$BAG")/ros2_nav/$(basename "$BAG")_nav"

ROS_SETUP="/opt/ros/humble/setup.bash"
PREAMBLE="source ${ROS_SETUP}; export RMW_IMPLEMENTATION=rmw_zenoh_cpp; cd '${SCRIPT_DIR}'"

if ! command -v tmux &>/dev/null; then
  echo "ERROR: tmux not found. Install with: sudo apt install tmux" >&2
  exit 1
fi

if [ ! -d "$BAG" ] && [ ! -f "$BAG" ]; then
  echo "ERROR: bag not found: $BAG" >&2
  exit 1
fi

HAVE_NAV_BAG=1
if [ ! -d "$NAV_BAG" ] && [ ! -f "$NAV_BAG" ]; then
  HAVE_NAV_BAG=0
  echo "WARNING: no calibrated-nav companion bag at $NAV_BAG" >&2
  echo "         (expected for camera bags 081709 / 114301 — see CLAUDE.md)." >&2
  echo "         ros2_pose_broadcaster.py / ros2_pose_validation.py will have no /episea/nav/* data." >&2
fi

# The nav bag always starts a few seconds after the camera bag but both stop within
# milliseconds of each other (true across every bag pair in this dataset — checked via
# `ros2 bag info`). Skipping that head start on the camera bag via --start-offset makes
# both bags the same duration, so looping them both stays phase-locked indefinitely
# instead of drifting apart (they'd otherwise loop on different periods and pair camera
# frames with nav data from the wrong point in time once out of phase). --start-offset
# is re-applied on every --loop iteration, not just the first — verified empirically by
# watching /nav header stamps snap back to the offset point (not the bag's true start)
# on each loop.
CAM_START_OFFSET="0"
if [ "$HAVE_NAV_BAG" -eq 1 ]; then
  OFFSET_SCRIPT="$(mktemp --suffix=.py)"
  trap 'rm -f "$OFFSET_SCRIPT"' EXIT
  cat > "$OFFSET_SCRIPT" <<'PYEOF'
import sys
import rosbag2_py
cam = rosbag2_py.Info().read_metadata(sys.argv[1], "sqlite3")
nav = rosbag2_py.Info().read_metadata(sys.argv[2], "sqlite3")
if nav.duration.nanoseconds / 1e9 < 1.0:
    print("NONE")  # degenerate companion bag (e.g. 111018_nav has 0 messages) -- treat as absent
else:
    offset = (nav.starting_time.nanoseconds - cam.starting_time.nanoseconds) / 1e9
    print(max(offset, 0.0))
PYEOF
  CAM_START_OFFSET="$(bash -c "${PREAMBLE}; python3 '${OFFSET_SCRIPT}' '${BAG}' '${NAV_BAG}'")"
  if [ "$CAM_START_OFFSET" = "NONE" ]; then
    echo "WARNING: nav companion bag at $NAV_BAG has ~zero duration (no real nav data) — treating as missing." >&2
    HAVE_NAV_BAG=0
    CAM_START_OFFSET="0"
  fi
fi

# Kill any existing session with the same name
tmux kill-session -t "$SESSION" 2>/dev/null && echo "Killed existing session '${SESSION}'" || true

echo "Starting pipeline with bag: $BAG"
[ "$HAVE_NAV_BAG" -eq 1 ] && echo "  + calibrated nav bag: $NAV_BAG (camera start-offset: ${CAM_START_OFFSET}s)"
echo ""

# ── 1. Zenoh daemon ──────────────────────────────────────────────────────────
tmux new-session -d -s "$SESSION" -n "zenoh" \
  "bash -c '${PREAMBLE}; echo \"[zenoh] Starting...\"; ros2 run rmw_zenoh_cpp rmw_zenohd; exec bash'"
echo " [1/14] zenoh daemon"
sleep 2

# ── 2. ROS bag (camera/sensor) ─────────────────────────────────────────────
# --start-offset skips the camera bag's head start relative to the nav bag (computed
# above) so both bags share the same loop period — see the comment above HAVE_NAV_BAG.
tmux new-window -t "$SESSION" -n "bag" \
  "bash -c '${PREAMBLE}; echo \"[bag] Playing ${BAG} --loop --clock --start-offset ${CAM_START_OFFSET}\"; ros2 bag play \"${BAG}\" --loop --rate ${RATE} --clock 200 --start-offset ${CAM_START_OFFSET} --read-ahead-queue-size 5000; exec bash'"
echo " [2/14] ros bag (camera/sensor)"
sleep 1

# ── 3. ROS bag (calibrated nav companion) ───────────────────────────────────
# The camera bag (above) is the sole /clock source. Two problems would otherwise give
# us two independent /clock streams: (1) passing --clock here would add a second
# synthetic clock publisher, and (2) the _nav bag was itself recorded WITH a /clock
# topic baked in (it has its own rosgraph_msgs/Clock messages on disk), so even without
# --clock the player replays those recorded /clock messages as ordinary data. Fix:
# don't pass --clock, AND remap the bag's own recorded /clock topic elsewhere so it
# can't reach the real /clock. Message timestamps (TF stamps etc.) come from each
# message's own header, not from /clock, so the nav bag doesn't need to publish it at
# all. (The two bags' *durations* are equalized separately via the camera bag's
# --start-offset above, so their --loop cycles also stay phase-locked.)
if [ "$HAVE_NAV_BAG" -eq 1 ]; then
  tmux new-window -t "$SESSION" -n "navbag" \
    "bash -c '${PREAMBLE}; echo \"[navbag] Playing ${NAV_BAG} --loop (no --clock, /clock remapped, see comment)\"; ros2 bag play \"${NAV_BAG}\" --loop --rate ${RATE} --remap /clock:=/_navbag_clock_unused; exec bash'"
  echo " [3/14] ros bag (calibrated nav)"
  sleep 1
else
  echo " [3/14] ros bag (calibrated nav) — SKIPPED, no companion bag found"
fi

# ── 4. Pose broadcaster (map -> base_link from /episea/nav/lla) ──────────────
tmux new-window -t "$SESSION" -n "pose" \
  "bash -c '${PREAMBLE}; echo \"[pose] Starting broadcaster...\"; python3 Nodes/ros2_pose_broadcaster.py --ros-args -p use_sim_time:=true; exec bash'"
echo " [4/14] pose broadcaster"
sleep 1

# ── 5. Static TF: base_link -> camera_left_rect ──────────────────────────────
# Real lever arm from the rig CAD (urdf_estrutura_ondas, link Left_cam): translation
# (0.044867, 0.51476, -0.0045012) m. Rotation stays IDENTITY: ros2_pointcloud_node.py
# already remaps the optical cloud into base-aligned FLU (X-fwd/Y-left/Z-up) internally,
# and the camera is mounted looking forward+level (Left_cam's URDF rpy is the canonical
# FLU->optical rotation, verified against the stereo calibration baseline to ~3 mm and to
# the node's remap within 0.05deg), so the post-remap cloud is already in base_link
# orientation -- only the translation is needed here. See WAVE_PARAMETERS_PLAN.md §3.5.
tmux new-window -t "$SESSION" -n "tf_cam" \
  "bash -c '${PREAMBLE}; echo \"[tf] base_link -> camera_left_rect\"; ros2 run tf2_ros static_transform_publisher 0.044867 0.51476 -0.0045012 0 0 0 base_link camera_left_rect; exec bash'"
echo " [5/14] static TF: base_link -> camera_left_rect"

# ── 6. Static TF: base_link -> rslidar ───────────────────────────────────────
tmux new-window -t "$SESSION" -n "tf_lidar" \
  "bash -c '${PREAMBLE}; echo \"[tf] base_link -> rslidar\"; ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link rslidar; exec bash'"
echo " [6/14] static TF: base_link -> rslidar"
sleep 1

# ── 7. Rig URDF visualization (robot_state_publisher + base_link -> estrutura_ondas) ──
# urdf_estrutura_ondas (see CLAUDE.md) is the physical rig CAD: real measured fixed-joint
# offsets for every sensor (cameras, lidar, altimeters, IMUs, GPS antennas) relative to a
# common rig origin ("estrutura_ondas", coincident with the umix IMU). It lives outside
# this repo (on the bag drive) and is built into the ~/ros2_ws overlay as a symlinked
# package — needs `source ~/ros2_ws/install/setup.bash` in addition to the usual ROS
# setup, hence the separate preamble here. The identity static TF below attaches the rig
# (and its meshes) to the live, nav-driven base_link so it moves with the airship in RViz
# instead of sitting at a disconnected origin — justified because umix sits at the rig's
# (0,0,0) and the camera baseline derived from the rig's Left_cam/right_cam offsets
# (~0.9998 m) matches the known ~1 m stereo calibration baseline, evidence the rig frame
# shares this project's body FLU convention (see CLAUDE.md "Altimeter unit
# re-verification & real lever-arm wiring").
tmux new-window -t "$SESSION" -n "rig_urdf" \
  "bash -c '${PREAMBLE}; source ~/ros2_ws/install/setup.bash; echo \"[rig_urdf] Starting robot_state_publisher...\"; ros2 launch urdf_estrutura_ondas estrutura_ondas.launch.py & echo \"[rig_urdf] static TF base_link -> estrutura_ondas\"; ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link estrutura_ondas; exec bash'"
echo " [7/14] rig URDF visualization (robot_state_publisher)"
sleep 1

# ── 8. RViz ──────────────────────────────────────────────────────────────────
# LD_PRELOAD forces the system libpthread over snap's copy (avoids GLIBC_PRIVATE error).
# ~/ros2_ws is sourced here too (not just for the rig_urdf step) because RViz is the
# process that actually resolves package://urdf_estrutura_ondas/meshes/... STL paths for
# the RobotModel display — without it those mesh lookups fail even though robot_state_
# publisher itself is running fine.
tmux new-window -t "$SESSION" -n "rviz" \
  "bash -c '${PREAMBLE}; source ~/ros2_ws/install/setup.bash; export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpthread.so.0:/usr/lib/x86_64-linux-gnu/libc.so.6; echo \"[rviz] Starting...\"; ros2 run rviz2 rviz2 --ros-args -p use_sim_time:=true; exec bash'"
echo " [8/14] rviz2"
sleep 2

# ── 9. Stereo rectifier ───────────────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "rectify" \
  "bash -c '${PREAMBLE}; echo \"[rectify] Starting...\"; python3 Nodes/ros2_stereo_rectifier.py --ros-args -p use_sim_time:=true; exec bash'"
echo " [9/14] stereo rectifier"
sleep 2

# ── 10. WAFT-Stereo disparity ──────────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "waft" \
  "bash -c '${PREAMBLE}; echo \"[waft] Starting...\"; python3 Nodes/ros2_waft_disparity.py --ros-args -p use_sim_time:=true -p config_file:=WAFT-Stereo/configs/SynLarge/DAv2S-4.yaml -p ckpt:=WAFT-Stereo/ckpts/SynLarge/DAv2S-4.pth; exec bash'"
echo " [10/14] WAFT-Stereo disparity"
sleep 2

# ── 11. Point cloud ────────────────────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "pointcloud" \
  "bash -c '${PREAMBLE}; echo \"[pointcloud] Starting...\"; python3 Nodes/ros2_pointcloud_node.py --ros-args -p use_sim_time:=true; exec bash'"
echo " [11/14] point cloud"

# ── 12. Altimeter publisher (sensor_msgs/Range + static TF frames) ────────────
tmux new-window -t "$SESSION" -n "altimeters" \
  "bash -c '${PREAMBLE}; echo \"[altimeters] Starting...\"; python3 Nodes/ros2_altimeter_publisher.py --ros-args -p use_sim_time:=true; exec bash'"
echo " [12/14] altimeter publisher"

# ── 13. Altimeter wave parameters (temporal spectrum -> /waves/altimeter) ──────
tmux new-window -t "$SESSION" -n "alt_waves" \
  "bash -c '${PREAMBLE}; echo \"[alt_waves] Starting...\"; python3 Nodes/ros2_altimeter_waves.py --ros-args -p use_sim_time:=true; exec bash'"
echo " [13/14] altimeter wave parameters"

# ── 14. Point-cloud wave parameters (plane fit + spatial spectrum -> /waves/pointcloud) ──
tmux new-window -t "$SESSION" -n "pc_waves" \
  "bash -c '${PREAMBLE}; echo \"[pc_waves] Starting...\"; python3 Nodes/ros2_pointcloud_waves.py --ros-args -p use_sim_time:=true; exec bash'"
echo " [14/14] point-cloud wave parameters"

# Select the bag window on attach (most useful to watch for clock ticking)
tmux select-window -t "$SESSION:bag"

echo ""
echo "Pipeline running in tmux session '${SESSION}'"
echo "  Next/prev window: Ctrl-b n / Ctrl-b p"
echo "  Pick window:      Ctrl-b w  (interactive list)"
echo "  Detach (keep running): Ctrl-b d"
echo "  Kill pipeline:    tmux kill-session -t ${SESSION}"
echo ""

tmux attach -t "$SESSION"
