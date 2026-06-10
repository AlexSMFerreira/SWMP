#!/usr/bin/env bash
# Starts the full SWMP pipeline in a tmux session.
# Usage:  ./start_pipeline.sh [bag_path]
# Default bag: /media/alex/External/2026_LEIXOES_LOGS/airship_20260528_115149

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAG="${1:-/media/alex/External/2026_LEIXOES_LOGS/airship_20260528_115149}"
SESSION="swmp"

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

# Kill any existing session with the same name
tmux kill-session -t "$SESSION" 2>/dev/null && echo "Killed existing session '${SESSION}'" || true

echo "Starting pipeline with bag: $BAG"
echo ""

# ── 1. Zenoh daemon ──────────────────────────────────────────────────────────
tmux new-session -d -s "$SESSION" -n "zenoh" \
  "bash -c '${PREAMBLE}; echo \"[zenoh] Starting...\"; ros2 run rmw_zenoh_cpp rmw_zenohd; exec bash'"
echo " [1/8] zenoh daemon"
sleep 2

# ── 2. ROS bag ───────────────────────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "bag" \
  "bash -c '${PREAMBLE}; echo \"[bag] Playing ${BAG} --loop\"; ros2 bag play \"${BAG}\" --loop --rate 1.0 --clock 200; exec bash'"
echo " [2/8] ros bag"
sleep 1

# ── 3. Pose broadcaster (map -> base_link from /nav) ─────────────────────────
tmux new-window -t "$SESSION" -n "pose" \
  "bash -c '${PREAMBLE}; echo \"[pose] Starting broadcaster...\"; python3 Nodes/ros2_pose_broadcaster.py --ros-args -p use_sim_time:=true; exec bash'"
echo " [3/8] pose broadcaster"
sleep 1

# ── 4. Static TF: base_link -> camera_left_rect ──────────────────────────────
tmux new-window -t "$SESSION" -n "tf_cam" \
  "bash -c '${PREAMBLE}; echo \"[tf] base_link -> camera_left_rect\"; ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link camera_left_rect; exec bash'"
echo " [4/8] static TF: base_link -> camera_left_rect"

# ── 5. Static TF: base_link -> rslidar ───────────────────────────────────────
tmux new-window -t "$SESSION" -n "tf_lidar" \
  "bash -c '${PREAMBLE}; echo \"[tf] base_link -> rslidar\"; ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link rslidar; exec bash'"
echo " [5/8] static TF: base_link -> rslidar"
sleep 1

# ── 6. RViz ──────────────────────────────────────────────────────────────────
# LD_PRELOAD forces the system libpthread over snap's copy (avoids GLIBC_PRIVATE error)
tmux new-window -t "$SESSION" -n "rviz" \
  "bash -c '${PREAMBLE}; export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpthread.so.0:/usr/lib/x86_64-linux-gnu/libc.so.6; echo \"[rviz] Starting...\"; ros2 run rviz2 rviz2 --ros-args -p use_sim_time:=true; exec bash'"
echo " [6/8] rviz2"
sleep 2

# ── 7. Stereo rectifier ───────────────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "rectify" \
  "bash -c '${PREAMBLE}; echo \"[rectify] Starting...\"; python3 Nodes/ros2_stereo_rectifier.py --ros-args -p use_sim_time:=true; exec bash'"
echo " [7/8] stereo rectifier"
sleep 2

# ── 8. HITNet disparity ───────────────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "hitnet" \
  "bash -c '${PREAMBLE}; echo \"[hitnet] Starting...\"; python3 Nodes/ros2_hitnet_disparity.py --ros-args -p use_sim_time:=true; exec bash'"
echo " [8/8] HITNet disparity"
sleep 2

# ── 9. Point cloud ────────────────────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "pointcloud" \
  "bash -c '${PREAMBLE}; echo \"[pointcloud] Starting...\"; python3 Nodes/ros2_pointcloud_node.py --ros-args -p use_sim_time:=true; exec bash'"
echo " [9/9] point cloud"

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
