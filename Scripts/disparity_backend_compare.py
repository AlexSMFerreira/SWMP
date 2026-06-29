#!/usr/bin/env python3
"""
disparity_backend_compare.py — collect the cross-backend disparity comparison numbers
for the report (see Relatorio/Report/REPORT_PLAN.md, Architecture subsection).

Run once per disparity backend (with that backend's node + the rest of the pipeline,
including ros2_pointcloud_waves.py, up) over the SAME bag segment, passing --backend
<name> each time. Subscribes to:
  /stereo/disparity_quality/photo_error     std_msgs/Float32  (no-reference quality, see
                                             stereo_common.photometric_consistency_error)
  /stereo/disparity_quality/valid_fraction  std_msgs/Float32
  /stereo/disparity_quality/latency_ms      std_msgs/Float32
  /waves/pointcloud                         diagnostic_msgs/DiagnosticArray (for
                                             n_bad_dropped/n_frames — the per-frame
                                             max_frame_hs gate already in
                                             ros2_pointcloud_waves.py — plus Hs/Hmax as a
                                             sanity cross-check)

On exit (duration elapsed or Ctrl-C) writes:
  <out>/<backend>_quality_raw.csv     every per-frame sample collected
  <out>/disparity_backend_comparison.csv   one summary row APPENDED per run (creates
                                       the file with a header on the first run) — run
                                       this script once per backend to build up the
                                       full comparison table across all of them.

Usage:
  ./disparity_backend_compare.py --backend hitnet [--duration S] [--bag PATH] [--out DIR]
  (run with the pipeline up: source /opt/ros/humble/setup.bash;
   export RMW_IMPLEMENTATION=rmw_zenoh_cpp)
  Ctrl-C also writes whatever has been collected so far.
"""
import argparse
import csv
import os
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Float32
from diagnostic_msgs.msg import DiagnosticArray


def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


class Collector(Node):
    def __init__(self, duration):
        super().__init__('disparity_backend_compare')
        self.rows = []  # list of dicts {time_s, photo_err, valid_frac, latency_ms}
        self._t0 = None
        self._latest = {}
        self._pc_n_frames = np.nan
        self._pc_n_dropped = np.nan
        self._pc_hs = []
        self._pc_hmax = []

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST,
                         durability=DurabilityPolicy.VOLATILE, depth=10)
        self.create_subscription(Float32, '/stereo/disparity_quality/photo_error',
                                 lambda m: self._cb_metric('photo_err', m), qos)
        self.create_subscription(Float32, '/stereo/disparity_quality/valid_fraction',
                                 lambda m: self._cb_metric('valid_frac', m), qos)
        self.create_subscription(Float32, '/stereo/disparity_quality/latency_ms',
                                 lambda m: self._cb_metric('latency_ms', m), qos)

        pc_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                            durability=DurabilityPolicy.VOLATILE, depth=10)
        self.create_subscription(DiagnosticArray, '/waves/pointcloud', self._cb_pointcloud, pc_qos)

        self._stop = False
        if duration > 0:
            self.create_timer(duration, self._done)
        self.get_logger().info(f'Recording disparity-quality topics for {duration:.0f}s '
                               '(Ctrl-C to stop early) ...')

    # The three Float32 quality topics are published once per disparity frame but as
    # separate messages, not one combined message — latch the most recent value of each
    # and flush a row whenever latency_ms arrives (it's always published last in every
    # node's callback), so each row is one frame's three metrics together.
    def _cb_metric(self, key, msg: Float32):
        t = time.time()
        if self._t0 is None:
            self._t0 = t
        self._latest[key] = msg.data
        if key == 'latency_ms' and all(k in self._latest for k in ('photo_err', 'valid_frac', 'latency_ms')):
            self.rows.append({'time_s': t - self._t0, **self._latest})
            if len(self.rows) % 50 == 0:
                self.get_logger().info(f'  {len(self.rows)} frames collected', throttle_duration_sec=5.0)

    def _cb_pointcloud(self, msg: DiagnosticArray):
        if not msg.status:
            return
        kv = {v.key: _to_float(v.value) for v in msg.status[0].values}
        if 'n_frames' in kv:
            self._pc_n_frames = kv['n_frames']
        if 'n_bad_dropped' in kv:
            self._pc_n_dropped = kv['n_bad_dropped']
        if 'Hs' in kv and np.isfinite(kv['Hs']):
            self._pc_hs.append(kv['Hs'])
        if 'Hmax' in kv and np.isfinite(kv['Hmax']):
            self._pc_hmax.append(kv['Hmax'])

    def _done(self):
        self._stop = True


DEFAULT_BAG = '/media/alex/External/2026_LEIXOES_LOGS/airship_20260528_115912'


def _bag_duration_s(bag):
    try:
        import rosbag2_py
        return rosbag2_py.Info().read_metadata(bag, 'sqlite3').duration.nanoseconds / 1e9
    except Exception as e:
        print(f'(could not read bag duration: {e})')
        return None


def _pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float('nan')


def _med(v):
    return float(np.median(v)) if len(v) else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', required=True,
                    help='label for this run, e.g. hitnet/raftstereo/waft/sgbm/sbm/sgm_cuda')
    ap.add_argument('--duration', type=float, default=0.0,
                    help='seconds to record. 0 (default) = auto: one whole bag pass '
                         '(bag duration + warm-up margin, read from --bag). <0 = until Ctrl-C.')
    ap.add_argument('--bag', default=DEFAULT_BAG,
                    help='bag path, used only to auto-size the recording to one full pass')
    ap.add_argument('--margin', type=float, default=75.0,
                    help='extra seconds added to the bag duration to cover node warm-up')
    ap.add_argument('--out', default='disparity_backend_compare')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    duration = args.duration
    if duration == 0.0:
        bag_dur = _bag_duration_s(args.bag)
        if bag_dur:
            duration = bag_dur + args.margin
            print(f'auto duration = whole bag ({bag_dur:.0f}s) + {args.margin:.0f}s warm-up '
                  f'= {duration:.0f}s')
        else:
            duration = 420.0
            print(f'auto duration fallback = {duration:.0f}s')
    elif duration < 0.0:
        duration = 0.0

    rclpy.init()
    node = Collector(duration)
    try:
        while rclpy.ok() and not node._stop:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass

    rows = node.rows
    n_frames, n_dropped = node._pc_n_frames, node._pc_n_dropped
    pc_hs, pc_hmax = node._pc_hs, node._pc_hmax
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    raw_path = os.path.join(args.out, f'{args.backend}_quality_raw.csv')
    with open(raw_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_s', 'photo_err', 'valid_frac', 'latency_ms'])
        for r in rows:
            w.writerow([f"{r['time_s']:.3f}", f"{r['photo_err']:.4f}",
                       f"{r['valid_frac']:.4f}", f"{r['latency_ms']:.2f}"])
    print(f'wrote {raw_path} ({len(rows)} frames)')

    if not rows:
        print('No quality samples received — is the pipeline running with this '
              'backend up, and RMW_IMPLEMENTATION=rmw_zenoh_cpp set?')
        return

    photo_err = [r['photo_err'] for r in rows if np.isfinite(r['photo_err'])]
    valid_frac = [r['valid_frac'] for r in rows]
    latency_ms = [r['latency_ms'] for r in rows]
    bad_frame_rate = n_dropped / n_frames if (np.isfinite(n_dropped) and np.isfinite(n_frames) and n_frames > 0) else float('nan')

    summary = {
        'backend': args.backend,
        'n_frames': len(rows),
        'photo_err_median': _med(photo_err),
        'photo_err_p90': _pct(photo_err, 90),
        'valid_fraction_median': _med(valid_frac),
        'latency_ms_median': _med(latency_ms),
        'latency_ms_p90': _pct(latency_ms, 90),
        'pc_n_bad_dropped': n_dropped,
        'pc_n_frames': n_frames,
        'pc_bad_frame_rate': bad_frame_rate,
        'pc_hs_median': _med(pc_hs),
        'pc_hmax_median': _med(pc_hmax),
    }

    comparison_path = os.path.join(args.out, 'disparity_backend_comparison.csv')
    write_header = not os.path.exists(comparison_path)
    with open(comparison_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            w.writeheader()
        w.writerow(summary)
    print(f'appended summary row for "{args.backend}" to {comparison_path}')

    print('\nsummary for this run:')
    for k, v in summary.items():
        print(f'  {k}: {v:.4g}' if isinstance(v, float) else f'  {k}: {v}')


if __name__ == '__main__':
    main()
