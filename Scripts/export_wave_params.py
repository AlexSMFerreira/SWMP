#!/usr/bin/env python3
"""
export_wave_params.py — record the wave parameters published by the two estimator nodes
(ros2_altimeter_waves.py -> /waves/altimeter, ros2_pointcloud_waves.py -> /waves/pointcloud)
to CSV and a comparison plot.

Subscribes to the two diagnostic_msgs/DiagnosticArray topics while the live pipeline runs
(needs RMW_IMPLEMENTATION=rmw_zenoh_cpp), collects every key/value parameter as a time
series, and on exit writes:
  <out>/waves_altimeter.csv
  <out>/waves_pointcloud.csv
  <out>/waves_params.png   (Hs, Hmax, period, wavelength — altimeter vs point cloud)

By default it records ONE WHOLE BAG PASS: the duration is auto-sized to the bag length
(from --bag metadata) plus a warm-up margin. Override with --duration, or pass a negative
--duration to record until Ctrl-C.

Usage:
  ./export_wave_params.py [--duration S] [--bag PATH] [--out DIR]
  (run with the pipeline up:  source /opt/ros/humble/setup.bash; export RMW_IMPLEMENTATION=rmw_zenoh_cpp)
  Ctrl-C also writes whatever has been collected so far.
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from diagnostic_msgs.msg import DiagnosticArray


def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


class Collector(Node):
    def __init__(self, duration):
        super().__init__('wave_param_exporter')
        # rows[topic] = list of dicts {time_s, <key>: value, ...}
        self.rows = {'altimeter': [], 'pointcloud': []}
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                         durability=DurabilityPolicy.VOLATILE, depth=10)
        self.create_subscription(DiagnosticArray, '/waves/altimeter',
                                 lambda m: self._cb(m, 'altimeter'), qos)
        self.create_subscription(DiagnosticArray, '/waves/pointcloud',
                                 lambda m: self._cb(m, 'pointcloud'), qos)
        self._t0 = None
        if duration > 0:
            self.create_timer(duration, self._done)
        self.get_logger().info(f'Recording /waves/altimeter + /waves/pointcloud '
                               f'for {duration:.0f}s (Ctrl-C to stop early) ...')
        self._stop = False

    def _cb(self, msg, which):
        if not msg.status:
            return
        # Wall-clock receive time, NOT msg.header.stamp: the nodes stamp with sim time,
        # which jumps backwards every bag --loop and produced negative/garbled time axes.
        t = time.time()
        if self._t0 is None:
            self._t0 = t
        row = {'time_s': t - self._t0}
        for kv in msg.status[0].values:
            row[kv.key] = _to_float(kv.value)
        self.rows[which].append(row)
        n_a, n_p = len(self.rows['altimeter']), len(self.rows['pointcloud'])
        self.get_logger().info(f'  altimeter={n_a}  pointcloud={n_p} samples', throttle_duration_sec=5.0)

    def _done(self):
        self._stop = True


def _df(rows):
    """list of dicts -> dict of np arrays (union of keys), sorted by time."""
    if not rows:
        return {}
    keys = set()
    for r in rows:
        keys.update(r.keys())
    rows = sorted(rows, key=lambda r: r['time_s'])
    return {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys}


def _write_csv(path, d):
    if not d:
        return 0
    cols = ['time_s'] + sorted(k for k in d if k != 'time_s')
    n = len(d['time_s'])
    with open(path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for i in range(n):
            f.write(','.join(f'{d[c][i]:.5g}' for c in cols) + '\n')
    return n


def _get(d, k):
    return d.get(k, np.array([]))


DEFAULT_BAG = '/media/alex/External/2026_LEIXOES_LOGS/airship_20260528_115912'


def _bag_duration_s(bag):
    """Bag content duration (s) from rosbag2 metadata, or None."""
    try:
        import rosbag2_py
        return rosbag2_py.Info().read_metadata(bag, 'sqlite3').duration.nanoseconds / 1e9
    except Exception as e:
        print(f'(could not read bag duration: {e})')
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=float, default=0.0,
                    help='seconds to record. 0 (default) = auto: cover one whole bag pass '
                         '(bag duration + warm-up margin, read from --bag). <0 = until Ctrl-C.')
    ap.add_argument('--bag', default=DEFAULT_BAG,
                    help='bag path, used only to auto-size the recording to one full pass')
    ap.add_argument('--margin', type=float, default=75.0,
                    help='extra seconds added to the bag duration to cover node warm-up')
    ap.add_argument('--out', default='wave_params_export')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    duration = args.duration
    if duration == 0.0:  # auto: whole bag + warm-up
        bag_dur = _bag_duration_s(args.bag)
        if bag_dur:
            duration = bag_dur + args.margin
            print(f'auto duration = whole bag ({bag_dur:.0f}s) + {args.margin:.0f}s warm-up '
                  f'= {duration:.0f}s  (note: bag may play slower than real-time if the '
                  f'player reports "queue starved")')
        else:
            duration = 420.0
            print(f'auto duration fallback = {duration:.0f}s')
    elif duration < 0.0:
        duration = 0.0  # Collector treats 0 as "until Ctrl-C"

    rclpy.init()
    node = Collector(duration)
    try:
        while rclpy.ok() and not node._stop:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass

    alt = _df(node.rows['altimeter'])
    pc = _df(node.rows['pointcloud'])
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    na = _write_csv(os.path.join(args.out, 'waves_altimeter.csv'), alt)
    npc = _write_csv(os.path.join(args.out, 'waves_pointcloud.csv'), pc)
    print(f'wrote waves_altimeter.csv ({na} rows), waves_pointcloud.csv ({npc} rows) in {args.out}/')
    if na == 0 and npc == 0:
        print('No data received — is the pipeline running and RMW_IMPLEMENTATION=rmw_zenoh_cpp set?')
        return

    # Stale-feed detection: a frozen producer republishes identical values. Check BOTH:
    #  (a) global: almost no unique values over the whole run;
    #  (b) mid-run freeze: the tail is a long identical run (the producer stalled partway —
    #      e.g. the cloud feed froze at ~130s while the first part was fine), which (a) misses.
    stale = []
    for name, d in (('altimeter', alt), ('pointcloud', pc)):
        hs = _get(d, 'Hs')
        t = _get(d, 'time_s')
        ok = np.isfinite(hs)
        hs, t = (hs[ok], t[ok]) if hs.size else (hs, t)
        if hs.size < 4:
            continue
        if len(np.unique(np.round(hs, 4))) <= max(1, int(0.1 * hs.size)):
            stale.append(name)
            print(f'WARNING: {name} feed looks STALE over the whole run — '
                  f'{len(np.unique(np.round(hs,4)))} unique Hs / {hs.size} samples.')
            continue
        # Mid-run freeze: count the trailing run of identical Hs.
        last = hs[-1]
        frozen = 0
        for v in hs[::-1]:
            if abs(v - last) < 1e-4:
                frozen += 1
            else:
                break
        if frozen >= max(3, int(0.2 * hs.size)):
            stale.append(name)
            t_freeze = t[len(hs) - frozen]
            print(f'WARNING: {name} feed FROZE mid-run at t≈{t_freeze:.0f}s — last {frozen} '
                  f'samples identical (producer stalled; data after ~{t_freeze:.0f}s not trustworthy).')

    # ── plot: 4 panels, altimeter vs point cloud ──
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    A, P = 'tab:blue', 'tab:orange'

    # Hs
    if alt: ax[0, 0].plot(_get(alt, 'time_s'), _get(alt, 'Hs'), '.-', color=A, label='altimeter Hs')
    if 'Hs_zerocross' in alt:
        ax[0, 0].plot(_get(alt, 'time_s'), _get(alt, 'Hs_zerocross'), ':', color=A, alpha=0.6, label='altimeter Hs (zero-cross)')
    if pc: ax[0, 0].plot(_get(pc, 'time_s'), _get(pc, 'Hs'), '.-', color=P, label='point cloud Hs')
    ax[0, 0].set_ylabel('Hs (m)'); ax[0, 0].set_title('Significant wave height'); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

    # Hmax
    if alt: ax[0, 1].plot(_get(alt, 'time_s'), _get(alt, 'Hmax'), '.-', color=A, label='altimeter Hmax')
    if pc: ax[0, 1].plot(_get(pc, 'time_s'), _get(pc, 'Hmax'), '.-', color=P, label='point cloud Hmax')
    ax[0, 1].set_ylabel('Hmax (m)'); ax[0, 1].set_title('Max wave height'); ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

    # Period
    if alt: ax[1, 0].plot(_get(alt, 'time_s'), _get(alt, 'Tp'), '.-', color=A, label='altimeter Tp')
    if pc: ax[1, 0].plot(_get(pc, 'time_s'), _get(pc, 'peak_period'), '.-', color=P, label='point cloud T (from λ)')
    ax[1, 0].set_ylabel('Period (s)'); ax[1, 0].set_xlabel('time (s)'); ax[1, 0].set_title('Peak period'); ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

    # Wavelength (point cloud only) + altimeter peak frequency on twin axis
    if pc: ax[1, 1].plot(_get(pc, 'time_s'), _get(pc, 'peak_wavelength'), '.-', color=P, label='point cloud λ')
    ax[1, 1].set_ylabel('wavelength (m)'); ax[1, 1].set_xlabel('time (s)'); ax[1, 1].set_title('Peak wavelength (point cloud)')
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

    title = 'Wave parameters — altimeter vs point cloud'
    if stale:
        title += f'   [STALE FEED: {", ".join(stale)} — producer stalled]'
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    png = os.path.join(args.out, 'waves_params.png')
    fig.savefig(png, dpi=130)
    print(f'wrote {png}')

    # quick numeric summary (medians)
    def med(d, k):
        v = _get(d, k); v = v[np.isfinite(v)] if v.size else v
        return float(np.median(v)) if v.size else float('nan')
    print('\nmedians over the recording:')
    print(f'  altimeter : Hs={med(alt,"Hs"):.2f}  Hmax={med(alt,"Hmax"):.2f}  Tp={med(alt,"Tp"):.2f}')
    print(f'  pointcloud: Hs={med(pc,"Hs"):.2f}  Hmax={med(pc,"Hmax"):.2f}  T={med(pc,"peak_period"):.2f}  λ={med(pc,"peak_wavelength"):.2f}')


if __name__ == '__main__':
    main()
