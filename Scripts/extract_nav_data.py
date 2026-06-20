#!/usr/bin/env python3
"""
extract_nav_data.py — Offline CSV export + plot of altimeter, position, and
orientation (roll/pitch/yaw) data straight from a camera bag and its
calibrated nav companion bag (see CLAUDE.md). No ROS node / live pipeline
needed — reads the rosbag2 SQLite files directly.

── Topics read ─────────────────────────────────────────────────────────────
  Camera bag:
    /airship/left/altimeter/height       geometry_msgs/PoseStamped   AGL (m) is
                                                                      position.x; position.y
                                                                      is a magnitude/quality
                                                                      value, not height (per
                                                                      José Carlos Fernandes,
                                                                      INESC TEC).
    /airship/right/altimeter/height      geometry_msgs/PoseStamped   same.
    /lightware_altimeter/left/altimeter  geometry_msgs/PointStamped  slant range, point.z
                                                                      (m); -1.0 = no return
                                                                      -> converted to NaN.
  Nav companion bag (ros2_nav/<bag_name>_nav):
    /episea/nav/lla     nav_msgs/Odometry           position = (lat_deg, lon_deg, alt_m)
    /episea/nav/euler   geometry_msgs/Vector3Stamped vector = (roll_deg, pitch_deg, yaw_deg)

Timestamps are each message's header.stamp (sensor time), not bag recording
time, so the camera bag and its separately-recorded nav companion line up
correctly even though they were captured by different processes.

── Outputs (in --out-dir) ───────────────────────────────────────────────────
  position.csv     time_s, lat_deg, lon_deg, alt_m
  orientation.csv  time_s, roll_deg, pitch_deg, yaw_deg
  altimeter.csv    time_s, alt_left_m, alt_right_m, alt_lightware_m
  nav_data.png     4-panel plot: ground track, altitude, attitude, altimeter ranges

Usage:
  ./extract_nav_data.py [bag_path] [--nav-bag PATH] [--out-dir DIR] [--show]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

DEFAULT_BAG = '/media/alex/External/2026_LEIXOES_LOGS/airship_20260528_115912'


def derive_nav_bag(bag_path: str) -> str:
    """Same convention as start_pipeline.sh: ros2_nav/<bag_name>_nav next to the bag."""
    bag_path = bag_path.rstrip('/')
    return os.path.join(os.path.dirname(bag_path), 'ros2_nav', os.path.basename(bag_path) + '_nav')


def read_topics(bag_path: str, topics: list) -> dict:
    """Return {topic: [(header_stamp_sec, msg), ...]}, sorted by recording order."""
    if not os.path.exists(bag_path):
        print(f'WARNING: bag not found, skipping: {bag_path}', file=sys.stderr)
        return {t: [] for t in topics}

    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [t for t in topics if t not in type_map]
    if missing:
        print(f'WARNING: topics not in {bag_path}: {missing}', file=sys.stderr)
    wanted = [t for t in topics if t in type_map]
    if not wanted:
        return {t: [] for t in topics}
    reader.set_filter(rosbag2_py.StorageFilter(topics=wanted))

    out = {t: [] for t in wanted}
    while reader.has_next():
        topic, data, _recv_time_ns = reader.read_next()
        msg = deserialize_message(data, get_message(type_map[topic]))
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        out[topic].append((stamp, msg))
    for t in topics:
        out.setdefault(t, [])
    return out


def to_df(entries, fields: dict) -> pd.DataFrame:
    """fields: {column_name: fn(msg) -> value}"""
    rows = [{'time_s': t, **{col: fn(m) for col, fn in fields.items()}} for t, m in entries]
    cols = ['time_s'] + list(fields.keys())
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values('time_s').reset_index(drop=True)


def merge_nearest(dfs, tolerance_s: float = 0.1) -> pd.DataFrame:
    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        return pd.DataFrame(columns=['time_s'])
    merged = dfs[0]
    for d in dfs[1:]:
        merged = pd.merge_asof(merged, d, on='time_s', direction='nearest', tolerance=tolerance_s)
    return merged


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag', nargs='?', default=DEFAULT_BAG, help='camera bag path')
    parser.add_argument('--nav-bag', default=None,
                         help='nav companion bag path (default: derived next to the camera bag)')
    parser.add_argument('--out-dir', default=None,
                         help='output directory (default: ./<bag_name>_extracted)')
    parser.add_argument('--show', action='store_true', help='open an interactive plot window too')
    parser.add_argument('--lightware-max-range', type=float, default=10.0,
                         help='Lightware readings beyond this (m) are treated as bogus/no-return '
                              'and dropped to NaN (see CLAUDE.md: outlier spikes up to 100-200 m '
                              'observed in several bags, not real over this low-AGL rig). '
                              'Default 10.0; use 0 to disable.')
    args = parser.parse_args()

    bag = args.bag.rstrip('/')
    nav_bag = args.nav_bag or derive_nav_bag(bag)
    out_dir = args.out_dir or f'{os.path.basename(bag)}_extracted'
    os.makedirs(out_dir, exist_ok=True)

    print(f'Camera bag: {bag}')
    print(f'Nav bag:    {nav_bag}')

    cam_topics = read_topics(bag, [
        '/airship/left/altimeter/height',
        '/airship/right/altimeter/height',
        '/lightware_altimeter/left/altimeter',
    ])
    nav_topics = read_topics(nav_bag, [
        '/episea/nav/lla',
        '/episea/nav/euler',
    ])

    pos_df = to_df(nav_topics['/episea/nav/lla'], {
        'lat_deg': lambda m: m.pose.pose.position.x,
        'lon_deg': lambda m: m.pose.pose.position.y,
        'alt_m': lambda m: m.pose.pose.position.z,
    })
    ori_df = to_df(nav_topics['/episea/nav/euler'], {
        'roll_deg': lambda m: m.vector.x,
        'pitch_deg': lambda m: m.vector.y,
        'yaw_deg': lambda m: m.vector.z,
    })
    left_df = to_df(cam_topics['/airship/left/altimeter/height'], {
        # position.x is the AGL reading in metres; position.y is a magnitude/quality
        # value, not height (confirmed against Lightware's range on the same bag).
        'alt_left_m': lambda m: m.pose.position.x,
    })
    right_df = to_df(cam_topics['/airship/right/altimeter/height'], {
        'alt_right_m': lambda m: m.pose.position.x,
    })
    lw_max = args.lightware_max_range
    def _lightware_value(m):
        # -1.0 means "no return". Beyond lw_max is also bogus: outlier spikes up to
        # 100-200 m have been observed in several bags (laser losing lock / multipath),
        # implausible for this low-AGL rig — see CLAUDE.md "Altimeter unit
        # re-verification". Both are dropped to NaN rather than wrecking the plot scale.
        if m.point.z < 0.0:
            return np.nan
        if lw_max > 0.0 and m.point.z > lw_max:
            return np.nan
        return m.point.z
    lw_df = to_df(cam_topics['/lightware_altimeter/left/altimeter'], {
        'alt_lightware_m': _lightware_value,
    })
    alt_df = merge_nearest([left_df, right_df, lw_df])

    for name, df in (('position', pos_df), ('orientation', ori_df), ('altimeter', alt_df)):
        path = os.path.join(out_dir, f'{name}.csv')
        df.to_csv(path, index=False)
        print(f'  wrote {path} ({len(df)} rows)')

    # ── Plot ─────────────────────────────────────────────────────────────────
    t0 = min(
        [d['time_s'].iloc[0] for d in (pos_df, ori_df, alt_df) if not d.empty] or [0.0]
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    if not pos_df.empty:
        ax.plot(pos_df['lon_deg'], pos_df['lat_deg'], '.-', ms=2)
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
    ax.set_title('Ground track')

    ax = axes[0, 1]
    if not pos_df.empty:
        ax.plot(pos_df['time_s'] - t0, pos_df['alt_m'])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Altitude (m, geodetic)')
    ax.set_title('Altitude')

    ax = axes[1, 0]
    if not ori_df.empty:
        t = ori_df['time_s'] - t0
        ax.plot(t, ori_df['roll_deg'], label='roll')
        ax.plot(t, ori_df['pitch_deg'], label='pitch')
        ax.plot(t, ori_df['yaw_deg'], label='yaw')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (deg)')
        ax.legend()
    ax.set_title('Attitude (NED)')

    ax = axes[1, 1]
    if not alt_df.empty:
        t = alt_df['time_s'] - t0
        if 'alt_left_m' in alt_df:
            ax.plot(t, alt_df['alt_left_m'], label='left')
        if 'alt_right_m' in alt_df:
            ax.plot(t, alt_df['alt_right_m'], label='right')
        if 'alt_lightware_m' in alt_df:
            ax.plot(t, alt_df['alt_lightware_m'], label='lightware')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Range (m)')
        ax.legend()
    ax.set_title('Altimeters (AGL)')

    fig.tight_layout()
    png_path = os.path.join(out_dir, 'nav_data.png')
    fig.savefig(png_path, dpi=150)
    print(f'  wrote {png_path}')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
