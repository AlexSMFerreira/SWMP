#!/usr/bin/env python3
"""
sgm_cuda_wls_eval.py — offline evaluation of SGM-CUDA + WLS disparity quality.

Reads raw stereo frames directly from the bag (no live pipeline needed), applies
the same rectification as ros2_stereo_rectifier.py (output_scale=0.25), runs
the SGM-CUDA matcher with and without WLS, and reports the same metrics used in
the disparity_backend_comparison.csv table:
  - photo_err  (photometric consistency error, lower = better)
  - valid_frac (fraction of pixels with a valid disparity)
  - latency_ms (per-frame wall time, matcher only)

Usage:
  python3 Scripts/sgm_cuda_wls_eval.py [--bag PATH] [--max-frames N]
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Nodes'))
from stereo_common import photometric_consistency_error, to_float_disparity

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

DEFAULT_BAG = '/media/alex/External/2026_LEIXOES_LOGS/airship_20260528_115912'

# ── Stereo extrinsics (from ros2_stereo_rectifier.py) ────────────────────────
R_STEREO = np.array([
    [ 0.99998433,  0.00309469,  0.00466459],
    [-0.00307396,  0.9999854,  -0.0044443 ],
    [-0.00467827,  0.00442989,  0.99997924],
])
T_STEREO = np.array([-1.00029476e+00, -1.10997479e-04,  8.05032395e-03])
OUTPUT_SCALE = 0.25

# ── SGM-CUDA tuned parameters (matching ros2_sgm_cuda_disparity.py defaults) ─
NUM_DISP   = 256
BLOCK_SIZE = 9
UNIQ_RATIO = 10
MODE       = 1   # STEREO_SGBM_MODE_HH (full 8-direction)

# ── WLS parameters ────────────────────────────────────────────────────────────
WLS_LAMBDA = 8000.0
WLS_SIGMA  = 1.5


def read_camera_info(bag, topic):
    so = rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3')
    reader = rosbag2_py.SequentialReader()
    reader.open(so, rosbag2_py.ConverterOptions('', ''))
    tmap = {t.name: t.type for t in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    cls = get_message(tmap[topic])
    while reader.has_next():
        _, data, _ = reader.read_next()
        return deserialize_message(data, cls)
    raise RuntimeError(f'{topic} not found in {bag}')


def read_image_pairs(bag, max_frames):
    """Read and temporally pair left/right compressed images by nearest timestamp."""
    left_topic  = '/airship/camera/left/image_color/compressed'
    right_topic = '/airship/camera/right/image_color/compressed'

    so = rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3')
    reader = rosbag2_py.SequentialReader()
    reader.open(so, rosbag2_py.ConverterOptions('', ''))
    tmap = {t.name: t.type for t in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=[left_topic, right_topic]))
    cls_l = get_message(tmap[left_topic])
    cls_r = get_message(tmap[right_topic])

    lefts, rights = {}, {}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == left_topic:
            msg = deserialize_message(data, cls_l)
            t = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            lefts[t] = msg
        else:
            msg = deserialize_message(data, cls_r)
            t = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            rights[t] = msg

    left_ts  = sorted(lefts.keys())
    right_ts = sorted(rights.keys())
    pairs = []
    ri = 0
    for lt in left_ts:
        while ri + 1 < len(right_ts) and abs(right_ts[ri+1] - lt) < abs(right_ts[ri] - lt):
            ri += 1
        dt_ms = abs(right_ts[ri] - lt) / 1e6
        if dt_ms < 50:
            pairs.append((lefts[lt], rights[right_ts[ri]]))
        if max_frames > 0 and len(pairs) >= max_frames:
            break

    print(f'  {len(pairs)} paired frames (slop <50ms) from {len(left_ts)} left / {len(right_ts)} right')
    return pairs


def decode_compressed(msg):
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def build_rectify_maps(info_l, info_r, output_scale):
    K_l = np.array(info_l.k, dtype=np.float64).reshape(3, 3)
    D_l = np.array(info_l.d, dtype=np.float64)
    K_r = np.array(info_r.k, dtype=np.float64).reshape(3, 3)
    D_r = np.array(info_r.d, dtype=np.float64)

    native_w = info_l.width
    native_h = info_l.height
    out_w = int(native_w * output_scale)
    out_h = int(native_h * output_scale)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l, D_l, K_r, D_r,
        (native_w, native_h),
        R_STEREO, T_STEREO,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
        newImageSize=(out_w, out_h),
    )
    map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, (out_w, out_h), cv2.CV_32FC1)
    map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, (out_w, out_h), cv2.CV_32FC1)
    return map_lx, map_ly, map_rx, map_ry, out_w, out_h


def build_matcher():
    bs = BLOCK_SIZE
    return cv2.cuda.createStereoSGM(
        minDisparity=0,
        numDisparities=NUM_DISP,
        P1=8  * bs * bs,
        P2=32 * bs * bs,
        uniquenessRatio=UNIQ_RATIO,
        mode=MODE,
    )


def build_wls():
    f = cv2.ximgproc.createDisparityWLSFilterGeneric(use_confidence=False)
    f.setLambda(WLS_LAMBDA)
    f.setSigmaColor(WLS_SIGMA)
    return f


def eval_backend(pairs, maps, matcher, wls_filter, gpu_l, gpu_r, use_wls, label):
    map_lx, map_ly, map_rx, map_ry = maps
    results = []
    min_disp = 0

    for left_msg, right_msg in pairs:
        left_bgr  = decode_compressed(left_msg)
        right_bgr = decode_compressed(right_msg)

        left_rect  = cv2.remap(left_bgr,  map_lx, map_ly, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_bgr, map_rx, map_ry, cv2.INTER_LINEAR)

        left_gray  = cv2.cvtColor(left_rect,  cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        t0 = time.perf_counter()
        gpu_l.upload(left_gray)
        gpu_r.upload(right_gray)
        raw = matcher.compute(gpu_l, gpu_r).download()

        if use_wls:
            cuda_valid = raw > 0
            lf = cv2.flip(left_gray,  1)
            rf = cv2.flip(right_gray, 1)
            gpu_l.upload(rf)
            gpu_r.upload(lf)
            raw_r = cv2.flip(matcher.compute(gpu_l, gpu_r).download(), 1)
            raw_r = (raw_r.astype(np.int32) * -1).astype(np.int16)
            raw = wls_filter.filter(raw, left_gray, disparity_map_right=raw_r)
            raw[~cuda_valid] = -16

        latency = (time.perf_counter() - t0) * 1000.0
        disp = to_float_disparity(raw, min_disp)

        photo_err, valid_frac = photometric_consistency_error(left_gray, right_gray, disp)
        results.append((photo_err, valid_frac, latency))

    photo_errs  = [r[0] for r in results if np.isfinite(r[0])]
    valid_fracs = [r[1] for r in results]
    latencies   = [r[2] for r in results]

    print(f'\n{label} ({len(results)} frames):')
    print(f'  photo_err  median={np.median(photo_errs):.2f}  p90={np.percentile(photo_errs, 90):.2f}')
    print(f'  valid_frac median={np.median(valid_fracs):.4f} ({np.median(valid_fracs)*100:.1f}%)')
    print(f'  latency_ms median={np.median(latencies):.1f}  p90={np.percentile(latencies, 90):.1f}')

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', default=DEFAULT_BAG)
    ap.add_argument('--max-frames', type=int, default=200,
                    help='max frames to process (0 = all)')
    args = ap.parse_args()

    if not cv2.cuda.getCudaEnabledDeviceCount():
        print('FATAL: no CUDA device', file=sys.stderr)
        sys.exit(1)

    print(f'Reading from {args.bag}')
    print('Reading camera info...')
    info_l = read_camera_info(args.bag, '/airship/camera/left/camera_info')
    info_r = read_camera_info(args.bag, '/airship/camera/right/camera_info')
    print(f'  native resolution: {info_l.width}x{info_l.height}')
    print(f'  rectified at {OUTPUT_SCALE}x: {int(info_l.width*OUTPUT_SCALE)}x{int(info_l.height*OUTPUT_SCALE)}')

    print('Building rectification maps...')
    map_lx, map_ly, map_rx, map_ry, out_w, out_h = build_rectify_maps(info_l, info_r, OUTPUT_SCALE)
    maps = (map_lx, map_ly, map_rx, map_ry)

    print(f'Reading image pairs (max {args.max_frames or "all"})...')
    pairs = read_image_pairs(args.bag, args.max_frames)

    matcher = build_matcher()
    wls     = build_wls()
    gpu_l   = cv2.cuda_GpuMat()
    gpu_r   = cv2.cuda_GpuMat()

    # warm up GPU
    dummy = np.zeros((out_h, out_w), dtype=np.uint8)
    gpu_l.upload(dummy); gpu_r.upload(dummy)
    matcher.compute(gpu_l, gpu_r)

    eval_backend(pairs, maps, matcher, wls, gpu_l, gpu_r, use_wls=False,
                 label='SGM-CUDA (no WLS)')
    eval_backend(pairs, maps, matcher, wls, gpu_l, gpu_r, use_wls=True,
                 label='SGM-CUDA + WLS')


if __name__ == '__main__':
    main()
