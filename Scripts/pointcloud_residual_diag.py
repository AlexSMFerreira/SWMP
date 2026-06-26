#!/usr/bin/env python3
"""
pointcloud_residual_diag.py — characterise the plane-fit residual distribution of the live
/stereo/points cloud, to decide a sensible spatial max-wave-height (Hmax) definition.

Subscribes to /stereo/points (needs the pipeline running + RMW_IMPLEMENTATION=rmw_zenoh_cpp),
fits the same sigma-clipped LS plane as ros2_pointcloud_waves.py, and prints residual
percentiles per frame: this shows where the tail is (real waves vs stereo outliers) and
hence what trim / aggregation makes Hmax physical. Plane residuals are frame-independent,
so no TF is needed. Collects `--n` clouds then exits.
"""
import os, sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Nodes'))


def fit_plane_sigmaclip(P, sigma=4.0, iters=10, min_points=500):
    idx = np.ones(len(P), dtype=bool)
    nrm = np.array([0, 0, 1.0]); d = 0.0
    for _ in range(iters):
        Q = P[idx]
        if len(Q) < min_points:
            return None
        c = Q.mean(0)
        _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
        nrm = vt[-1]; d = nrm @ c
        res = P @ nrm - d
        s = res[idx].std()
        if s < 1e-9:
            break
        new = np.abs(res) < sigma * s
        if int(new.sum()) == int(idx.sum()):
            idx = new; break
        idx = new
    if nrm[2] < 0:
        nrm, d = -nrm, -d
    return nrm, d, idx


class Diag(Node):
    def __init__(self, n):
        super().__init__('pc_residual_diag')
        self._n = n
        self._count = 0
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(PointCloud2, '/stereo/points', self._cb, qos)
        self.get_logger().info(f'waiting for {n} clouds on /stereo/points ...')

    def _cb(self, msg):
        pts = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        xyz = np.column_stack((pts['x'], pts['y'], pts['z'])).astype(np.float64) \
            if isinstance(pts, np.ndarray) else \
            np.array([(p[0], p[1], p[2]) for p in pts], dtype=np.float64)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if len(xyz) < 500:
            return
        fit = fit_plane_sigmaclip(xyz)
        if fit is None:
            return
        nrm, d, inl = fit
        r = xyz[inl] @ nrm - d
        a = np.abs(r)
        hs = 4 * r.std()
        pcts = np.percentile(a, [50, 90, 95, 99, 99.9])
        # crest-to-trough at a few trims
        def ct(p):
            lo, hi = np.percentile(r, [p, 100 - p]); return hi - lo
        self._count += 1
        self.get_logger().info(
            f'[{self._count:2d}] N={len(xyz):5d} inl={inl.sum():5d}  Hs={hs:.2f}  '
            f'|res| p50={pcts[0]:.2f} p90={pcts[1]:.2f} p95={pcts[2]:.2f} '
            f'p99={pcts[3]:.2f} p99.9={pcts[4]:.2f} max={a.max():.2f}  ||  '
            f'crest-trough: lit={ct(0):.2f} p1={ct(1):.2f} p2={ct(2):.2f} p5={ct(5):.2f}')
        if self._count >= self._n:
            raise SystemExit


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rclpy.init()
    node = Diag(n)
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
