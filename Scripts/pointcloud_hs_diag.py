#!/usr/bin/env python3
"""
pointcloud_hs_diag.py — why is point-cloud Hs lower than altimeter Hs?

Transforms the live /stereo/points to map (ENU) and, per cloud, compares Hs computed
several ways to isolate the cause:
  hs_plane  = 4*std(residual to the best-fit TILTED plane)   <- what the node does
  hs_vert   = 4*std(z - mean z)  (deviation about a HORIZONTAL plane, true vertical)
  hs_near   = hs_plane but only points within `near_m` of the cloud centroid (map XY)
Also reports the plane tilt and the map-XY horizontal extent of the patch.

If hs_vert >> hs_plane, the tilted-plane fit is absorbing long-wave elevation as "tilt"
(patch comparable to / smaller than the wavelength) — the suspected bias. If hs_near
differs a lot, depth range / far-point stereo error matters.
"""
import os, sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


def fit_plane(P, sigma=4.0, iters=10, minp=500):
    idx = np.ones(len(P), dtype=bool); nrm = np.array([0, 0, 1.0]); d = 0.0
    for _ in range(iters):
        Q = P[idx]
        if len(Q) < minp: return None
        c = Q.mean(0); _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
        nrm = vt[-1]; d = nrm @ c; res = P @ nrm - d; s = res[idx].std()
        if s < 1e-9: break
        new = np.abs(res) < sigma * s
        if int(new.sum()) == int(idx.sum()): idx = new; break
        idx = new
    if nrm[2] < 0: nrm, d = -nrm, -d
    return nrm, d, idx


class Diag(Node):
    def __init__(self, n, near_m):
        super().__init__('pc_hs_diag')
        self._n = n; self._near = near_m; self._count = 0
        self._tfbuf = tf2_ros.Buffer(); self._tfl = tf2_ros.TransformListener(self._tfbuf, self)
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(PointCloud2, '/stereo/points', self._cb, qos)
        self.get_logger().info(f'collecting {n} clouds (near_m={near_m}) ...')

    def _cb(self, msg):
        try:
            tf = self._tfbuf.lookup_transform('map', msg.header.frame_id,
                                              Time.from_msg(msg.header.stamp),
                                              rclpy.duration.Duration(seconds=0.1))
        except Exception:
            try:
                tf = self._tfbuf.lookup_transform('map', msg.header.frame_id, Time(),
                                                  rclpy.duration.Duration(seconds=0.1))
            except Exception as e:
                self.get_logger().warn(f'no TF: {e}', throttle_duration_sec=3.0); return
        cloud = do_transform_cloud(msg, tf)
        pts = point_cloud2.read_points(cloud, field_names=('x', 'y', 'z'), skip_nans=True)
        P = np.column_stack((pts['x'], pts['y'], pts['z'])).astype(np.float64) \
            if isinstance(pts, np.ndarray) else \
            np.array([(p[0], p[1], p[2]) for p in pts])
        P = P[np.isfinite(P).all(axis=1)]
        if len(P) < 500: return
        fit = fit_plane(P)
        if fit is None: return
        nrm, d, inl = fit; Q = P[inl]
        hs_plane = 4 * (Q @ nrm - d).std()
        hs_vert = 4 * (Q[:, 2] - Q[:, 2].mean()).std()
        tilt = np.degrees(np.arccos(min(1.0, abs(nrm[2]))))
        cxy = Q[:, :2].mean(0); rad = np.hypot(Q[:, 0] - cxy[0], Q[:, 1] - cxy[1])
        extent = float(rad.max())
        near = rad < self._near
        if near.sum() > 200:
            fitn = fit_plane(Q[near])
            hs_near = 4 * (Q[near] @ fitn[0] - fitn[1]).std() if fitn else float('nan')
        else:
            hs_near = float('nan')
        self._count += 1
        self.get_logger().info(
            f'[{self._count:2d}] N={len(Q):5d} extent={extent:5.1f}m tilt={tilt:4.1f}deg  '
            f'Hs_plane={hs_plane:.2f}  Hs_vert={hs_vert:.2f}  Hs_near(<{self._near:.0f}m)={hs_near:.2f}')
        if self._count >= self._n: raise SystemExit


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    near = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    rclpy.init(); node = Diag(n, near)
    try: rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt): pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == '__main__':
    main()
