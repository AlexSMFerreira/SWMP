#!/usr/bin/env python3
"""
PointCloudWavesNode — wave parameters from the stereo point cloud (spatial analysis).

A single stereo frame is a spatial snapshot of the sea surface. We fit the mean surface
with a robust plane (iterative sigma-clipped least-squares: reject only gross outliers
like birds/spray beyond N·sigma, keeping the whole wave — self-tuning, so no per-sea-state
threshold), take each point's signed distance to that plane as the instantaneous elevation
field, and derive:
  - Hs              significant wave height = 4*std(residuals)   (spatial variance)
  - Hmax            max wave height = largest per-frame robust crest-to-trough over the buffer
                    (percentile-trimmed so a stray stereo point can't blow it up)
  - peak_wavelength dominant wavelength from a 2-D FFT of the gridded elevation field
  - peak_period     from peak_wavelength via deep-water linear dispersion (T=sqrt(2π·λ/g))

Wave DIRECTION is intentionally not produced: from a single snapshot it is 180° ambiguous
and, on this rig's small/noisy patch, was empirically near-random (per-frame std ~115°).
Only the wavelength magnitude (|k| at the spectral peak) is used.

── Inputs ───────────────────────────────────────────────────────────────────────────────
  /stereo/points        sensor_msgs/PointCloud2   XYZ(+RGB) in frame camera_left_rect
  TF  map ← camera_left_rect                       to express the surface in ENU

The cloud is transformed to the `map` (ENU) frame with tf2_sensor_msgs.do_transform_cloud()
— never manual per-point math (CLAUDE.md). Working in ENU makes "up" well-defined so the
plane normal (hence Hs and the in-plane wavelength) is physically meaningful.

── Outputs (rolling mean ± std over the last `buffer_frames`) ─────────────────────────────
  /waves/pointcloud                       diagnostic_msgs/DiagnosticArray
  /waves/pointcloud/significant_height    std_msgs/Float32   Hs (m)
  /waves/pointcloud/max_wave_height       std_msgs/Float32   Hmax (m), worst crest-to-trough in buffer
  /waves/pointcloud/peak_wavelength       std_msgs/Float32   λ (m)
  /waves/pointcloud/peak_period           std_msgs/Float32   T (s)
  /waves/pointcloud/surface               sensor_msgs/PointCloud2
                                          plane-fit inliers in map frame; fields x,y,z + elevation
                                          (signed residual, m above/below the mean surface plane)

── Caveats (see CLAUDE.md / WAVE_PARAMETERS_PLAN.md) ─────────────────────────────────────
  * Hs (variance) is robust to frame placeholders — it depends only on the cloud's internal
    geometry.
  * Deep-water dispersion is assumed (no valid depth in these bags).
  * Single-frame coverage is the camera FOV at altitude → wavelengths longer than the patch
    are unresolved; the snapshot favours short/steep waves. Rolling averaging mitigates
    per-frame noise but not this fundamental coverage limit.
"""

import math
import time
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from sensor_msgs_py import point_cloud2

import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

import wave_common as wc


def _read_xyz(cloud: PointCloud2) -> np.ndarray:
    """Return an (N,3) float64 array of finite XYZ points from a PointCloud2."""
    pts = point_cloud2.read_points(cloud, field_names=('x', 'y', 'z'), skip_nans=True)
    if isinstance(pts, np.ndarray):
        xyz = np.column_stack((pts['x'], pts['y'], pts['z'])).astype(np.float64)
    else:  # older API returns an iterable of tuples
        xyz = np.array([(p[0], p[1], p[2]) for p in pts], dtype=np.float64)
    if xyz.size == 0:
        return xyz.reshape(0, 3)
    return xyz[np.isfinite(xyz).all(axis=1)]


class PointCloudWavesNode(Node):
    def __init__(self):
        super().__init__('pointcloud_waves_node')

        # ── PARAMETERS ──────────────────────────────────────────────────────────────
        self.declare_parameter('cloud_topic', '/stereo/points')
        self.declare_parameter('map_frame', 'map')
        # Robust plane fit: iterative sigma-clipped least-squares. outlier_sigma rejects
        # only gross outliers (birds/spray) while keeping the wave — must stay well above
        # ~1 (which would clip the surface itself); 4 keeps the wave and drops birds.
        self.declare_parameter('outlier_sigma', 4.0)
        self.declare_parameter('plane_iters', 10)        # max sigma-clip iterations
        self.declare_parameter('min_points', 500)
        self.declare_parameter('max_points', 30000)      # random-subsample cap for speed
        # Keep only points within this range (m) of the camera before analysis. The raw
        # cloud spans ~2-99 m but stereo depth error grows ~quadratically — far points are
        # unreliable and bias Hs. Restricting to the accurate near field makes Hs honest
        # (see "Point-cloud Hs: range filter + median (2026-06-22)" in CLAUDE.md). 0 = off.
        self.declare_parameter('max_range', 100.0)
        self.declare_parameter('grid_res', 0.5)          # m, 2-D FFT grid cell
        self.declare_parameter('max_grid_cells', 200000) # guard against huge grids
        self.declare_parameter('tf_timeout_s', 0.1)
        self.declare_parameter('buffer_frames', 1)
        self.declare_parameter('report_period_s', 5.0)
        self.declare_parameter('min_process_interval_s', 0.3)  # throttle heavy per-frame work
        self.declare_parameter('gravity', 9.81)
        # Hmax (max wave height) = robust per-frame crest-to-trough, maxed over the buffer.
        # hmax_percentile trims this % off each tail of the residuals so a single stray
        # stereo point doesn't blow up the literal max-min (raise it if outliers persist).
        self.declare_parameter('hmax_percentile', 1.0)
        # Per-frame quality gate: a frame whose surface RMS implies Hs above this (m) is a
        # stereo/disparity failure — the WHOLE patch is spread out (verified: per-frame Hs
        # swings 0.04..3.0 m on a sea that's really ~0.3-0.5 m; see
        # Scripts/pointcloud_residual_diag.py). Such frames are dropped so they can't
        # corrupt Hs/Hmax/λ. Raise for genuinely rougher seas. 0 disables the gate.
        self.declare_parameter('max_frame_hs', 1.0)
        # Publish the plane-fit inlier cloud on /waves/pointcloud/surface. Each point
        # carries an 'elevation' field = signed residual (m above/below the mean plane),
        # useful for colorising the wave surface in RViz. Set false to save bandwidth.
        self.declare_parameter('publish_surface', True)

        p = self.get_parameter
        self._map_frame = p('map_frame').value
        self._sigma = float(p('outlier_sigma').value)
        self._plane_iters = int(p('plane_iters').value)
        self._min_points = int(p('min_points').value)
        self._max_points = int(p('max_points').value)
        self._max_range = float(p('max_range').value)
        self._grid_res = float(p('grid_res').value)
        self._max_cells = int(p('max_grid_cells').value)
        self._tf_timeout = float(p('tf_timeout_s').value)
        self._min_interval = float(p('min_process_interval_s').value)
        self._g = float(p('gravity').value)
        self._hmax_pct = float(p('hmax_percentile').value)
        self._max_frame_hs = float(p('max_frame_hs').value)
        self._publish_surface = bool(p('publish_surface').value)
        self._dropped = 0          # bad-disparity frames rejected since last report
        self._rng = np.random.default_rng(0)

        # Rolling per-frame results.
        n = int(p('buffer_frames').value)
        self._hs = deque(maxlen=n)
        self._hmax = deque(maxlen=n)        # per-frame crest-to-trough (max over buffer = Hmax)
        self._lam = deque(maxlen=n)
        self._tp = deque(maxlen=n)
        self._tilt = deque(maxlen=n)
        self._last_proc = None

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        sub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(PointCloud2, p('cloud_topic').value, self._cb_cloud, sub_qos)

        self._pub_diag = self.create_publisher(DiagnosticArray, '/waves/pointcloud', 10)
        self._pub_hs = self.create_publisher(Float32, '/waves/pointcloud/significant_height', 10)
        self._pub_hmax = self.create_publisher(Float32, '/waves/pointcloud/max_wave_height', 10)
        self._pub_lam = self.create_publisher(Float32, '/waves/pointcloud/peak_wavelength', 10)
        self._pub_tp = self.create_publisher(Float32, '/waves/pointcloud/peak_period', 10)
        self._pub_latency = self.create_publisher(Float32, '/waves/pointcloud/latency_ms', 10)
        self._pub_surface = self.create_publisher(PointCloud2, '/waves/pointcloud/surface', 10)

        self.create_timer(float(p('report_period_s').value), self._report)
        self.get_logger().info(
            f"PointCloudWaves ready. Plane fit: sigma-clip LS (sigma={self._sigma}, "
            f"iters={self._plane_iters}), grid={self._grid_res} m. "
            f"Transforming to '{self._map_frame}'. Publishing /waves/pointcloud."
        )

    # ── cloud callback ────────────────────────────────────────────────────────────

    def _cb_cloud(self, msg: PointCloud2):
        now = self.get_clock().now()
        if self._last_proc is not None and self._min_interval > 0.0:
            if (now - self._last_proc).nanoseconds * 1e-9 < self._min_interval:
                return
        self._last_proc = now

        t0 = time.perf_counter()

        tf = self._lookup_tf(msg)
        if tf is None:
            return
        cloud_map = do_transform_cloud(msg, tf)
        xyz = _read_xyz(cloud_map)
        if len(xyz) < self._min_points:
            return

        # Range filter: keep only points within max_range of the camera. Stereo depth error
        # grows ~quadratically with range, and the raw cloud spans ~2-99 m — the far points
        # are unreliable and, over such a long oblique strip, the plane fit absorbs real
        # long-wave elevation as "tilt" (biasing Hs low) while far-point scatter inflates a
        # vertical variance (see Scripts/pointcloud_hs_diag.py). Restricting to the accurate
        # near field makes Hs an honest, well-defined quantity. The camera origin in `map`
        # is the transform's translation. 0 disables.
        if self._max_range > 0.0:
            cam = np.array([tf.transform.translation.x, tf.transform.translation.y,
                            tf.transform.translation.z])
            within = np.linalg.norm(xyz - cam, axis=1) < self._max_range
            xyz = xyz[within]
            if len(xyz) < self._min_points:
                return
        if len(xyz) > self._max_points:
            xyz = xyz[self._rng.choice(len(xyz), self._max_points, replace=False)]

        fit = self._fit_plane(xyz)
        if fit is None:
            return
        nrm, d, inliers = fit
        P_in = xyz[inliers]
        residual = P_in @ nrm - d            # signed elevation about the mean plane

        hs = wc.hs_from_elevation(residual)
        # Per-frame quality gate: drop disparity-failure frames (whole-patch spread) so
        # they can't corrupt Hs/Hmax/λ. See the max_frame_hs parameter.
        if self._max_frame_hs > 0.0 and math.isfinite(hs) and hs > self._max_frame_hs:
            self._dropped += 1
            return

        hmax = wc.crest_to_trough(residual, self._hmax_pct)
        tilt = math.degrees(math.acos(min(1.0, abs(float(nrm[2])))))
        lam = self._peak_wavelength(P_in, residual, nrm)
        tp = wc.deepwater_period(lam, self._g)

        if self._publish_surface:
            self._publish_surface_cloud(P_in, residual, cloud_map.header)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._pub_latency.publish(Float32(data=latency_ms))

        if math.isfinite(hs):
            self._hs.append(hs)
        if math.isfinite(hmax):
            self._hmax.append(hmax)
        self._tilt.append(tilt)
        if math.isfinite(lam):
            self._lam.append(lam)
            self._tp.append(tp)
        self._report()

    def _lookup_tf(self, msg: PointCloud2):
        """Transform map ← cloud frame, at the cloud stamp if available else latest.
        (The cloud's raw stamp resets on bag --loop while map→base_link is unwrapped to
        stay monotonic — see CLAUDE.md — so an exact-stamp lookup can fail at a loop
        boundary; falling back to the latest transform is safe for a slow platform.)"""
        src = msg.header.frame_id
        timeout = rclpy.duration.Duration(seconds=self._tf_timeout)
        try:
            return self._tf_buffer.lookup_transform(self._map_frame, src,
                                                    Time.from_msg(msg.header.stamp), timeout)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass
        try:
            return self._tf_buffer.lookup_transform(self._map_frame, src, Time(), timeout)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF {self._map_frame}<-{src} unavailable: {e}',
                                   throttle_duration_sec=5.0)
            return None

    # ── geometry ──────────────────────────────────────────────────────────────────

    def _fit_plane(self, P: np.ndarray):
        """Robust mean-surface plane via iterative sigma-clipped least-squares. Returns
        (normal, d, inlier_mask) with n·x = d, normal oriented +Z (up), or None.

        Each pass fits the plane (SVD on the current inliers' centroid), then keeps only
        points within `outlier_sigma`·std of the residuals — this rejects gross outliers
        (birds/spray, many sigma away) while keeping the entire wave surface, so it needs
        no per-sea-state distance threshold (unlike fixed-band RANSAC, which clips any
        wave taller than its band). Converges in a few iterations."""
        idx = np.ones(len(P), dtype=bool)
        nrm = np.array([0.0, 0.0, 1.0])
        d = 0.0
        for _ in range(max(1, self._plane_iters)):
            Q = P[idx]
            if len(Q) < self._min_points:
                return None
            c = Q.mean(axis=0)
            _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
            nrm = vt[-1]
            d = nrm @ c
            res = P @ nrm - d
            s = res[idx].std()
            if s < 1e-9:
                break
            new = np.abs(res) < self._sigma * s
            if int(new.sum()) == int(idx.sum()):
                idx = new
                break
            idx = new
        if int(idx.sum()) < self._min_points:
            return None
        if nrm[2] < 0.0:        # orient up
            nrm, d = -nrm, -d
        return nrm, float(d), idx

    # x,y,z (float32) + elevation (float32, signed residual in metres above/below mean plane)
    _SURFACE_FIELDS = [
        PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='elevation', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    _SURFACE_POINT_STEP = 16  # 4 × float32

    def _publish_surface_cloud(self, P_in: np.ndarray, residual: np.ndarray, header) -> None:
        """Pack plane-fit inliers into a PointCloud2 pinned to the z=0 plane.

        x, y are taken from the map-frame point positions (which are centred at the map
        origin because ros2_pose_broadcaster pins base_link to XY=0).  z is the signed
        plane residual (elevation above/below the mean surface), so every point sits near
        z=0 with wave crests/troughs as the only Z variation.  The separate `elevation`
        field carries the same value for use with named-field colourizers in RViz."""
        data = np.empty((len(P_in), 4), dtype=np.float32)
        data[:, :2] = P_in[:, :2].astype(np.float32)   # x, y in map frame
        data[:, 2]  = residual.astype(np.float32)        # z = wave elevation
        data[:, 3]  = residual.astype(np.float32)        # elevation field (same)

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(P_in)
        msg.fields = self._SURFACE_FIELDS
        msg.is_bigendian = False
        msg.point_step = self._SURFACE_POINT_STEP
        msg.row_step = self._SURFACE_POINT_STEP * len(P_in)
        msg.is_dense = True
        msg.data = data.tobytes()
        self._pub_surface.publish(msg)

    def _peak_wavelength(self, P_in, residual, nrm):
        """Dominant wavelength (m) = 2π/|k| at the peak of a 2-D FFT of the gridded
        elevation field. Direction (the angle of k) is deliberately not returned — it is
        180° ambiguous from a single snapshot and was empirically near-random on this
        rig's patch. Returns nan if coverage is too sparse to resolve."""
        nan = float('nan')
        # In-plane orthonormal basis (u, v); plane is ~horizontal so these lie ~in ENU XY.
        zhat = np.array([0.0, 0.0, 1.0])
        u = np.cross(nrm, zhat)
        if np.linalg.norm(u) < 1e-6:
            u = np.array([1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(nrm, u)
        v /= np.linalg.norm(v)

        c = P_in.mean(axis=0)
        uu = (P_in - c) @ u
        vv = (P_in - c) @ v
        umin, umax = uu.min(), uu.max()
        vmin, vmax = vv.min(), vv.max()
        nu = int((umax - umin) / self._grid_res) + 1
        nv = int((vmax - vmin) / self._grid_res) + 1
        if nu < 8 or nv < 8 or nu * nv > self._max_cells:
            return nan

        iu = np.clip(((uu - umin) / self._grid_res).astype(int), 0, nu - 1)
        iv = np.clip(((vv - vmin) / self._grid_res).astype(int), 0, nv - 1)
        flat = iu * nv + iv
        ssum = np.bincount(flat, weights=residual, minlength=nu * nv)
        scnt = np.bincount(flat, minlength=nu * nv)
        filled = scnt > 0
        if filled.sum() < 0.25 * nu * nv:    # too gappy to trust a spectrum
            return nan
        grid = np.zeros(nu * nv)
        grid[filled] = ssum[filled] / scnt[filled]
        grid = grid.reshape(nu, nv)
        grid -= grid[filled.reshape(nu, nv)].mean()

        # 2-D Hann window to suppress spectral leakage from the patch edges.
        win = np.outer(np.hanning(nu), np.hanning(nv))
        F = np.fft.fft2(grid * win)
        power = np.abs(F) ** 2

        ku = 2.0 * np.pi * np.fft.fftfreq(nu, d=self._grid_res)   # rad/m, axis 0 (u)
        kv = 2.0 * np.pi * np.fft.fftfreq(nv, d=self._grid_res)   # rad/m, axis 1 (v)
        KU, KV = np.meshgrid(ku, kv, indexing='ij')
        kmag = np.sqrt(KU ** 2 + KV ** 2)
        # Mask DC + wavelengths longer than the patch (can't be resolved).
        max_len = min(umax - umin, vmax - vmin)
        power[kmag < (2.0 * np.pi / max_len)] = 0.0
        if not np.any(power > 0.0):
            return nan

        pk = np.unravel_index(np.argmax(power), power.shape)
        kabs = math.hypot(KU[pk], KV[pk])
        if kabs <= 0.0:
            return nan
        return float(2.0 * np.pi / kabs)

    # ── reporting ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _stat(dq):
        """Central value = MEDIAN (robust to the occasional residual bad frame, so the
        quality gate's removal of high frames no longer biases it), spread = std.
        With buffer_frames=1 (per-frame mode) this is just the latest value."""
        if not dq:
            return float('nan'), float('nan')
        a = np.array(dq, dtype=float)
        return float(np.median(a)), float(a.std())

    def _report(self):
        if not self._hs and not self._tilt:
            self.get_logger().info('… waiting for point cloud + TF …')
            return
        hs_m, hs_s = self._stat(self._hs)
        lam_m, lam_s = self._stat(self._lam)
        tp_m, _ = self._stat(self._tp)
        tilt_m, _ = self._stat(self._tilt)
        # Hmax = the worst (largest) crest-to-trough seen over the buffer, not a mean.
        hmax = max(self._hmax) if self._hmax else float('nan')

        for pub, val in ((self._pub_hs, hs_m), (self._pub_hmax, hmax),
                         (self._pub_lam, lam_m), (self._pub_tp, tp_m)):
            m = Float32()
            m.data = float(val) if math.isfinite(val) else float('nan')
            pub.publish(m)

        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        st = DiagnosticStatus()
        st.name = 'wave_params'
        st.hardware_id = 'pointcloud'
        st.level = DiagnosticStatus.OK if math.isfinite(hs_m) else DiagnosticStatus.WARN
        st.message = 'wave parameters from point cloud (sigma-clip plane + spatial spectrum)'

        def kv(k, val):
            x = KeyValue()
            x.key = k
            x.value = (f'{val:.4f}' if isinstance(val, float) and math.isfinite(val) else str(val))
            return x

        st.values = [
            kv('Hs', hs_m), kv('Hs_std', hs_s), kv('Hmax', hmax),
            kv('peak_wavelength', lam_m), kv('peak_wavelength_std', lam_s),
            kv('peak_period', tp_m),
            kv('plane_tilt_deg', tilt_m),
            kv('n_frames', len(self._hs)), kv('n_bad_dropped', self._dropped),
        ]
        arr.status = [st]
        self._pub_diag.publish(arr)

        def f(x):
            return f'{x:.3f}' if math.isfinite(x) else 'nan'
        self.get_logger().info(
            f'[{len(self._hs):2d} frames, {self._dropped} bad dropped] '
            f'Hs(med)={f(hs_m)}±{f(hs_s)} m  Hmax={f(hmax)} m  '
            f'λ(med)={f(lam_m)}±{f(lam_s)} m  T={f(tp_m)} s  tilt={f(tilt_m)}°',
            throttle_duration_sec=2.0,
        )
        self._dropped = 0


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudWavesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
