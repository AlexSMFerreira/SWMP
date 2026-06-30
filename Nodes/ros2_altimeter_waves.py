#!/usr/bin/env python3
"""
AltimeterWavesNode — wave parameters from the altimeters (temporal spectrum).

Builds a sea-surface-elevation time series from a downward altimeter and the calibrated
nav altitude, then estimates significant wave height / periods by spectral analysis.

── Elevation reconstruction (and why nav subtraction is OFF by default) ──────────────────
    r_vert(t) = r(t) · cos(pitch) · cos(roll)        # range tilt-corrected to vertical drop
    eta(t)    = -r_vert(t)  − linear trend            # wave elevation (up positive)
The per-window linear detrend removes the slow airship altitude drift; the wave-band limit
([band_low_hz, band_high_hz], default 0.05-0.5 Hz) then removes the altimeter's broadband
sensor-noise floor — without it, integrating noise to Nyquist inflates Hs ~4x.

Originally this node subtracted the nav altitude (`nav_alt - r_vert`) to remove airship
heave. The offline diagnostic (Scripts/altimeter_wave_diagnostic.py) showed that is
counterproductive on this rig: the airship has negligible heave in the wave band (nav_alt
energy is all <0.1 Hz drift), while nav_alt's own EKF noise (~0.4 m RMS) is LARGER than the
wave signal, so subtracting it nearly doubled the in-band RMS. Band-limited raw range gives
Hs≈0.55 m / Tp≈2.1 s, matching the point-cloud estimate (~0.5 m, ~2.5-3 s); nav-subtracted
gave 0.84 m at a wrong 10 s period. Re-enable with `subtract_nav:=true` only if a future bag
shows real airship heave in the wave band.

── Inputs (see CLAUDE.md) ──────────────────────────────────────────────────────────────
  /airship/left/altimeter/height       geometry_msgs/PoseStamped   range = position.x (m)  [default source]
  /airship/right/altimeter/height      geometry_msgs/PoseStamped   range = position.x (m)  (obstructed in first 10 bags)
  /lightware_altimeter/left/altimeter  geometry_msgs/PointStamped  range = point.z (m); -1.0 = no return;
                                                                   values > max_range clamped out (spike bug)
  /episea/nav/lla                      nav_msgs/Odometry           nav_alt = position.z (m, WGS84)
  /episea/nav/euler                    geometry_msgs/Vector3Stamped roll/pitch/yaw (deg, NED) for tilt correction

── Doppler / encounter-frequency correction ─────────────────────────────────────────────
A boat moving through the waves measures an ENCOUNTER period, not the true wave period.
Using the boat speed (nav twist) and a deep-water encounter relation, `Tp`/`f_peak` are
converted to the TRUE wave values (`true_freq_from_encounter`); `Tp_encounter` keeps the
raw value. Verified: U≈2.6 m/s turns the raw 2.1-2.4 s peak into ~3.2-3.5 s, matching the
(motion-independent) point-cloud period ~3.7 s. `Hs` is unaffected (variance invariant).
`encounter_angle_deg` defaults to 180° (head seas — the survey case here). `Tm01`/`Tm02`
stay encounter-frame (moment remap not applied).

── Wave direction from the 3-altimeter array (diagnostic) ────────────────────────────────
The three altimeters lie ~collinear along the boat Y axis (a small wave-gauge array). A
least-squares fit of cross-spectrum phase vs Y-position at the wave peak gives the along-Y
wavenumber → the angle OFF head seas (port/stbd + fore/aft mirror ambiguous). Coherence is
high (~0.95), but the phase is small (~6° max over a 0.9 m aperture vs ~19 m waves) and
NON-STATIONARY (the relative wave angle drifts as the boat maneuvers), so the off-axis angle
swings ~3-20° — informative ("roughly head seas") but too noisy to trust as a number. It is
therefore REPORTED (`off_axis_angle_deg`, `direction_coherence`) but does NOT drive μ unless
`apply_array_mu:=true`. No full 2-D heading (1-D array).

── Outputs ─────────────────────────────────────────────────────────────────────────────
  /waves/altimeter                      diagnostic_msgs/DiagnosticArray  (Hs, Hmax, Tp[true],
                            Tp_encounter, Tm01, Tm02, f_peak, Hs_zerocross, speed_mps, n_samples, source)
  /waves/altimeter/significant_height   std_msgs/Float32   Hs (m)
  /waves/altimeter/max_wave_height      std_msgs/Float32   Hmax (m), largest individual wave
  /waves/altimeter/peak_period          std_msgs/Float32   Tp (s) — Doppler-corrected (true)
  /waves/altimeter/mean_period          std_msgs/Float32   Tm01 (s)
  /waves/altimeter/peak_frequency       std_msgs/Float32   f_peak (Hz) — true

No wave direction is produced (a single altimeter footprint has no spatial information).
The point-cloud node does not produce direction either — see ros2_pointcloud_waves.py.
"""

import math
import time
from collections import deque

import numpy as np
from scipy import signal

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped
from std_msgs.msg import Float32
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

import wave_common as wc


def _stamp_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class AltimeterWavesNode(Node):
    def __init__(self):
        super().__init__('altimeter_waves_node')

        # ── PARAMETERS ──────────────────────────────────────────────────────────────
        # Which altimeter feeds the elevation series. 'left' default — 'right' is
        # obstructed in the first 10 bags (CLAUDE.md); 'lightware' is the Lightware laser.
        self.declare_parameter('source', 'left')
        self.declare_parameter('alt_left_topic', '/airship/left/altimeter/height')
        self.declare_parameter('alt_right_topic', '/airship/right/altimeter/height')
        self.declare_parameter('lightware_topic', '/lightware_altimeter/left/altimeter')
        self.declare_parameter('nav_topic', '/episea/nav/lla')
        self.declare_parameter('euler_topic', '/episea/nav/euler')

        self.declare_parameter('fs', 10.0)               # uniform resample rate (Hz)
        self.declare_parameter('window_s', 60.0)         # Welch segment length / min span to analyse
        self.declare_parameter('buffer_s', 120.0)        # rolling buffer length (must be >= window_s)
        self.declare_parameter('report_period_s', 5.0)   # analysis + publish cadence
        self.declare_parameter('max_range', 10.0)        # clamp Lightware outlier spikes (m)
        # Ocean-wave band (Hz) for the spectral moments / Hs. The raw altimeter range
        # carries a broadband sensor-noise floor far above the wave band; integrating the
        # full 0..Nyquist range inflates Hs ~4x (verified, Scripts/altimeter_wave_diagnostic.py).
        # 0.05-0.5 Hz = periods 2-20 s; raise band_high for short wind chop if the noise
        # floor allows.
        self.declare_parameter('band_low_hz', 0.05)
        self.declare_parameter('band_high_hz', 0.5)
        # Subtract nav altitude to remove airship heave? Default False: the diagnostic
        # found the airship has negligible heave in the wave band, while nav_alt's own EKF
        # noise (~0.4 m RMS) is LARGER than the wave signal, so subtracting it injects
        # noise and worsens the in-band estimate. Detrending the raw range is cleaner. Set
        # True only if a future bag shows real airship heave in the wave band.
        self.declare_parameter('subtract_nav', False)
        # Doppler / encounter-frequency correction. A boat moving through the waves measures
        # a SHORTER (encounter) period than the true wave period — verified here: boat speed
        # ~2.6 m/s makes the altimeter read Tp≈2.4 s while the (motion-independent) point
        # cloud reads ≈3.7 s; the correction reconciles them (→3.5 s). Uses boat speed from
        # the nav twist; encounter_angle_deg = μ (180 = head seas, the survey case and what
        # this data shows). Hs is unaffected (variance is invariant under the remap).
        self.declare_parameter('doppler_correct', True)
        self.declare_parameter('encounter_angle_deg', 180.0)
        # Wave-direction estimate from the 3 collinear altimeters (a small wave-gauge array).
        # They lie ~along the boat Y axis (athwartships), so we recover the off-head-seas
        # ANGLE (with a port/starboard + fore/aft mirror ambiguity, irrelevant for cos μ),
        # which REFINES the Doppler encounter angle instead of assuming a fixed 180°. Done
        # by a least-squares fit of cross-spectrum phase vs Y-position at the wave peak;
        # gated on coherence + fit residual. Y offsets are the rig-CAD altimeter positions.
        self.declare_parameter('direction_estimate', True)
        self.declare_parameter('altimeter_y', [0.46953, -0.43967, -0.20444])  # left, right, lightware
        self.declare_parameter('dir_coherence_min', 0.6)
        self.declare_parameter('dir_resid_max_deg', 15.0)
        # Whether the array-measured off-axis angle DRIVES the Doppler μ. Default False:
        # verified that the directional phase is small (~6° max) and non-stationary on this
        # boat (the relative wave angle drifts as the boat maneuvers), so the off-axis angle
        # swings ~3-20° window-to-window — informative ("roughly head seas") but too noisy to
        # override μ, and it only moves Tp ~5% anyway. So we MEASURE & REPORT it but keep μ at
        # encounter_angle_deg (head seas). Set True to let the array refine μ.
        self.declare_parameter('apply_array_mu', False)

        p = self.get_parameter
        self._source = p('source').value.lower()
        self._fs = float(p('fs').value)
        self._window_s = float(p('window_s').value)
        self._buffer_s = float(p('buffer_s').value)
        self._max_range = float(p('max_range').value)
        self._band = (float(p('band_low_hz').value), float(p('band_high_hz').value))
        self._subtract_nav = bool(p('subtract_nav').value)
        self._doppler = bool(p('doppler_correct').value)
        self._enc_angle = float(p('encounter_angle_deg').value)
        self._direction = bool(p('direction_estimate').value)
        ay = list(p('altimeter_y').value)
        self._gauge_y = {'left': ay[0], 'right': ay[1], 'lightware': ay[2]}
        self._coh_min = float(p('dir_coherence_min').value)
        self._resid_max = float(p('dir_resid_max_deg').value)
        self._apply_array_mu = bool(p('apply_array_mu').value)

        # Latest auxiliary state.
        self._nav_alt = None
        self._speed = None                 # boat horizontal speed (m/s), from nav twist
        self._roll = self._pitch = None    # radians

        # Rolling (tu, elevation) buffer for the PRIMARY source (Hs/Tp etc.).
        self._buf = deque()
        self._last_raw_t = None            # last raw stamp, for backward-jump (bag loop) detection
        self._stamp_offset = 0.0           # cumulative offset to unwrap loop wraps (keeps buffer continuous)

        # Per-gauge raw-elevation buffers for the directional array (always -r_vert, no nav),
        # each with its own loop-unwrap state (all share the bag clock so offsets stay aligned).
        self._gbuf = {g: deque() for g in self._gauge_y}
        self._glast = {g: None for g in self._gauge_y}
        self._goff = {g: 0.0 for g in self._gauge_y}

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                         durability=DurabilityPolicy.VOLATILE, depth=10)

        if self._source not in self._gauge_y:
            raise ValueError(f"source must be left|right|lightware, got '{self._source}'")
        self.create_subscription(Odometry, p('nav_topic').value, self._cb_nav, qos)
        self.create_subscription(Vector3Stamped, p('euler_topic').value, self._cb_euler, qos)
        # Subscribe to ALL three altimeters: the source feeds the primary spectrum, and all
        # three feed the directional array (each callback buffers its own gauge).
        self.create_subscription(PoseStamped, p('alt_left_topic').value,
                                 lambda m: self._cb_pose_alt(m, 'left'), qos)
        self.create_subscription(PoseStamped, p('alt_right_topic').value,
                                 lambda m: self._cb_pose_alt(m, 'right'), qos)
        self.create_subscription(PointStamped, p('lightware_topic').value, self._cb_lw_alt, qos)

        # ── PUBLISHERS ──────────────────────────────────────────────────────────────
        self._pub_diag = self.create_publisher(DiagnosticArray, '/waves/altimeter', 10)
        self._pub_hs = self.create_publisher(Float32, '/waves/altimeter/significant_height', 10)
        self._pub_hmax = self.create_publisher(Float32, '/waves/altimeter/max_wave_height', 10)
        self._pub_tp = self.create_publisher(Float32, '/waves/altimeter/peak_period', 10)
        self._pub_tm = self.create_publisher(Float32, '/waves/altimeter/mean_period', 10)
        self._pub_fp = self.create_publisher(Float32, '/waves/altimeter/peak_frequency', 10)
        self._pub_latency = self.create_publisher(Float32, '/waves/altimeter/latency_ms', 10)

        self.create_timer(float(p('report_period_s').value), self._analyse)
        self.get_logger().info(
            f"AltimeterWaves ready. source={self._source}, fs={self._fs} Hz, "
            f"window={self._window_s}s, band={self._band[0]}-{self._band[1]} Hz, "
            f"subtract_nav={self._subtract_nav}. Publishing /waves/altimeter."
        )

    # ── callbacks ───────────────────────────────────────────────────────────────────

    def _cb_nav(self, msg: Odometry):
        self._nav_alt = msg.pose.pose.position.z
        # Horizontal boat speed for the Doppler/encounter-frequency correction.
        v = msg.twist.twist.linear
        self._speed = math.hypot(v.x, v.y)

    def _cb_euler(self, msg: Vector3Stamped):
        # /episea/nav/euler is roll/pitch/yaw in degrees, NED convention.
        self._roll = math.radians(msg.vector.x)
        self._pitch = math.radians(msg.vector.y)

    def _cb_pose_alt(self, msg: PoseStamped, gauge: str):
        # rfbeam altimeter: AGL height is position.x (position.y is a magnitude value).
        t = _stamp_sec(msg.header.stamp)
        r = float(msg.pose.position.x)
        self._gauge_push(gauge, t, r)
        if gauge == self._source:
            self._push(t, r)

    def _cb_lw_alt(self, msg: PointStamped):
        raw = float(msg.point.z)
        # -1.0 = no return; reject spikes beyond max_range (Lightware multipath, CLAUDE.md).
        if raw < 0.0 or raw > self._max_range:
            return
        t = _stamp_sec(msg.header.stamp)
        self._gauge_push('lightware', t, raw)
        if self._source == 'lightware':
            self._push(t, raw)

    def _gauge_push(self, gauge: str, t: float, r: float):
        """Buffer tilt-corrected elevation (-r_vert) for one gauge, for the directional
        array. Independent per-gauge loop-unwrap (all gauges share the bag clock)."""
        if self._roll is None or self._pitch is None:
            return
        lr = self._glast[gauge]
        if lr is not None and t < lr - 1.0:
            self._goff[gauge] += (lr - t) + 0.03
        self._glast[gauge] = t
        tu = t + self._goff[gauge]
        elev = -(r * math.cos(self._pitch) * math.cos(self._roll))
        buf = self._gbuf[gauge]
        buf.append((tu, elev))
        t_min = tu - self._buffer_s
        while buf and buf[0][0] < t_min:
            buf.popleft()

    def _push(self, t: float, r: float):
        """Convert one altimeter sample to a surface-elevation sample and buffer it."""
        if self._roll is None or self._pitch is None:
            return
        if self._subtract_nav and self._nav_alt is None:
            return
        # Unwrap bag --loop restarts: on a backward jump in the raw stamp, bump a cumulative
        # offset so the buffered time keeps increasing instead of clearing. The bag loops the
        # same (stationary) sea, so a continuous looped series is fine for the spectrum — and
        # this avoids re-warming the whole window (e.g. 200 s) after every loop, which would
        # otherwise leave the node silent most of the time. Mirrors ros2_pose_broadcaster.py.
        if self._last_raw_t is not None and t < self._last_raw_t - 1.0:
            self._stamp_offset += (self._last_raw_t - t) + 0.03   # +~1 sample to stay monotonic
        self._last_raw_t = t
        tu = t + self._stamp_offset

        r_vert = r * math.cos(self._pitch) * math.cos(self._roll)
        # Elevation (up positive). Default: -r_vert (range shrinks as the surface rises);
        # the slow airship drift it still contains is removed by the per-window detrend +
        # the wave-band limit. Optional nav subtraction (off by default, see params).
        elev = (self._nav_alt - r_vert) if self._subtract_nav else (-r_vert)
        self._buf.append((tu, elev))

        # Trim to buffer_s.
        t_min = tu - self._buffer_s
        while self._buf and self._buf[0][0] < t_min:
            self._buf.popleft()

    # ── directional array (3 collinear altimeters) ──────────────────────────────────

    def _estimate_ky(self, f_target):
        """Least-squares fit of cross-spectrum phase vs altimeter Y-position at f_target Hz,
        across the 3 collinear altimeters. Returns (k_y [rad/m], mean_coherence,
        fit_residual_deg) or None if there isn't enough overlapping data. The phase slope vs
        position is the along-Y wavenumber component; a low residual means a clean plane wave."""
        order = ('left', 'right', 'lightware')
        series = {}
        for g in order:
            buf = self._gbuf[g]
            if len(buf) < 16:
                return None
            series[g] = (np.fromiter((b[0] for b in buf), float),
                         np.fromiter((b[1] for b in buf), float))
        t0 = max(t[0] for t, _ in series.values())
        t1 = min(t[-1] for t, _ in series.values())
        if t1 - t0 < self._window_s:
            return None
        tg = np.arange(t0, t1, 1.0 / self._fs)
        if tg.size < 16:
            return None
        sig = {g: signal.detrend(np.interp(tg, t, y)) for g, (t, y) in series.items()}
        nper = int(min(tg.size, max(16, round(self._window_s * self._fs))))
        ref = sig['left']
        f, _ = signal.welch(ref, fs=self._fs, nperseg=nper)
        i = int(np.argmin(np.abs(f - f_target)))
        ys, phs, cohs = [], [], []
        for g in order:
            _, C = signal.csd(sig[g], ref, fs=self._fs, nperseg=nper)
            _, co = signal.coherence(sig[g], ref, fs=self._fs, nperseg=nper)
            ys.append(self._gauge_y[g]); phs.append(np.angle(C[i])); cohs.append(co[i])
        ys = np.array(ys); phs = np.unwrap(np.array(phs))
        mean_coh = float(np.mean(cohs[1:]))          # exclude left-vs-left (=1)
        A = np.vstack([ys, np.ones_like(ys)]).T
        (ky, b), *_ = np.linalg.lstsq(A, phs, rcond=None)
        resid = float(np.degrees(np.sqrt(np.mean((phs - A @ np.array([ky, b])) ** 2))))
        return float(ky), mean_coh, resid

    # ── analysis ──────────────────────────────────────────────────────────────────

    def _analyse(self):
        t0 = time.perf_counter()
        if len(self._buf) < 8:
            self.get_logger().info('… waiting for altimeter + nav data …')
            return
        t = np.fromiter((b[0] for b in self._buf), dtype=float)
        y = np.fromiter((b[1] for b in self._buf), dtype=float)
        span = t[-1] - t[0]
        if span < self._window_s:
            self.get_logger().info(f'… buffering {span:.0f}/{self._window_s:.0f}s …')
            return

        tg, yg = wc.resample_uniform(t, y, self._fs)
        eta = wc.detrend_linear(yg)
        freqs, psd = wc.welch_psd(eta, self._fs, self._window_s)
        prm = wc.wave_params_from_psd(freqs, psd, band=self._band)
        # Zero-crossing Hs and max wave height on the band-passed series so they match the
        # spectral band (raw eta has broadband noise that would create spurious crossings).
        eta_bp = wc.bandpass(eta, self._fs, *self._band)
        hs_zc = wc.zero_crossing_hs(eta_bp)
        hmax = wc.max_wave_height(eta_bp)
        n = int(eta.size)

        # Doppler / encounter-frequency correction: the measured peak is the ENCOUNTER
        # frequency on a moving boat; convert it to the TRUE wave frequency. Hs is
        # unaffected (variance invariant). Tp becomes the true period; Tp_encounter keeps
        # the raw value. Tm01/Tm02 remain encounter-frame (moment remap not applied).
        tp_enc = prm['Tp']
        f_enc = prm['f_peak']
        mu_used = self._enc_angle
        off_axis = float('nan')          # measured |angle off head seas| (deg), if available
        dir_coh = float('nan')
        if (self._doppler and self._speed is not None and self._speed > 0.1
                and math.isfinite(f_enc)):
            # Refine the encounter angle μ from the 3-altimeter array (else keep the default).
            if self._direction:
                est = self._estimate_ky(f_enc)
                f1 = wc.true_freq_from_encounter(f_enc, self._speed, 180.0)  # head-seas guess for k
                if est is not None and f1 is not None and math.isfinite(f1) and f1 > 0.0:
                    ky, coh, resid = est
                    if coh >= self._coh_min and resid <= self._resid_max:
                        k_tot = (2.0 * math.pi * f1) ** 2 / 9.81   # deep-water k from true freq
                        sin_off = abs(ky) / k_tot if k_tot > 0.0 else 2.0
                        if sin_off < 1.0:
                            # Measured |angle off head seas| (port/stbd + fore/aft mirror
                            # ambiguous — irrelevant for cos μ). Reported as a diagnostic;
                            # only drives μ if apply_array_mu (off by default — see param).
                            off_axis = math.degrees(math.asin(sin_off))
                            dir_coh = coh
                            if self._apply_array_mu:
                                mu_used = math.degrees(math.acos(-math.sqrt(1.0 - sin_off ** 2)))
            f_true = wc.true_freq_from_encounter(f_enc, self._speed, mu_used)
            if f_true is not None and math.isfinite(f_true) and f_true > 0.0:
                prm['Tp'] = 1.0 / f_true
                prm['f_peak'] = f_true

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._publish(prm, hs_zc, hmax, n, tp_enc, mu_used, off_axis, dir_coh, latency_ms)

        def f(x):
            return f'{x:.3f}' if (x is not None and math.isfinite(x)) else 'nan'
        spd = self._speed if self._speed is not None else float('nan')
        self.get_logger().info(
            f'[{n:4d} samp, {span:.0f}s] Hs={f(prm["Hs"])} m  Hmax={f(hmax)} m  '
            f'Tp={f(prm["Tp"])} s (enc {f(tp_enc)})  Tm02={f(prm["Tm02"])} s  '
            f'U={f(spd)} m/s  off-axis={f(off_axis)}° (coh {f(dir_coh)})  '
            f'Hs_zc={f(hs_zc)} m  latency={latency_ms:.1f} ms'
        )

    def _publish(self, prm, hs_zc, hmax, n, tp_enc, mu_used, off_axis, dir_coh, latency_ms=float('nan')):
        nan = float('nan')
        scalars = ((self._pub_hs, prm.get('Hs', nan)), (self._pub_tp, prm.get('Tp', nan)),
                   (self._pub_tm, prm.get('Tm01', nan)), (self._pub_fp, prm.get('f_peak', nan)),
                   (self._pub_hmax, hmax))
        for pub, v in scalars:
            m = Float32()
            m.data = float(v) if v is not None and math.isfinite(v) else nan
            pub.publish(m)
        self._pub_latency.publish(Float32(data=float(latency_ms)))

        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        st = DiagnosticStatus()
        st.name = 'wave_params'
        st.hardware_id = f'altimeter:{self._source}'
        st.level = DiagnosticStatus.OK if (prm['Hs'] is not None and math.isfinite(prm['Hs'])) \
            else DiagnosticStatus.WARN
        st.message = 'wave parameters from altimeter (temporal spectrum)'

        def kv(k, v):
            x = KeyValue()
            x.key = k
            x.value = (f'{v:.4f}' if isinstance(v, float) and math.isfinite(v) else str(v))
            return x

        spd = self._speed if self._speed is not None else nan
        st.values = [
            kv('Hs', prm['Hs']), kv('Hmax', hmax),
            kv('Tp', prm['Tp']), kv('Tp_encounter', tp_enc),
            kv('Tm01', prm['Tm01']), kv('Tm02', prm['Tm02']), kv('f_peak', prm['f_peak']),
            kv('Hs_zerocross', hs_zc),
            kv('speed_mps', float(spd)),
            kv('doppler_corrected', str(self._doppler)),
            kv('encounter_angle_deg', float(mu_used)),
            kv('off_axis_angle_deg', off_axis),      # |wave angle off head seas| from the array; NaN if not measured
            kv('direction_coherence', dir_coh),
            kv('n_samples', n), kv('source', self._source),
        ]
        arr.status = [st]
        self._pub_diag.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = AltimeterWavesNode()
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
