#!/usr/bin/env python3
"""
wave_common.py — shared sea-state / wave-parameter math for the two wave nodes
(ros2_altimeter_waves.py, ros2_pointcloud_waves.py). Pure numpy/scipy, no ROS, so the
spectral conventions stay identical across both backends for a fair comparison (mirrors
the role of stereo_common.py for the disparity nodes).

All "wave parameters" are derived from a sea-surface-elevation signal eta (metres,
mean-removed):
  - Hs (= H_m0)  significant wave height   = 4*sqrt(m0)
  - Tp           peak period               = 1 / f_peak
  - Tm01         mean period               = m0 / m1
  - Tm02         zero-crossing period       = sqrt(m0 / m2)
  - f_peak       peak frequency (Hz)
where mn = ∫ fⁿ·S(f) df are spectral moments of the elevation PSD S(f).

The point-cloud node works on a SPATIAL elevation field (residuals to a RANSAC plane);
it reuses hs_from_elevation() (spatial variance) and deepwater_period() (wavelength →
period via linear dispersion), and does its own 2-D FFT for direction.
"""

import numpy as np
from scipy import signal

GRAVITY = 9.81


# ── time-series conditioning ────────────────────────────────────────────────────────

def resample_uniform(t, y, fs):
    """Resample irregularly-sampled (t, y) onto a uniform grid at `fs` Hz via linear
    interpolation. Returns (t_grid, y_grid). Welch/FFT need even sampling; sensor message
    arrival is mildly irregular. `t` must be monotonically increasing."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 2:
        return t, y
    t_grid = np.arange(t[0], t[-1], 1.0 / fs)
    if t_grid.size < 2:
        return t, y
    return t_grid, np.interp(t_grid, t, y)


def detrend_linear(y):
    """Remove a linear trend (slow platform drift + datum offset), leaving the wave
    oscillation. Returns the residual, mean-removed."""
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return y - np.mean(y) if y.size else y
    return signal.detrend(y, type='linear')


def bandpass(eta, fs, lo, hi):
    """Zero-phase Butterworth band-pass to the wave band [lo, hi] Hz. Used so the
    time-domain zero-crossing Hs sees the same band as the spectral Hs (raw altimeter
    range carries a broadband noise floor well above the wave band — see
    Scripts/altimeter_wave_diagnostic.py). Falls back to the input if too short."""
    eta = np.asarray(eta, dtype=float)
    nyq = 0.5 * fs
    lo = max(lo, 1e-4) / nyq
    hi = min(hi, nyq * 0.999) / nyq
    if eta.size < 27 or not (0 < lo < hi < 1.0):
        return eta - np.mean(eta) if eta.size else eta
    b, a = signal.butter(4, [lo, hi], btype='band')
    return signal.filtfilt(b, a, eta)


# ── spectral analysis ───────────────────────────────────────────────────────────────

def welch_psd(eta, fs, window_s):
    """One-sided PSD of `eta` via Welch (Hann window, 50 % overlap). Segment length is
    min(len(eta), window_s*fs) so a short buffer still yields a (lower-resolution) PSD.
    Returns (freqs Hz, psd m²/Hz)."""
    eta = np.asarray(eta, dtype=float)
    nperseg = int(min(len(eta), max(8, round(window_s * fs))))
    if nperseg < 8:
        return np.empty(0), np.empty(0)
    freqs, psd = signal.welch(eta, fs=fs, window='hann',
                              nperseg=nperseg, noverlap=nperseg // 2,
                              detrend='constant', scaling='density')
    return freqs, psd


def spectral_moments(freqs, psd, band=None):
    """Zeroth/first/second spectral moments (m0, m1, m2) via trapezoidal integration.
    The DC bin (f=0) is always excluded. `band=(lo, hi)` restricts integration to that
    frequency range (Hz) — important for altimeter data, whose broadband sensor-noise
    floor extends far above the wave band and otherwise dominates m0 (inflating Hs); see
    Scripts/altimeter_wave_diagnostic.py."""
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    if freqs.size < 2:
        return 0.0, 0.0, 0.0
    mask = freqs > 0.0
    if band is not None:
        mask &= (freqs >= band[0]) & (freqs <= band[1])
    f, s = freqs[mask], psd[mask]
    if f.size < 2:
        return 0.0, 0.0, 0.0
    m0 = np.trapz(s, f)
    m1 = np.trapz(f * s, f)
    m2 = np.trapz(f * f * s, f)
    return float(m0), float(m1), float(m2)


def wave_params_from_psd(freqs, psd, band=None):
    """Spectral wave parameters from an elevation PSD. `band=(lo, hi)` Hz restricts the
    moments and the peak search to the ocean-wave band (recommended for altimeters).
    Returns a dict with Hs, Tp, Tm01, Tm02, f_peak, m0 (NaN where undefined)."""
    nan = float('nan')
    out = dict(Hs=nan, Tp=nan, Tm01=nan, Tm02=nan, f_peak=nan, m0=nan)
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    if freqs.size < 2:
        return out
    m0, m1, m2 = spectral_moments(freqs, psd, band)
    out['m0'] = m0
    if m0 <= 0.0:
        return out
    out['Hs'] = 4.0 * np.sqrt(m0)
    if m1 > 0.0:
        out['Tm01'] = m0 / m1
    if m2 > 0.0:
        out['Tm02'] = np.sqrt(m0 / m2)
    mask = freqs > 0.0
    if band is not None:
        mask &= (freqs >= band[0]) & (freqs <= band[1])
    f, s = freqs[mask], psd[mask]
    if f.size:
        f_peak = float(f[int(np.argmax(s))])
        out['f_peak'] = f_peak
        if f_peak > 0.0:
            out['Tp'] = 1.0 / f_peak
    return out


# ── statistical (time-domain) cross-check ───────────────────────────────────────────

def zero_crossing_heights(eta):
    """Individual wave heights (crest-to-trough between successive zero up-crossings).
    Returns an empty array if there are fewer than two up-crossings. The primitive behind
    both zero_crossing_hs (significant) and max_wave_height (Hmax)."""
    eta = np.asarray(eta, dtype=float)
    eta = eta - np.mean(eta)
    if eta.size < 4:
        return np.empty(0)
    # Indices where the signal crosses zero going upward.
    up = np.where((eta[:-1] < 0.0) & (eta[1:] >= 0.0))[0]
    if up.size < 2:
        return np.empty(0)
    return np.array([eta[a:b + 1].max() - eta[a:b + 1].min()
                     for a, b in zip(up[:-1], up[1:])])


def zero_crossing_hs(eta):
    """Significant wave height H1/3: mean of the highest third of zero-crossing wave
    heights. An independent cross-check of the spectral 4*sqrt(m0). NaN if too few waves."""
    h = zero_crossing_heights(eta)
    if h.size == 0:
        return float('nan')
    h = np.sort(h)[::-1]
    return float(np.mean(h[:max(1, h.size // 3)]))


def max_wave_height(eta):
    """Maximum individual wave height Hmax (largest crest-to-trough between zero
    up-crossings) in a temporal record. NaN if too few waves."""
    h = zero_crossing_heights(eta)
    return float(h.max()) if h.size else float('nan')


# ── spatial helpers (point-cloud node) ──────────────────────────────────────────────

def hs_from_elevation(eta):
    """Significant wave height from an elevation sample's variance: Hs = 4*std(eta).
    Used for the spatial (point-cloud) estimate where there is no time axis."""
    eta = np.asarray(eta, dtype=float)
    if eta.size < 2:
        return float('nan')
    return float(4.0 * np.std(eta))


def crest_to_trough(eta, pct=1.0):
    """Robust peak-to-trough range of an elevation sample — the spatial max-wave-height
    proxy (largest crest-to-trough across the patch in one snapshot). Uses the
    [pct, 100-pct] percentile span rather than literal max-min, because a single stray
    stereo point (mismatch/specular reflection) several metres off the surface would
    otherwise dominate the literal range — even one such point that survives the
    sigma-clipped plane fit. For a clean wave it still recovers ~2*amplitude (pct=1 trims
    only the extreme 1% tails, far below a real broad crest). pct=0 gives literal max-min."""
    eta = np.asarray(eta, dtype=float)
    if eta.size < 2:
        return float('nan')
    pct = min(max(pct, 0.0), 49.0)
    if pct <= 0.0:
        return float(eta.max() - eta.min())
    lo, hi = np.percentile(eta, [pct, 100.0 - pct])
    return float(hi - lo)


def deepwater_period(wavelength, g=GRAVITY):
    """Wave period from wavelength via deep-water linear dispersion (ω²=g·k):
    T = sqrt(2π·λ / g). Deep water is assumed (no valid depth in these bags)."""
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        return float('nan')
    return float(np.sqrt(2.0 * np.pi * wavelength / g))


def true_freq_from_encounter(f_enc, speed, angle_deg=180.0, g=GRAVITY):
    """Absolute (true) wave frequency [Hz] from an encounter frequency measured on a
    platform moving through the waves at `speed` [m/s]. Deep-water encounter relation:

        ω_e = ω₀ − (U/g)·cos(μ)·ω₀²,   μ = angle between wave-travel dir and heading.

    μ=180° = head seas (vessel into the waves — the common survey case, and what this
    dataset shows: following seas give no real root at this speed). A moving platform
    measures a *shorter* (encounter) period than the true wave period; this inverts that.
    Returns the physical positive root (→ f_enc as speed→0), or NaN at the following-sea
    singularity (no real solution). Variance/Hs is unaffected by this remap — only the
    frequency/period parameters shift."""
    if not np.isfinite(f_enc) or f_enc <= 0.0 or speed <= 0.0:
        return f_enc
    we = 2.0 * np.pi * f_enc
    k = speed * np.cos(np.radians(angle_deg)) / g     # ω_e = ω₀ − k·ω₀²  →  k·ω₀² − ω₀ + ω_e = 0
    if abs(k) < 1e-12:
        return f_enc
    disc = 1.0 - 4.0 * k * we
    if disc < 0.0:
        return float('nan')                            # singularity (e.g. following seas)
    sq = np.sqrt(disc)
    roots = [(1.0 - sq) / (2.0 * k), (1.0 + sq) / (2.0 * k)]
    roots = [r for r in roots if r > 0.0]
    if not roots:
        return float('nan')
    w0 = min(roots, key=lambda w: abs(w - we))         # physical branch (nearest encounter)
    return float(w0 / (2.0 * np.pi))
