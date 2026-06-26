#!/usr/bin/env python3
"""
altimeter_wave_diagnostic.py — diagnose the implausible altimeter wave result
(Hs~2.3 m at Tp~1.7 s / f_peak~0.583 Hz, see live run). Reads the bags directly
(no live pipeline) and compares the elevation spectrum three ways to locate the
0.583 Hz oscillation and decide whether nav_alt subtraction helps or hurts:

  (A) raw altimeter range (tilt-corrected)         = waves + platform heave + sensor noise
  (B) nav_alt alone                                = platform vertical motion per the nav EKF
  (C) nav_alt - tilt-corrected range (the node)    = the node's "wave" estimate

If the 0.583 Hz peak is in (A) but not (B), nav can't cancel it (it's surface chop or
altimeter noise). If it's mostly in (B) and (C) is noisier than (A), nav subtraction is
INJECTING it (nav EKF artifact) and we should not subtract.

Usage: ./altimeter_wave_diagnostic.py [bag] [--nav-bag PATH] [--source left|right] [--fs 10]
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Nodes'))
import wave_common as wc

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

DEFAULT_BAG = '/media/alex/External/2026_LEIXOES_LOGS/airship_20260528_115912'


def derive_nav_bag(bag):
    bag = bag.rstrip('/')
    return os.path.join(os.path.dirname(bag), 'ros2_nav', os.path.basename(bag) + '_nav')


def read_topic(bag, topic):
    """Return (stamps[N], msgs[N]) for one topic, by header stamp."""
    so = rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3')
    reader = rosbag2_py.SequentialReader()
    reader.open(so, rosbag2_py.ConverterOptions('', ''))
    tmap = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in tmap:
        print(f'  topic {topic} not in {bag}', file=sys.stderr)
        return np.empty(0), []
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    ts, ms = [], []
    cls = get_message(tmap[topic])
    while reader.has_next():
        _, data, _ = reader.read_next()
        m = deserialize_message(data, cls)
        ts.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        ms.append(m)
    return np.array(ts), ms


def band_hs(f, p, lo, hi):
    """Hs and peak period from the PSD restricted to [lo, hi] Hz (the ocean-wave band)."""
    m = (f >= lo) & (f <= hi)
    if m.sum() < 2:
        return float('nan'), float('nan')
    m0 = np.trapz(p[m], f[m])
    fpk = f[m][int(np.argmax(p[m]))]
    return 4*np.sqrt(m0), (1/fpk if fpk > 0 else float('nan'))


def psd_and_params(t, y, fs, window_s, label, band=(0.05, 0.5)):
    """Resample -> detrend -> Welch; print full-band and wave-band Hs/Tp; return PSD."""
    tg, yg = wc.resample_uniform(t, y, fs)
    eta = wc.detrend_linear(yg)
    f, p = wc.welch_psd(eta, fs, window_s)
    prm = wc.wave_params_from_psd(f, p)            # full 0..Nyquist band (what the node does)
    bhs, btp = band_hs(f, p, *band)               # wave band only
    print(f'  {label:30s} full: Hs={prm["Hs"]:.3f} Tp={prm["Tp"]:.2f}  |  '
          f'band[{band[0]}-{band[1]}Hz]: Hs={bhs:.3f} Tp={btp:.2f}  (RMS={np.std(eta):.3f})')
    return f, p, eta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag', nargs='?', default=DEFAULT_BAG)
    ap.add_argument('--nav-bag', default=None)
    ap.add_argument('--source', default='left', choices=['left', 'right'])
    ap.add_argument('--fs', type=float, default=10.0)
    ap.add_argument('--window-s', type=float, default=60.0)
    ap.add_argument('--out', default='altimeter_diag.png')
    args = ap.parse_args()

    bag = args.bag.rstrip('/')
    nav_bag = args.nav_bag or derive_nav_bag(bag)
    print(f'Camera bag: {bag}\nNav bag:    {nav_bag}\nsource={args.source} fs={args.fs} Hz\n')

    alt_topic = f'/airship/{args.source}/altimeter/height'
    t_alt, m_alt = read_topic(bag, alt_topic)
    t_nav, m_nav = read_topic(nav_bag, '/episea/nav/lla')
    t_eu, m_eu = read_topic(nav_bag, '/episea/nav/euler')
    if t_alt.size == 0 or t_nav.size == 0 or t_eu.size == 0:
        print('Missing data, aborting.'); return

    r = np.array([m.pose.position.x for m in m_alt])        # AGL range (m)
    nav_alt = np.array([m.pose.pose.position.z for m in m_nav])
    roll = np.radians([m.vector.x for m in m_eu])
    pitch = np.radians([m.vector.y for m in m_eu])

    # Sampling stats.
    dt = np.diff(t_alt)
    print(f'altimeter: {t_alt.size} msgs over {t_alt[-1]-t_alt[0]:.1f}s  '
          f'rate≈{1/np.median(dt):.1f} Hz (dt med={np.median(dt)*1e3:.0f}ms '
          f'min={dt.min()*1e3:.0f} max={dt.max()*1e3:.0f})')
    print(f'nav lla:   {t_nav.size} msgs  rate≈{t_nav.size/(t_nav[-1]-t_nav[0]):.1f} Hz')
    print(f'nav euler: {t_eu.size} msgs  rate≈{t_eu.size/(t_eu[-1]-t_eu[0]):.1f} Hz\n')

    # Interpolate nav onto altimeter timestamps (the node holds latest; interp is cleaner).
    nav_alt_i = np.interp(t_alt, t_nav, nav_alt)
    roll_i = np.interp(t_alt, t_eu, roll)
    pitch_i = np.interp(t_alt, t_eu, pitch)
    r_vert = r * np.cos(pitch_i) * np.cos(roll_i)
    surface_abs = nav_alt_i - r_vert

    print('Spectral comparison (each detrended over the whole record):')
    fA, pA, eA = psd_and_params(t_alt, r_vert, args.fs, args.window_s, '(A) raw range (tilt-corr)')
    fB, pB, eB = psd_and_params(t_alt, nav_alt_i, args.fs, args.window_s, '(B) nav_alt alone')
    fC, pC, eC = psd_and_params(t_alt, surface_abs, args.fs, args.window_s, '(C) nav_alt - range (node)')

    # ── plot ──
    fig, ax = plt.subplots(2, 1, figsize=(11, 9))
    tg = np.arange(0, t_alt[-1]-t_alt[0], 1/args.fs)
    n = min(len(tg), len(eA), len(eB), len(eC))
    ax[0].plot(tg[:n], eA[:n], label='(A) raw range', lw=0.8)
    ax[0].plot(tg[:n], eB[:n], label='(B) nav_alt', lw=0.8)
    ax[0].plot(tg[:n], eC[:n], label='(C) nav_alt - range (node)', lw=0.8)
    ax[0].set_xlabel('time (s)'); ax[0].set_ylabel('elevation (m, detrended)')
    ax[0].set_title('Elevation time series'); ax[0].legend(); ax[0].grid(alpha=0.3)

    for f, p, lab in ((fA, pA, '(A) raw range'), (fB, pB, '(B) nav_alt'),
                      (fC, pC, '(C) node')):
        ax[1].semilogy(f, p, label=lab)
    ax[1].axvline(0.583, color='k', ls='--', lw=0.8, label='0.583 Hz (observed peak)')
    ax[1].set_xlabel('frequency (Hz)'); ax[1].set_ylabel('PSD (m²/Hz)')
    ax[1].set_xlim(0, 2.5); ax[1].set_title('Elevation PSD'); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
