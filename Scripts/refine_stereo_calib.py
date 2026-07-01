"""
Refine the stereo extrinsics (R_stereo / T_stereo) from image data.

The hardcoded R_stereo/T_stereo in ros2_stereo_rectifier.py leave a residual
vertical epipolar error (the horizon, at infinity, lands a few px lower in the
right rectified image than the left — it should be on the same row). This script
re-estimates the relative rotation/translation between the two cameras directly
from stereo correspondences:

  1. Accumulate SIFT/ORB matches between left/right across many frames.
  2. Undistort the matched points to normalized rays (per-camera K, D).
  3. Estimate the essential matrix (RANSAC) and recover (R, t).
  4. Scale t to the known baseline (|T_stereo| ~ 1.0 m).
  5. Compare old vs refined rectification: residual vertical disparity of
     matches, and StereoBM near-field point count (the sensitive canary).

Prints the refined R_stereo / T_stereo ready to paste into
ros2_stereo_rectifier.py and Scripts/waft_offline_figure.py.

Usage:  python3 Scripts/refine_stereo_calib.py [bag] [--frames N] [--stride S]
"""

import argparse
import os
import sys

import cv2
import numpy as np

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_SCRIPTS)
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'Nodes'))

from waft_offline_figure import (  # noqa: E402
    R_STEREO, T_STEREO, read_topic_messages, _decode_compressed,
    LEFT_IMG, RIGHT_IMG, LEFT_INFO, RIGHT_INFO, DEFAULT_BAG,
)
import backend_offline_figure as B  # noqa: E402
from waft_offline_figure import rectify_pair, reproject  # noqa: E402


def collect_pairs(bag, n_frames, stride):
    targets = set(range(0, n_frames * stride, stride))
    li = ri = None
    idx = 0
    rbuf = []
    pairs = {}
    for topic, ts, msg in read_topic_messages(bag, [LEFT_IMG, RIGHT_IMG, LEFT_INFO, RIGHT_INFO]):
        if topic == LEFT_INFO and li is None:
            li = msg
        elif topic == RIGHT_INFO and ri is None:
            ri = msg
        elif topic == RIGHT_IMG:
            rbuf.append((ts, msg)); rbuf = rbuf[-4:]
        elif topic == LEFT_IMG:
            if idx in targets and rbuf and li is not None and ri is not None:
                nn = min(rbuf, key=lambda x: abs(x[0] - ts))
                pairs[idx] = (_decode_compressed(msg), _decode_compressed(nn[1]))
            idx += 1
            if idx >= n_frames * stride and len(pairs) == len(targets):
                break
    return li, ri, pairs


def match_pair(lg, rg, det):
    k1, d1 = det.detectAndCompute(lg, None)
    k2, d2 = det.detectAndCompute(rg, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return np.empty((0, 2)), np.empty((0, 2))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(d1, d2, k=2)
    pl, pr = [], []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:          # Lowe ratio
            pl.append(k1[m.queryIdx].pt)
            pr.append(k2[m.trainIdx].pt)
    return np.array(pl), np.array(pr)


def rect_build(K_l, D_l, K_r, D_r, Rst, Tst, wh_raw, scale=0.25):
    W_raw, H_raw = wh_raw
    W_out, H_out = int(round(W_raw * scale)), int(round(H_raw * scale))
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, (W_raw, H_raw), Rst, Tst,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=(W_out, H_out))
    mlx, mly = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, (W_out, H_out), cv2.CV_32FC1)
    mrx, mry = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, (W_out, H_out), cv2.CV_32FC1)
    return (mlx, mly, mrx, mry), Q, (W_out, H_out)


def residual_dy(pairs, maps, det):
    """Median |vertical disparity| of matches in the rectified pair."""
    mlx, mly, mrx, mry = maps
    dys = []
    for f, (l, r) in pairs.items():
        lr, rr = rectify_pair(l, r, mlx, mly, mrx, mry)
        lg = cv2.cvtColor(lr, cv2.COLOR_BGR2GRAY)
        rg = cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY)
        pl, pr = match_pair(lg, rg, det)
        if len(pl) == 0:
            continue
        dx = pl[:, 0] - pr[:, 0]
        dy = pl[:, 1] - pr[:, 1]
        ok = (dx > -3) & (dx < 250) & (np.abs(dy) < 40)
        dys.extend(dy[ok].tolist())
    dys = np.array(dys)
    return np.median(dys), np.median(np.abs(dys)), len(dys)


def sbm_points(pairs, maps, Q, wh, frame):
    mlx, mly, mrx, mry = maps
    l, r = pairs[frame]
    lr, rr = rectify_pair(l, r, mlx, mly, mrx, mry)
    d = B.disparity_sbm(lr, rr)
    X, _, _ = reproject(d, Q.astype(np.float32), wh, (d.shape[1], d.shape[0]), max_range=40.0)
    return len(X)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag_path', nargs='?', default=DEFAULT_BAG)
    ap.add_argument('--frames', type=int, default=40)
    ap.add_argument('--stride', type=int, default=8)
    args = ap.parse_args()

    print('Reading bag…')
    li, ri, pairs = collect_pairs(args.bag_path, args.frames, args.stride)
    print(f'  {len(pairs)} frames')
    K_l = np.array(li.k).reshape(3, 3); D_l = np.array(li.d)
    K_r = np.array(ri.k).reshape(3, 3); D_r = np.array(ri.d)
    wh_raw = (li.width, li.height)

    try:
        det = cv2.SIFT_create(4000)
        print('Using SIFT')
    except Exception:
        det = cv2.ORB_create(6000)
        print('Using ORB')

    # ── Accumulate correspondences in raw images, undistort to normalized rays ──
    pl_all, pr_all = [], []
    for f, (l, r) in pairs.items():
        lg = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
        rg = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        pl, pr = match_pair(lg, rg, det)
        if len(pl):
            pl_all.append(pl); pr_all.append(pr)
    pl_all = np.vstack(pl_all); pr_all = np.vstack(pr_all)
    print(f'  {len(pl_all)} raw correspondences')

    # Robust inlier set via essential-matrix RANSAC (used only to reject outlier
    # matches, NOT to re-estimate pose — that is degenerate on the water plane).
    nl = cv2.undistortPoints(pl_all.reshape(-1, 1, 2), K_l, D_l).reshape(-1, 2)
    nr = cv2.undistortPoints(pr_all.reshape(-1, 1, 2), K_r, D_r).reshape(-1, 2)
    _, mask = cv2.findEssentialMat(nl, nr, np.eye(3), method=cv2.RANSAC,
                                   prob=0.999, threshold=1e-3)
    inl = mask.ravel().astype(bool)
    pl_in, pr_in = pl_all[inl], pr_all[inl]
    print(f'  {inl.sum()} inlier correspondences kept for refinement')

    R_old = np.array(R_STEREO)
    T_col = np.array(T_STEREO).reshape(3, 1)
    W_raw, H_raw = wh_raw

    def rectify_dy(Rst):
        """Vertical disparity of the inlier matches after rectification with Rst."""
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            K_l, D_l, K_r, D_r, (W_raw, H_raw), Rst, T_col,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=(W_raw, H_raw))
        rl = cv2.undistortPoints(pl_in.reshape(-1, 1, 2), K_l, D_l, R=R1, P=P1).reshape(-1, 2)
        rr = cv2.undistortPoints(pr_in.reshape(-1, 1, 2), K_r, D_r, R=R2, P=P2).reshape(-1, 2)
        return rl[:, 1] - rr[:, 1]

    def objective(rvec):
        Rc, _ = cv2.Rodrigues(rvec)
        dy = rectify_dy(Rc @ R_old)
        return float(np.median(np.abs(dy)))    # robust to remaining outliers

    from scipy.optimize import minimize  # noqa: PLC0415
    print('  Optimising corrective rotation…')
    res = minimize(objective, np.zeros(3), method='Nelder-Mead',
                   options={'xatol': 1e-5, 'fatol': 1e-3, 'maxiter': 400})
    Rc, _ = cv2.Rodrigues(res.x)
    R_new = Rc @ R_old
    corr_deg = np.degrees(res.x)
    np.set_printoptions(precision=8, suppress=True)
    print(f'  corrective rotation (deg, xyz): {corr_deg}')
    print(f'  objective median|dy|: {objective(np.zeros(3)):.2f} -> {res.fun:.2f} px (native res)')

    print('\n--- refined R_stereo ---'); print(R_new)
    print('--- T_stereo (unchanged) ---'); print(np.array(T_STEREO))
    print('(old R_stereo)'); print(R_old)

    # ── Verify: residual dy and SBM points, old vs refined ──
    maps_old, Q_old, wh = rect_build(K_l, D_l, K_r, D_r, R_old, T_col, wh_raw)
    maps_new, Q_new, _ = rect_build(K_l, D_l, K_r, D_r, R_new, T_col, wh_raw)

    med_o, absmed_o, n_o = residual_dy(pairs, maps_old, det)
    med_n, absmed_n, n_n = residual_dy(pairs, maps_new, det)
    print(f'\nResidual vertical disparity (median / median-abs):')
    print(f'  OLD:     {med_o:+.2f} / {absmed_o:.2f} px  ({n_o} matches)')
    print(f'  REFINED: {med_n:+.2f} / {absmed_n:.2f} px  ({n_n} matches)')

    test_frame = sorted(pairs)[len(pairs) // 2]
    print(f'\nStereoBM near-field points (frame {test_frame}):')
    print(f'  OLD:     {sbm_points(pairs, maps_old, Q_old, wh, test_frame)}')
    print(f'  REFINED: {sbm_points(pairs, maps_new, Q_new, wh, test_frame)}')


if __name__ == '__main__':
    main()
