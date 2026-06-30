"""
Offline WAFT-Stereo figure generator.

Reads one stereo pair from a bag, rectifies it (same calibration as
ros2_stereo_rectifier.py), runs WAFT-Stereo, reprojects to 3D, and writes a
4-panel figure to Relatorio/Report/figures/waft_figure_<bag>.png:
    top-left:  raw left camera frame
    top-right: colourised WAFT disparity map
    bottom-left:  point cloud – side view (X across, Z depth)
    bottom-right: point cloud – elevation profile (Z depth, elevation Y)

Usage (from repo root):
    python3 Scripts/waft_offline_figure.py [bag_path] [--frame N] [--scale S] [--out path]

Defaults:
    bag_path = /media/alex/External/2026_LEIXOES_LOGS/airship_20260528_115912
    frame    = middle frame of the bag
    scale    = 0.5  (WAFT working resolution relative to the rectified image)
    out      = Relatorio/Report/figures/waft_figure_<bag_name>.png
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# ── rosbag2_py ───────────────────────────────────────────────────────────────
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

DEFAULT_BAG  = '/media/alex/External/2026_LEIXOES_LOGS/airship_20260528_115912'
LEFT_IMG     = '/airship/camera/left/image_color/compressed'
RIGHT_IMG    = '/airship/camera/right/image_color/compressed'
LEFT_INFO    = '/airship/camera/left/camera_info'
RIGHT_INFO   = '/airship/camera/right/camera_info'

# ── Stereo calibration (from ros2_stereo_rectifier.py) ───────────────────────
R_STEREO = np.array([
    [ 0.99998433,  0.00309469,  0.00466459],
    [-0.00307396,  0.9999854,  -0.0044443 ],
    [-0.00467827,  0.00442989,  0.99997924],
])
T_STEREO = np.array([-1.00029476e+00, -1.10997479e-04, 8.05032395e-03])
RECT_OUTPUT_SCALE = 0.25  # matches ros2_stereo_rectifier.py default


# ── Bag helpers ───────────────────────────────────────────────────────────────

def read_topic_messages(bag_path, topics):
    """Yield (topic, timestamp_ns, deserialized_msg) for the requested topics."""
    storage_opts = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    conv_opts    = rosbag2_py.ConverterOptions('', '')
    reader       = rosbag2_py.SequentialReader()
    reader.open(storage_opts, conv_opts)
    type_map  = {t.name: t.type for t in reader.get_all_topics_and_types()}
    filter_   = rosbag2_py.StorageFilter(topics=[t for t in topics if t in type_map])
    reader.set_filter(filter_)
    msg_types = {name: get_message(type_map[name]) for name in topics if name in type_map}
    while reader.has_next():
        topic, raw, ts = reader.read_next()
        if topic in msg_types:
            yield topic, ts, deserialize_message(raw, msg_types[topic])


def _decode_compressed(msg):
    """CompressedImage → BGR uint8 ndarray."""
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def collect_stereo_pair(bag_path, frame_index=None):
    """
    Return (left_bgr, right_bgr, left_info_msg, right_info_msg) for one frame.

    Early-exits as soon as both CameraInfo messages and the target frame pair
    are found — does not scan the entire bag.  frame_index=None picks the first
    frame (avoids needing a full pre-scan to find the middle).
    """
    if frame_index is None:
        frame_index = 0

    left_info = right_info = None
    left_seen = 0   # how many left frames have been skipped
    right_buf  = [] # recent right frames, to match the chosen left by timestamp

    left_target_ts = None
    left_target_msg = None

    topics = [LEFT_IMG, RIGHT_IMG, LEFT_INFO, RIGHT_INFO]
    for topic, ts, msg in read_topic_messages(bag_path, topics):
        if topic == LEFT_INFO and left_info is None:
            left_info = msg
        elif topic == RIGHT_INFO and right_info is None:
            right_info = msg
        elif topic == LEFT_IMG:
            if left_seen < frame_index:
                left_seen += 1
                continue
            if left_target_msg is None:
                left_target_ts  = ts
                left_target_msg = msg
                left_seen += 1
        elif topic == RIGHT_IMG:
            right_buf.append((ts, msg))
            # Keep only last 4 right frames (more than enough to find the match)
            if len(right_buf) > 4:
                right_buf.pop(0)

        # Done once we have both info + the target left frame + at least one right frame
        if (left_info is not None and right_info is not None
                and left_target_msg is not None and len(right_buf) >= 1):
            break

    if left_target_msg is None:
        raise RuntimeError(f'Frame {frame_index} not found in {bag_path}')
    if left_info is None or right_info is None:
        raise RuntimeError('CameraInfo not found before target frame')
    if not right_buf:
        raise RuntimeError('No right frames found near target left frame')

    # Nearest right frame by recording timestamp
    nearest = min(right_buf, key=lambda x: abs(x[0] - left_target_ts))
    dt_ms = abs(nearest[0] - left_target_ts) / 1e6
    print(f'  Frame {frame_index}, stereo dt={dt_ms:.1f} ms')

    return _decode_compressed(left_target_msg), _decode_compressed(nearest[1]), left_info, right_info


# ── Rectification ─────────────────────────────────────────────────────────────

def build_rectification(left_info, right_info):
    """Build remap LUTs + Q at RECT_OUTPUT_SCALE using the calibration from ros2_stereo_rectifier.py."""
    K_l = np.array(left_info.k).reshape(3, 3)
    D_l = np.array(left_info.d)
    K_r = np.array(right_info.k).reshape(3, 3)
    D_r = np.array(right_info.d)

    W_raw, H_raw = left_info.width, left_info.height
    W_out = max(1, int(round(W_raw * RECT_OUTPUT_SCALE)))
    H_out = max(1, int(round(H_raw * RECT_OUTPUT_SCALE)))
    print(f'  Rectifying {W_raw}x{H_raw} → {W_out}x{H_out} (scale={RECT_OUTPUT_SCALE})')

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l, D_l, K_r, D_r,
        (W_raw, H_raw), R_STEREO, T_STEREO,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        newImageSize=(W_out, H_out),
    )

    map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, (W_out, H_out), cv2.CV_32FC1)
    map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, (W_out, H_out), cv2.CV_32FC1)

    return map_lx, map_ly, map_rx, map_ry, Q, P1, (W_out, H_out)


def rectify_pair(left_bgr, right_bgr, map_lx, map_ly, map_rx, map_ry):
    left_rect  = cv2.remap(left_bgr,  map_lx, map_ly, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_bgr, map_rx, map_ry, cv2.INTER_LINEAR)
    return left_rect, right_rect


# ── WAFT inference ────────────────────────────────────────────────────────────

def load_waft(repo_dir, config_file, ckpt_path, device='cuda'):
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from bridgedepth.config import get_cfg
    from algorithms.waft import WAFT
    from peft import PeftModel

    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.freeze()

    cwd_before = os.getcwd()
    try:
        os.chdir(repo_dir)
        model = WAFT(cfg)
    finally:
        os.chdir(cwd_before)

    model.eval()
    model = model.to(device)

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    weights = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(weights, strict=False)

    for _, module in model.named_modules():
        if isinstance(module, PeftModel):
            module.merge_and_unload()

    return model


def run_waft(model, left_bgr, right_bgr, scale_factor, device='cuda'):
    """Downscale, infer, return float32 disparity at the working resolution."""
    h_orig, w_orig = left_bgr.shape[:2]
    w_work = max(1, int(round(w_orig * scale_factor)))
    h_work = max(1, int(round(h_orig * scale_factor)))
    print(f'  WAFT working resolution: {w_work}x{h_work}')

    def to_tensor(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).float()[None]

    left_small  = cv2.resize(left_bgr,  (w_work, h_work), interpolation=cv2.INTER_LINEAR)
    right_small = cv2.resize(right_bgr, (w_work, h_work), interpolation=cv2.INTER_LINEAR)

    t0 = time.perf_counter()
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            sample = {
                'img1': to_tensor(left_small).to(device),
                'img2': to_tensor(right_small).to(device),
            }
            out = model.inference(sample, size=None, factor=1.0)
    dt = time.perf_counter() - t0
    print(f'  WAFT inference: {dt*1000:.0f} ms')

    disp = out['disp_pred'][0].float().cpu().numpy().astype(np.float32)
    disp = np.maximum(disp, 0.0)

    # Trim any padding the model added internally
    if disp.shape != (h_work, w_work):
        disp = cv2.resize(disp, (w_work, h_work), interpolation=cv2.INTER_LINEAR)

    # Remove invisible matches (match would fall off left edge of right image)
    xx = np.arange(w_work, dtype=np.float32)[None, :]
    disp[xx - disp < 0] = 0.0

    return disp, left_small


# ── 3-D reprojection ──────────────────────────────────────────────────────────

def reproject(disp, Q_native, rect_wh, work_wh, max_range=25.0):
    """
    Reproject disparity to 3-D.  disp is at work_wh; Q_native is for rect_wh.
    Scales Q's pixel-based terms the same way ros2_pointcloud_node.py does.
    max_range caps depth to the near field where stereo is accurate (same as
    ros2_pointcloud_waves.py's max_range param — stereo depth error grows as
    depth²/(fx·B), so long-range points dominate the residual variance without
    adding wave information).
    """
    rect_w, _ = rect_wh
    work_w, _ = work_wh
    scale = work_w / rect_w
    Q = Q_native.copy()
    Q[0, 3] *= scale
    Q[1, 3] *= scale
    Q[2, 3] *= scale
    Q[3, 3] *= scale

    pts3d = cv2.reprojectImageTo3D(disp, Q)  # (H, W, 3)

    depth = pts3d[:, :, 2]
    mask = (disp > 0.0) & (depth > 0.5) & (depth < max_range)

    xyz = pts3d[mask]
    # camera-frame FLU remap (same as ros2_pointcloud_node.py)
    # optical: +X right, +Y down, +Z forward → FLU: +X forward, +Y left, +Z up
    X =  xyz[:, 2]   # forward  (depth)
    Y = -xyz[:, 0]   # left     (-optical X)
    Z = -xyz[:, 1]   # up       (-optical Y)
    return X, Y, Z


# ── Plane fit + residuals ─────────────────────────────────────────────────────

def plane_residuals(X, Y, Z, sigma_iters=3, sigma_thresh=2.5):
    """
    Fit Z = aX + bY + c by least-squares (iterative sigma clipping).
    Returns (residuals, a, b, c).  Same approach as ros2_pointcloud_waves.py.
    """
    pts = np.stack([X, Y, np.ones_like(X)], axis=1)
    mask = np.ones(len(X), dtype=bool)
    a = b = c = 0.0
    for _ in range(sigma_iters):
        if mask.sum() < 10:
            break
        coef, _, _, _ = np.linalg.lstsq(pts[mask], Z[mask], rcond=None)
        a, b, c = coef
        res = Z - (a * X + b * Y + c)
        std = float(np.std(res[mask]))
        mask = np.abs(res) < sigma_thresh * std
    res = Z - (a * X + b * Y + c)
    return res, a, b, c


# ── Figure rendering ──────────────────────────────────────────────────────────

def _colorise_disparity(disp):
    valid = disp[disp > 0]
    max_d = float(np.percentile(valid, 95)) if valid.size else 64.0
    scaled = np.clip(disp / max_d * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    color[disp <= 0] = 0
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def make_figure(left_bgr, left_small_bgr, disp, X, Y, Z, out_path, near_range=15.0):
    """
    4-panel figure.  The Hs estimate uses only points within near_range (m) of the
    camera, where stereo depth error is small enough to see wave signal rather than
    the quadratic depth-error trend that dominates at long range.  The live pipeline
    avoids this by transforming to map frame (nav TF + attitude) before fitting —
    this offline figure does not have nav data, so the near-field restriction is
    the practical substitute.
    """
    # Near-field mask for Hs only (full cloud shown in scatter)
    near = X < near_range
    res_all, a, b, c = plane_residuals(X, Y, Z)
    res_near = Z[near] - (a * X[near] + b * Y[near] + c)
    tilt_deg = float(np.degrees(np.arctan(np.sqrt(a**2 + b**2))))
    hs_near  = 4.0 * float(np.std(res_near))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('WAFT-Stereo — offline single-frame validation', fontsize=13)

    # (0,0) Raw left camera image
    axes[0, 0].imshow(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Left camera (raw, native resolution)')
    axes[0, 0].axis('off')

    # (0,1) Disparity map
    axes[0, 1].imshow(_colorise_disparity(disp))
    axes[0, 1].set_title(f'WAFT disparity map ({disp.shape[1]}×{disp.shape[0]} px, sky masked)')
    axes[0, 1].axis('off')

    # Subsample for scatter (keep at most 40k points)
    n = len(X)
    if n > 40_000:
        idx = np.random.choice(n, 40_000, replace=False)
        Xp, Yp, Zp, rp = X[idx], Y[idx], Z[idx], res_all[idx]
    else:
        Xp, Yp, Zp, rp = X, Y, Z, res_all

    # (1,0) Bird's-eye view: lateral (Y) vs forward (X), coloured by plane-fit residual
    r_lim2 = max(abs(np.percentile(rp, 2)), abs(np.percentile(rp, 98)))
    sc = axes[1, 0].scatter(Xp, Yp, c=rp, cmap='RdYlBu_r', s=0.8, alpha=0.6,
                            vmin=-r_lim2, vmax=r_lim2)
    plt.colorbar(sc, ax=axes[1, 0], label='Wave elevation residual (m)')
    axes[1, 0].set_xlabel('Forward X (m)')
    axes[1, 0].set_ylabel('Lateral Y (m, left=+)')
    axes[1, 0].set_title(f'Bird\'s-eye view (colour = elevation after plane fit, tilt {tilt_deg:.1f}°)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal', adjustable='datalim')

    # (1,1) Near-field plane residuals — where stereo is accurate
    r_near_all = res_all[near]
    r_lim = max(abs(np.percentile(r_near_all, 1)), abs(np.percentile(r_near_all, 99)), 0.05) * 1.3
    axes[1, 1].hist(r_near_all, bins=60, range=(-r_lim, r_lim),
                    color='steelblue', alpha=0.8, edgecolor='none')
    axes[1, 1].axvline(0, color='r', linestyle='--', linewidth=1.2)
    axes[1, 1].set_xlabel('Wave elevation residual (m)')
    axes[1, 1].set_ylabel('Point count')
    axes[1, 1].set_title(
        f'Near-field residuals (X < {near_range:.0f} m, tilt {tilt_deg:.1f}°)\n'
        f'Hs = 4σ = {hs_near:.3f} m  [single frame, no nav TF — approximate]'
    )
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {out_path}')
    print(f'  {len(X)} total pts, near-field ({near_range}m): {near.sum()} pts, '
          f'tilt={tilt_deg:.1f}°, Hs≈{hs_near:.3f} m')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag_path', nargs='?', default=DEFAULT_BAG)
    ap.add_argument('--frame', type=int, default=None,
                    help='Frame index (default: middle of bag)')
    ap.add_argument('--scale', type=float, default=0.5,
                    help='WAFT working scale relative to rectified image (default 0.5)')
    ap.add_argument('--max-range', type=float, default=25.0,
                    help='Max stereo depth (m) to include in point cloud (default 25)')
    ap.add_argument('--out', default=None,
                    help='Output PNG path (default: Relatorio/Report/figures/waft_figure_<bag>.png)')
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    waft_repo  = os.path.join(repo_root, 'WAFT-Stereo')
    config     = os.path.join(waft_repo, 'configs/SynLarge/DAv2S-4.yaml')
    ckpt       = os.path.join(waft_repo, 'ckpts/SynLarge/DAv2S-4.pth')

    bag_name = os.path.basename(args.bag_path.rstrip('/'))
    out_path = args.out or os.path.join(repo_root, f'Relatorio/Report/figures/waft_figure_{bag_name}.png')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    print('Reading bag…')
    left_bgr, right_bgr, left_info, right_info = collect_stereo_pair(args.bag_path, args.frame)

    print('Building rectification maps…')
    map_lx, map_ly, map_rx, map_ry, Q, P1, rect_wh = build_rectification(left_info, right_info)

    print('Rectifying stereo pair…')
    left_rect, right_rect = rectify_pair(left_bgr, right_bgr, map_lx, map_ly, map_rx, map_ry)

    print('Loading WAFT-Stereo…')
    model = load_waft(waft_repo, config, ckpt, device=device)

    print('Running WAFT inference…')
    disp, left_small = run_waft(model, left_rect, right_rect, args.scale, device=device)

    work_wh = (disp.shape[1], disp.shape[0])
    print('Reprojecting to 3-D…')
    X, Y, Z = reproject(disp, Q.astype(np.float32), rect_wh, work_wh,
                        max_range=args.max_range)

    print('Rendering figure…')
    make_figure(left_bgr, left_small, disp, X, Y, Z, out_path, near_range=15.0)


if __name__ == '__main__':
    main()
