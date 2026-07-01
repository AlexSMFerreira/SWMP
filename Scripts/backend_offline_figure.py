"""
Offline multi-backend figure generator.

Produces the same 4-panel figure as waft_offline_figure.py, for any of the six
disparity backends used in the comparison:
    {sbm, sgbm, sgm_cuda, hitnet, raft, waft}

    top-left:     rectified left camera
    top-right:    rectified right camera (same instant)
    bottom-left:  colourised disparity map (sky masked)
    bottom-right: point cloud, bird's-eye view, coloured by wave elevation

Each backend's disparity is computed exactly as its live ROS node does (same
parameters, same working resolution, same sky/horizon mask, same rescale back to
the full rectified resolution), so the figures are a faithful stand-in for the
live pipeline. The spine (bag read, rectify, reproject, render) is reused from
waft_offline_figure.py.

Usage (from repo root):
    python3 Scripts/backend_offline_figure.py <backend> [bag_path] [--frame N]
                                              [--max-range M] [--out path]

Examples:
    python3 Scripts/backend_offline_figure.py sgbm
    python3 Scripts/backend_offline_figure.py hitnet --frame 0
    python3 Scripts/backend_offline_figure.py all           # every backend, one call
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_SCRIPTS)
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, _REPO)                       # for the `hitnet` package
sys.path.insert(0, os.path.join(_REPO, 'Nodes'))

# Reused spine + WAFT implementation from the original single-backend script.
from waft_offline_figure import (  # noqa: E402
    DEFAULT_BAG, collect_stereo_pair, build_rectification, rectify_pair,
    horizon_correct_pair, reproject, make_figure, _HORIZON, load_waft, run_waft,
)
from stereo_common import downscale_pair, to_float_disparity, rescale_disparity  # noqa: E402

# Classical/learned CPU-GPU backends all match at this working resolution, then
# rescale the disparity back up to the full rectified size (mirrors the nodes'
# input_width/input_height defaults).
CLASSICAL_INPUT = (320, 240)
WLS_LAMBDA = 8000.0
WLS_SIGMA = 1.5


def _apply_sky(disp, left_rect):
    """Zero the sky above the detected horizon, exactly like the live nodes."""
    sky, _ = _HORIZON.compute_mask(left_rect, mask_shape=disp.shape[:2])
    disp[sky] = 0.0
    return disp


# ── Classical CPU matchers (StereoBM / StereoSGBM) ───────────────────────────

def _run_cpu_matcher(matcher, left_rect, right_rect, min_disp=0):
    """Downscale, match (+ WLS left/right refinement), rescale to full res, mask."""
    full_w, full_h = left_rect.shape[1], left_rect.shape[0]
    pl, pr = downscale_pair(left_rect, right_rect, *CLASSICAL_INPUT)
    lg = cv2.cvtColor(pl, cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(pr, cv2.COLOR_BGR2GRAY)

    raw = matcher.compute(lg, rg)
    right_matcher = cv2.ximgproc.createRightMatcher(matcher)
    wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcher)
    wls.setLambda(WLS_LAMBDA)
    wls.setSigmaColor(WLS_SIGMA)
    raw_r = right_matcher.compute(rg, lg)
    raw = wls.filter(raw, lg, disparity_map_right=raw_r)

    disp = to_float_disparity(raw, min_disp)
    disp = rescale_disparity(disp, (full_w, full_h))
    return _apply_sky(disp, left_rect)


def disparity_sbm(left_rect, right_rect):
    m = cv2.StereoBM_create(numDisparities=128, blockSize=21)
    m.setMinDisparity(0)
    m.setTextureThreshold(0)
    m.setUniquenessRatio(15)
    m.setSpeckleWindowSize(200)
    m.setSpeckleRange(2)
    m.setDisp12MaxDiff(1)
    m.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
    m.setPreFilterSize(9)
    m.setPreFilterCap(63)
    return _run_cpu_matcher(m, left_rect, right_rect)


def disparity_sgbm(left_rect, right_rect):
    bs = 5
    m = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=bs,
        P1=8 * bs * bs, P2=32 * bs * bs,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=6, speckleRange=1,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM,
    )
    return _run_cpu_matcher(m, left_rect, right_rect)


# ── GPU classical matcher (cv2.cuda.StereoSGM) ───────────────────────────────

def disparity_sgm_cuda(left_rect, right_rect):
    full_w, full_h = left_rect.shape[1], left_rect.shape[0]
    bs = 9
    matcher = cv2.cuda.createStereoSGM(
        minDisparity=0, numDisparities=256,
        P1=8 * bs * bs, P2=32 * bs * bs,
        uniquenessRatio=10, mode=1,   # MODE_HH
    )
    pl, pr = downscale_pair(left_rect, right_rect, *CLASSICAL_INPUT)
    lg = cv2.cvtColor(pl, cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(pr, cv2.COLOR_BGR2GRAY)

    gpu_l, gpu_r = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
    gpu_l.upload(lg)
    gpu_r.upload(rg)
    raw = matcher.compute(gpu_l, gpu_r).download()

    # WLS with the flip trick for the right disparity, then restore the CUDA
    # matcher's own valid mask so WLS stays a pure smoother (matches the node).
    cuda_valid = raw > 0
    wls = cv2.ximgproc.createDisparityWLSFilterGeneric(use_confidence=False)
    wls.setLambda(WLS_LAMBDA)
    wls.setSigmaColor(WLS_SIGMA)
    left_flip, right_flip = cv2.flip(lg, 1), cv2.flip(rg, 1)
    gpu_l.upload(right_flip)
    gpu_r.upload(left_flip)
    raw_right = cv2.flip(matcher.compute(gpu_l, gpu_r).download(), 1)
    raw_right = (raw_right.astype(np.int32) * -1).astype(np.int16)
    raw = wls.filter(raw, lg, disparity_map_right=raw_right)
    raw[~cuda_valid] = -16

    disp = to_float_disparity(raw, 0)
    disp = rescale_disparity(disp, (full_w, full_h))
    return _apply_sky(disp, left_rect)


# ── HITNet (ONNX) ────────────────────────────────────────────────────────────

_HITNET = None


def disparity_hitnet(left_rect, right_rect):
    global _HITNET
    from hitnet import HitNet, ModelType, CameraConfig  # noqa: PLC0415
    if _HITNET is None:
        model_path = os.path.join(_REPO, 'models/eth3d/saved_model_240x320/model_float32.onnx')
        # baseline/f only drive HITNet's depth read-out, not the disparity itself.
        _HITNET = HitNet(model_path, ModelType.eth3d, CameraConfig(baseline=1.0, f=1000.0), 100.0)
    disp = _HITNET(left_rect, right_rect)
    disp = rescale_disparity(disp, (left_rect.shape[1], left_rect.shape[0]))
    return _apply_sky(disp, left_rect)


# ── RAFT-Stereo (PyTorch) ────────────────────────────────────────────────────

_RAFT = None


def disparity_raft(left_rect, right_rect):
    global _RAFT
    import argparse as _argparse  # noqa: PLC0415
    import torch  # noqa: PLC0415
    repo = os.path.join(_REPO, 'RAFT-Stereo')
    for d in (repo, os.path.join(repo, 'core')):
        if d not in sys.path:
            sys.path.insert(0, d)
    from raft_stereo import RAFTStereo  # noqa: PLC0415
    from utils.utils import InputPadder  # noqa: PLC0415

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if _RAFT is None:
        args = _argparse.Namespace(
            mixed_precision=True, valid_iters=12, corr_implementation='alt',
            shared_backbone=True, n_downsample=3, n_gru_layers=2, slow_fast_gru=True,
            hidden_dims=[128] * 3, context_dims=[128] * 3,
            corr_levels=4, corr_radius=4, context_norm='batch',
        )
        ckpt = os.path.join(repo, 'models/raftstereo-realtime.pth')
        state = torch.load(ckpt, map_location=device, weights_only=False)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        state = {k.replace('module.', ''): v for k, v in state.items()}
        model = RAFTStereo(args)
        model.load_state_dict(state)
        _RAFT = model.to(device).eval()

    orig_h, orig_w = left_rect.shape[:2]
    inp_w, inp_h = CLASSICAL_INPUT
    ls = cv2.resize(left_rect, (inp_w, inp_h), interpolation=cv2.INTER_LINEAR)
    rs = cv2.resize(right_rect, (inp_w, inp_h), interpolation=cv2.INTER_LINEAR)

    def to_tensor(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).float()[None].to(device)

    img1, img2 = to_tensor(ls), to_tensor(rs)
    padder = InputPadder(img1.shape, divis_by=32)
    img1, img2 = padder.pad(img1, img2)
    with torch.no_grad():
        _, flow_up = _RAFT(img1, img2, iters=12, test_mode=True)
    flow_up = padder.unpad(flow_up)
    disp = (-flow_up[0, 0]).cpu().numpy().astype(np.float32)
    disp = np.maximum(disp, 0.0)
    disp = cv2.resize(disp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) * (orig_w / inp_w)
    return _apply_sky(disp, left_rect)


# ── WAFT (reuse the original script's model) ─────────────────────────────────

_WAFT = None


def disparity_waft(left_rect, right_rect):
    global _WAFT
    if _WAFT is None:
        waft_repo = os.path.join(_REPO, 'WAFT-Stereo')
        config = os.path.join(waft_repo, 'configs/SynLarge/DAv2S-4.yaml')
        ckpt = os.path.join(waft_repo, 'ckpts/SynLarge/DAv2S-4.pth')
        _WAFT = load_waft(waft_repo, config, ckpt,
                          device='cuda' if _has_cuda() else 'cpu')
    disp, _ = run_waft(_WAFT, left_rect, right_rect, 0.5,
                       device='cuda' if _has_cuda() else 'cpu')
    # run_waft already applies the sky mask; rescale to full rect resolution so
    # the figure and reprojection are uniform across backends.
    return rescale_disparity(disp, (left_rect.shape[1], left_rect.shape[0]))


def _has_cuda():
    import torch  # noqa: PLC0415
    return torch.cuda.is_available()


# ── Dispatch ─────────────────────────────────────────────────────────────────

BACKENDS = {
    'sbm': ('StereoBM', disparity_sbm),
    'sgbm': ('StereoSGBM', disparity_sgbm),
    'sgm_cuda': ('SGM-CUDA', disparity_sgm_cuda),
    'hitnet': ('HITNet', disparity_hitnet),
    'raft': ('RAFT-Stereo', disparity_raft),
    'waft': ('WAFT-Stereo', disparity_waft),
}


def _run_one(backend, left_rect, right_rect, Q, rect_wh, max_range, out_path):
    label, fn = BACKENDS[backend]
    print(f'[{backend}] computing disparity…')
    t0 = time.perf_counter()
    disp = fn(left_rect, right_rect)
    print(f'[{backend}] disparity {disp.shape[1]}x{disp.shape[0]} in {(time.perf_counter()-t0)*1000:.0f} ms')

    work_wh = (disp.shape[1], disp.shape[0])
    X, Y, Z = reproject(disp, Q.astype(np.float32), rect_wh, work_wh, max_range=max_range)
    make_figure(left_rect, right_rect, disp, X, Y, Z, out_path, title=label)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('backend', choices=list(BACKENDS) + ['all'])
    ap.add_argument('bag_path', nargs='?', default=DEFAULT_BAG)
    ap.add_argument('--frame', type=int, default=None)
    ap.add_argument('--max-range', type=float, default=40.0,
                    help='Max stereo depth (m); 40 matches the wave node max_range')
    ap.add_argument('--out', default=None, help='Output PNG (single backend only)')
    ap.add_argument('--no-horizon-correct', action='store_true',
                    help='disable the online horizon-based roll/offset correction '
                         '(ros2_stereo_rectifier_horizon.py); use plain rectification')
    args = ap.parse_args()

    print('Reading bag…')
    left_bgr, right_bgr, left_info, right_info = collect_stereo_pair(args.bag_path, args.frame)
    print('Building rectification maps…')
    map_lx, map_ly, map_rx, map_ry, Q, _, rect_wh = build_rectification(left_info, right_info)
    print('Rectifying stereo pair…')
    left_rect, right_rect = rectify_pair(left_bgr, right_bgr, map_lx, map_ly, map_rx, map_ry)
    if not args.no_horizon_correct:
        left_rect, right_rect, hz = horizon_correct_pair(left_rect, right_rect)
        if hz is not None:
            print(f'  Horizon correction (differential): roll L={hz["roll_l"]:+.2f}° '
                  f'R={hz["roll_r"]:+.2f}° (diff {hz["roll_diff"]:+.2f}°), '
                  f'vert offset {hz["off"]:+.1f}px → right image aligned to left')
        else:
            print('  Horizon correction: no horizon detected, using plain rectification')

    fig_dir = os.path.join(_REPO, 'Relatorio/Report/figures')
    backends = list(BACKENDS) if args.backend == 'all' else [args.backend]
    for b in backends:
        out = args.out if (args.out and args.backend != 'all') \
            else os.path.join(fig_dir, f'backend_figure_{b}.png')
        _run_one(b, left_rect, right_rect, Q, rect_wh, args.max_range, out)


if __name__ == '__main__':
    main()
