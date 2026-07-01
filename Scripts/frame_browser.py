"""
Frame browser — helps pick which frame to use for each backend figure.

Produces two things for a bag:
  1. A contact sheet of the rectified LEFT camera frames, each labelled with its
     frame index (so you can eyeball which frames have nice, well-lit waves).
     Saved to Relatorio/Report/figures/_frame_contact_sheet.png (gitignore-able).
  2. A printed table of near-field (<40 m) point counts for the classical backends
     (StereoBM / StereoSGBM / SGM-CUDA) at each sampled frame — the learned
     backends reconstruct essentially every frame, so frame choice for them is
     purely aesthetic (use the contact sheet).

Usage (from repo root):
    python3 Scripts/frame_browser.py [bag_path] [--stride N] [--max-frame M]
                                     [--cols C] [--no-counts]

Then, once you've chosen a frame index per backend, regenerate each figure with:
    python3 Scripts/backend_offline_figure.py <backend> --frame <N>
(each backend writes only its own backend_figure_<backend>.png).
"""

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_SCRIPTS)
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'Nodes'))

from waft_offline_figure import (  # noqa: E402
    DEFAULT_BAG, read_topic_messages, _decode_compressed, build_rectification,
    rectify_pair, reproject, LEFT_IMG, RIGHT_IMG, LEFT_INFO, RIGHT_INFO,
)
import backend_offline_figure as B  # noqa: E402


def collect_frames(bag, stride, max_frame):
    """Return (info_l, info_r, {frame_index: (left_bgr, right_bgr)})."""
    targets = set(range(0, max_frame + 1, stride))
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
            rbuf.append((ts, msg))
            rbuf = rbuf[-4:]
        elif topic == LEFT_IMG:
            if idx in targets and rbuf and li is not None and ri is not None:
                nn = min(rbuf, key=lambda x: abs(x[0] - ts))
                pairs[idx] = (_decode_compressed(msg), _decode_compressed(nn[1]))
            idx += 1
            if idx > max_frame and len(pairs) == len(targets):
                break
    return li, ri, pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag_path', nargs='?', default=DEFAULT_BAG)
    ap.add_argument('--stride', type=int, default=10, help='sample every Nth frame')
    ap.add_argument('--max-frame', type=int, default=400)
    ap.add_argument('--cols', type=int, default=6)
    ap.add_argument('--no-counts', action='store_true',
                    help='skip the classical point-count table (faster)')
    args = ap.parse_args()

    print('Reading bag…')
    li, ri, pairs = collect_frames(args.bag_path, args.stride, args.max_frame)
    print(f'  {len(pairs)} frames sampled (stride {args.stride})')
    mlx, mly, mrx, mry, Q, _, rect_wh = build_rectification(li, ri)

    frames = sorted(pairs)
    rects = {}
    for f in frames:
        l, r = pairs[f]
        rects[f] = rectify_pair(l, r, mlx, mly, mrx, mry)

    # ── Contact sheet ────────────────────────────────────────────────────────
    cols = args.cols
    rows = (len(frames) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.2))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis('off')
    for i, f in enumerate(frames):
        lr, _ = rects[f]
        axes[i].imshow(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f'frame {f}', fontsize=9)
    plt.tight_layout()
    out = os.path.join(_REPO, 'Relatorio/Report/figures/_frame_contact_sheet.png')
    plt.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'Contact sheet → {out}')

    # ── Classical point-count table ──────────────────────────────────────────
    if not args.no_counts:
        print('\nNear-field (<40 m) point counts — classical backends:')
        print(f'{"frame":>6} {"sbm":>7} {"sgbm":>7} {"sgm_cuda":>9}')
        for f in frames:
            lr, rr = rects[f]
            row = {}
            for nm, fn in (('sbm', B.disparity_sbm), ('sgbm', B.disparity_sgbm),
                           ('sgm_cuda', B.disparity_sgm_cuda)):
                d = fn(lr, rr)
                X, _, _ = reproject(d, Q.astype(np.float32), rect_wh,
                                    (d.shape[1], d.shape[0]), max_range=40.0)
                row[nm] = len(X)
            print(f'{f:>6} {row["sbm"]:>7} {row["sgbm"]:>7} {row["sgm_cuda"]:>9}')
        print('\n(learned backends reconstruct ~every frame — pick those from the '
              'contact sheet by wave appearance)')

    print('\nNext: python3 Scripts/backend_offline_figure.py <backend> --frame <N>')


if __name__ == '__main__':
    main()
