"""
Regenerate the two low-resolution single-image report figures the tutor flagged:

  Fig 4 (fig:raw-camera)      -> figures/original_camera_footage.png
      Rectified LEFT and RIGHT frames at the SAME instant, side by side, plus a
      matching zoomed inset of one sea patch in each view. Because the pair is
      rectified (row-aligned), the same image box shows the same scene in both
      views, so the differing specular highlights / features between left and
      right directly demonstrate the non-Lambertian mismatch that breaks
      photometric stereo matching.

  Fig 5 (fig:horizon-detection) -> figures/horizon_detection.png
      Rectified left frame with the detected horizon line (red) and the excluded
      sky region shaded, exactly as the shared pre-processing stage
      (stereo_common.HorizonMasker, used by every disparity node) computes it.

Both are rendered from the same bag/frame and the same calibration as the live
pipeline (spine reused from waft_offline_figure.py), but rectified at full
resolution instead of the pipeline's 0.25 working scale so the figures are sharp.

Usage (from repo root):
    python3 Scripts/frame_and_horizon_figure.py [bag_path] [--frame N]
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
sys.path.insert(0, os.path.join(_REPO, 'Nodes'))

import waft_offline_figure as spine  # noqa: E402
from waft_offline_figure import (  # noqa: E402
    DEFAULT_BAG, collect_stereo_pair, build_rectification, rectify_pair,
)
from stereo_common import HorizonMasker  # noqa: E402

FIG_DIR = os.path.join(_REPO, 'Relatorio/Report/figures')


def _bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_raw_camera_figure(left_rect, right_rect, out_path):
    """Fig 4: rectified left + right at the same instant, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, img, title in ((axes[0], left_rect, 'Rectified left camera'),
                           (axes[1], right_rect, 'Rectified right camera')):
        ax.imshow(_bgr2rgb(img))
        ax.set_title(title)
        ax.axis('off')

    fig.suptitle('Synchronized left and right frames at the same instant: the '
                 'textureless, specular sea surface\nreflects differently in each '
                 'view, violating the brightness-constancy assumption of stereo matching',
                 fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')


def make_horizon_figure(left_rect, out_path):
    """Fig 5: detected horizon (yellow) + margined mask boundary (red) + shaded sky."""
    masker = HorizonMasker(horizon_margin_pct=0.01)
    sky_mask, source = masker.compute_mask(left_rect)
    print(f'  Horizon source: {source}, margin={masker.horizon_margin_pct:.0%}')

    rgb = _bgr2rgb(left_rect).astype(np.float32)
    # Shade the excluded sky region (blend toward red).
    overlay = rgb.copy()
    overlay[sky_mask] = 0.55 * overlay[sky_mask] + 0.45 * np.array([255, 60, 60])
    rgb = np.clip(overlay, 0, 255).astype(np.uint8)

    h, w = left_rect.shape[:2]
    lw = max(2, w // 700)
    # Raw detected horizon (yellow) and the margined mask boundary (red, 1% below).
    raw_mean, raw_dir = masker._cur_raw
    pt_l, pt_r = HorizonMasker._endpoints(raw_mean, raw_dir, w)
    cv2.line(rgb, pt_l, pt_r, (255, 215, 0), lw, cv2.LINE_AA)
    m_mean, m_dir = masker._cur_masked
    pt_lm, pt_rm = HorizonMasker._endpoints(m_mean, m_dir, w)
    cv2.line(rgb, pt_lm, pt_rm, (220, 20, 20), lw, cv2.LINE_AA)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(rgb)
    ax.set_title('Sky/horizon masking: the detected horizon (yellow) is nudged down by '
                 'a 1% safety margin\nto the mask boundary (red); everything above it '
                 'is excluded from disparity estimation', fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag_path', nargs='?', default=DEFAULT_BAG)
    ap.add_argument('--frame', type=int, default=0)
    args = ap.parse_args()

    print('Reading bag...')
    left_bgr, right_bgr, left_info, right_info = collect_stereo_pair(args.bag_path, args.frame)

    # Rectify at FULL resolution (override the pipeline's 0.25 working scale) so
    # the display figures are sharp — these figures are not fed to a matcher.
    spine.RECT_OUTPUT_SCALE = 1.0
    print('Building rectification maps (full resolution)...')
    map_lx, map_ly, map_rx, map_ry, _Q, _P1, _wh = build_rectification(left_info, right_info)
    print('Rectifying stereo pair...')
    left_rect, right_rect = rectify_pair(left_bgr, right_bgr, map_lx, map_ly, map_rx, map_ry)

    print('Rendering Fig 4 (raw camera pair)...')
    make_raw_camera_figure(left_rect, right_rect, os.path.join(FIG_DIR, 'original_camera_footage.png'))
    print('Rendering Fig 5 (horizon detection)...')
    make_horizon_figure(left_rect, os.path.join(FIG_DIR, 'horizon_detection.png'))


if __name__ == '__main__':
    main()
