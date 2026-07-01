"""
Stereo rectifier with online horizon-based roll correction — drop-in replacement.

The static calibration (K/D/R_stereo/T_stereo) is unchanged and still assumed
correct, but the cameras are subject to VIBRATION, which adds a small, time-varying
in-plane rotation (roll about the optical axis) to each camera independently. A static
rectification cannot remove a per-frame, per-camera rotation, so the horizon — a line
at infinity that must appear perfectly horizontal AND on the same row in both rectified
images — ends up tilted by a fraction of a degree that changes frame to frame, and the
two cameras' tilts differ (see the ~1.1 deg horizon-roll finding in CLAUDE.md /
memory `stereo-rectification-defect`). That residual roll defeats horizontal-search
disparity (StereoBM in particular) and biases every backend.

This node fixes it ONLINE using the horizon itself as the calibration reference (the
tutor's suggestion — the horizon is useful for online inter-camera calibration):

  1. Standard undistort + rectify (identical to ros2_stereo_rectifier.py: same maps,
     same R_stereo/T_stereo, same P1/P2/Q, so downstream depth/baseline are unchanged).
  2. Detect the sea/sky horizon line in EACH rectified image (Canny + Hough, reusing
     stereo_common.HorizonMasker — the same detector the disparity nodes already use).
  3. Measure each camera's horizon roll angle and the row it crosses image centre.
  4. Correct the residual stereo misalignment (see the two modes below).

WHY ALIGN RIGHT-TO-LEFT, NOT BOTH-TO-HORIZONTAL (mode, default 'differential'):
The measured absolute horizon roll is dominated by REAL boat roll, which is common to
both cameras (measured: L +5.3 deg / R +4.3 deg on one frame — a ~5 deg common roll on
top of the ~1 deg inter-camera difference). That common roll is a true attitude the
downstream cloud must keep: `/stereo/points` is expressed in `camera_left_rect` and then
placed in `map` via the nav TF, so silently levelling the LEFT image to absolute
horizontal would rotate the reconstructed surface away from the physical camera
orientation the pose chain assumes. What actually breaks epipolar matching is only the
DIFFERENTIAL roll+offset between the two cameras (the documented ~1.1 deg / ~7 px, which
is stable frame-to-frame). So by default this node leaves the LEFT rectified image
untouched (it stays the reference frame it already is) and rotates+shifts only the RIGHT
image so its horizon matches the left's tilt and row — removing exactly the inter-camera
error while preserving the true scene orientation and all depths.
  - mode='differential' (default): align right to left. Left untouched.
  - mode='absolute': level BOTH to true horizontal (also removes real boat roll). Only
    use this if you specifically want a gravity-levelled rectified pair and are NOT
    feeding the cloud through the nav-based TF (or are re-levelling downstream too).

The correction angle/offset is smoothed over time (EMA) so single-frame Hough noise
doesn't jitter the output, and is sanity-clamped (a bad detection is ignored, the last
good correction is held). If no horizon has ever been seen, the node behaves exactly
like the plain rectifier (identity correction).

Assumptions / scope:
  - Corrects the differential ROLL and vertical offset — the stereo-relevant,
    horizon-observable degrees of freedom. It does NOT correct pitch/yaw vibration
    (those move the horizon in common mode / out of plane and are not separable from
    scene geometry with a single horizon line), nor does it re-estimate intrinsics.
  - Needs a visible horizon. Over open sea that is the normal case; if the horizon is
    lost the last good correction is held (see horizon_hold_frames).

Deliberately a SEPARATE node so the original validated rectifier is untouched: run this
instead of ros2_stereo_rectifier.py in start_pipeline.sh (window `rectify`); revert by
switching back. Both publish the same node name and topics, so exactly one runs at a
time. It also supersedes ros2_stereo_rectifier_refined.py (a static one-shot rotation
tweak) — this handles the same roll error but time-varyingly, which is what vibration
actually needs.
"""

import numpy as np
import cv2
import rclpy

from ros2_stereo_rectifier import RectifyNode  # original node, unchanged
from stereo_common import HorizonMasker


def _horizon_angle_deg(direction):
    """In-plane roll of a horizon whose unit direction is `direction` (image coords,
    x right / y down). Returns degrees in (-90, 90], 0 == perfectly horizontal."""
    ang = np.degrees(np.arctan2(direction[1], direction[0]))
    # The line is undirected: fold to the nearest-horizontal representative.
    if ang > 90.0:
        ang -= 180.0
    elif ang <= -90.0:
        ang += 180.0
    return float(ang)


def _horizon_row_at(mean, direction, x):
    """y-coordinate where the horizon line crosses column `x`."""
    if abs(direction[0]) < 1e-9:
        return float(mean[1])
    t = (x - mean[0]) / direction[0]
    return float(mean[1] + t * direction[1])


class HorizonRectifyNode(RectifyNode):
    def __init__(self):
        super().__init__()

        # ── PARAMETERS ────────────────────────────────────────────────────────
        # Master switch: False makes this behave exactly like the plain rectifier.
        self.declare_parameter('horizon_correct', True)
        # 'differential' (default): align right to left, left untouched — corrects only
        # the inter-camera roll/offset, preserves real boat roll + cloud orientation.
        # 'absolute': level both cameras to true horizontal (removes boat roll too).
        # See the module docstring for why 'differential' is the safe default.
        self.declare_parameter('horizon_mode', 'differential')
        # EMA smoothing on the correction (0..1). Lower = smoother / slower to react.
        # Vibration roll is small and fast; a moderate value tracks it while rejecting
        # per-frame Hough jitter.
        self.declare_parameter('horizon_smooth_alpha', 0.25)
        # Also shift the right image vertically to match the left horizon row (fixes the
        # residual ~7-13 px vertical epipolar offset). Rotation-only if False.
        self.declare_parameter('horizon_align_vertical', True)
        # Reject a horizon whose implied DIFFERENTIAL roll exceeds this (deg) as a
        # mis-detection — the real inter-camera roll is ~1 deg, so a differential beyond
        # a few degrees is a spurious line. (In 'absolute' mode this bounds the absolute
        # roll instead; raise it if the boat rolls harder than this.)
        self.declare_parameter('horizon_max_roll_deg', 4.0)
        # Reject a differential vertical offset larger than this fraction of image
        # height (a bad detection can otherwise yank the whole image).
        self.declare_parameter('horizon_max_offset_pct', 0.15)
        # How many consecutive frames to keep applying the last good correction while
        # the horizon is lost before decaying back toward identity. 0 = hold forever.
        self.declare_parameter('horizon_hold_frames', 0)

        p = self.get_parameter
        self._hz_on         = p('horizon_correct').value
        self._hz_mode       = str(p('horizon_mode').value).lower()
        self._hz_alpha      = float(p('horizon_smooth_alpha').value)
        self._hz_align_vert = p('horizon_align_vertical').value
        self._hz_max_roll   = float(p('horizon_max_roll_deg').value)
        self._hz_max_off    = float(p('horizon_max_offset_pct').value)
        self._hz_hold       = int(p('horizon_hold_frames').value)
        self._hz_absolute   = self._hz_mode == 'absolute'

        # One detector per camera (each caches its own last-good horizon). detect at
        # the output resolution directly (it is already small, e.g. 616 px wide).
        self._masker_l = HorizonMasker(detect_max_dim=0)
        self._masker_r = HorizonMasker(detect_max_dim=0)

        # Smoothed correction state (None until first good horizon).
        # differential mode: _corr_l stays 0, _corr_r = (roll_r - roll_l).
        # absolute mode:     _corr_l = roll_l, _corr_r = roll_r.
        self._corr_l: float | None = None        # smoothed left roll to apply (deg)
        self._corr_r: float | None = None        # smoothed right roll to apply (deg)
        self._voff: float | None = None          # smoothed (row_l - row_r) offset (px)
        self._miss = 0                           # consecutive frames with no horizon

        if self._hz_on:
            self.get_logger().warn(
                f'Using HORIZON-CORRECTED rectifier (ros2_stereo_rectifier_horizon.py), '
                f'mode={self._hz_mode}: online roll/offset alignment from the horizon '
                f'line. Revert to ros2_stereo_rectifier.py to disable.')
        else:
            self.get_logger().warn(
                'ros2_stereo_rectifier_horizon.py loaded but horizon_correct=false — '
                'behaving as the plain rectifier.')

    # ── Horizon measurement + smoothing ──────────────────────────────────────

    def _measure_roll(self, masker, img_bgr):
        """Return (roll_deg, row_at_center) or None if no horizon was found."""
        raw = masker._find_raw(img_bgr)
        if raw is None:
            return None
        mean, direction = raw
        cx = img_bgr.shape[1] * 0.5
        return _horizon_angle_deg(direction), _horizon_row_at(mean, direction, cx)

    def _update_corrections(self, left_rect, right_rect):
        """Detect horizons on both rectified images and update the smoothed
        correction state. Returns True if a fresh measurement was folded in."""
        H = left_rect.shape[0]
        ml = self._measure_roll(self._masker_l, left_rect)
        mr = self._measure_roll(self._masker_r, right_rect)

        if ml is None or mr is None:
            self._miss += 1
            if self._hz_hold and self._miss > self._hz_hold and self._corr_l is not None:
                # Decay corrections back toward identity when the horizon stays lost.
                a = self._hz_alpha
                self._corr_l *= (1 - a)
                self._corr_r *= (1 - a)
                self._voff *= (1 - a)
            return False

        roll_l, row_l = ml
        roll_r, row_r = mr

        # In 'differential' mode the sanity gate is on the inter-camera roll (the thing
        # we correct); in 'absolute' mode it bounds each camera's own roll.
        if self._hz_absolute:
            if abs(roll_l) > self._hz_max_roll or abs(roll_r) > self._hz_max_roll:
                self._miss += 1
                return False
            corr_l, corr_r = roll_l, roll_r
        else:
            if abs(roll_r - roll_l) > self._hz_max_roll:
                self._miss += 1
                return False
            corr_l, corr_r = 0.0, (roll_r - roll_l)

        voff = row_l - row_r
        if abs(voff) > self._hz_max_off * H:
            voff = self._voff if self._voff is not None else 0.0

        self._miss = 0
        a = self._hz_alpha
        self._corr_l = corr_l if self._corr_l is None else (1 - a) * self._corr_l + a * corr_l
        self._corr_r = corr_r if self._corr_r is None else (1 - a) * self._corr_r + a * corr_r
        self._voff   = voff   if self._voff   is None else (1 - a) * self._voff   + a * voff
        return True

    def _level(self, img, roll_deg, extra_dy=0.0):
        """Rotate `img` in-plane by -roll (about its centre column, at the horizon
        row) to make the horizon horizontal, plus an optional vertical shift."""
        h, w = img.shape[:2]
        # Rotate about the image centre column; the exact pivot row only introduces a
        # sub-pixel horizontal shift for these <2 deg angles, negligible for disparity.
        cx, cy = w * 0.5, h * 0.5
        M = cv2.getRotationMatrix2D((cx, cy), roll_deg, 1.0)
        M[1, 2] += extra_dy
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # ── Override the publish path to insert the correction ────────────────────

    def _cb_images(self, left_msg, right_msg):
        if not self._maps_ok:
            return

        stamp_sec = left_msg.header.stamp.sec + left_msg.header.stamp.nanosec * 1e-9
        if self._last_msg_stamp is not None and stamp_sec < self._last_msg_stamp - 1.0:
            self._reset_sync(f'time jump {self._last_msg_stamp:.1f}→{stamp_sec:.1f}s')
            return
        self._last_msg_stamp = stamp_sec
        import time as _t
        self._last_frame_time = _t.monotonic()

        stamp = left_msg.header.stamp
        left_raw  = self._bridge.compressed_imgmsg_to_cv2(left_msg,  desired_encoding='bgr8')
        right_raw = self._bridge.compressed_imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

        left_rect  = cv2.remap(left_raw,  self._map_lx, self._map_ly, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_raw, self._map_rx, self._map_ry, cv2.INTER_LINEAR)

        if self._hz_on:
            self._update_corrections(left_rect, right_rect)
            if self._corr_l is not None:
                # Apply the roll correction to each camera (left is a no-op in
                # differential mode) and put the right horizon on the left's row by
                # folding the differential vertical offset into the right image.
                if abs(self._corr_l) > 1e-3:
                    left_rect = self._level(left_rect, self._corr_l)
                dy = self._voff if self._hz_align_vert else 0.0
                right_rect = self._level(right_rect, self._corr_r, extra_dy=dy)

        for img, pub in ((left_rect, self._pub_left), (right_rect, self._pub_right)):
            ros_img = self._bridge.cv2_to_imgmsg(img, encoding='bgr8')
            ros_img.header.stamp    = stamp
            ros_img.header.frame_id = 'camera_left_rect'
            pub.publish(ros_img)

        self._rect_info_msg.header.stamp = stamp
        self._pub_info.publish(self._rect_info_msg)


def main(args=None):
    rclpy.init(args=args)
    node = HorizonRectifyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
