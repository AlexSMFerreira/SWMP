"""
Shared helpers for the classical stereo disparity nodes (SBM / SGBM / CUDA-SGM).

Keeps the four disparity backends consistent: identical sky masking (so the
comparison against HITNet is fair), identical disparity message format
(32FC1, pixels), and identical colourised preview.

The HITNet node intentionally keeps its own inline copy of the horizon logic;
this module mirrors it so the classical nodes mask the sky the same way.
"""

import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage


# ── CameraInfo unpack ───────────────────────────────────────────────────────

def extract_baseline_fx(camera_info):
    """Returns (baseline_m, fx_px) using the same packing as the rectifier:
    P[0] = fx, and P2 (right projection) is stored ';'-joined in distortion_model.
    baseline = |-P2[3] / fx|."""
    fx = camera_info.p[0]
    p2 = [float(v) for v in camera_info.distortion_model.split(';')]
    baseline = abs(-p2[3] / fx)
    return baseline, fx


# ── Disparity scaling ───────────────────────────────────────────────────────

def to_float_disparity(raw_int16, min_disparity):
    """OpenCV matchers return CV_16S fixed-point disparity (x16). Convert to
    float pixels and zero-out invalid pixels (OpenCV marks them below the valid
    range), matching HITNet's 'disparity > 0' convention used downstream."""
    disp = raw_int16.astype(np.float32) / 16.0
    disp[disp < float(min_disparity)] = 0.0
    return disp


# ── Publishing ──────────────────────────────────────────────────────────────

def downscale_pair(left_bgr, right_bgr, width, height):
    """Resizes a rectified stereo pair down to a working resolution for the matcher.
    width/height <= 0 disables this (pair is returned unchanged) — used when a
    backend should run directly on the rectifier's native-resolution output."""
    if width <= 0 or height <= 0:
        return left_bgr, right_bgr
    return (cv2.resize(left_bgr,  (width, height), interpolation=cv2.INTER_LINEAR),
            cv2.resize(right_bgr, (width, height), interpolation=cv2.INTER_LINEAR))


def rescale_disparity(disp, target_wh):
    """Resizes a disparity map computed at its own (possibly downscaled) resolution
    up to target_wh = (width, height), scaling pixel values by the width ratio so
    they remain correct pixel-disparities at the new resolution. target_wh is
    normally the full rectified image size: ros2_pointcloud_node.py indexes the
    disparity and the left image elementwise, so their shapes must match exactly
    regardless of what resolution a given backend chose to run its matcher at."""
    h, w = disp.shape[:2]
    target_w, target_h = target_wh
    if (w, h) == (target_w, target_h):
        return disp
    scale = target_w / w
    resized = cv2.resize(disp, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return resized * scale


def make_disparity_msg(bridge, disp_float, header):
    """Builds a 32FC1 disparity Image with the matching step, stamped with the
    left-image header (keeps TF/time sync)."""
    msg = bridge.cv2_to_imgmsg(disp_float, encoding='32FC1')
    msg.header = header
    msg.step = disp_float.shape[1] * 4
    return msg


def colorize_disparity(disp_float, num_disparities):
    """Shared colour preview so every backend looks the same in rqt/RViz.
    Zero (invalid) stays black."""
    scaled = np.clip(disp_float / float(max(num_disparities, 1)) * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    color[disp_float <= 0.0] = 0
    return color


def make_color_msg(color_bgr, header):
    """Builds a jpeg CompressedImage preview."""
    success, buf = cv2.imencode('.jpg', color_bgr)
    if not success:
        return None
    msg = CompressedImage()
    msg.header = header
    msg.format = 'jpeg'
    msg.data = buf.tobytes()
    return msg


# ── Horizon detection / sky mask (mirrors ros2_hitnet_disparity.py) ─────────

class HorizonMasker:
    """Detects the sea/sky horizon via Hough lines and produces a boolean sky
    mask (True = sky, above the horizon). Caches the last good horizon and falls
    back to a fixed crop row on a cold start, exactly like the HITNet node."""

    def __init__(self, fallback_crop_pct=0.40, horizon_margin_pct=0.03, detect_max_dim=640):
        self.fallback_crop_pct = fallback_crop_pct
        self.horizon_margin_pct = horizon_margin_pct
        # The rectifier now publishes at native camera resolution (e.g. 2464x2056) —
        # Canny/Hough only need to see the horizon line's coarse shape, so detect on a
        # downscaled copy and scale the result back up. <= 0 disables (detect at native
        # res). The direction vector is scale-invariant; only the mean point needs
        # rescaling — see _find_raw.
        self.detect_max_dim = detect_max_dim
        self._last_masked = None   # (mean, direction) with margin baked in
        self._last_raw = None      # pre-nudge, for debug
        self._cur_raw = None       # horizon used on the most recent frame (raw)
        self._cur_masked = None    # horizon used on the most recent frame (nudged)

    def _find_raw(self, img_bgr):
        h, w = img_bgr.shape[:2]
        scale = 1.0
        if self.detect_max_dim > 0 and max(h, w) > self.detect_max_dim:
            scale = self.detect_max_dim / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))),
                                  interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        min_line_len = img_bgr.shape[1] * 0.2
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100,
            minLineLength=min_line_len, maxLineGap=50,
        )
        if lines is None:
            return None

        candidates = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            if angle < 20.0 or angle > 160.0:
                candidates.append((x1, y1, x2, y2))

        if not candidates:
            return None

        pts = np.array([[x1, y1] for x1, y1, _, _ in candidates] +
                       [[x2, y2] for _, _, x2, y2 in candidates], dtype=np.float32)
        mean = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - mean)
        if scale != 1.0:
            mean = mean / scale
        return mean, vt[0]

    def _apply_margin(self, horizon_raw, img_h):
        mean, direction = horizon_raw
        nudge = img_h * self.horizon_margin_pct
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        if normal[1] > 0:
            normal = -normal  # point toward sky (decreasing y)
        return mean + nudge * (-normal), direction

    def _sky_mask(self, shape, horizon_masked):
        h, w = shape[:2]
        mean, direction = horizon_masked
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        if normal[1] > 0:
            normal = -normal
        # dot(x,y) = normal[0]*(x-mean[0]) + normal[1]*(y-mean[1]) is separable —
        # build it via broadcasting of two 1-D terms instead of two full (h,w)
        # np.meshgrid arrays plus the dot itself; same result, fewer/smaller temp
        # allocations (matters here since this can run at native camera resolution).
        col_term = normal[0] * (np.arange(w, dtype=np.float32) - mean[0])
        row_term = normal[1] * (np.arange(h, dtype=np.float32) - mean[1])
        return (row_term[:, None] + col_term[None, :]) > 0

    def compute_mask(self, img_bgr, mask_shape=None):
        """Returns (sky_mask, source) where source is hough/cached/fallback.
        mask_shape (h, w), if given, builds the returned mask at this resolution
        instead of img_bgr.shape's. Useful when img_bgr is only needed at full
        resolution for horizon *detection* quality, but the mask itself is about to be
        applied to a smaller array (e.g. a disparity map computed at a downscaled
        working resolution) — paying for a full-resolution mask just to immediately
        downsize it is wasted work (this used to dominate ros2_waft_disparity.py's
        per-frame cost: ~90ms building a 2464x2056 mask that was thrown away a few
        lines later)."""
        h, w = img_bgr.shape[:2]
        raw = self._find_raw(img_bgr)

        if raw is not None:
            masked = self._apply_margin(raw, h)
            self._last_raw = raw
            self._last_masked = masked
            source = 'hough'
        elif self._last_masked is not None:
            raw = self._last_raw
            masked = self._last_masked
            source = 'cached'
        else:
            row = int(h * self.fallback_crop_pct)
            raw = (np.array([w / 2, row], dtype=np.float32),
                   np.array([1.0, 0.0], dtype=np.float32))
            masked = self._apply_margin(raw, h)
            source = 'fallback'

        self._cur_raw = raw
        self._cur_masked = masked

        if mask_shape is None:
            return self._sky_mask((h, w), masked), source

        # mean/direction are in img_bgr's coordinate space (native) — rescale the
        # point (direction/angle is scale-invariant) before building the mask at the
        # smaller target shape.
        mh, mw = mask_shape
        scale = mw / w
        mean, direction = masked
        return self._sky_mask((mh, mw), (mean * scale, direction)), source

    # ── Debug overlay (mirrors ros2_hitnet_disparity.py) ────────────────────

    @staticmethod
    def _endpoints(mean, direction, w):
        """Where the horizon line crosses x=0 and x=w-1."""
        if abs(direction[0]) > 1e-6:
            t_left = (0 - mean[0]) / direction[0]
            t_right = (w - 1 - mean[0]) / direction[0]
        else:
            t_left = t_right = 0.0
        pt_left = (0, int(mean[1] + t_left * direction[1]))
        pt_right = (w - 1, int(mean[1] + t_right * direction[1]))
        return pt_left, pt_right

    def make_debug_image(self, left_bgr, source, scale=1.0):
        """Draws the yellow raw-Hough horizon + red nudged mask boundary on a copy
        of left_bgr. Call after compute_mask() for the same frame. left_bgr may be at
        a smaller resolution than detection ran at (e.g. a disparity backend's own
        working-resolution image) — pass the same scale used for compute_mask's
        mask_shape (mask_w / detection_w) so the lines land in the right place instead
        of at native-resolution coordinates. Returns a BGR image (wrap with
        make_color_msg to publish)."""
        if self._cur_raw is None or self._cur_masked is None:
            return None
        debug = left_bgr.copy()
        w = debug.shape[1]

        raw_mean, raw_dir = self._cur_raw
        masked_mean, masked_dir = self._cur_masked
        if scale != 1.0:
            raw_mean = raw_mean * scale
            masked_mean = masked_mean * scale

        pt_l, pt_r = self._endpoints(raw_mean, raw_dir, w)
        cv2.line(debug, pt_l, pt_r, (0, 255, 255), 1, cv2.LINE_AA)        # yellow = raw

        pt_ln, pt_rn = self._endpoints(masked_mean, masked_dir, w)
        cv2.line(debug, pt_ln, pt_rn, (0, 0, 255), 2, cv2.LINE_AA)        # red = mask boundary

        cv2.putText(debug, f'horizon [{source}]', (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        return debug
