"""
WAFT-Stereo disparity node.

Deep-learning stereo backend using WAFT-Stereo (https://github.com/princeton-vl/WAFT-Stereo,
warping-alone field transforms, no cost volume — ranked #1 on ETH3D/KITTI/Middlebury at
publication time). Drop-in replacement for ros2_hitnet_disparity.py /
ros2_raftstereo_disparity.py for the input topics and the horizon-based sky mask/debug
overlay, but NOT for the output resolution — see below.

The rectifier (ros2_stereo_rectifier.py) publishes at native camera resolution.
input_width/input_height/scale_factor below downscale the left/right pair to this
node's own working resolution for the matcher, same idea as every other disparity
backend in this repo — but unlike them, WAFT does NOT rescale the result back up to
native resolution before publishing on /stereo/disparity: at native res (e.g.
2464x2056) that upscale plus publishing a ~20 MB Image message every frame was the
dominant remaining cost once inference itself was already running on a small image.
/stereo/disparity from this node is therefore at the node's own working resolution,
not native — ros2_pointcloud_node.py has been updated to resize the native left image
down to match whatever resolution it actually receives instead of assuming the two are
always equal. Sky-mask/horizon detection still runs on the full native left image
(cheap — see stereo_common.HorizonMasker's own internal downscale), independent of
how aggressively the matcher's working resolution is set.

Setup (one-time):
    WAFT-Stereo is already vendored at WAFT-Stereo/ in the project root. You only need
    a config+checkpoint pair from https://huggingface.co/MemorySlices/WAFT-Stereo/tree/main
    — the three SynLarge presets trade VRAM/speed for quality (see 'waft_repo_path' /
    'config_file' / 'ckpt' below). Download e.g.:
        mkdir -p WAFT-Stereo/ckpts/SynLarge
        curl -L -o WAFT-Stereo/ckpts/SynLarge/DAv2S-4.pth \\
            https://huggingface.co/MemorySlices/WAFT-Stereo/resolve/main/SynLarge/DAv2S-4.pth
    This node also needs timm/peft/yacs in the same Python that runs rclpy (system
    python3.10 here, not the separate `waft-stereo` conda env used for the standalone
    demo) — installed via `pip install --user timm peft yacs`. torch/torchvision,
    einops, matplotlib and PyYAML were already present (shared with
    ros2_raftstereo_disparity.py / the rest of the ROS stack). xformers is
    intentionally NOT installed here: WAFT-Stereo's attention falls back to plain
    PyTorch attention when xformers is absent, which sidesteps an xformers bf16
    requirement (Ampere/compute-capability>=8.0 only) that breaks on older GPUs like
    the RTX 2060 (Turing, 7.5) — see the 'precision' parameter below.

Run (from repo root):
    python3 Nodes/ros2_waft_disparity.py --ros-args \\
        -p config_file:=WAFT-Stereo/configs/SynLarge/DAv2S-4.yaml \\
        -p ckpt:=WAFT-Stereo/ckpts/SynLarge/DAv2S-4.pth \\
        -p use_sim_time:=true

See the bottom of this file ("TUNING GUIDE") for what every parameter does and how to
trade off quality/speed/VRAM.
"""

import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
from cv_bridge import CvBridge

import cv2
import numpy as np
import torch

from stereo_common import (
    HorizonMasker, extract_baseline_fx, make_disparity_msg, make_color_msg, downscale_pair,
)

_PRECISION_DTYPES = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp32': torch.float32,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _colorize(disp_float, max_disp=None):
    """JET colormap; zero/invalid pixels stay black.
    max_disp auto-computed from the 95th percentile if not given (<=0)."""
    if max_disp is None or max_disp <= 0:
        valid = disp_float[disp_float > 0]
        max_disp = float(np.percentile(valid, 95)) if valid.size > 0 else 64.0
    scaled = np.clip(disp_float / max_disp * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    color[disp_float <= 0.0] = 0
    return color


# ── Node ───────────────────────────────────────────────────────────────────────

class WAFTDisparityNode(Node):
    def __init__(self):
        super().__init__('waft_disparity_node')

        self.declare_parameter('left_rect_topic',  '/stereo/left/image_rect')
        self.declare_parameter('right_rect_topic', '/stereo/right/image_rect')
        self.declare_parameter('rect_info_topic',  '/stereo/camera_info_rect')
        self.declare_parameter('disp_raw_topic',   '/stereo/disparity')
        self.declare_parameter('disp_color_topic', '/stereo/disparity_color/compressed')

        # WAFT-Stereo repo root + config/checkpoint pair (must match each other).
        self.declare_parameter('waft_repo_path', 'WAFT-Stereo')
        self.declare_parameter('config_file',    'WAFT-Stereo/configs/SynLarge/DAv2S-4.yaml')
        self.declare_parameter('ckpt',           'WAFT-Stereo/ckpts/SynLarge/DAv2S-4.pth')
        self.declare_parameter('device',         'cuda')   # 'cuda' or 'cpu'
        self.declare_parameter('merge_lora',     True)     # fuse LoRA adapters into base weights once at load time (faster per-frame, no quality change)

        # Inference-time speed/quality knobs — safe to change live via `ros2 param set`,
        # no restart needed (re-read every frame).
        self.declare_parameter('precision',        'fp16')     # 'fp16' | 'bf16' | 'fp32'
        # The rectifier publishes at native camera resolution. input_width/height (if
        # both > 0) downscale the left/right images on the CPU to this exact working
        # resolution *before* they're colour-converted/uploaded to the GPU — cheaper
        # than uploading the full native-res image and letting WAFT's own internal
        # F.interpolate downscale it afterwards (same compute, far less data moved per
        # frame). Set either to <= 0 to fall back to the fractional 'scale_factor' below
        # instead (applied relative to the incoming image's actual size). Unlike the
        # other disparity nodes, this is also the resolution /stereo/disparity gets
        # published at — there's no rescale-back-to-native step (see the module
        # docstring above), so ros2_pointcloud_node.py must tolerate this resolution.
        self.declare_parameter('input_width',      -1)
        self.declare_parameter('input_height',    -1)
        self.declare_parameter('scale_factor',     0.5)        # fractional fallback used only when input_width/height <= 0
        self.declare_parameter('infer_iters',      -1)         # recurrent refinement steps; -1 = checkpoint's trained count
        self.declare_parameter('hiera',            False)      # tiled hierarchical inference, for high-res input only
        self.declare_parameter('tile_height',      480)        # tile size for hiera mode (match training crop, e.g. 480x640 for SynLarge)
        self.declare_parameter('tile_width',       640)
        self.declare_parameter('hiera_factors',    '0.5,1.0')  # comma-separated coarse->fine scale passes, used only when hiera=true
        self.declare_parameter('remove_invisible', True)       # zero disparity where the match would fall outside the right image

        # Colour preview
        self.declare_parameter('color_max_disp', 0.0)  # 0 = auto (95th percentile/frame); >0 = fixed pixel scale for stable comparisons
        # The raw 32FC1 /stereo/disparity publish (point cloud input) stays native
        # resolution — only the JPEG preview/debug images below are downscaled, since
        # colourising + cv2.imencode at full ~2464x2056 resolution is pure RViz-viewing
        # cost paid every frame for no benefit. <= 0 disables (publish at native res).
        self.declare_parameter('preview_max_dim', 640)

        # Sky/horizon masking — identical to the other disparity nodes
        self.declare_parameter('sky_crop_pct',       0.40)
        self.declare_parameter('horizon_margin_pct', 0.01)
        self.declare_parameter('debug_horizon',      True)

        p = self.get_parameter

        self._bridge       = CvBridge()
        self._model        = None
        self._device       = None
        self._model_ready  = False
        self._default_iters = None
        self._warned_cpu_fp16 = False

        self._horizon = HorizonMasker(
            fallback_crop_pct=p('sky_crop_pct').value,
            horizon_margin_pct=p('horizon_margin_pct').value,
        )

        pub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=5)
        vis_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=5)

        self._pub_disp_raw   = self.create_publisher(Image,           p('disp_raw_topic').value,   pub_qos)
        self._pub_disp_color = self.create_publisher(CompressedImage, p('disp_color_topic').value, vis_qos)
        self._pub_debug       = self.create_publisher(CompressedImage, '/stereo/debug/horizon/compressed', vis_qos)

        self._sub_info = self.create_subscription(
            CameraInfo, p('rect_info_topic').value, self._cb_camera_info, pub_qos
        )
        self.get_logger().info('WAFT-Stereo node waiting for rectified CameraInfo…')

        self._sync_qos = pub_qos
        self._sub_left = self._sub_right = self._sync = None
        self._last_msg_stamp: float | None = None
        self._last_frame_time: float       = time.monotonic()
        self._build_sync()
        self.create_timer(3.0, self._cb_watchdog)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _resolve_path(self, path: str) -> str:
        """Resolve a path that may be relative to the project root."""
        if os.path.isabs(path):
            return path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, path)

    def _load_model(self):
        p = self.get_parameter
        repo_dir = self._resolve_path(p('waft_repo_path').value)
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

        from bridgedepth.config import get_cfg  # noqa: PLC0415
        from algorithms.waft import WAFT         # noqa: PLC0415
        from peft import PeftModel               # noqa: PLC0415

        cfg = get_cfg()
        cfg.merge_from_file(self._resolve_path(p('config_file').value))
        cfg.freeze()

        want_device = p('device').value
        if want_device == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn('CUDA requested but not available — falling back to CPU.')
            want_device = 'cpu'
        self._device = torch.device(want_device)
        self.get_logger().info(f'WAFT-Stereo device: {self._device}')

        # DAv2Encoder looks up 'depth-anything-ckpts/<arch>.pth' relative to cwd, not
        # __file__ — chdir into the repo while constructing so that resolves correctly
        # regardless of where this node was launched from. (Harmless either way: any
        # checkpoint found there gets overwritten by our own state_dict load below;
        # this just avoids the "using random weights" log line when it IS present.)
        cwd_before = os.getcwd()
        try:
            os.chdir(repo_dir)
            model = WAFT(cfg)
        finally:
            os.chdir(cwd_before)

        model.eval()
        model = model.to(self._device)
        #model = torch.compile(model, dynamic=False)
        
        ckpt_path = self._resolve_path(p('ckpt').value)
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        weights = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        model.load_state_dict(weights, strict=False)

        if p('merge_lora').value:
            for _, module in model.named_modules():
                if isinstance(module, PeftModel):
                    module.merge_and_unload()

        self._model         = model
        self._default_iters = model.iters
        self.get_logger().info(
            f'WAFT-Stereo checkpoint loaded: {ckpt_path} (trained iters={self._default_iters})'
        )

    # ── Sync management (identical pattern to ros2_raftstereo_disparity.py) ────

    def _build_sync(self):
        p = self.get_parameter
        if self._sub_left is not None:
            self.destroy_subscription(self._sub_left.sub)
        if self._sub_right is not None:
            self.destroy_subscription(self._sub_right.sub)
        self._sub_left  = message_filters.Subscriber(
            self, Image, p('left_rect_topic').value,  qos_profile=self._sync_qos)
        self._sub_right = message_filters.Subscriber(
            self, Image, p('right_rect_topic').value, qos_profile=self._sync_qos)
        self._sync = message_filters.TimeSynchronizer(
            [self._sub_left, self._sub_right], queue_size=10)
        self._sync.registerCallback(self._cb_images)

    def _reset_sync(self, reason: str):
        self.get_logger().warn(f'WAFT-Stereo: resetting sync — {reason}')
        self._build_sync()
        self._last_frame_time = time.monotonic()
        self._last_msg_stamp  = None

    def _cb_watchdog(self):
        elapsed = time.monotonic() - self._last_frame_time
        if elapsed > 3.0:
            self._reset_sync(f'no frame for {elapsed:.1f}s')

    # ── Camera info ───────────────────────────────────────────────────────────

    def _cb_camera_info(self, msg: CameraInfo):
        if self._model_ready:
            return
        baseline, fx = extract_baseline_fx(msg)
        self.get_logger().info(f'Baseline: {baseline:.4f} m  fx: {fx:.2f} px')
        try:
            self._load_model()
            self._model_ready = True
            self.get_logger().info('WAFT-Stereo inference engine online.')
            self.destroy_subscription(self._sub_info)
        except Exception as exc:
            self.get_logger().error(f'Failed to load WAFT-Stereo: {exc}')

    # ── Inference ─────────────────────────────────────────────────────────────

    def _to_tensor(self, img_bgr):
        """BGR uint8 -> float32 RGB tensor [1, 3, H, W], range 0-255 (WAFT normalizes internally)."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).float()[None]

    def _working_size(self, orig_w: int, orig_h: int) -> tuple[int, int]:
        """The (width, height) the matcher runs at — see input_width/input_height/
        scale_factor. This is also the resolution /stereo/disparity is published at
        (no rescale-back-to-native step), so changing this changes that topic's size."""
        p = self.get_parameter
        inp_w = p('input_width').value
        inp_h = p('input_height').value
        if inp_w > 0 and inp_h > 0:
            return inp_w, inp_h
        sf = p('scale_factor').value
        return max(1, int(round(orig_w * sf))), max(1, int(round(orig_h * sf)))

    def _infer_disparity(self, left_img, right_img) -> np.ndarray:
        """Returns float32 disparity array (pixels, positive), same H×W as the inputs
        (caller is responsible for pre-downscaling left_img/right_img to the desired
        working resolution — see _working_size — except in hiera mode, which expects
        the real native-resolution image and tiles internally)."""
        p = self.get_parameter

        infer_iters = p('infer_iters').value
        self._model.iters = self._default_iters if infer_iters <= 0 else int(infer_iters)

        dtype = _PRECISION_DTYPES.get(p('precision').value, torch.float16)
        if self._device.type == 'cpu' and dtype == torch.float16:
            if not self._warned_cpu_fp16:
                self.get_logger().warn("precision='fp16' is not supported on CPU autocast — using fp32 instead.")
                self._warned_cpu_fp16 = True
            dtype = torch.float32
        autocast_enabled = dtype != torch.float32

        img1 = self._to_tensor(left_img).to(self._device)
        img2 = self._to_tensor(right_img).to(self._device)
        sample = {'img1': img1, 'img2': img2}

        with torch.inference_mode():
            with torch.autocast(device_type=self._device.type, dtype=dtype, enabled=autocast_enabled):
                if p('hiera').value:
                    factors = [float(x) for x in p('hiera_factors').value.split(',') if x.strip()]
                    output = self._model.heirarchical_inference(
                        sample,
                        size=(p('tile_height').value, p('tile_width').value),
                        factor_list=factors,
                    )
                else:
                    output = self._model.inference(sample, size=None, factor=1.0)

        disp_np = output['disp_pred'][0].float().cpu().numpy().astype(np.float32)
        disp_np = np.maximum(disp_np, 0.0)

        h, w = left_img.shape[:2]
        if disp_np.shape != (h, w):
            # Padding/rounding inside the model's own resize path can land a few
            # rows/cols short of the input size — not a real resolution change.
            disp_np = cv2.resize(disp_np, (w, h), interpolation=cv2.INTER_LINEAR)

        if p('remove_invisible').value:
            # Pixels whose match would land left of the right image's edge are
            # unverifiable matches, not "far away" — zero them out (0 = invalid,
            # matching the convention ros2_pointcloud_node.py expects), rather than
            # demo.py's np.inf (which would mean "infinitely far" downstream instead).
            xx = np.arange(w, dtype=np.float32)[None, :]
            us_right = xx - disp_np
            disp_np[us_right < 0] = 0.0

        return disp_np

    # ── Preview helper ────────────────────────────────────────────────────────

    @staticmethod
    def _shrink(img, max_dim):
        """Downscale img (if needed) so its longer side is at most max_dim. Used only
        for the colour preview / debug overlay (normally a no-op for the preview now
        that disparity_map is already at the matcher's small working resolution) —
        purely to cut JPEG-encode cost on images nobody reads at full resolution
        anyway (RViz/rqt viewing)."""
        if max_dim <= 0:
            return img
        h, w = img.shape[:2]
        if max(h, w) <= max_dim:
            return img
        s = max_dim / max(h, w)
        new_w, new_h = max(1, int(round(w * s))), max(1, int(round(h * s)))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ── Main callback ─────────────────────────────────────────────────────────

    def _cb_images(self, left_msg: Image, right_msg: Image):
        if not self._model_ready:
            return

        stamp_sec = left_msg.header.stamp.sec + left_msg.header.stamp.nanosec * 1e-9
        if self._last_msg_stamp is not None and stamp_sec < self._last_msg_stamp - 1.0:
            self._reset_sync(f'time jump {self._last_msg_stamp:.1f}→{stamp_sec:.1f}s')
            return
        self._last_msg_stamp  = stamp_sec
        self._last_frame_time = time.monotonic()

        # Per-stage timestamps for the breakdown logged at the bottom — cheap
        # (time.perf_counter() calls only), kept in permanently since "where did the
        # time go" comes up every time this node's speed is tuned.
        marks = [('t0', time.perf_counter())]

        def mark(name):
            marks.append((name, time.perf_counter()))

        left_cv  = self._bridge.imgmsg_to_cv2(left_msg,  desired_encoding='bgr8')
        right_cv = self._bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
        mark('decode')

        # Decide the matcher's working resolution up front (needed below for both the
        # sky mask and the inference downscale) — hiera mode wants the real
        # native-resolution image and tiles internally, every other mode runs the
        # matcher (and therefore /stereo/disparity's published resolution) at a
        # downscaled size, see module docstring / input_width/input_height/scale_factor.
        hiera = self.get_parameter('hiera').value
        if hiera:
            mask_shape = None
            debug_scale = 1.0
        else:
            small_w, small_h = self._working_size(left_cv.shape[1], left_cv.shape[0])
            mask_shape = (small_h, small_w)
            debug_scale = small_w / left_cv.shape[1]

        # 1. Detect horizon + build sky mask. Detection always runs on the native
        # image (cheap — see stereo_common.HorizonMasker's own internal downscale —
        # and kept independent of the matcher's working resolution, so a very small
        # scale_factor doesn't degrade horizon-line detection quality), but the mask
        # itself is built directly at mask_shape — building it at native resolution
        # just to immediately downsize it cost ~90ms/frame (see compute_mask docstring).
        sky_mask, source = self._horizon.compute_mask(left_cv, mask_shape=mask_shape)
        mark('horizon')

        # 2. Run disparity inference at the matcher's own working resolution.
        # downscale_pair uses INTER_LINEAR, same as every other disparity backend's
        # pre-matcher downscale in this repo — cheaper than INTER_AREA, and the
        # anti-aliasing difference doesn't matter for a learned matcher's input.
        if hiera:
            work_left, work_right = left_cv, right_cv
        else:
            work_left, work_right = downscale_pair(left_cv, right_cv, small_w, small_h)
        mark('downscale')

        disparity_map = self._infer_disparity(work_left, work_right)
        mark('infer')

        # 3. Zero out sky (mask already matches disparity_map's shape — see step 1).
        disparity_map[sky_mask] = 0.0
        mark('mask')

        # 4. Publish raw disparity (32FC1, pixels) at its own working resolution — see
        # module docstring for why this node, unlike the others, doesn't rescale back
        # up to native before publishing.
        disp_msg = make_disparity_msg(self._bridge, disparity_map, left_msg.header)
        self._pub_disp_raw.publish(disp_msg)
        mark('pub_raw')

        # 5. Publish colourised preview — downscaled further only if still bigger than
        # preview_max_dim (normally a no-op now that disparity_map is already small).
        # Sky is already zero in disparity_map (step 4), and _colorize treats <=0 as
        # invalid/black, so no separate sky_mask indexing is needed on the small copy.
        preview_dim = self.get_parameter('preview_max_dim').value
        small_disp = self._shrink(disparity_map, preview_dim)
        color_img = _colorize(small_disp, max_disp=self.get_parameter('color_max_disp').value)
        color_msg = make_color_msg(color_img, left_msg.header)
        if color_msg:
            self._pub_disp_color.publish(color_msg)
        mark('preview')

        # 6. Publish horizon debug overlay — drawn directly on work_left (the matcher's
        # working-resolution image, already decoded above) instead of the native image
        # + downscale-after: same cheap copy/draw/encode benefit as the colour preview,
        # without paying for a full native-resolution .copy() + JPEG encode first.
        # make_debug_image's scale= rescales the (native-coordinate) horizon line to
        # match work_left's resolution; direction/angle is scale-invariant.
        if self.get_parameter('debug_horizon').value:
            dbg = self._horizon.make_debug_image(work_left, source, scale=debug_scale)
            if dbg is not None:
                dbg = self._shrink(dbg, preview_dim)
                dbg_msg = make_color_msg(dbg, left_msg.header)
                if dbg_msg:
                    self._pub_debug.publish(dbg_msg)
        mark('debug')

        total_ms = (marks[-1][1] - marks[0][1]) * 1000
        breakdown = ' '.join(
            f'{name}={(t1 - t0) * 1000:.1f}'
            for (_, t0), (name, t1) in zip(marks, marks[1:])
        )
        self.get_logger().info(
            f'WAFT-Stereo | [{source}] | iters={self._model.iters} | '
            f'total={total_ms:.1f}ms | {breakdown}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = WAFTDisparityNode()
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


# ── TUNING GUIDE ─────────────────────────────────────────────────────────────
#
# Restart-required (load-time) parameters:
#
#   waft_repo_path, config_file, ckpt
#       A config and checkpoint must come from the same training run. The three
#       SynLarge presets in WAFT-Stereo/configs/SynLarge/ trade quality for
#       VRAM/speed (all use the same ViT-based DepthAnythingV2 encoder, just a
#       different backbone size):
#         DAv2S-4.yaml / DAv2S-4.pth   vits, ~748 MB  — fastest, fits 6 GB VRAM (RTX 2060)
#         DAv2B-4.yaml / DAv2B-4.pth   vitb, ~1.13 GB — balanced
#         DAv2L-5.yaml / DAv2L-5.pth   vitl, ~2.13 GB — best quality, may not fit 6 GB VRAM
#       To compare backends, change config_file+ckpt together and restart the node.
#       Architecture knobs inside the yaml (LORA_RANK, ARCH, ITERATIVE_MODULE...) are
#       NOT exposed as ROS parameters — they're baked into the checkpoint's weight
#       shapes, so changing them without retraining will break `load_state_dict`.
#
#   device
#       'cuda' (default) or 'cpu'. Falls back to CPU automatically with a warning if
#       CUDA isn't available. CPU inference will be far slower than even SBM/SGBM.
#
#   merge_lora
#       Fuses each LoRA adapter into the base encoder weights once, at load time.
#       Pure latency optimization (skips an extra matmul per attention block on every
#       frame) — leave True unless you're debugging adapter behaviour itself.
#
# Live-tunable (re-read every frame, change anytime via `ros2 param set
# /waft_disparity_node <name> <value>`, no restart needed):
#
#   precision  ('fp16' | 'bf16' | 'fp32')
#       Autocast dtype for the forward pass. 'bf16' requires Ampere or newer
#       (compute capability >= 8.0) — on Turing cards (RTX 2060/2070/2080, capability
#       7.5) it will silently underperform or error depending on what ops run, so use
#       'fp16' there instead (full tensor-core speed, no precision-related GPU
#       requirement). 'fp32' disables autocast entirely — slowest, use only to rule out
#       precision as the cause of a visibly wrong disparity map.
#
#   input_width, input_height  (<= 0 disables, falls back to scale_factor below)
#       The rectifier (ros2_stereo_rectifier.py) always publishes at the camera's
#       native resolution (e.g. 2464x2056). When both are > 0, the left/right images
#       are resized with cv2 to exactly this resolution *before* colour-conversion/GPU
#       upload and inference (factor=1.0 — no further internal resize). This is also
#       the resolution /stereo/disparity gets published at — unlike every other
#       disparity backend in this repo, WAFT does NOT resize the result back up to
#       native (see the module docstring for why: at native res that upscale plus
#       publishing a ~20 MB Image every frame was the dominant remaining cost).
#       ros2_pointcloud_node.py resizes the native left image down to match whatever
#       resolution it actually receives, so this is safe to change live. Bigger values
#       keep more fine detail (small waves, foam texture) at a quadratic cost in
#       runtime *and* in every downstream cost that scales with pixel count (sky mask
#       resize, JPEG preview, point-cloud reprojection); smaller are faster but
#       blurrier/sparser. Need not match the native aspect ratio (each axis is resized
#       independently); set both to -1 to fall back to scale_factor below instead.
#
#   scale_factor  (0 < f <= 1.0, default 0.1; only used when input_width/height <= 0)
#       Fractional alternative to input_width/input_height — working resolution (and
#       therefore /stereo/disparity's published resolution) is round(native_dim *
#       scale_factor) per axis, same CPU-resize-first behaviour. Same tradeoff.
#
#   infer_iters  (-1 or a positive int, default -1)
#       Number of recurrent disparity-refinement passes. -1 uses the checkpoint's
#       trained count (3 for every SynLarge config). The refinement module is a
#       shared/recurrent block, so it's valid to run it for *more* steps than it was
#       trained for at test time (the same trick RAFT-Stereo's `valid_iters` parameter
#       uses) — try 4-6 if edges look soft and you can spare the latency; iters below
#       2-3 will look noticeably blockier. Each step costs roughly one extra encoder
#       pass's worth of compute.
#
#   hiera, tile_height, tile_width, hiera_factors
#       Tiled hierarchical inference for high-resolution input (the WAFT-Stereo README
#       recommends this above ~1080p): splits the image into overlapping tiles sized
#       tile_height x tile_width (defaults 480x640, matching what SynLarge was trained
#       on), runs a coarse pass at hiera_factors[0], then progressively finer passes
#       seeded by the previous pass's disparity. When True, this node feeds the real
#       native-resolution image (input_width/input_height/scale_factor are ignored —
#       see _cb_images) and /stereo/disparity is published at native resolution too,
#       same as the other disparity backends. Leave hiera=False at WAFT's normal small
#       working resolution — at that resolution tiling only adds overhead for no
#       benefit; turn it on only if you actually want native-resolution disparity from
#       WAFT (accepting the native-res publish cost this node otherwise avoids).
#
#   remove_invisible
#       When True, zeroes disparity for pixels near the left edge whose stereo match
#       would fall outside the right image (an unverifiable, not just distant, match).
#       Turn off only if you want to see the model's raw extrapolation there (e.g. to
#       judge how much of the frame is actually being thrown away near the rig's edge
#       occlusion zone).
#
#   color_max_disp
#       0 (default) auto-scales the JET colour preview to each frame's own 95th
#       percentile disparity — convenient for general viewing, but means the same
#       colour means different distances frame to frame. Set a fixed pixel value
#       (e.g. whatever you observe via `ros2 topic echo /stereo/disparity` on a calm
#       sea) if you want colours to be comparable across frames or against another
#       disparity backend's preview side by side in RViz.
#
#   preview_max_dim  (default 640; <= 0 disables, publishes preview/debug at whatever
#   resolution they were built at)
#       Caps the longer side of the colour disparity preview and the horizon debug
#       overlay before JPEG-encoding them. The colour preview is normally already at
#       WAFT's small working resolution (this is a no-op for it in that case) — the
#       debug overlay is still drawn on the full native left image (horizon detection
#       quality reasons, see _cb_images), so this is what keeps *that* one cheap to
#       encode. Matters most if input_width/input_height/scale_factor or hiera mode
#       push the working/published resolution back up near native.
#
#   sky_crop_pct, horizon_margin_pct, debug_horizon
#       Same Hough-line horizon detector as every other disparity node in this repo
#       (see stereo_common.HorizonMasker) — tune these identically: sky_crop_pct only
#       matters as the fallback crop row before any horizon has ever been detected;
#       horizon_margin_pct nudges the mask boundary further into the sky (raise it if
#       wave crests near the horizon are getting zeroed out as "sky"); debug_horizon
#       publishes the yellow/red overlay on /stereo/debug/horizon/compressed for
#       visually checking both lines in RViz.
