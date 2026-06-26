# Plan — Wave-Parameter Extraction Nodes

Two new ROS 2 nodes, each publishing **wave parameters** from an independent source so the
two can be cross-validated:

1. `ros2_altimeter_waves.py` — temporal spectrum from the **altimeters** (point measurement
   under the airship, sampled over time).
2. `ros2_pointcloud_waves.py` — spatial (and optionally temporal) analysis from the
   **stereo point cloud**, after fitting it to a robust plane (the local still-water
   surface). *(Implementation note: fixed-band RANSAC clips any wave taller than its
   inlier band — verified — so the node uses a self-tuning iterative **sigma-clipped
   least-squares** plane fit instead, which rejects only gross outliers while keeping the
   whole wave. Same goal — a robust mean-surface plane — done robustly without per-sea-
   state tuning.)*

Both are drop-in pipeline nodes following the existing conventions (parameterized topics,
`use_sim_time`, BEST_EFFORT/RELIABLE QoS matched to the source, rolling-window reporting
like `ros2_pose_validation.py`).

---

## 0. What "wave parameters" means here

Standard sea-state descriptors derived from a sea-surface-elevation signal η:

| Symbol | Name | Definition | From |
|---|---|---|---|
| `Hs` (`H_m0`) | Significant wave height | `4·√m0` (m0 = elevation variance / zeroth spectral moment) | both |
| `Hmax` | Max wave height | altimeter: largest zero-crossing wave height in the window; cloud: largest per-frame *robust* (percentile-trimmed) crest-to-trough over the buffer | both |
| `Tp` | Peak period | `1 / f_peak` | both |
| `Tm01` | Mean period | `m0 / m1` | altimeter (temporal) |
| `Tm02` | Zero-crossing period | `√(m0 / m2)` | altimeter (temporal) |
| `f_peak` | Peak frequency | argmax of spectrum (Hz) | both |
| `λ_peak` | Peak wavelength | `2π / |k_peak|` (magnitude only) | point cloud (spatial) |

where `mn = ∫ fⁿ·S(f) df` are spectral moments. The point cloud converts spatial
wavelength → period via **deep-water linear dispersion** `ω² = g·k`, i.e.
`Tp = √(2π·λ / g)` (justified: `/episea/nav/depth` reads 0/invalid in these bags, so we
cannot apply a finite-depth correction — assume deep water and flag it).

**Wave direction is not produced by either node** (dropped per user decision 2026-06-21).
The altimeter is a point sample (no spatial information); the point cloud *could* give a
direction from the 2-D FFT peak angle, but from a single snapshot it is 180° ambiguous and
on this rig's small/noisy patch was empirically near-random (per-frame std ~115°). Only the
wavelength **magnitude** `|k_peak|` is used (rotation-invariant, so it doesn't even depend on
the camera-frame orientation).

---

## 1. Shared helper: `Nodes/wave_common.py`

Mirror `stereo_common.py` — keep the spectral math in one place so both nodes report
consistently. Pure numpy/scipy, no ROS.

- `spectral_moments(freqs, psd) -> (m0, m1, m2)` — trapezoid integration.
- `wave_params_from_psd(freqs, psd) -> dict(Hs, Tp, Tm01, Tm02, f_peak)`.
- `welch_psd(eta, fs, window_s) -> (freqs, psd)` — thin wrapper over
  `scipy.signal.welch` (Hann window, ~50 % overlap, segment length = min(window, len)).
- `detrend_linear(t, y)` — remove slow airship drift / datum before spectral analysis
  (`scipy.signal.detrend` with a linear fit; the residual is the wave signal).

---

## 2. Node 1 — `ros2_altimeter_waves.py` (altimeter → temporal spectrum)

### Principle (revised after the live diagnostic — see §2.5)
```
r_vert(t) = r(t) · cos(pitch) · cos(roll)        # range tilt-corrected to vertical drop
η(t)      = -r_vert(t) − linear_trend             # wave elevation (up positive)
```

- `r_vert(t)`  = altimeter AGL tilt-corrected to a vertical drop:
  `r_vert = r · cos(pitch) · cos(roll)` — reuse the exact formula from
  `ros2_pose_validation.py:_accumulate`, with roll/pitch from `/episea/nav/euler` (deg).
- Linear detrend over the analysis window removes the slow airship drift; a **wave-band
  limit** (`band_low_hz`–`band_high_hz`, default 0.05–0.5 Hz) on the spectral moments
  removes the altimeter's broadband sensor-noise floor.
- **nav-altitude subtraction is OFF by default** (`subtract_nav:=false`). The original
  design subtracted `nav_alt` to remove airship heave, but the live diagnostic (§2.5)
  showed that hurts on this rig — see there. `nav_alt` is still subscribed and used only
  if `subtract_nav:=true`.

### Inputs
| Topic | Type | Use |
|---|---|---|
| `/airship/left/altimeter/height` | `geometry_msgs/PoseStamped` | range = `position.x` (m) — default source |
| `/airship/right/altimeter/height` | `geometry_msgs/PoseStamped` | range = `position.x` (m) — alt. source |
| `/lightware_altimeter/left/altimeter` | `geometry_msgs/PointStamped` | range = `point.z` (m); `-1.0` = no-return → drop; clamp `>max_range` outliers (Lightware spike bug, CLAUDE.md) |
| `/episea/nav/lla` | `nav_msgs/Odometry` | `position.z` = nav_alt |
| `/episea/nav/euler` | `geometry_msgs/Vector3Stamped` | roll/pitch (deg) for tilt correction |

`source` parameter selects which altimeter feeds η (`left` default / `right` / `lightware`).
Right channel is obstructed in the first 10 bags (CLAUDE.md) → keep `left` default.

### Processing
1. On each altimeter callback, hold latest roll/pitch (and nav_alt if `subtract_nav`),
   compute the elevation sample, append `(t, η)` to a ring buffer (`buffer_s` ≈ 120 s).
2. Every `report_period_s` (≈ 5 s), if buffer spans ≥ `window_s` (≈ 60 s):
   - Resample buffer onto a uniform grid at `fs` (≈ 10 Hz) via `np.interp` (Welch needs
     even sampling; altimeter arrival is mildly irregular, ~31 Hz native).
   - `detrend_linear` → η.
   - `welch_psd(η, fs, window_s)` → `wave_params_from_psd(..., band)` — **band-limited** to
     the wave band so the broadband altimeter noise floor doesn't inflate Hs.
   - Also compute a **zero-up-crossing** `Hs` on the band-passed η (`wave_common.bandpass`)
     as an independent cross-check of the spectral `Hs`.

### Outputs
- `diagnostic_msgs/DiagnosticArray` on `/waves/altimeter` — one `DiagnosticStatus` named
  `wave_params` with key/value pairs: `Hs`, `Hmax`, `Tp`, `Tm01`, `Tm02`, `f_peak`,
  `Hs_zerocross`, `n_samples`, `source`.
- Convenience scalar topics (`std_msgs/Float32`): `/waves/altimeter/significant_height`,
  `/waves/altimeter/max_wave_height`, `/waves/altimeter/peak_period`,
  `/waves/altimeter/mean_period`, `/waves/altimeter/peak_frequency`.
- Rolling log line (mean ± std over the window) in the `ros2_pose_validation.py` style.

### Parameters
`source`, `fs` (10.0), `window_s` (60.0), `buffer_s` (120.0), `report_period_s` (5.0),
`max_range` (10.0, Lightware outlier clamp), `band_low_hz` (0.05), `band_high_hz` (0.5),
`subtract_nav` (False), topic names.

---

## 2.5 Live diagnostic finding (2026-06-21) — band-limit + drop nav subtraction

First live run gave Hs≈2.3 m at Tp≈1.7 s (`f_peak`≈0.583 Hz) — physically impossible
(wave steepness ~3.5× past breaking). Diagnosed offline straight from the bags with
`Scripts/altimeter_wave_diagnostic.py` (no live pipeline), comparing three elevation
spectra: (A) raw tilt-corrected range, (B) `nav_alt` alone, (C) `nav_alt − range` (the
node's original method). Findings on `airship_20260528_115912`:

1. **The altimeter range has a broadband sensor-noise floor** (PSD nearly flat to 2.5 Hz
   instead of rolling off above a wave peak). The "0.583 Hz peak" was just the argmax of
   that noise; integrating it over the full 0–Nyquist band inflated Hs ~4×. **Fix:**
   band-limit the spectral moments to the wave band (0.05–0.5 Hz). Raw range then gives
   **Hs≈0.55 m, Tp≈2.1 s**.
2. **`nav_alt` has no wave-band energy** — its PSD is all <0.1 Hz drift, i.e. the airship
   doesn't heave at wave frequencies. Subtracting it removes no heave there and **injects**
   its own ~0.4 m-RMS EKF noise (in-band RMS 0.23 → 0.40 m; band Hs 0.55 → 0.84 m at a
   wrong 10 s period). **Fix:** `subtract_nav=False` by default; detrend the raw range.

After both fixes the altimeter (Hs≈0.4–0.6 m, Tp≈2–2.5 s) **agrees with the point-cloud**
estimate (Hs≈0.5 m, T≈2.5–3 s) — two independent sensors cross-validating. The diagnostic
script is kept in `Scripts/` for re-running on other bags / rougher seas.

---

## 3. Node 2 — `ros2_pointcloud_waves.py` (point cloud → robust plane → spatial spectrum)

### Principle
A single stereo frame is a **spatial** snapshot of the sea surface. Fit the mean surface
with a robust plane (sigma-clipped LS), take each point's signed distance to that plane as the
instantaneous elevation field η(x,y), and analyse it both statistically (variance → Hs)
and spectrally (2-D FFT → wavelength magnitude; direction dropped, see §0).

### Inputs
| Topic | Type | Use |
|---|---|---|
| `/stereo/points` | `sensor_msgs/PointCloud2` | XYZ(+RGB), frame `camera_left_rect` |
| TF `map ← camera_left_rect` | tf2 | transform cloud to ENU before analysis |

Read points with `sensor_msgs_py.point_cloud2.read_points` (structured numpy, skip NaNs).
Transform to the `map` (ENU) frame with **`tf2_sensor_msgs.do_transform_cloud()`** — never
manual per-point math in Python (CLAUDE.md open item #3). Use a `tf2_ros.Buffer` +
`TransformListener`, looking up the transform at the cloud's header stamp (with a short
timeout; drop the frame if TF isn't ready). Working in ENU makes "up" well-defined so the
plane normal (hence Hs and the in-plane wavelength) is physically meaningful.

**Range filter (`max_range`, default 40 m).** The raw cloud spans ~2–99 m depth, but stereo
error grows ~quadratically — the far points are unreliable and, over such a long oblique
strip, the plane fit absorbs real long-wave elevation as "tilt" (biasing Hs low) while
far-point scatter makes a vertical variance explode (1–4 m). Keeping only points within
`max_range` of the camera (its origin in `map` is the lookup transform's translation) makes
Hs an honest, well-defined quantity and drops the plane tilt from ~6–8° to ~4°. Verified:
range-filtered cloud `Hs(med)≈0.37 m` matches the altimeter (`Hs≈0.40 m`) at the same
instant. See `Scripts/pointcloud_hs_diag.py`.

### Processing
0. **Per-frame quality gate (`max_frame_hs`, default 1.0 m).** The disparity backend
   produces frequent whole-frame failures: per-frame Hs was measured swinging 0.04–3.0 m on
   a sea that is really ~0.3–0.5 m (~35 % of frames bad — see
   `Scripts/pointcloud_residual_diag.py`). A frame whose surface RMS implies Hs above this
   threshold is a stereo failure (the *whole* patch is spread, not a few outliers) and is
   dropped before it can corrupt Hs/Hmax/λ. Gating the live data took Hs 0.92→0.38 m and
   Hmax 4.2→1.0 m (Hmax/Hs 4.6→2.7, physical). Raise for genuinely rougher seas; 0 disables.
1. **Robust plane fit — iterative sigma-clipped least-squares** (numpy only): fit a plane
   (SVD on the current inliers' centroid), reject points beyond `outlier_sigma`·std of the
   residuals (gross outliers — birds/spray — only), refit; converges in a few iterations.
   Skip the frame if inliers `< min_points`. *Why not fixed-band RANSAC:* its inlier band
   would have to exceed the (unknown) wave amplitude or it clips wave crests as outliers —
   verified to under-read Hs and halve the wavelength. Sigma-clipping is self-tuning to the
   sea state; on synthetic data it recovers Hs/λ and rejects injected outliers.
2. **Mean-surface sanity:** plane normal `n` should be ≈ vertical in ENU; log the tilt
   angle `acos(n·ẑ)` — a large tilt flags a bad pose/calibration, not real waves.
3. **Elevation field:** signed distance `η_i = n·x_i − d` for every inlier (use plane
   coords: project inliers onto the two in-plane basis vectors → `(u, v, η)`).
4. **Statistical Hs + Hmax:** `Hs_spatial = 4·std(η)` (= `4·√variance`); per-frame
   crest-to-trough as a **percentile-trimmed** range (`hmax_percentile`, default 1% each
   tail — NOT literal max−min, which a single stray stereo point blows up), the worst over
   the buffer reported as `Hmax`. Both are robust to frame placeholders — they depend only
   on the cloud's *internal* geometry/scale, not its absolute pose.
5. **Wavelength spectrum:**
   - Bin `(u, v, η)` onto a regular 2-D grid (`grid_res` ~ point spacing, e.g. 0.5 m),
     averaging η per cell; fill empty cells (zero) and apply a 2-D Hann window.
   - 2-D FFT → power spectrum in `(k_u, k_v)`. Peak (excluding DC + sub-patch wavelengths)
     → `|k_peak|`, `λ_peak = 2π/|k_peak|`. The peak's **angle** (direction) is discarded
     (see §0) — only the magnitude is used, which is also rotation-invariant.
   - `Tp = √(2π·λ_peak / g)` (deep-water dispersion, see §0).
6. Maintain a rolling buffer of per-frame `(Hs, crest-to-trough, λ, Tp, tilt)` and publish
   the **median** for Hs/λ/T/tilt (robust — so the quality gate's removal of high frames no
   longer biases the central value) and the **max** for Hmax.

### Outputs
- `diagnostic_msgs/DiagnosticArray` on `/waves/pointcloud` — `wave_params` status with
  `Hs`, `Hs_std`, `Hmax`, `peak_wavelength`, `peak_wavelength_std`, `peak_period`,
  `plane_tilt_deg`, `n_frames`.
- Scalar `std_msgs/Float32`: `/waves/pointcloud/significant_height`,
  `/waves/pointcloud/max_wave_height`, `/waves/pointcloud/peak_wavelength`,
  `/waves/pointcloud/peak_period`.
- *(Optional, debug)* republish plane inliers as a `PointCloud2` and/or the fitted plane
  as a `visualization_msgs/Marker` for RViz.

### Parameters
`cloud_topic` (`/stereo/points`), `map_frame` (`map`), `outlier_sigma` (4.0),
`plane_iters` (10), `min_points` (500), `max_points` (30000), `grid_res` (0.5),
`max_range` (40.0, near-field range filter), `max_grid_cells` (200000), `tf_timeout_s` (0.1),
`min_process_interval_s` (0.3), `buffer_frames` (30), `report_period_s` (5.0),
`gravity` (9.81), `hmax_percentile` (1.0), `max_frame_hs` (1.0, bad-disparity-frame gate).

### Key caveats (to put in the node docstring)
- **Direction is not produced** (dropped 2026-06-21, see §0). With it gone, every remaining
  wave output is rotation-invariant: `Hs` is a variance and `λ_peak` is `|k|`, so neither
  depends on the `base_link → camera_left_rect` orientation. (The transform is still wired —
  §3.5 — for the rest of the pipeline / georeferencing, just no longer required here.)
- Absolute Z offset is meaningless until the lever arm is calibrated — but wave height is
  a *relative* (variance) quantity, so it survives the placeholder.
- Deep-water dispersion assumed (no valid depth in these bags).
- Single-frame spatial coverage is limited (~camera FOV at altitude) → wavelengths longer
  than the patch are unresolved; the spectrum favours short/steep waves. Note as a known
  limitation; temporal accumulation over the pass is a future extension.

---

## 3.5 Prerequisite — make `base_link → camera_left_rect` real from the URDF — DONE (2026-06-21)

Direction (§3 step 5) is only meaningful once this static transform is real. The URDF
(`urdf_estrutura_ondas/urdf/estrutura_ondas.urdf.xacro`) lets us replace the identity
placeholder. `estrutura_ondas`/`umix` sits at the rig origin = `base_link` (FLU), so the
`Left_cam` joint *is* `base_link → camera_left` directly:

```
Left_cam   xyz = (0.044867,  0.51476, -0.0045012)   rpy = (-1.5708, 0, -1.5717)  ≈ (-90°, 0°, -90°)
right_cam  xyz = (0.043926, -0.48504, -0.0045011)   rpy = (-1.5708, 0, -1.5717)
```

**Wired (`start_pipeline.sh` window `tf_cam` + `init_pipeline.txt`):**
`static_transform_publisher 0.044867 0.51476 -0.0045012  0 0 0  base_link camera_left_rect`
— real **translation, identity rotation**.

- **Translation:** `(0.044867, 0.51476, -0.0045012)` m (Left_cam link origin ≈ optical
  centre to ~cm; baseline cross-check 0.9998 m URDF vs 1.0003 m calibration).
- **Rotation = identity, NOT the optical quaternion** — this is the key subtlety.
  `ros2_pointcloud_node.py` already remaps the optical cloud into base-aligned FLU
  internally (`x=optZ, y=-optX, z=-optY`, lines ~202-204) and labels it `camera_left_rect`.
  That remap matches the URDF optical→base rotation to **0.05°** (camera mounted
  forward+level), so the published cloud is already in `base_link` orientation. Putting the
  optical quaternion `(-0.5, 0.5, -0.5, 0.5)` in the TF as well would **double-rotate** it.
  The verification below (camera rpy *is* the canonical FLU→optical rotation) is exactly
  what justifies leaving the TF rotation identity. *(Alternative not taken: strip the
  node's internal remap so the cloud is true optical, then put the quaternion in the TF —
  the standard ROS layout, a larger change to a working node.)*

**Sign-ambiguity resolution — VERIFIED (corrects CLAUDE.md open item #2).** The earlier
"right camera lands at −X" came from applying the URDF rpy with the wrong Euler
convention. URDF rpy is **fixed-axis / extrinsic** `R = Rz(yaw)·Ry(pitch)·Rx(roll)` =
scipy `Rotation.from_euler('xyz', [roll, pitch, yaw])` (**lowercase = extrinsic**;
uppercase `'XYZ'` is *intrinsic* and gives a different matrix — **CLAUDE.md has these two
labels swapped**, which is the source of the flip).

Confirmed numerically against an independent ground truth (the stereo calibration in
`ros2_stereo_rectifier.py`): the right camera's position in the left optical frame from
the calibration is `-R_stereoᵀ·T_stereo ≈ (+1.0003, +0.0032, -0.0034)` = **+X optical**.
The URDF rpy applied as **extrinsic** places `right_cam` at `≈ (+0.9998, 0, 0)` = **+X
optical** — the two agree to ~3 mm. (Applied as intrinsic `'XYZ'` it lands at
`≈ (0, 0, -0.9998)` = −Z, i.e. wrong — reproducing the old flip.) The resulting
quaternion is `(-0.4998, 0.5002, -0.5002, 0.4998) ≈ (-0.5, 0.5, -0.5, 0.5)`, the
canonical FLU→optical. So the transform is correct and consistent.

**Remaining caveat — rectified vs physical frame (the ~3 mm residual above).** The URDF
describes the *physical* left camera; `camera_left_rect` is the *rectified* optical frame,
which differs by the stereo toe-in (`R_stereo`) + rectification rotation `R1` (a few
mrad/mm, baked into `ros2_stereo_rectifier.py`, not the URDF) — that is exactly the small
URDF-vs-calibration difference seen above. Negligible for wave direction; ignore for the
first cut. For full rigor, compose `base_link → physical_left` (URDF) with
`physical_left → rectified` (`R1`).

**Optional RViz sanity check (no longer a gate):** the sign is settled quantitatively
above, but a quick look in RViz with the rig meshes (camera frustum/cloud points
forward+down) is still worth doing as a gross-orientation sanity check before trusting
absolute `θ_peak` values.

**Wiring — done:** `start_pipeline.sh`'s `tf_cam` window and `init_pipeline.txt` now
publish `static_transform_publisher 0.044867 0.51476 -0.0045012 0 0 0 base_link
camera_left_rect` (real translation, identity rotation per the subtlety above). CLAUDE.md
updated ("Camera lever arm wired in (2026-06-21)"). This benefits the whole pipeline
(georeferenced clouds), not just the wave node.

---

## 4. Cross-validation (why two nodes)

The two are independent estimators of the **same** `Hs`:
- Altimeter: temporal variance of η at one footprint.
- Point cloud: spatial variance of η over the surface patch.

For a stationary sea state these should agree. A persistent disagreement flags a
calibration/scale problem in one branch. Optionally extend `ros2_pose_validation.py` (or a
small new comparator) to log `Hs_altimeter` vs `Hs_pointcloud` side by side. The available
bags may be relatively calm, so expect possibly-small `Hs` — validate the *machinery*
(spectrum recovers a known synthetic signal, the two estimates agree) on these, and re-check
absolute magnitudes on a rougher-sea bag when available.

---

## 5. Integration & deliverables

1. `Nodes/wave_common.py` — shared spectral helpers (§1).
2. `Nodes/ros2_altimeter_waves.py` — Node 1 (§2).
3. `Nodes/ros2_pointcloud_waves.py` — Node 2 (§3).
4. **Real `base_link → camera_left_rect` static transform (§3.5)** — replace the identity
   `static_transform_publisher` in `start_pipeline.sh` / `init_pipeline.txt` with the
   URDF-derived translation + quaternion. Prerequisite for point-cloud direction.
5. `start_pipeline.sh` — two new tmux windows (`alt_waves`, `pc_waves`), `use_sim_time:=true`.
6. `init_pipeline.txt` — equivalent manual `python3 Nodes/...` lines.
7. `CLAUDE.md` — add both nodes to the pipeline diagram / supporting-nodes list and the new
   `/waves/*` topics; update open work item #2 (camera transform now resolved/wired, with
   the extrinsic-vs-intrinsic correction) and the direction caveat.

### Dependencies to confirm before coding
- `scipy.signal` (Welch/detrend) — scipy already required (`requirements.txt`).
- `sensor_msgs_py` + `tf2_sensor_msgs` (apt `ros-humble-tf2-sensor-msgs`) — verify
  installed; needed for cloud read + `do_transform_cloud`.
- numpy `<2.0` pin already enforced (CLAUDE.md) — plane fit/FFT are pure numpy, fine.

---

## 6. Suggested build order

1. `wave_common.py` + a tiny offline test (synthetic sine η → check Hs/Tp recovered).
2. `ros2_altimeter_waves.py` (simpler, single time series) — validate live on a bag with
   the pipeline running, eyeball `Hs`/`Tp` for plausibility (sub-metre `Hs` on calm bags).
3. Wire the real `base_link → camera_left_rect` transform (§3.5) — done; benefits the whole
   pipeline (note: no longer strictly required by the wave node now that direction is dropped).
4. `ros2_pointcloud_waves.py` — sigma-clip plane + statistical `Hs` first (most robust), then add the
   2-D FFT wavelength stage.
5. Wire the wave nodes into `start_pipeline.sh` / `init_pipeline.txt`, then the
   cross-validation log.
