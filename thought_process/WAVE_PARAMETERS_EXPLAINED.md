# Wave-Parameter Extraction — How It Works

This document explains, step by step, how SWMP turns sensor data into sea-state (wave)
parameters. Two **independent** estimators run in parallel so their results can be
cross-checked:

| Node | Sensor | Domain | File |
|---|---|---|---|
| Altimeter waves | downward altimeter (rfbeam / Lightware) | **temporal** (one point, over time) | `Nodes/ros2_altimeter_waves.py` |
| Point-cloud waves | stereo point cloud | **spatial** (a surface patch, per frame) | `Nodes/ros2_pointcloud_waves.py` |

Shared maths (spectra, statistics, dispersion) lives in `Nodes/wave_common.py`.

```
                        ┌─────────────────────────────┐
  /airship/.../altimeter ─►│ AltimeterWavesNode          │─► /waves/altimeter (+ scalars)
  /episea/nav/euler ──────►│  (temporal spectrum)        │
                        └─────────────────────────────┘
                        ┌─────────────────────────────┐
  /stereo/points ────────►│ PointCloudWavesNode         │─► /waves/pointcloud (+ scalars)
  TF map←camera_left_rect ►│  (spatial plane + spectrum) │
                        └─────────────────────────────┘
```

---

## 1. The parameters

All wave parameters come from a **sea-surface-elevation signal** η — the surface height
relative to its own mean, with the platform's motion removed.

| Symbol | Name | Definition | From |
|---|---|---|---|
| `Hs` (`H_m0`) | Significant wave height | `4·√m0`, `m0` = variance of η | both |
| `Hmax` | Maximum wave height | largest single wave (crest-to-trough) | both |
| `Tp` | Peak period | `1 / f_peak` | both |
| `Tm01` | Mean period | `m0 / m1` | altimeter |
| `Tm02` | Zero-crossing period | `√(m0 / m2)` | altimeter |
| `f_peak` | Peak frequency | argmax of the spectrum | both |
| `λ_peak` | Peak wavelength | `2π / |k_peak|` | point cloud |

`mn = ∫ fⁿ·S(f) df` are the spectral moments of the elevation power spectral density
`S(f)`. **Wave direction is not produced** — a single altimeter footprint has no spatial
information, and the point cloud's single-snapshot direction was 180°-ambiguous and
empirically near-random, so only the rotation-invariant magnitudes are kept.

---

## 2. Altimeter path — temporal spectrum

The altimeter looks straight down and measures its range to the water. As waves pass under
the (slowly drifting) airship, that range fluctuates. The pipeline turns a stream of range
readings into a wave spectrum.

### 2.1 Reconstruct surface elevation
For each altimeter sample of range `r`, with roll/pitch from `/episea/nav/euler`:

```
r_vert = r · cos(pitch) · cos(roll)      # tilt-correct the slant range to a vertical drop
η      = −r_vert                          # elevation (up positive); range shrinks as surface rises
```

Samples `(t, η)` go into a rolling buffer (`buffer_s`, default 120 s).

> **Why no nav-altitude subtraction?** The original design subtracted the nav altitude to
> remove airship heave (`η = nav_alt − r_vert`). Measurement showed that *hurts* here: the
> airship has negligible heave in the wave band (nav altitude energy is all < 0.1 Hz drift),
> while the nav filter's own ~0.4 m-RMS noise is larger than the wave signal — subtracting it
> nearly doubled the in-band noise. So `subtract_nav` defaults to **off**; the slow drift it
> would have removed is taken out by the detrend + band limit below instead.

### 2.2 Condition the signal
1. **Resample** the (slightly irregular, ~31 Hz native) samples onto a uniform grid at `fs`
   (default 10 Hz) by linear interpolation — the FFT needs even spacing.
2. **Linear detrend** the analysis window — removes the slow airship altitude drift and the
   ~56 m geodetic datum offset, leaving the wave oscillation.

### 2.3 Spectrum and parameters
1. **Welch PSD** (Hann window, 50 % overlap) → `S(f)`.
2. **Band-limit** the spectral moments to the ocean-wave band (`band_low_hz`–`band_high_hz`,
   default **0.05–0.5 Hz** = periods 2–20 s), then `Hs = 4·√m0`, `Tp`, `Tm01`, `Tm02`,
   `f_peak`.

> **Why band-limit?** The raw altimeter range carries a broadband sensor-noise floor that
> extends far above the wave band (its PSD is nearly flat to several Hz). Integrating that
> noise over the full 0–Nyquist range inflated `Hs` by ~4× and put `f_peak` on a noise bin.
> Restricting to the wave band fixes both.

3. **Zero-crossing cross-check:** band-pass η, find zero up-crossings, measure each
   individual wave's crest-to-trough height →
   - `Hs_zerocross` = mean of the highest third (an independent check on `4·√m0`),
   - `Hmax` = the single largest wave height.
4. **Doppler / encounter-frequency correction.** This is a *boat*, moving through the waves
   at ~2.6 m/s, so the measured peak is the **encounter** frequency, not the true wave
   frequency. Convert it with the deep-water encounter relation
   `ω_e = ω₀ − (U/g)·cos(μ)·ω₀²` (U = boat speed from nav twist, μ = `encounter_angle_deg`,
   default 180° = head seas): `Tp`/`f_peak` become the **true** values; `Tp_encounter` keeps
   the raw. `Hs` is unaffected (variance is invariant under the remap). This is what makes
   the altimeter period agree with the point cloud — see §4.

### 2.4 Outputs
`/waves/altimeter` (`diagnostic_msgs/DiagnosticArray`: `Hs`, `Hmax`, `Tp`, `Tm01`, `Tm02`,
`f_peak`, `Hs_zerocross`, …) plus scalar `std_msgs/Float32` on
`/waves/altimeter/{significant_height, max_wave_height, peak_period, mean_period,
peak_frequency}`. Recomputed every `report_period_s` (default 5 s) once the buffer spans
`window_s` (default 60 s). `source` selects `left` (default) / `right` / `lightware`.

---

## 3. Point-cloud path — spatial plane + spectrum

A single stereo frame is a spatial snapshot of the sea surface. The pipeline fits the mean
surface, measures the surface's deviation from it, and analyses that deviation field.

### 3.1 Put the cloud in a world frame
Transform `/stereo/points` to the `map` (ENU) frame with
`tf2_sensor_msgs.do_transform_cloud()` (never manual per-point maths). Working in ENU makes
"up" well-defined, so the fitted plane and its residuals are physically meaningful.

### 3.2 Range filter — keep only accurate stereo
Keep points within `max_range` (default **40 m**) of the camera (its position in `map` is the
lookup transform's translation).

> **Why?** The raw cloud spans ~2–99 m depth, but stereo depth error grows ~quadratically
> with range. Including the far points makes the analysis patch a ~90 m oblique strip whose
> far end is unreliable; the plane fit then tilts to chase it and absorbs real wave energy.
> Restricting to the near field makes `Hs` an honest, well-defined quantity (plane tilt drops
> from ~6–8° to ~4°).

### 3.3 Fit the mean surface (robust plane)
Iterative **sigma-clipped least-squares**: fit a plane (SVD about the inlier centroid),
reject points beyond `outlier_sigma`·std of the residuals, refit; converge in a few passes.
Orient the normal up. The per-point **signed distance to the plane is the elevation field η**.

> **Why not RANSAC?** Fixed-band RANSAC needs an inlier band larger than the (unknown) wave
> amplitude, or it clips wave crests as outliers — verified to halve the wavelength and
> under-read `Hs`. Sigma-clipping is self-tuning: it removes only gross outliers (birds,
> spray) while keeping the whole wave.

### 3.4 Per-frame quality gate
The disparity backend frequently fails over water (specular, texture-poor surface):
per-frame `Hs` was seen swinging 0.04–3.0 m, ~35 % of frames being whole-patch garbage. A
frame whose surface RMS implies `Hs > max_frame_hs` (default 1.0 m) is a disparity failure
and is **dropped** before it can corrupt anything (the dropped count is reported as
`n_bad_dropped`).

### 3.5 Per-frame parameters
- **`Hs = 4·std(η)`** — significant wave height from the spatial variance.
- **Crest-to-trough** — a percentile-trimmed range (`hmax_percentile`, default 1 % each tail,
  *not* literal max−min, which one stray stereo point would blow up). The largest of these
  over the buffer becomes `Hmax`.
- **Wavelength** — project η onto the in-plane axes, bin onto a regular grid (`grid_res`,
  default 0.5 m), apply a 2-D Hann window, take the 2-D FFT; the spectral peak `|k_peak|`
  gives `λ = 2π/|k_peak|`. The peak's *angle* (direction) is discarded.
- **Period** — from `λ` via deep-water linear dispersion `T = √(2π·λ / g)` (deep water
  assumed: `/episea/nav/depth` is invalid in these bags).
- **Plane tilt** — `acos(n·ẑ)`, a sanity/diagnostic value.

### 3.6 Robust aggregation over the buffer
Keep a rolling buffer of the last `buffer_frames` (default 30). Publish the **median** of
`Hs`, `λ`, `T`, `tilt` (robust — so the quality gate's removal of high frames doesn't bias
the central value) and the **max** of the per-frame crest-to-trough as `Hmax`.

### 3.7 Outputs
`/waves/pointcloud` (`diagnostic_msgs/DiagnosticArray`: `Hs`, `Hmax`, `peak_wavelength`,
`peak_period`, `plane_tilt_deg`, `n_frames`, `n_bad_dropped`, …) plus scalar
`std_msgs/Float32` on `/waves/pointcloud/{significant_height, max_wave_height,
peak_wavelength, peak_period}`.

---

## 4. Cross-validation and known limitations

The two estimators are independent, so comparing them is a built-in health check.

- **`Hs` agrees to ~factor 2**, both sub-metre on the calm test bags. They do *not* track
  tightly, and there are reasons each is imperfect:
  - **Point cloud likely reads low:** even the 40 m patch retains ~4° plane tilt (a long wave
    across the patch is partly absorbed as "tilt"), and the cloud is spatially smoothed /
    downsampled, which damps variance.
  - **Altimeter may read high:** a low `Hmax/Hs` (~1.15, vs the natural ~1.5–2) signals a
    near-monochromatic signal — i.e. a possible residual in-band platform tone inflating
    `Hs` rather than real waves.
  - The true `Hs` is probably **between** them (~0.3–0.5 m on these bags). Neither is clean
    ground truth — compare them *contemporaneously*, since the real sea state also varies
    over a pass (altimeter `Hs` ranged ~0.4–0.8 m across one bag).
- **Period — they now agree once the boat motion is accounted for.** This was a puzzle: the
  altimeter read Tp≈2.1–2.4 s while the point cloud read ≈3.7 s. Cause: the altimeter's raw
  value is the **encounter** period (boat moving ~2.6 m/s through the waves) while the point
  cloud measures the **true** period from spatial wavelength (motion-independent). Applying
  the Doppler correction (§2.3 step 4) turns the altimeter's 2.1–2.4 s into ~3.2–3.5 s — i.e.
  the point cloud was right all along, and the corrected altimeter matches it to ~10–13 %.
  (The point cloud's wavelength is still coarsely resolved over a ~40 m patch, so treat its
  period as approximate; but it provided the independent truth that validated the Doppler fix.)
- **Direction:** not produced by either (see §1).
- **Absolute Z** is not needed — every wave parameter is a *relative* quantity (a variance, a
  crest-to-trough, or `|k|`), so they are unaffected by the lever-arm/datum calibration that
  still limits absolute georeferencing.

---

## 5. Diagnostic tooling (offline, no live pipeline)

These read the bags / live topics directly and were used to derive the choices above; keep
them for re-tuning on other bags or rougher seas:

- `Scripts/altimeter_wave_diagnostic.py` — compares altimeter elevation spectra (raw range vs
  nav-subtracted vs nav-only) and band-limited `Hs`; established the band limit + no-nav
  decision.
- `Scripts/pointcloud_residual_diag.py` — per-frame plane-fit residual percentiles; revealed
  the bad-disparity-frame problem behind the quality gate.
- `Scripts/pointcloud_hs_diag.py` — compares cloud `Hs` computed several ways (tilted-plane
  vs vertical vs near-only) against patch extent; established the range filter.

---

## 6. Parameter quick reference

**Altimeter** (`ros2_altimeter_waves.py`): `source` (left), `fs` (10), `window_s` (60),
`buffer_s` (120), `report_period_s` (5), `band_low_hz` (0.05), `band_high_hz` (0.5),
`subtract_nav` (false), `doppler_correct` (true), `encounter_angle_deg` (180),
`max_range` (Lightware clamp, 10).

**Point cloud** (`ros2_pointcloud_waves.py`): `max_range` (40), `outlier_sigma` (4),
`plane_iters` (10), `min_points` (500), `max_points` (30000), `grid_res` (0.5),
`hmax_percentile` (1.0), `max_frame_hs` (1.0), `buffer_frames` (30), `report_period_s` (5),
`min_process_interval_s` (0.3), `gravity` (9.81).

See `thought_process/WAVE_PARAMETERS_PLAN.md` for the design history and the dated entries in
`CLAUDE.md` for the rationale behind each tuning decision.
