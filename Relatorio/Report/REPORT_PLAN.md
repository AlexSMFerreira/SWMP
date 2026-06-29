# Report plan — "Sea Wave Modelling and Prediction Based on Multisensor Fusion"

Target: max 25 pages including annexes. Report language: English (template translated in
`capstoneproject.tex`). Maps onto the existing `sections/*.tex` includes.

Budget: ~1.5 pg Introduction, ~2-3 pg Methodology, ~12-14 pg Development (bulk of the
work), ~2-3 pg Conclusions, remainder as annexes/figures.

## 1. Introduction (~1.5 pg)

- **Context.tex** — done (translated). SEAWINGS/AIRSHIP, WIG ground-effect rationale, why
  real-time sea-surface perception is a safety requirement, not just mapping.
- **Objectives.tex** — reconstruct and georeference the sea surface from the multi-sensor
  rig (stereo cameras, laser altimeters, lidar, thermal camera, GNSS/INS) and extract wave
  parameters (Hs, Tp, direction) from recorded ROS 2 bags; validate the estimates against
  an independent navigation solution. Expected results: a working ROS 2 pipeline plus two
  independently cross-validated wave estimators.
- **Structure.tex** — one paragraph mapping to the sections below.

## 2. Methodology (~2 pg)

- **Methodology.tex** — iterative, data-first engineering process: every milestone is
  driven by first inspecting real bag data (topic schemas, units, clock domains) before
  writing code; a tight measure → implement → re-verify loop, documented as a running log
  in `CLAUDE.md` and in `thought_process/*.md`. Tools: ROS 2 Humble, `rmw_zenoh_cpp`,
  Python/rclpy, git.
- **Intervenients.tex** — student (Alexandre Ferreira), Prof. Pedro Ribeiro (U.Porto
  tutor), Prof. Eduardo Silva (company/proposer supervisor), José Carlos Fernandes
  (INESC TEC — domain feedback that corrected two altimeter calibration/unit bugs; cite as
  evidence of iteration with a domain expert).
- **Activities.tex** — Gantt-style timeline reconstructed from `git log` commit dates and
  the dated entries in `CLAUDE.md`/`thought_process`: rectify pipeline → disparity-backend
  search (classical SBM/SGBM/SGM-CUDA, then HITNet/RAFT-Stereo/WAFT-Stereo) → point cloud
  → pose/nav migration → wave nodes (altimeter + point cloud) → tuning/cross-validation →
  Doppler correction (2026-06-18 through 2026-06-26, plus the two earliest commits for the
  base pipeline). The disparity-backend search spans most of the project timeline (it
  reappears as late as `ros2_waft_disparity.py`, the most recently touched node) and should
  read as an ongoing thread, not a single one-off milestone.

## 3. Solution development (~12-14 pg, core of the report)

- **Requirements.tex** —
  - Functional: real-time disparity → point-cloud reconstruction; wave-parameter
    extraction (Hs/Tp/direction) from two independent sensing modalities; georeferencing
    via nav fusion (map ↔ base_link TF).
  - Non-functional: must run against recorded bags (no live airship available), GPU
    budget of an RTX 2060-class card, ROS 2 Humble + Zenoh RMW.
  - Constraints: exfat bag-drive throughput limits, two-bag (camera + nav) clock
    synchronization, sensor outputs with undocumented/incorrect units in vendor docs.
- **Arquitecture.tex** — pipeline diagram (RectifyNode → DisparityNode → PointCloudNode →
  PoseBroadcasterNode, plus the altimeter publisher and the two wave nodes); technology
  stack table. The main technical-difficulty narrative is **finding a disparity backend
  that works on the sea surface**, not incidental integration bugs — this is the real
  engineering problem and should carry most of this subsection:
  - **Why it's hard**: the sea surface is low-texture, specular, and non-Lambertian —
    exactly the conditions that break classical block-matching stereo (textureless
    regions give ambiguous/noisy matches, specular highlights move with viewpoint and
    violate the brightness-constancy assumption matching relies on).
  - **Six backends were actually implemented and tried**, in two families:
    - Classical (CPU/GPU, no training data needed): `ros2_sbm_disparity.py` (StereoBM —
      fastest, weakest on water), `ros2_sgbm_disparity.py` (StereoSGBM — global P1/P2
      smoothness term, the strongest classical option on low-texture water, but slower),
      `ros2_sgm_cuda_disparity.py` (`cv2.cuda.StereoSGM` — SGM quality at GPU speed,
      requires an OpenCV build with CUDA support).
    - Deep-learning (pretrained, GPU): `ros2_hitnet_disparity.py` (HITNet ONNX — the
      pipeline's default, best quality/speed tradeoff on the RTX 2060), 
      `ros2_raftstereo_disparity.py` (RAFT-Stereo — tuned via mixed precision + reduced
      iteration count for sub-100ms on 6 GB VRAM), `ros2_waft_disparity.py` (WAFT-Stereo —
      warping-alone field transforms, no cost volume, state-of-the-art on ETH3D/KITTI/
      Middlebury at publication time; three checkpoint sizes — vits/vitb/vitl — trading
      VRAM/quality, vits chosen to fit the 6 GB card).
  - **All six share the same I/O contract** (`stereo_common.py` helpers: sky/horizon
    masking, disparity scaling, message packing) specifically so they are drop-in
    interchangeable and a fair comparison is possible without touching the rest of the
    pipeline.
  - **Comparison table — DATA COLLECTED AND TUNED, final numbers below.** Built the
    tooling (`stereo_common.photometric_consistency_error` + `Scripts/
    disparity_backend_compare.py`) and ran it live: lean pipeline (zenoh, camera+nav
    bag, pose broadcaster, rectifier, one disparity backend, point cloud, point-cloud
    wave node) against `airship_20260528_115912`, ~90s capture per backend after a 35s
    warm-up, same segment every time, for all six backends including a tuned-parameter
    pass on the three classical ones (see below for why).

    | Backend | photo_err (med) | valid % (med) | latency ms (med / p90) | pc bad:good ratio | pc Hs (med) |
    |---|---|---|---|---|---|
    | SBM, default params | n/a (0 valid px) | 0.0% | 9.4 / 10.9 | n/a — point cloud always empty | n/a |
    | SBM, **tuned** | 14.06 | 19.2% | 8.7 / 10.1 | 11:3 (3.7) | 0.97 m |
    | SGBM, default params | 12.49 | 23.8% | 43.9 / 53.0 | 11:5 (2.2) | 0.91 m |
    | SGBM, **tuned** | 12.50 | 23.8% | 40.3 / 45.9 | 10:10 (1.0) | 0.90 m |
    | SGM-CUDA, default params | 13.22 | 7.1% | 10.1 / 11.2 | n/a — point cloud always empty | n/a |
    | SGM-CUDA, **tuned** | 12.92 | 17.0% | 8.6 / 9.8 | n/a — point cloud always empty | n/a |
    | HITNet | 14.27 | 35.8% | 37.3 / 40.0 | 6:30 (0.20) | 0.79 m |
    | RAFT-Stereo | 12.90 | 33.7% | 115.0 / 162.2 | 2:30 (0.07) | 0.50 m |
    | WAFT-Stereo | 12.92 | 36.0% | 114.2 / 160.6 | 0:30 (0.00) | 0.29 m |

    "Tuned" = `uniqueness_ratio:=0` for all three, plus `texture_threshold:=0
    speckle_window_size:=0` for SBM, `disp12_max_diff:=64 speckle_window_size:=0` for
    SGBM, and `num_disparities:=128` (matching SGBM, down from the CUDA default of 256)
    for SGM-CUDA — found by testing short live samples first (see below), then
    confirmed with full 90s captures. Both the default and tuned rows are kept in the
    table on purpose: the delta between them IS part of the finding.

    (`pc bad:good ratio` = `n_bad_dropped`/`n_frames` from the point-cloud wave node's
    last `/waves/pointcloud` report received during the capture window. The gate itself,
    in `ros2_pointcloud_waves.py`: every accepted cloud frame gets a robust plane fit
    (iterative sigma-clipped least-squares, NOT RANSAC — see Architecture notes below)
    and a per-frame Hs from the residuals; if that single frame's Hs exceeds
    `max_frame_hs` (default 1.0 m) the whole frame is dropped as a disparity-failure
    artifact (`self._dropped += 1`) before it can contaminate the running Hs/Hmax/λ
    buffers. `_report()` fires every `report_period_s` (default 5s), publishes the
    median over the buffer, and resets the dropped-counter — so `n_bad_dropped` is "since
    the last report," not a cumulative total, and `n_frames` is the *good*-frame buffer
    size (capped at `buffer_frames`, default 30), not a total-attempted count. So this
    ratio is "bad thrown out per good one kept, in the most recent ~5s window," not a
    0–1 fraction (SGBM's 2.2 means more bad frames were dropped than good ones survived
    in that window) — label it that way in the report rather than "rate." "n/a — point
    cloud always empty" means the wave node never received enough valid disparity to fit
    a plane even once during the whole 90s capture, so it never published a diagnostic
    at all.)

    **The SGM-CUDA "couldn't run" finding was actually an environment bug, not a real
    limitation — found and fixed, worth a sentence in Architecture.** This machine has a
    separately-compiled, CUDA-enabled OpenCV (4.14-pre, `WITH_CUDA=ON`, built from
    `/home/alex/opencv` + `opencv_contrib`, installed to `/usr/local/lib/python3.10/
    dist-packages`) — but it was being **shadowed** by the non-CUDA `opencv-contrib-python`
    pip wheel in `~/.local/lib/python3.10/site-packages`, which sits earlier in
    `sys.path` (same class of bug as the numpy-ABI-shadowing issue already documented in
    `CLAUDE.md`'s environment notes). Verified the custom build has everything the other
    nodes need too (`ximgproc` for SGBM's WLS filter, `cuda.createStereoSGM`) and passes
    a basic numpy/cv2 ABI sanity check, then ran SGM-CUDA with `PYTHONPATH=/usr/local/
    lib/python3.10/dist-packages:$PYTHONPATH` scoped to just that one process — not a
    global environment change, so the rest of the already-validated pipeline (which
    depends on the pip wheel for `ximgproc` etc. and was never re-tested against the
    custom build globally) is untouched.

    **What the default-vs-tuned comparison actually shows — tested, not assumed, after
    being asked exactly this question:**
    - **SBM's apparent total failure (0% valid, never produced a point cloud) was
      mostly a parameter artifact, not an algorithmic dead end.** Tuned, it reaches
      19.2% valid and *does* produce point-cloud Hs estimates (0.97 m) — its defaults
      (`texture_threshold=10`, `uniqueness_ratio=15`) were simply too strict for this
      segment. The honest claim is "SBM's shipped defaults don't work here," not "SBM
      cannot work on water."
    - **SGBM barely moved** (23.8%→23.8% valid, 0.91→0.90 m Hs) — its defaults were
      already near its real ceiling for this scene, so SGBM's number is the more
      parameter-robust of the three classical results.
    - **SGM-CUDA improved substantially but never enough to produce a single usable
      point-cloud frame** (7.1%→17.0% valid, still nothing through the wave node's
      plane fit) — `uniqueness_ratio` and `num_disparities` both mattered, but a real
      gap to SGBM remains even tuned, plausibly because the CUDA binding exposes no
      `disp12MaxDiff`-equivalent left-right consistency control that SGBM has.
    - **Net effect on the backend ranking**: even tuned, all three classical methods
      remain clearly behind the three learned backends on every axis that matters for
      this use case (valid fraction, and — decisively — whether the point cloud wave
      node can produce an estimate at all: only SGBM and tuned-SBM ever did, and both
      with a worse bad:good ratio than any learned backend). Tuning closes the gap
      between the classical methods themselves; it does not close the classical-vs-learned
      gap. This is the fairer, defensible version of the comparison for the report.
    - **HITNet has the best bad:good ratio after WAFT** (0.20) and the lowest latency of
      the three learned backends (37 ms) — matches the pipeline's choice of it as the
      default.
    - **RAFT-Stereo and WAFT-Stereo have nearly identical photometric quality and valid
      fraction** (expected — closely related architectures) but WAFT has a notably
      better point-cloud bad:good ratio (0:30 vs 2:30) and the lowest median Hs (0.29 m)
      of all backends, closest to the altimeter's cross-validated ≈0.4-0.6 m once the
      dropped-frame bias is considered.
    - **Both RAFT-Stereo and WAFT-Stereo measured ~114-115 ms median latency here** —
      above the "sub-100 ms" design target quoted in `ros2_raftstereo_disparity.py`'s
      docstring. Caveat honestly: this was measured with the full lean pipeline
      (rectifier, point cloud, wave node) competing for the same GPU/CPU, not the
      disparity node in isolation — a *pipeline* latency figure, not a clean per-model
      benchmark number.
    - Photometric error is fairly flat (~12.5-14.3) across every backend that produces
      any valid disparity at all — it mainly distinguishes "works at all" from
      "doesn't," not fine-grained quality between the working backends. The point-cloud
      bad:good ratio and Hs values resolve that finer distinction better; don't
      over-read small photo_err differences between backends.

    Raw per-frame data: `Scripts/disparity_backend_compare_out/<backend>_quality_raw.csv`.
    Summary table: `Scripts/disparity_backend_compare_out/disparity_backend_comparison.csv`.
  - **Pros/cons discussion**: classical methods need no training data and are
    predictable, but plateau in quality on water no matter how tuned (even SGBM's global
    smoothness can't recover texture that isn't there); CUDA-SGM buys GPU speed without
    a quality jump over SGBM. The learned backends generalize better to the
    textureless/specular case because they don't rely purely on local photometric
    matching, at the cost of needing a GPU, a checkpoint, and (for WAFT) extra Python
    deps (`timm`/`peft`/`yacs`) and care around mixed-precision support on older GPUs
    (Turing/RTX 2060 lacks the bf16 support some attention kernels assume).
  - Briefly note, without dwelling on it: a handful of integration issues had to be
    resolved along the way for the pipeline to run reliably at all (bag playback
    requiring the Zenoh RMW, navigation-source migration, lever-arm calibration) — these
    are necessary plumbing, not the interesting engineering result, so they get at most
    one summary paragraph rather than individual case studies.

  **Citations for this subsection — checked against `refs.bib`, not invented:**
  - Already present and usable as-is: `hirschmuller2008stereo` (Hirschmüller, the
    original Semi-Global Matching paper — the algorithm behind both `ros2_sgbm_disparity.py`
    and, on the GPU, `cv2.cuda.StereoSGM`); `app12157447`, `jmse12010197`,
    `li2025thermal` (stereo vision specifically applied to water/maritime surfaces —
    good support for the "why is stereo on water hard" framing); `nowak2022weavenet`
    (depth completion, tangential but usable for the deep-learning angle).
  - **Resolved**: `DBLP:journals/corr/abs-2007-12140` (HITNet, Tankovich et al.),
    `DBLP:journals/corr/abs-2109-07547` (RAFT-Stereo, Lipson/Teed/Deng), and
    `wang2026waftstereowarpingalonefieldtransforms` (WAFT-Stereo, Wang & Deng) were added
    to `refs.bib` — verified the arXiv IDs (2007.12140, 2109.07547, 2603.24836) match the
    actual PDFs in `Papers/`, so these are real, not fabricated.
  - **Added and read in full:** `bergamasco2017wass` (Bergamasco et al., "WASS: An
    open-source pipeline for 3D stereo reconstruction of ocean waves", Computers &
    Geosciences 107, 2017) and `ichikawa2024seasurface` (Ichikawa et al., "Sea Surface
    Height Measurements Using UAV Altimeters with Nadir LiDAR or Low-Cost GNSS
    Reflectometry", Remote Sensing 16(23), 2024). Both give concrete, citable material
    for specific design choices in this project (see below) rather than generic
    background.

  **What WASS actually contributes to the Architecture/Validation narrative:**
  - WASS's dense-stereo step uses Hirschmüller's SGM (`hirschmuller2008stereo`) for the
    same reason this project's `ros2_sgbm_disparity.py`/`ros2_sgm_cuda_disparity.py` do —
    it is described there as the standard trade-off between local block matching and
    fully global methods, which validates SGM/SGBM as the right classical baseline to
    compare the learned backends against, not an arbitrary choice.
  - WASS estimates its mean sea-plane via **RANSAC** (random 3-point plane hypotheses +
    inlier counting) followed by weighted least-squares refinement. This is a useful,
    concrete point of contrast: `ros2_pointcloud_waves.py` deliberately does NOT use
    RANSAC for exactly this step (see `CLAUDE.md`'s "robust mean-surface plane —
    iterative sigma-clipped least-squares, NOT fixed-band RANSAC, which clips waves
    taller than its band"). WASS's setup is a tighter, near-fixed-platform geometry where
    that's less of an issue; citing `bergamasco2017wass` here lets the report justify the
    sigma-clipped-LS choice against a named, published alternative instead of asserting
    it unsupported.
  - WASS also filters disparity-map outliers with a graph built over 3D-point z-axis
    continuity — directly comparable to this project's per-frame `max_frame_hs` quality
    gate (same goal, different mechanism: WASS prunes points/edges within a frame, this
    project drops whole bad frames). Worth one sentence in Architecture noting the same
    "non-Lambertian water surface produces spurious stereo points" problem was
    independently solved twice, differently.
  - WASS's reported accuracy (quantization error of a few mm, RMS reconstruction error
    sub-cm) was achieved with a controlled, near-fixed oceanographic-tower/vessel rig
    (2.5 m baseline, 50° look-down angle, 15 Hz, 5 MP cameras, baseline-to-distance ratio
    tuned to ~0.10). This is a useful honest contrast for Validation: this project's much
    higher disparity-failure/bad-frame rate (~35% of frames before gating) is plausibly
    explained by a longer oblique range (2-99 m vs. WASS's tight near-field design) and a
    less specialized rig, not just a weaker disparity backend — worth saying explicitly
    rather than implying the backends alone are the limiting factor.

  **What the Ichikawa et al. UAV-altimeter paper contributes:**
  - Their Table 4 result — nadir LiDAR has low per-sample noise (SD 0.13-0.25 m) but
    ~50-70% data dropout when the platform tilts (because off-nadir returns don't come
    back to the sensor), while the low-cost GNSS-R altimeter has high per-sample noise
    (SD ~1.6-1.8 m) but a comparably small mean error once enough reflection points are
    averaged — is a strong, citable parallel to two things already found in this
    project: (1) the Lightware/rfbeam altimeters' own outlier/dropout behaviour (see
    "Altimeter unit re-verification" and the unfiltered "100+ m" spikes in `CLAUDE.md`),
    and (2) the general principle behind band-limiting the altimeter wave spectrum
    instead of trusting raw single-sample range — both this project and Ichikawa et al.
    independently lean on averaging/robust statistics to recover a usable signal from a
    noisy single-point range sensor.
  - Their finding that LiDAR dropout is driven specifically by **platform tilt** (off-
    nadir returns are lost) is a good citable explanation for why a boat's roll/pitch
    matters for altimeter reliability generally, supporting this project's own
    tilt-correction step in `ros2_altimeter_waves.py`.
  - Use this paper in Validation as scientific support for the cross-validation
    methodology itself (independent sensors checked against each other / a reference)
    rather than just as a UAV-specific citation.
- **Solution.tex** — user-facing walkthrough: `start_pipeline.sh` tmux session layout,
  RViz views (point cloud, pose/path, disparity), the four interchangeable disparity
  backends as a deliberate quality/speed/hardware tradeoff. Screenshots of RViz,
  disparity, and point-cloud outputs.
- **Validation.tex** — the strongest material, presented as method → bug found → fix →
  before/after numbers:
  - Two independent wave estimators (altimeter temporal spectrum vs. point-cloud spatial
    FFT) used to cross-validate each other, since no ground truth is available.
  - Noise-floor / band-limiting fix on the altimeter spectrum (Hs dropped from a
    physically impossible ≈2.3 m to ≈0.4-0.6 m).
  - Per-frame quality gate on the point cloud (drops disparity-failure frames; Hmax/Hs
    ratio went from 4.6 to a physical 2.7).
  - Range filter + median aggregation reconciling point-cloud Hs with the altimeter's.
  - Doppler/encounter-frequency correction that resolved the long-standing Tp
    disagreement between the two estimators (2.1 s encounter → 3.2 s true, vs. the
    cloud's independent 3.7 s, agreement to ~10-13%).

## 4. Conclusions (~2-3 pg)

- **Results.tex** — working end-to-end pipeline; six interchangeable disparity backends
  evaluated to find one that copes with the low-texture/specular sea surface; two
  cross-validated, physically plausible wave estimates (Hs≈0.4-0.6 m, Tp≈3.2 s) on real
  boat-collected data; map back to the stated objectives.
- **Lessons.tex** — value of verifying every sensor/topic against raw bag data before
  trusting vendor docs; unit and convention bugs (extrinsic vs. intrinsic Euler, wrong
  field read as cm vs. m) are common and only caught by empirical cross-checks; the
  importance of independent cross-validation when no ground truth exists.
- **Future.tex** — open items already tracked: `rslidar` clock-domain fix + TF wiring,
  GPS-antenna → datum absolute calibration, more robust wave-direction estimation (current
  3-altimeter array baseline is too short/noisy), validation on a rougher-sea bag.
- **ODS.tex** — SDG 13 (Climate Action / ocean monitoring) and SDG 9 (Industry,
  Innovation and Infrastructure); short paragraph linking to maritime safety and
  autonomous platform development.

## Bibliography

Already populated in `refs.bib` (wave theory — Airy, Stokes, JONSWAP/Pierson-Moskowitz;
estimation theory — Kalman/particle filters; sensing surveys — LiDAR, mmWave, stereo wave
measurement). Cite inline while writing Development/Architecture/Validation; no bulk
additions planned.

## Annexes (if space allows within the 25-page cap)

- Pipeline architecture diagram.
- RViz screenshots (point cloud, pose/path, disparity).
- Key plots: PSD before/after band-limiting; Hs convergence after the bad-frame gate;
  encounter-vs-true period correction.
