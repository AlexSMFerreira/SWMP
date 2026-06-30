# TODO before submission

Things the report text already flags inline (search for `[TODO:` in the `.tex` files) plus a
few editorial items. Nothing below is fabricated data — every number already in the report
came from `CLAUDE.md`/`thought_process/*` or the `REPORT_PLAN.md` comparison table.

## Figures still missing (report currently has placeholder `[TODO: ...]` text instead)

1. **Pipeline architecture diagram** (`sections/Development/Arquitecture.tex`). A block
   diagram of RectifyNode → DisparityNode → PointCloudNode → PoseBroadcasterNode, plus the
   altimeter publisher and the two wave-parameter nodes, with topic names on the arrows.
2. **RViz screenshots** (`sections/Development/Solution.tex`): point cloud view, pose/path
   overlay, and the live disparity output. Run `./start_pipeline.sh` and grab these from the
   `swmp` tmux session's RViz window.
3. **Supporting validation plots** (`sections/Development/Validation.tex`):
   - Altimeter PSD before vs. after band-limiting (the noise-floor fix).
   - Point-cloud `Hmax`/`Hs` ratio before vs. after the bad-frame gate.
   - Encounter-period vs. true-period correction (the Doppler fix).

   These can likely be regenerated from the existing diagnostic scripts
   (`Scripts/altimeter_wave_diagnostic.py`, `Scripts/pointcloud_residual_diag.py`,
   `Scripts/pointcloud_hs_diag.py`) — check whether they already save plots or only print
   numbers; may need a small `matplotlib.savefig` added to each.

## Editorial / decisions for you to make

4. **Title page date** — currently set to "June 29, 2026" (today). Confirm the actual
   submission date before printing.
5. **Photo/headshot or other front-matter** — none added; only present if your template
   requires it (current template doesn't).
6. **Page budget** — current draft compiles to 17 pages (vs. the 25-page cap), so there is
   room for the figures above plus the annexes section from `REPORT_PLAN.md` (architecture
   diagram, RViz screenshots, key plots) if you want to also duplicate small thumbnails there
   instead of only inline.
7. **Double-check the disparity comparison table numbers** in
   `sections/Development/Arquitecture.tex` against your own copy of
   `Scripts/disparity_backend_compare_out/disparity_backend_comparison.csv` before
   submission — they were transcribed from `REPORT_PLAN.md`, not re-read from the CSV during
   this writing pass.
8. **Proposal deviations** — the report (Objectives, Results, Future Work) notes that the
   original proposal's Phase 4 (short-term wave-occurrence *forecasting*) was descoped in
   favour of consolidating wave-parameter *estimation*, and that data comes from a boat
   platform rather than an in-flight airship. Confirm this framing matches how you want to
   present it to evaluators (i.e. as a deliberate scope decision, not a shortfall).

## Not included, on purpose (per your instructions)

- No code listings anywhere in the report.
- No mention of the Zenoh RMW workaround, numpy/OpenCV shadowing bugs, or other
  driver/compatibility plumbing — these are referenced only obliquely as "a handful of
  integration issues" in Architecture, not detailed.
