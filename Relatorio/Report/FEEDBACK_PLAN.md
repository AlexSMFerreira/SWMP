# Report v1 — Feedback Revision Plan

Plan for addressing tutor (José) annotations in `capstoneproject_v1_feedback.pdf`.
Report pages below are the **printed** page numbers; source files are under `sections/`.

Author language note: all comments are in Portuguese; each item below gives the
English gist plus the concrete action.

---

## Recurring / cross-cutting themes (fix everywhere, not just where flagged)

1. **Every figure must be referenced in the text *before* it appears.** Flagged
   explicitly for Fig 1, Fig 3 (referenced only on p18), Fig 4, Fig 5, Fig 12, Fig 13,
   and the Doppler/pipeline figures. → Audit *all* figures: add a `Figure~\ref{}`
   mention in the body text preceding each `\begin{figure}`, and never introduce a
   figure for the first time in the Conclusion.
2. **Tone: several passages read as AI-generated ("está muito a chatgpt").** Rewrite
   flagged sentences in a plainer, more technical voice; remove rhetorical framings
   (e.g. `"works at all" vs "does not"`, arrow notation in prose).
3. **Figure quality / usefulness.** Many disparity/point-cloud figures are low
   resolution and hard to read; several need the source camera image(s) shown
   alongside for context.

---

## Section 1 — Introduction

- **Fig 1 (WIG vehicle), p3** — *"Acrescentar referência da imagem"*: add an in-text
  reference to Figure 1 in `sections/Introduction/Context.tex`.

## Section 2 — Methodology

- **Title "2 Methodology and activities carried out", p6** — *"'and activities carried
  out' → não é necessário"*: shorten chapter title to **"Methodology"**
  (`capstoneproject.tex` / `sections/Methodology/*` `\section` title).
- **Subsection "2.1 Methodology used", p6** — *"Não precisas de subcapítulo aqui. Passa
  logo de capítulo para texto"*: remove the `2.1 Methodology used` subsection heading;
  let the chapter flow straight into text (`sections/Methodology/Methodology.tex`).
- **"measure → implement → re-verify", p6** — *"Reescrever frase sem setas"*: rewrite
  without arrow notation (prose form, e.g. "characterise the data, implement, then
  re-verify"). Same fix applies to the identical arrow phrasing in Validation (Sec 3.4).
- **Table 1 project timeline, p7** — *"Isto é necessário? Não basta apenas um Gantt
  chart com explicação sucinta das atividades em texto?"*: replace the milestone table
  with a **Gantt chart + a short prose summary** of activities
  (`sections/Methodology/Activities.tex`).

## Section 3 — Solution development

### 3.1 Requirements (p8)
- **Constraints bullet (sea surface is adversarial)** — *"Não considerava isto como
  constraint, é o motivo do tema da tese"*: reframe Constraints. Move the "sea surface
  is non-Lambertian" point out of Constraints (it's the problem motivation, belongs in
  Context/Architecture). Real constraints to list instead: **available hardware
  (PC/GPU, sensors, calibration tools, test rig), number of available datasets, and
  inability to run more test campaigns due to time / adverse weather.**
  (`sections/Development/Requirements.tex`)

### 3.2 Architecture and technologies (p9)
- **Missing technology analysis** — *"Saltaste logo para a implementação... era uma
  mais valia teres uma análise das tecnologias existentes"*: add, before the
  implementation description, a **survey of candidate sensing technologies**: briefly
  introduce the sensors and add a **table of advantages/disadvantages of each**, then
  **justify the final sensor choice**. (`sections/Development/Arquitecture.tex`)
- **Fig 3 (frame tree) referenced only on p18** — reference it in the text where it is
  introduced (Architecture), not 9 pages later.
- **Fig 4 (typical left frame), p10** — *"Não é mencionada no texto. Não tens imagens
  com melhor definição? Podes mostrar a imagem da esquerda e da direita... para o mesmo
  instante têm diferentes features (non-Lambertian)"*: (a) reference it in text; (b) use
  a higher-resolution image; (c) show **left + right frames at the same instant** to
  visually demonstrate the non-Lambertian mismatch that breaks photometric matching.
- **Fig 5 (Hough horizon), p11** — *"Não é mencionada no texto. Melhor definição? Falta
  explicar a transformada de Hough... o motivo de excluíres o céu. Podes referir que o
  horizonte é útil para calibrar as câmaras online"*: (a) reference in text; (b) higher
  resolution; (c) add a paragraph **explaining the Hough transform and why the sky is
  excluded**; (d) note the horizon can be used for **online camera calibration**
  (inter-camera rotation).
- **Table 2 caption / "end-to-end application-level quality signal" sentence, p12** —
  *"Frase está confusa"*: rewrite the confusing sentence describing the PC bad:good /
  wave-usability metric more clearly.
- **Figs 6–11 (disparity map + point cloud), p13–15** — *"Ajuda ter pelo menos uma das
  imagens das câmaras (idealmente as duas) para saber a qualidade dos mapas de
  disparidade e das pointclouds"*: for each backend figure, include at least one (ideally
  both) source camera image next to the disparity map and point cloud.
- **Fig 10/11 point cloud, p15** — *"Não tens uma melhor imagem da pointcloud onde se
  possa ver melhor as ondas? Nesta imagem parece quase plano"*: replace with a
  point-cloud view where the wave structure is actually visible (better viewpoint /
  colour-by-elevation).
- **"point-cloud wave estimator", p15** — *"Onde é que está este estimador?"*: make the
  reference explicit — point to the section/node where the point-cloud wave estimator is
  described (forward-reference to Sec 3.4 / the node), so the term isn't used before it's
  defined.
- **"it mainly distinguishes 'works at all' from 'does not'", p15** — *"Está muito a
  chatgpt"*: rewrite in a technical tone without the rhetorical quoted phrasing.
- **"Tuning matters more between the classical methods", p16** — *"Que parâmetros
  mudaste? Porque é que funcionou melhor depois de alterar os parâmetros? Mostrar que
  percebeste o algoritmo bem"*: add concrete detail on **which parameters were tuned
  (per backend) and why each change improved the result**, demonstrating understanding
  of the algorithms and the parameters that drive their output (space permitting).
- **WASS comparison, p17** — *(sticky note)* *"Podes acrescentar que WASS não funciona
  em tempo real. Nunca usei WASSfast, mas se tiveres tempo experimenta"*: (a) add that
  **WASS is not real-time**; (b) *optional/if time*: try **WASSfast** and report its
  runtime + point-cloud quality. (`sections/Development/Arquitecture.tex` / Solution) IMPORTANT: wassfast would not work in this case because it relies on the camera relative position to the mean sea plane stay constant. mentioned in the wassfast paper, which is in the refs.bib

### 3.4 Validation (p18)
- **Fig 12 (RViz), p18** — *"Imagem não é mencionada no texto"*: reference Figure 12 in
  the body text. (`sections/Development/Validation.tex` / `Solution.tex`)

## Section 4 — Conclusions

- **Fig 13 (altimeter pipeline) & Doppler/pipeline figures, p21–22** — *"Figura não é
  mencionada no texto. Não podes demonstrar pela primeira vez a arquitetura na
  conclusão"*: these architecture/processing-pipeline figures must not first appear in
  the Conclusion. **Move the pipeline/processing figures (Fig 13/14/15) into Section 3
  (Architecture / Validation)** where the algorithms are described, and reference them
  there. Keep the Conclusion figure-light.

---

## Suggested execution order

1. ✅ **DONE — Structural/text-only, low effort:** title trimmed to "Methodology",
   removed 2.1 heading, de-arrowed the two `measure→implement→re-verify` phrases,
   reframed Constraints, rewrote the two "chatgpt"/confusing sentences.
2. ✅ **DONE — Figure referencing pass:** in-text refs added for Figs 1, 3, 4, 5, 6–11,
   12, 13–15; Figs 4/5 reordered after their lead-in text; Figs 13/14/15 moved next to
   the paragraphs they illustrate (now render in Sec 3.4, before Conclusions).
3. **New content (DONE):**
   - ✅ technology/sensor comparison table (3.2, adapted from the bibliographic review).
     Full per-sensor survey prose left as-is by request.
   - ✅ Hough transform explanation + sky-exclusion rationale + online-calibration note (3.2).
   - ✅ parameter-tuning rationale per backend (3.2) — grounded in the real node params
     (numDisparities, textureThreshold=0, X-Sobel/preFilterCap, block size, speckle, WLS,
     MODE_HH, disp12 control).
   - ✅ WASS-not-real-time note + WASSfast N/A (needs constant camera-to-mean-sea-plane
     geometry; cite rs13183780).
4. **Figure regeneration (IN PROGRESS):**
   - ✅ **Backend figures (Figs 6–11)** regenerated as 4-panel combined figures
     (rectified left+right on top, disparity bottom-left, bird's-eye point cloud
     bottom-right; near-field 40 m; all six on the **same frame 0**, tuned classical
     params). Frame 0 gives the richest output (brighter sunset light + visible skyline
     for context, dense classical disparity — SGBM 70k, SGM-CUDA 7k points) with clean
     WAFT/RAFT/HITNet surfaces.
     **Rectification is now the online horizon-corrected variant** (mirrors
     `Nodes/ros2_stereo_rectifier_horizon.py`, differential mode: right aligned to left,
     ~1.2° inter-camera roll + ~6.5px offset removed per frame) — `backend_offline_figure.py`
     applies it by default (`--no-horizon-correct` to disable).
     New generator `Scripts/backend_offline_figure.py --frame 0 all` (all six backends,
     faithful to the live nodes); figures `Relatorio/Report/figures/backend_figure_{sbm,
     sgbm,sgm_cuda,hitnet,raft,waft}.png`; captions updated.
     Note: **StereoBM produces an empty cloud on every frame of this dark bag** (raw
     matches ~5 at nDisp=128, rejected to 0 by the tuned uniqueness/disp12/speckle
     filters) — genuinely the weakest backend here; caption says so and points to Table 2.
     Investigated whether SBM could hit its Table-2 valid % (19.2%): confirmed that is a
     **406-frame live-node mean, not a single-frame value**; best achievable offline is
     ~6.6% (far WLS-fill, empty near field), and a visibly non-empty cloud needs
     non-standard CLAHE + nDisp=16 and still looks broken (sparse off-centre streak).
     **Decision (user): keep SBM's figure honestly empty** — faithful to the node and to
     the report's thesis that classical block matching fails on the sea surface.
   - ✅ Orphaned per-backend images removed (13 files: `{backend}_disparity/pc.png`,
     `waft_figure_...png`).
   - ✅ **Fig 4 (typical frame)** regenerated as the **left+right rectified pair at the
     same instant**, full-resolution (2464×2056 rectified → ~2741px-wide PNG, up from the
     old 282px), so the specular glints visibly differ between the two synchronized views
     (directly answers the tutor's "mostra esquerda e direita... têm diferentes features").
     Caption + lead-in text updated; widened to `\textwidth`. (An auto-aligned zoom-inset
     variant was tried and dropped: template-matching the low-texture water is itself
     unreliable — the very failure the report is about — so the clean two-panel pair is the
     honest figure.) Generator: `Scripts/frame_and_horizon_figure.py`.
   - ✅ **Fig 5 (horizon)** regenerated full-resolution (up from 277px) with the excluded
     sky region shaded and **both** the Hough-detected horizon (yellow) and the margined
     mask boundary (red) drawn, computed by the real `stereo_common.HorizonMasker` (same
     code the disparity nodes use). Margin set to 1% of image height for the figure
     (`horizon_margin_pct=0.01`); caption notes the safety margin. Same generator.

5. ✅ **DONE — Timeline → Gantt chart** in Methodology: replaced the milestone table
   with a `pgfgantt` 15-week chart (7 activity threads) + a concise prose summary
   (`sections/Methodology/Activities.tex`; `pgfgantt` added to preamble).
