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

1. **Structural/text-only, low effort:** title trim, remove 2.1 heading, de-arrow the
   two phrases, reframe Constraints, rewrite the two "chatgpt"/confusing sentences.
2. **Figure referencing pass:** add in-text refs for Figs 1, 3, 4, 5, 12, 13; move
   Fig 13/14/15 out of Conclusions into Sec 3.
3. **New content:** technology/sensor comparison table + justification (3.2); Hough
   transform explanation + sky-exclusion rationale (3.2); parameter-tuning rationale per
   backend (3.2); WASS-not-real-time note.
4. **Figure regeneration:** higher-res Fig 4 (left+right same instant), Fig 5, and
   disparity/point-cloud figures with source camera image alongside; better point-cloud
   viewpoint showing waves. 

   For the backend figure regeneration, let's do as we did to generate the waft figure with the 4 parts and the pointcloud plot, but remove near field residuals plot and place the left and right camera images on top, bottom left disparity and bottom right pointcloud plot.

5. **Timeline → Gantt chart** in Methodology.
6. *(Optional, time permitting):* WASSfast experiment.
