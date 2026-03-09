


review the response if the ticket landed in the repo as expected (zipped repo attached), if not, fold change requests into the next ticket. now, proceed with this repo agent ticket:

# WA-01 — Detailed outline, figure plan, and section-level claim map

## Goal

Lock the narrative architecture before prose drafting.

## Repo agent tasks

Create or fill:

* `paper/outline.md`
* `paper/figure_plan.md`
* update `paper/evidence_map.csv`

### `paper/outline.md`

Build a detailed outline with:

* section titles
* subsection titles
* 1–3 sentence purpose for each subsection
* explicit placement of the “six primitives” paragraph in the opening section

The target section structure should be:

1. Introduction
2. Six Birds recap and a strict definition of currency
3. Currency–constraint duality across closure layers
4. Finite-state realization and measurement pipeline
5. Results

   * Honest audits under coarse-graining
   * Price emergence and slack
   * Currency dimension grows with resolution
   * Proxy currencies fail
   * Budgets buy object stability
6. Discussion
7. Conclusion
   Appendices

### `paper/figure_plan.md`

For each final figure, record:

* target section/subsection
* one-sentence narrative role
* core quantitative takeaway
* claim IDs supported

### `paper/evidence_map.csv`

Update so every claim row has a nonempty `section_target`.

## Pass criteria

* outline exists and covers the full manuscript
* every figure has a planned location
* every claim-ledger row has a section target

## Repo agent response must include

* modified file paths
* build exit code
* the section/subsection list only
* a 5-line figure placement summary

---






review the response if the ticket landed in the repo as expected (zipped repo attached), if not, fold change requests into the next ticket. now, proceed with this repo agent ticket:

# WA-02 — Title, abstract, and introduction

## Goal

Draft the front matter and introduction so the paper already has its central narrative.

## Repo agent tasks

I will provide the full prose for:

* `paper/sections/00_title_abstract.tex`
* `paper/sections/01_intro.tex`

The repo agent should:

* insert the text verbatim
* add minimal LaTeX markup as needed
* ensure clean compilation
* add citations only if explicitly present in the supplied text
* update `paper/references.bib` only for citations actually used

### Required content in this ticket

The intro must include:

* the recurring “currencies across disciplines” puzzle
* the gap: why local duality is not yet a cross-layer explanation
* the six primitives listed early
* the paper thesis:

  * currencies are structural, not metaphors
  * lower-layer currencies become higher-layer constraints
  * higher-layer currencies emerge as dual prices
* a short “position in the Six Birds series” paragraph

## Pass criteria

* abstract and introduction compile cleanly
* the six primitives appear in the first two pages
* no unresolved citation keys introduced by the ticket

## Repo agent response must include

* modified file paths
* build exit code
* current PDF page count
* the abstract only, as rendered LaTeX source excerpt

---







review the response if the ticket landed in the repo as expected (zipped repo attached), if not, fold change requests into the next ticket. now, proceed with this repo agent ticket:

# WA-03 — Framework section: strict currency definition and the morphism principle

## Goal

Write the conceptual core of the paper.

## Repo agent tasks

I will provide the full prose for:

* `paper/sections/02_framework.tex`

The repo agent should insert it verbatim and keep notation consistent.

### Required content

This section must contain:

* the strict definition of currency

  * C1 constraint currency
  * C2 ledger currency
  * C3 potential currency
  * C4 dual currency
* structural vs proxy currencies
* state currency vs path currency
* the P5–P6–P2 chain
* the central principle:

  * lower-layer currency → higher-layer constraint
  * higher-layer constraint → shadow price
* canonical examples:

  * energy → temperature / free energy
  * bits → rate–distortion multiplier
  * budget → price
* the slogan-level framing:

  * a currency is what a layer spends to stay closed

## Pass criteria

* framework section compiles and cross-references cleanly
* notation is consistent with intro
* any claims made here have a corresponding entry or support path in `paper/evidence_map.csv`

## Repo agent response must include

* modified file paths
* build exit code
* current PDF page count
* subsection heading list from Section 2

---











review the response if the ticket landed in the repo as expected (zipped repo attached), if not, fold change requests into the next ticket. now, proceed with this repo agent ticket:

# WA-04 — Methods / finite-state realization / formal anchor

## Goal

Write the section that tells readers how the principle is measured and what is formally proved.

## Repo agent tasks

I will provide the full prose for:

* `paper/sections/03_methods.tex`

The repo agent should insert it verbatim and wire up figure/table refs if already present.

### Required content

This section must cover:

* finite Markov substrate
* lenses and coarse-graining
* path-reversal KL audit
* cycle-affinity audit
* MaxCal single-constraint dual recovery
* MLE recovery of lambda
* packaging endomap and idempotence defect
* how the final evidence pack is organized
* what the Lean theorem proves and what it does not prove

This section must explicitly mention the proved theorem:

* `CurrencyMorphism.finiteKL_map_le`

But it must describe it carefully as a finite-`klFun` / finite-KL deterministic pushforward monotonicity anchor, not as the whole cross-layer theorem.

## Pass criteria

* methods section compiles
* Lean theorem is named correctly
* all quantitative procedure descriptions correspond to frozen code/results only

## Repo agent response must include

* modified file paths
* build exit code
* current PDF page count
* the paragraph containing the Lean theorem mention

---










review the response if the ticket landed in the repo as expected (zipped repo attached), if not, fold change requests into the next ticket. now, proceed with this repo agent ticket:

# WA-05 — Results I: honest audits and price emergence

## Goal

Draft the first half of Results around the two cleanest exhibits.

## Repo agent tasks

I will provide the full prose for the first part of:

* `paper/sections/04_results.tex`

The repo agent should:

* insert prose
* place the DPI and budget figures
* add figure environments and labels
* add one compact summary table if specified in the supplied text

### Required content

This ticket must write:

1. **Honest audits under coarse-graining**

   * use `docs/experiments/final/dpi_summary.csv`
   * include the no-violating-seed-run message
   * report the key magnitude:

     * `min_delta_mean = 0.0010685921959992`
2. **Price emergence and slack**

   * use `docs/experiments/final/budget_summary.csv`
   * report:

     * `spearman_rho_mean = -0.9992304731449788`
     * `n_tail_zero_lambda = 5`

### Figures

Place:

* `fig_dpi_scan.png`
* `fig_budget_sweep.png`

with strong captions, but captions can be finalized later.

## Pass criteria

* both figures appear in the PDF
* all reported numbers match the frozen summary files
* results prose stays tied to these numbers only

## Repo agent response must include

* modified file paths
* build exit code
* current PDF page count
* the two figure labels used
* the two quantitative lines as they appear in the manuscript

---












review the response if the ticket landed in the repo as expected (zipped repo attached), if not, fold change requests into the next ticket. now, proceed with this repo agent ticket:

# WA-06 — Results II: currency ladder, proxy ablation, idempotence-budget

## Goal

Complete the Results section with the remaining three exhibits.

## Repo agent tasks

I will provide the rest of the prose for:

* `paper/sections/04_results.tex`

The repo agent should:

* append/insert the supplied text
* place the remaining three figures
* add any requested table environments

### Required content

This ticket must cover:

1. **Currency dimension grows with resolution**

   * from `ladder_summary.csv`
   * report `0.0, 5.0, 11.0`

2. **Proxy currencies fail**

   * from `proxy_summary.csv`
   * report:

     * `mean_nll_baseline = 5.287560793904439`
     * `mean_nll_good = 2.2196445818681854`
     * `mean_nll_bad = 2.76939916766148`
     * `rel_adv = 0.10397130306795872`
     * `std_lam_good = 2.1202318751289444`
     * `std_lam_bad = 3.36067109746149`

3. **Budgets buy object stability**

   * from `idempotence_summary.csv`
   * report:

     * `spearman_rho = -1.0` effectively
     * defect reduction from `0.11596447561541763` to `0.001443300174998845`

### Figures

Place:

* `fig_currency_ladder.png`
* `fig_proxy_ablation.png`
* `fig_idempotence_budget.png`

## Pass criteria

* Results section is complete and compiles
* all three figures appear
* all numeric claims match the frozen summaries

## Repo agent response must include

* modified file paths
* build exit code
* current PDF page count
* a 3-line summary of the reported metrics for ladder / proxy / idempotence exactly as written into the paper

---







review the response if the ticket landed in the repo as expected (zipped repo attached), if not, fold change requests into the next ticket. now, proceed with this repo agent ticket:

# WA-07 — Discussion and conclusion

## Goal

Write the interpretive end of the paper without overselling.

## Repo agent tasks

I will provide the full prose for:

* `paper/sections/05_discussion.tex`
* `paper/sections/06_conclusion.tex`

The repo agent should insert it and keep citations minimal.

### Required content

Discussion must include:

* why temperatures, prices, regularizers, and attention weights are the same mathematical kind of object
* why this is a cross-layer story, not just “Lagrange multipliers exist”
* why proxy currencies fail
* why cycle-space can appear before strong affinity norms
* limitations:

  * layer-relative notion
  * finite-state realization
  * proxy-ablation is controlled/synthetic
  * Lean theorem is finiteKL-form, not a full mechanization of the whole paper
* a short “what this adds to the Six Birds series” paragraph

Conclusion should:

* restate the main principle cleanly
* summarize the five empirical signatures
* end on the line that stable closures develop something price-like

## Pass criteria

* discussion and conclusion compile
* limitations paragraph is explicit and honest
* no unsupported new claims are introduced

## Repo agent response must include

* modified file paths
* build exit code
* current PDF page count
* the conclusion’s final paragraph only

---




review the response if the ticket landed in the repo as expected (zipped repo attached), if not, fold change requests into the next ticket. now, proceed with this repo agent ticket:

# WA-08 — Appendices, reproducibility text, bibliography completion

## Goal

Add the back matter that makes the paper self-contained and referee-resistant.

## Repo agent tasks

I will provide the prose for:

* `paper/sections/appendix_a_evidence.tex`
* `paper/sections/appendix_b_lean.tex`
* `paper/sections/appendix_c_repro.tex`

The repo agent should:

* insert text
* complete `paper/references.bib`
* ensure all citations resolve
* add appendix table(s) if requested

### Required appendix content

Appendix A:

* evidence map / claim ledger summary
* where each main exhibit comes from

Appendix B:

* short statement of the Lean formalization
* theorem signature or prose description
* what is formalized vs not formalized

Appendix C:

* reproducibility audit summary
* frozen assets and manifest structure
* where figures and summary CSVs came from

### Bibliography rule

Only include references actually cited in the manuscript.

## Pass criteria

* all citations resolve
* appendices compile
* bibliography has no unused or obviously extraneous entries if easy to avoid

## Repo agent response must include

* modified file paths
* build exit code
* unresolved refs/cites count
* bibliography entry count
* appendix titles list

---









review the response if the ticket landed in the repo as expected (zipped repo attached), if not, fold change requests into the next ticket. now, proceed with this repo agent ticket:

# WA-09 — Final editorial pass and submission-grade compile

## Goal

Turn the assembled draft into a clean, coherent manuscript.

## Repo agent tasks

Perform a manuscript-only polish pass across `paper/`.

### Required cleanup

* remove repeated sentences and duplicate claims
* unify notation (`u`, `lambda`, `Sigma_T`, `beta_1`, etc.)
* standardize caption style
* ensure title/subtitle formatting is correct
* ensure the six primitives appear early and only as much as needed
* ensure figure order matches first mention
* ensure intro promises match results delivered
* ensure discussion does not outrun the evidence
* ensure no TODO / FIXME / placeholder text remains
* ensure bibliography and cross-references are clean
* produce final compiled PDF

Optionally create:

* `paper/final_checklist.md`

with boxes for:

* abstract
* section order
* figure order
* citation resolution
* claim/evidence alignment
* Lean appendix
* reproducibility appendix

## Pass criteria

* final manuscript compiles with exit code 0
* no unresolved refs/cites
* no TODO/FIXME/placeholders
* PDF is complete and readable
* title is exactly:

**To Spend a Stone with Six Birds: Currency–Constraint Duality and Shadow Prices Across Closure Layers**

## Repo agent response must include

* modified file paths
* build exit code
* unresolved refs/cites count
* final PDF page count
* path to final PDF
* the final table of contents lines only

---

# Optional only after WA-09

If we later want submission extras, that should be a separate mini-phase:

* cover letter
* journal-specific formatting
* arXiv abstract block
* author checklist

But I would not mix that into manuscript drafting.

The next move is to issue **WA-00**.
