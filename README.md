# Six Birds: Currency Instantiation

This repository contains the **currency instantiation** for the paper:

> **To Spend a Stone with Six Birds: Currency--Constraint Duality and Shadow Prices Across Closure Layers**
>
> Archived at: https://zenodo.org/records/18926771
>
> DOI: https://doi.org/10.5281/zenodo.18926771

This paper is the economics-and-duality instantiation of the emergence calculus introduced in *Six Birds: Foundations of Emergence Calculus*. It studies when a lower-layer spending coordinate becomes a higher-layer feasibility law, and when a higher-layer dual variable should be interpreted as the shadow price of that inherited budget.

## What this repository provides

The currency instantiation implements:

- **Finite-state Markov laboratory**: controlled reversible and driven kernels for testing pathwise asymmetry, cycle structure, and budget-induced closure
- **Audit measurements**: path-reversal KL diagnostics and cycle-affinity summaries for honest lower-layer directionality currencies
- **Dual-recovery experiments**: MaxCal and conditional-logit pipelines for recovering higher-layer shadow prices from lower-layer budgets
- **Resolution-ladder and proxy-ablation studies**: experiments showing growth of visible currency dimension with resolution and failure of structurally misaligned proxy costs
- **Packaging-stability experiment**: idempotence-defect measurements showing stronger budget enforcement yields more nearly stable packaged objects
- **Lean/mathlib anchor**: a machine-checked finite-KL monotonicity theorem for deterministic pushforwards under coarse observation
- **Frozen evidence pack**: writing-facing CSVs, figures, manifest files, and reproducibility audits under `docs/experiments/final/`

## Scope and limitations

The paper is explicit about what this repository does and does not establish:

- The experiments are controlled finite-state demonstrations; they do not claim a universal empirical law for all scientific systems
- The shadow-price results depend on the chosen finite-state laboratory and measurement protocol; they are not a theorem about arbitrary cross-layer economies
- Proxy failure is demonstrated in a synthetic held-out setting designed to isolate structural alignment, not as a survey of all possible surrogate objectives
- The Lean development anchors one audit monotonicity principle for finite KL under deterministic pushforward; it does not mechanize the full currency--constraint narrative or verify the empirical experiments end to end

## Install

```bash
make setup
cd lean && lake build
```

## Test

```bash
make test
make lint
python scripts/smoke.py
```

## Run experiments

```bash
python scripts/exp_dpi_scan.py
python scripts/exp_budget_sweep.py
python scripts/exp_currency_ladder.py
python scripts/exp_proxy_ablation.py
python scripts/exp_idempotence_vs_budget.py
```

## Freeze evidence pack

```bash
python scripts/final_evidence_pack.py
python scripts/export_final_assets.py
```

The frozen writing-facing artifacts are placed under `docs/experiments/final/`, including:

- `run_manifest.json`
- `claim_ledger.csv`
- experiment summary CSVs
- final manuscript figure PNGs
- `lean_summary.json`
- reproducibility audit outputs

## Run reproducibility audit

```bash
python scripts/repro_audit.py
```

## Build paper

```bash
cd paper && make pdf
```

## Repository layout

- `paper/`: LaTeX manuscript sources
- `scripts/`: experiment drivers, evidence-pack export, and reproducibility tooling
- `src/currencymorphism/`: shared Python package code
- `results/`: run outputs and intermediate experiment artifacts
- `docs/experiments/final/`: frozen evidence pack used by the manuscript
- `lean/`: Lean formalization of the finite-KL monotonicity anchor
- `assets/zenodo_related_works.csv`: Zenodo related-works upload helper derived from the bibliography
