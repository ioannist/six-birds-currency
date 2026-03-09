from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export frozen evidence assets")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest.json from final_evidence_pack (defaults to latest)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="docs/experiments/final",
        help="Output directory for writing-facing final assets",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def find_latest_manifest(root: Path) -> Path:
    candidates = sorted(
        (root / "results" / "final_evidence").glob("pack_*/manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No manifests found under results/final_evidence/")
    return candidates[0]


def ensure_exists(path: Path, desc: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {desc}: {path}")
    return path


def write_one_row_csv(path: Path, row: dict[str, Any]) -> None:
    df = pd.DataFrame([row])
    df.to_csv(path, index=False)


def copy_figure(root: Path, src_rel: str, dst_path: Path) -> None:
    src = (root / src_rel).resolve()
    ensure_exists(src, f"figure source {src_rel}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_path)


def main() -> None:
    args = parse_args()
    root = repo_root()

    manifest_path = (
        find_latest_manifest(root)
        if args.manifest is None
        else ensure_exists((root / args.manifest).resolve(), "manifest")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    outdir = (root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    run_manifest_path = outdir / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summaries = manifest["summaries"]

    dpi = summaries["dpi_scan"]
    budget = summaries["budget_sweep"]
    ladder = summaries["currency_ladder"]
    proxy = summaries["proxy_ablation"]
    idem = summaries["idempotence_budget"]
    lean = summaries["lean"]

    dpi_summary_path = outdir / "dpi_summary.csv"
    write_one_row_csv(
        dpi_summary_path,
        {
            "run_id": dpi["run_id"],
            "min_delta_mean": dpi["min_delta_mean"],
            "max_delta_mean": dpi["max_delta_mean"],
            "n_seeds": dpi["n_seeds"],
            "tol": dpi.get("tol", None),
            "worst_seed_delta": dpi.get("worst_seed_delta", None),
            "dpi_seed_robust_pass": dpi.get("dpi_seed_robust_pass", None),
        },
    )

    budget_summary_path = outdir / "budget_summary.csv"
    write_one_row_csv(
        budget_summary_path,
        {
            "run_id": budget["run_id"],
            "spearman_rho_mean": budget["spearman_rho_mean"],
            "lam_mean_max": budget["lam_mean_max"],
            "n_tail_zero_lambda": budget["n_tail_zero_lambda"],
            "budget_pass": budget["budget_pass"],
        },
    )

    ladder_summary_path = outdir / "ladder_summary.csv"
    write_one_row_csv(
        ladder_summary_path,
        {
            "run_id": ladder["run_id"],
            "beta1_coarse_mean": ladder["beta1_coarse_mean"],
            "beta1_mid_mean": ladder["beta1_mid_mean"],
            "beta1_fine_mean": ladder["beta1_fine_mean"],
            "ladder_pass": ladder["ladder_pass"],
        },
    )

    proxy_summary_path = outdir / "proxy_summary.csv"
    write_one_row_csv(
        proxy_summary_path,
        {
            "run_id": proxy["run_id"],
            "mean_nll_baseline": proxy["mean_nll_baseline"],
            "mean_nll_good": proxy["mean_nll_good"],
            "mean_nll_bad": proxy["mean_nll_bad"],
            "rel_adv": proxy["rel_adv"],
            "std_lam_good": proxy["std_lam_good"],
            "std_lam_bad": proxy["std_lam_bad"],
            "proxy_pass": proxy["proxy_pass"],
        },
    )

    idem_summary_path = outdir / "idempotence_summary.csv"
    write_one_row_csv(
        idem_summary_path,
        {
            "run_id": idem["run_id"],
            "spearman_rho": idem["spearman_rho"],
            "defect_mean_min": idem["defect_mean_min"],
            "defect_mean_max": idem["defect_mean_max"],
            "defect_end_lt_start": idem["defect_end_lt_start"],
            "idempotence_pass": idem["idempotence_pass"],
        },
    )

    lean_summary_path = outdir / "lean_summary.json"
    lean_summary_path.write_text(json.dumps(lean, indent=2), encoding="utf-8")

    fig_dpi = outdir / "fig_dpi_scan.png"
    fig_budget = outdir / "fig_budget_sweep.png"
    fig_ladder = outdir / "fig_currency_ladder.png"
    fig_proxy = outdir / "fig_proxy_ablation.png"
    fig_idem = outdir / "fig_idempotence_budget.png"

    copy_figure(root, dpi["plot_png"], fig_dpi)
    copy_figure(root, budget["plot_png"], fig_budget)
    copy_figure(root, ladder["plot_png"], fig_ladder)
    copy_figure(root, proxy["plot_png"], fig_proxy)
    copy_figure(root, idem["plot_png"], fig_idem)

    claim_rows = [
        {
            "claim_id": "C1",
            "experiment": "dpi_scan",
            "metric_name": "dpi_seed_robust_pass",
            "observed_value": dpi.get("dpi_seed_robust_pass", False),
            "threshold": "True (no violating seed-run)",
            "pass": bool(dpi.get("dpi_seed_robust_pass", False)),
            "primary_artifact": dpi_summary_path.relative_to(root).as_posix(),
            "notes": f"min_delta_mean={dpi['min_delta_mean']:.6e}",
        },
        {
            "claim_id": "C2",
            "experiment": "budget_sweep",
            "metric_name": "spearman_rho_mean",
            "observed_value": budget["spearman_rho_mean"],
            "threshold": "< -0.9",
            "pass": bool(float(budget["spearman_rho_mean"]) < -0.9),
            "primary_artifact": budget_summary_path.relative_to(root).as_posix(),
            "notes": "price emerges monotonically",
        },
        {
            "claim_id": "C3",
            "experiment": "budget_sweep",
            "metric_name": "n_tail_zero_lambda",
            "observed_value": budget["n_tail_zero_lambda"],
            "threshold": ">= 1",
            "pass": bool(int(budget["n_tail_zero_lambda"]) >= 1),
            "primary_artifact": budget_summary_path.relative_to(root).as_posix(),
            "notes": "slack regime yields near-zero lambda",
        },
        {
            "claim_id": "C4",
            "experiment": "currency_ladder",
            "metric_name": "beta1 progression",
            "observed_value": (
                f"{ladder['beta1_coarse_mean']},"
                f"{ladder['beta1_mid_mean']},"
                f"{ladder['beta1_fine_mean']}"
            ),
            "threshold": "coarse==0, mid>0, fine>mid",
            "pass": bool(ladder["ladder_pass"]),
            "primary_artifact": ladder_summary_path.relative_to(root).as_posix(),
            "notes": "currency dimension grows with resolution",
        },
        {
            "claim_id": "C5",
            "experiment": "proxy_ablation",
            "metric_name": "mean_nll_good_vs_bad",
            "observed_value": (
                f"good={proxy['mean_nll_good']:.6e}, bad={proxy['mean_nll_bad']:.6e}"
            ),
            "threshold": "mean_nll_good < mean_nll_bad",
            "pass": bool(float(proxy["mean_nll_good"]) < float(proxy["mean_nll_bad"])),
            "primary_artifact": proxy_summary_path.relative_to(root).as_posix(),
            "notes": "good currency predicts better",
        },
        {
            "claim_id": "C6",
            "experiment": "proxy_ablation",
            "metric_name": "std_lam_bad_vs_good",
            "observed_value": (
                f"std_bad={proxy['std_lam_bad']:.6e}, "
                f"std_good={proxy['std_lam_good']:.6e}"
            ),
            "threshold": "std_lam_bad > std_lam_good",
            "pass": bool(float(proxy["std_lam_bad"]) > float(proxy["std_lam_good"])),
            "primary_artifact": proxy_summary_path.relative_to(root).as_posix(),
            "notes": "bad proxy lambda is less stable",
        },
        {
            "claim_id": "C7",
            "experiment": "idempotence_budget",
            "metric_name": "defect_end_lt_start",
            "observed_value": idem["defect_end_lt_start"],
            "threshold": "True and spearman_rho < -0.7",
            "pass": bool(
                bool(idem["defect_end_lt_start"]) and float(idem["spearman_rho"]) < -0.7
            ),
            "primary_artifact": idem_summary_path.relative_to(root).as_posix(),
            "notes": "stronger budget enforcement lowers defect",
        },
        {
            "claim_id": "C8",
            "experiment": "lean",
            "metric_name": "build_success",
            "observed_value": lean["build_success"],
            "threshold": "True",
            "pass": bool(lean["build_success"]),
            "primary_artifact": lean_summary_path.relative_to(root).as_posix(),
            "notes": lean.get("theorem_name", ""),
        },
    ]

    claim_path = outdir / "claim_ledger.csv"
    with claim_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "claim_id",
                "experiment",
                "metric_name",
                "observed_value",
                "threshold",
                "pass",
                "primary_artifact",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(claim_rows)

    for row in claim_rows:
        p = (root / row["primary_artifact"]).resolve()
        ensure_exists(p, f"claim primary artifact {row['claim_id']}")

    print(f"manifest_in={manifest_path.relative_to(root).as_posix()}")
    print(f"export_dir={outdir.relative_to(root).as_posix()}")
    print(f"claim_ledger={claim_path.relative_to(root).as_posix()}")


if __name__ == "__main__":
    main()
