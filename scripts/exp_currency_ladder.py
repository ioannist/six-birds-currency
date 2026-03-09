from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from currencymorphism.audits_cycles import (
    cycle_affinities,
    cycle_basis,
    cycle_rank,
    undirected_support_graph,
)
from currencymorphism.lens import lumped_kernel
from currencymorphism.markov import normalize_rows, stationary_dist
from currencymorphism.runlog import new_run_id, save_metadata


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the currency-ladder experiment."""
    parser = argparse.ArgumentParser(
        description="Currency ladder resolution experiment"
    )
    parser.add_argument("--M", type=int, default=6)
    parser.add_argument("--L", type=int, default=10)
    parser.add_argument("--bridge-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--affinities", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="results/currency_ladder")
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    """Parse comma-separated float list."""
    values = [x.strip() for x in raw.split(",") if x.strip()]
    return [float(v) for v in values]


def build_ladder_module_kernel(
    M: int,
    L: int,
    affinities: np.ndarray,
    bridge_weight: float,
) -> np.ndarray:
    """Build ladder-bridged rings with two bridge points per adjacent pair."""
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}.")
    if L < 4:
        raise ValueError(f"L must be >= 4, got {L}.")
    if L % 2 != 0:
        raise ValueError(f"L must be even, got {L}.")
    if bridge_weight < 0.0:
        raise ValueError(f"bridge_weight must be >= 0, got {bridge_weight}.")
    if affinities.shape != (M,):
        raise ValueError(f"affinities must have shape ({M},), got {affinities.shape}.")

    n_states = M * L
    W = np.zeros((n_states, n_states), dtype=np.float64)

    for m in range(M):
        A_m = float(affinities[m])
        w_cw = float(np.exp(+A_m / (2.0 * L)))
        w_ccw = float(np.exp(-A_m / (2.0 * L)))
        base = m * L
        for i in range(L):
            s = base + i
            W[s, base + ((i + 1) % L)] += w_cw
            W[s, base + ((i - 1) % L)] += w_ccw

    pos0 = 0
    pos1 = L // 2
    for m in range(M - 1):
        for pos in (pos0, pos1):
            u = m * L + pos
            v = (m + 1) * L + pos
            W[u, v] += bridge_weight
            W[v, u] += bridge_weight

    return normalize_rows(W, eps=0.0)


def make_partitions(M: int, L: int) -> dict[str, np.ndarray]:
    """Construct coarse, intermediate, and fine partitions."""
    n_states = M * L
    idx = np.arange(n_states, dtype=np.int64)
    part_coarse = idx // L
    part_mid = 2 * (idx // L) + ((idx % L) >= (L // 2)).astype(np.int64)
    part_fine = idx.copy()
    return {
        "coarse": part_coarse,
        "mid": part_mid,
        "fine": part_fine,
    }


def compute_level_metrics(
    P: np.ndarray,
    pi: np.ndarray,
    part: np.ndarray,
    name: str,
) -> dict[str, float | int | str]:
    """Compute lumped-kernel cycle metrics for one partition level."""
    Q = lumped_kernel(P, part, pi=pi)
    G = undirected_support_graph(Q, thresh=0.0)
    beta1 = cycle_rank(G)
    cycles = cycle_basis(G)
    aff = cycle_affinities(Q, cycles, orientation="canonical")
    aff_norm = float(np.linalg.norm(aff, ord=2)) if aff.size > 0 else 0.0

    return {
        "name": name,
        "k": int(np.max(part) + 1),
        "beta1": int(beta1),
        "basis_size": int(len(cycles)),
        "aff_norm": float(aff_norm),
    }


def run_one_seed(
    args: argparse.Namespace,
    seed: int,
) -> list[dict[str, float | int | str]]:
    """Compute per-level ladder metrics for one seed."""
    _ = seed
    if args.affinities is None:
        affinities = np.linspace(1.0, -0.5, args.M, dtype=np.float64)
    else:
        affinities = np.asarray(parse_float_list(args.affinities), dtype=np.float64)
        if affinities.shape != (args.M,):
            raise ValueError(
                f"Expected {args.M} affinities, got {affinities.shape[0]}."
            )

    P = build_ladder_module_kernel(
        M=args.M,
        L=args.L,
        affinities=affinities,
        bridge_weight=args.bridge_weight,
    )
    pi = stationary_dist(P, method="eigs")

    parts = make_partitions(args.M, args.L)
    order = ["coarse", "mid", "fine"]
    metrics = [compute_level_metrics(P, pi, parts[name], name) for name in order]
    return metrics


def pop_std(series: pd.Series) -> float:
    """Population standard deviation helper for grouped aggregation."""
    return float(np.std(series.to_numpy(dtype=np.float64), ddof=0))


def aggregate_levels(per_seed_df: pd.DataFrame, n_seeds: int) -> pd.DataFrame:
    """Aggregate per-seed ladder metrics by level name."""
    grouped = per_seed_df.groupby("name", sort=False)
    agg = grouped.agg(
        k=("k", "first"),
        beta1_mean=("beta1", "mean"),
        beta1_std=("beta1", pop_std),
        basis_size_mean=("basis_size", "mean"),
        basis_size_std=("basis_size", pop_std),
        aff_norm_mean=("aff_norm", "mean"),
        aff_norm_std=("aff_norm", pop_std),
    ).reset_index()
    agg["k"] = agg["k"].astype(int)
    agg["n_seeds"] = int(n_seeds)

    order = ["coarse", "mid", "fine"]
    agg["name"] = pd.Categorical(agg["name"], categories=order, ordered=True)
    agg = agg.sort_values("name").reset_index(drop=True)
    agg["name"] = agg["name"].astype(str)
    return agg


def plot_single(metrics: list[dict[str, float | int | str]], png_path: Path) -> None:
    """Plot single-run cycle rank beta1 across levels."""
    labels = [str(m["name"]) for m in metrics]
    y = [int(m["beta1"]) for m in metrics]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("resolution")
    ax.set_ylabel("beta1")
    ax.set_title("Currency Ladder: Cycle Rank vs Resolution")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def plot_aggregated(agg_df: pd.DataFrame, png_path: Path) -> None:
    """Plot aggregated beta1 means with std error bars across levels."""
    labels = agg_df["name"].tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        x,
        agg_df["beta1_mean"],
        yerr=agg_df["beta1_std"],
        marker="o",
        capsize=3,
        linestyle="-",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("resolution")
    ax.set_ylabel("beta1_mean")
    ax.set_title("Currency Ladder Robustness")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run ladder experiment and save JSON/CSV/PNG artifacts."""
    args = parse_args()

    if args.L % 2 != 0:
        raise ValueError(f"L must be even, got {args.L}.")

    seed_list = args.seeds if args.seeds else [int(args.seed)]
    multi_seed = args.seeds is not None and len(args.seeds) > 0

    run_suffix = "multi" if multi_seed else f"s{int(seed_list[0])}"
    run_id = f"ladder_{new_run_id()}_{run_suffix}"
    run_dir = Path(args.outdir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "summary.json"
    csv_path = run_dir / "metrics.csv"
    png_path = run_dir / "plot.png"

    if multi_seed:
        rows: list[dict[str, float | int | str]] = []
        for seed in seed_list:
            metrics = run_one_seed(args, int(seed))
            for m in metrics:
                row = dict(m)
                row["seed"] = int(seed)
                rows.append(row)

        per_seed_df = pd.DataFrame(rows)
        agg_df = aggregate_levels(per_seed_df, n_seeds=len(seed_list))

        coarse = float(agg_df.loc[agg_df["name"] == "coarse", "beta1_mean"].iloc[0])
        mid = float(agg_df.loc[agg_df["name"] == "mid", "beta1_mean"].iloc[0])
        fine = float(agg_df.loc[agg_df["name"] == "fine", "beta1_mean"].iloc[0])
        if coarse != 0.0:
            raise RuntimeError(f"Ladder robustness failed: beta1_coarse_mean={coarse}.")
        if mid <= 0.0:
            raise RuntimeError(f"Ladder robustness failed: beta1_mid_mean={mid}.")
        if fine <= mid:
            raise RuntimeError(
                f"Ladder robustness failed: beta1_fine_mean={fine} <= {mid}."
            )

        per_seed_csv_path = run_dir / "per_seed_metrics.csv"
        per_seed_df.to_csv(per_seed_csv_path, index=False)
        agg_df.to_csv(csv_path, index=False)
        plot_aggregated(agg_df, png_path)

        payload = {
            "run_id": run_id,
            "params": vars(args),
            "levels": agg_df.to_dict(orient="records"),
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        metric_map = {r["name"]: r for r in agg_df.to_dict(orient="records")}
        summary = {
            "n_seeds": int(len(seed_list)),
            "beta1_coarse_mean": float(metric_map["coarse"]["beta1_mean"]),
            "beta1_mid_mean": float(metric_map["mid"]["beta1_mean"]),
            "beta1_fine_mean": float(metric_map["fine"]["beta1_mean"]),
            "aff_norm_coarse_mean": float(metric_map["coarse"]["aff_norm_mean"]),
            "aff_norm_mid_mean": float(metric_map["mid"]["aff_norm_mean"]),
            "aff_norm_fine_mean": float(metric_map["fine"]["aff_norm_mean"]),
        }
        save_metadata(
            run_id=run_id,
            params=vars(args),
            metrics=summary,
            paths={
                "run_dir": run_dir,
                "metrics_csv": csv_path,
                "per_seed_metrics_csv": per_seed_csv_path,
                "plot_png": png_path,
                "summary_json": json_path,
            },
        )

        print(f"run_id={run_id}")
        print(f"json_path={json_path.as_posix()}")
        print(f"csv_path={csv_path.as_posix()}")
        print(f"png_path={png_path.as_posix()}")
        for row in agg_df.to_dict(orient="records"):
            print(
                f"level={row['name']} k={int(row['k'])} "
                f"beta1={float(row['beta1_mean']):.12e} "
                f"aff_norm={float(row['aff_norm_mean']):.12e}"
            )
    else:
        metrics = run_one_seed(args, int(seed_list[0]))
        pd.DataFrame(metrics).to_csv(csv_path, index=False)
        plot_single(metrics, png_path)

        payload = {
            "run_id": run_id,
            "params": vars(args),
            "levels": metrics,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        metric_map = {m["name"]: m for m in metrics}
        summary = {
            "beta1_coarse": int(metric_map["coarse"]["beta1"]),
            "beta1_mid": int(metric_map["mid"]["beta1"]),
            "beta1_fine": int(metric_map["fine"]["beta1"]),
            "aff_norm_coarse": float(metric_map["coarse"]["aff_norm"]),
            "aff_norm_mid": float(metric_map["mid"]["aff_norm"]),
            "aff_norm_fine": float(metric_map["fine"]["aff_norm"]),
        }
        save_metadata(
            run_id=run_id,
            params=vars(args),
            metrics=summary,
            paths={
                "run_dir": run_dir,
                "metrics_csv": csv_path,
                "plot_png": png_path,
                "summary_json": json_path,
            },
        )

        print(f"run_id={run_id}")
        print(f"json_path={json_path.as_posix()}")
        print(f"csv_path={csv_path.as_posix()}")
        print(f"png_path={png_path.as_posix()}")
        for m in metrics:
            print(
                f"level={m['name']} k={int(m['k'])} "
                f"beta1={int(m['beta1'])} aff_norm={float(m['aff_norm']):.12e}"
            )


if __name__ == "__main__":
    main()
