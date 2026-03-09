from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from currencymorphism.generators import module_rings_chain, ring_drift_kernel
from currencymorphism.lens import block_partition, random_partition
from currencymorphism.markov import simulate_many
from currencymorphism.maxcal_single import (
    expected_cost,
    maxent_kernel,
    solve_lambda_for_budget,
)
from currencymorphism.mle_logit import fit_lambda
from currencymorphism.runlog import new_run_id, save_metadata


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for budget sweep experiment."""
    parser = argparse.ArgumentParser(description="Budget sweep for price emergence")

    parser.add_argument("--kernel", choices=["module", "ring"], default="module")
    parser.add_argument("--M", type=int, default=6)
    parser.add_argument("--L", type=int, default=10)
    parser.add_argument("--bridge-weight", type=float, default=0.2)
    parser.add_argument("--affinities", type=str, default=None)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--affinity-A", type=float, default=2.0)
    parser.add_argument("--stay-prob", type=float, default=0.05)

    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--partition", choices=["random", "block"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)

    parser.add_argument("--N", type=int, default=8000)
    parser.add_argument("--traj-len", type=int, default=80)

    parser.add_argument("--n-budgets", type=int, default=25)
    parser.add_argument("--b-max-mult", type=float, default=1.10)
    parser.add_argument("--b-min-frac", type=float, default=0.05)
    parser.add_argument("--tol", type=float, default=1e-10)

    parser.add_argument("--mle-check", action="store_true")
    parser.add_argument("--mle-N-per-row", type=int, default=5000)

    parser.add_argument("--outdir", type=str, default="results/budget_sweep")
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    """Parse comma-separated float values."""
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return [float(v) for v in items]


def build_micro_kernel(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Construct micro kernel and uniform initial distribution."""
    if args.kernel == "module":
        if args.affinities is None:
            affinities = np.linspace(1.0, -0.5, args.M, dtype=np.float64)
        else:
            affinities = np.asarray(parse_float_list(args.affinities), dtype=np.float64)
            if affinities.shape != (args.M,):
                msg = f"Expected {args.M} affinities, got {affinities.shape[0]}."
                raise ValueError(msg)

        P_micro = module_rings_chain(args.M, args.L, affinities, args.bridge_weight)
        n_states = args.M * args.L
        rho0 = np.full(n_states, 1.0 / n_states, dtype=np.float64)
        return P_micro, rho0

    P_micro = ring_drift_kernel(args.n, args.affinity_A, args.stay_prob)
    rho0 = np.full(args.n, 1.0 / args.n, dtype=np.float64)
    return P_micro, rho0


def derive_macro_cost(
    P_micro: np.ndarray,
    part: np.ndarray,
    rho0: np.ndarray,
    N: int,
    traj_len: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate macro cost matrix from sampled micro transition EP increments."""
    paths = simulate_many(P_micro, x0_dist=rho0, T=traj_len - 1, N=N, rng=rng)

    i = paths[:, :-1].ravel()
    j = paths[:, 1:].ravel()

    a = part[i]
    b = part[j]

    pij = P_micro[i, j]
    pji = P_micro[j, i]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(pij, pji, out=np.ones_like(pij), where=pji > 0.0)
        ep = np.log(ratio, out=np.zeros_like(ratio), where=ratio > 0.0)

    invalid = (pij > 0.0) & (pji <= 0.0)
    if np.any(invalid):
        msg = (
            "Observed transitions with zero reverse probability; "
            "macro cost undefined."
        )
        raise RuntimeError(msg)

    cost = np.maximum(ep, 0.0)

    k = int(np.max(part) + 1)
    sum_u = np.zeros((k, k), dtype=np.float64)
    cnt = np.zeros((k, k), dtype=np.float64)
    np.add.at(sum_u, (a, b), cost)
    np.add.at(cnt, (a, b), 1.0)

    total_cnt = float(cnt.sum())
    global_mean = float(sum_u.sum() / max(total_cnt, 1.0))
    alpha = 1.0
    u = (sum_u + alpha * global_mean) / (cnt + alpha)
    u = np.clip(u, 0.0, None)

    zero_count = int(np.sum(cnt == 0.0))
    print(
        f"u_min={float(u.min()):.12e} u_max={float(u.max()):.12e} "
        f"mean={float(u.mean()):.12e} zero_count={zero_count}"
    )

    return u, sum_u, cnt


def maybe_mle_check(
    enabled: bool,
    u: np.ndarray,
    q_values: list[np.ndarray],
    lam_values: list[float],
    mle_n_per_row: int,
    rng: np.random.Generator,
) -> tuple[list[float], list[float], list[float]]:
    """Optionally run MLE confirmation and return lists for CSV columns."""
    n_points = len(lam_values)
    if not enabled:
        nan_list = [float("nan")] * n_points
        return nan_list, nan_list, nan_list

    lam_hat_vals: list[float] = []
    se_hat_vals: list[float] = []
    diff_vals: list[float] = []

    for q, lam in zip(q_values, lam_values, strict=True):
        C = np.vstack(
            [rng.multinomial(mle_n_per_row, q[i]) for i in range(q.shape[0])]
        ).astype(np.float64)
        lam_hat, se_hat = fit_lambda(C, u)
        lam_hat_vals.append(float(lam_hat))
        se_hat_vals.append(float(se_hat))
        diff_vals.append(float(lam_hat - lam))

    return lam_hat_vals, se_hat_vals, diff_vals


def build_partition(
    n_states: int,
    k: int,
    partition_name: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build macro partition from requested scheme."""
    if not (1 <= k <= n_states):
        raise ValueError(f"k must satisfy 1 <= k <= {n_states}, got {k}.")

    if partition_name == "random":
        return random_partition(n_states, k, rng)
    if partition_name == "block":
        return block_partition(n_states, k)
    raise ValueError(f"Unknown partition: {partition_name!r}.")


def pop_std(series: pd.Series) -> float:
    """Population std helper used for aggregation."""
    return float(np.std(series.to_numpy(dtype=np.float64), ddof=0))


def run_one_seed(
    args: argparse.Namespace,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    """Run one seed and return per-alpha rows plus scalar diagnostics."""
    rng = np.random.default_rng(seed)
    P_micro, rho0 = build_micro_kernel(args)
    n_states = P_micro.shape[0]
    part = build_partition(n_states, args.k, args.partition, rng)

    u, _, _ = derive_macro_cost(
        P_micro=P_micro,
        part=part,
        rho0=rho0,
        N=args.N,
        traj_len=args.traj_len,
        rng=rng,
    )

    q0 = maxent_kernel(u, lam=0.0)
    cost0 = expected_cost(q0, u, mu=None)
    costmin = float(np.mean(np.min(u, axis=1)))
    gap = cost0 - costmin

    alpha_values = np.linspace(args.b_min_frac, args.b_max_mult, args.n_budgets)
    b_values = costmin + alpha_values * gap

    lam_values: list[float] = []
    achieved_values: list[float] = []
    q_values: list[np.ndarray] = []
    rows: list[dict[str, float | int | str]] = []

    for idx, alpha in enumerate(alpha_values):
        b = float(b_values[idx])
        lam, q, achieved = solve_lambda_for_budget(
            u,
            b,
            bracket=(0.0, 50.0),
            tol=args.tol,
            mu=None,
        )
        lam_values.append(float(lam))
        achieved_values.append(float(achieved))
        q_values.append(q)
        rows.append(
            {
                "seed": int(seed),
                "alpha_idx": int(idx),
                "alpha": float(alpha),
                "b": b,
                "lam": float(lam),
                "achieved": float(achieved),
                "costmin": float(costmin),
                "cost0": float(cost0),
                "kernel": args.kernel,
                "k": int(args.k),
                "partition": args.partition,
                "N": int(args.N),
                "traj_len": int(args.traj_len),
            }
        )

    lam_hat_vals, se_hat_vals, diff_vals = maybe_mle_check(
        enabled=args.mle_check,
        u=u,
        q_values=q_values,
        lam_values=lam_values,
        mle_n_per_row=args.mle_N_per_row,
        rng=rng,
    )

    for idx, row in enumerate(rows):
        row["lam_hat"] = float(lam_hat_vals[idx])
        row["se_hat"] = float(se_hat_vals[idx])
        row["lam_hat_minus_lam"] = float(diff_vals[idx])

    rho, _ = spearmanr(b_values, np.asarray(lam_values, dtype=np.float64))
    for row in rows:
        row["spearman_rho"] = float(rho)

    tail_has_zero = bool(np.any(np.asarray(lam_values[-3:], dtype=np.float64) <= 1e-12))
    info = {
        "seed": int(seed),
        "spearman_rho": float(rho),
        "tail_has_zero": tail_has_zero,
        "costmin": float(costmin),
        "cost0": float(cost0),
        "lam_min": float(np.min(lam_values)),
        "lam_max": float(np.max(lam_values)),
    }
    return pd.DataFrame(rows), info


def aggregate_per_seed(per_seed_df: pd.DataFrame, n_seeds: int) -> pd.DataFrame:
    """Aggregate per-seed rows by alpha index into mean/std summary."""
    grouped = per_seed_df.groupby("alpha_idx", sort=True)
    agg = pd.DataFrame(
        {
            "alpha_idx": grouped["alpha_idx"].first().astype(int),
            "alpha": grouped["alpha"].mean().astype(float),
            "b_mean": grouped["b"].mean().astype(float),
            "b_std": grouped["b"].apply(pop_std),
            "lam_mean": grouped["lam"].mean().astype(float),
            "lam_std": grouped["lam"].apply(pop_std),
            "achieved_mean": grouped["achieved"].mean().astype(float),
            "achieved_std": grouped["achieved"].apply(pop_std),
            "spearman_rho_mean": grouped["spearman_rho"].mean().astype(float),
            "spearman_rho_std": grouped["spearman_rho"].apply(pop_std),
            "n_seeds": int(n_seeds),
        }
    )
    return agg.reset_index(drop=True)


def plot_single(
    b_values: np.ndarray,
    lam_values: list[float],
    run_id: str,
    png_path: Path,
) -> None:
    """Plot single-seed lambda vs budget."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(b_values, lam_values, marker="o")
    ax.set_xlabel("budget b")
    ax.set_ylabel("lambda")
    ax.set_title(f"Price emergence curve: {run_id}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def plot_aggregated(agg_df: pd.DataFrame, run_id: str, png_path: Path) -> None:
    """Plot aggregated lambda-vs-budget with std error bars."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        agg_df["b_mean"],
        agg_df["lam_mean"],
        yerr=agg_df["lam_std"],
        marker="o",
        capsize=3,
        linestyle="-",
    )
    ax.set_xlabel("budget b (mean)")
    ax.set_ylabel("lambda (mean)")
    ax.set_title(f"Price emergence curve: {run_id}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run budget sweep and save artifacts."""
    args = parse_args()
    if args.traj_len < 2:
        raise ValueError(f"traj-len must be >= 2, got {args.traj_len}.")

    seed_list = args.seeds if args.seeds else [int(args.seed)]
    multi_seed = args.seeds is not None and len(args.seeds) > 0

    run_suffix = "multi" if multi_seed else f"s{int(seed_list[0])}"
    run_id = f"budget_{new_run_id()}_{run_suffix}"
    run_dir = Path(args.outdir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "metrics.csv"
    png_path = run_dir / "plot.png"

    if multi_seed:
        frames: list[pd.DataFrame] = []
        seed_infos: list[dict[str, float | int | bool]] = []
        for seed in seed_list:
            df_seed, info = run_one_seed(args, int(seed))
            frames.append(df_seed)
            seed_infos.append(info)

            if float(info["spearman_rho"]) >= -0.9:
                msg = (
                    "Monotonicity robustness failed: "
                    f"seed={int(info['seed'])}, "
                    f"spearman_rho={float(info['spearman_rho']):.6e}"
                )
                raise RuntimeError(msg)
            if not bool(info["tail_has_zero"]):
                msg = f"Tail slack check failed for seed={int(info['seed'])}."
                raise RuntimeError(msg)

        per_seed_df = pd.concat(frames, ignore_index=True)
        agg_df = aggregate_per_seed(per_seed_df, n_seeds=len(seed_list))

        per_seed_csv_path = run_dir / "per_seed_metrics.csv"
        per_seed_df.to_csv(per_seed_csv_path, index=False)
        agg_df.to_csv(csv_path, index=False)
        plot_aggregated(agg_df, run_id, png_path)

        spearman_values = np.asarray(
            [float(x["spearman_rho"]) for x in seed_infos],
            dtype=np.float64,
        )
        n_tail_zero = int(sum(bool(x["tail_has_zero"]) for x in seed_infos))

        summary = {
            "n_seeds": int(len(seed_list)),
            "spearman_rho_mean": float(np.mean(spearman_values)),
            "spearman_rho_std": float(np.std(spearman_values, ddof=0)),
            "lam_mean_min": float(agg_df["lam_mean"].min()),
            "lam_mean_max": float(agg_df["lam_mean"].max()),
            "n_tail_zero_lambda": int(n_tail_zero),
        }
        params = vars(args).copy()
        save_metadata(
            run_id=run_id,
            params=params,
            metrics=summary,
            paths={
                "run_dir": run_dir,
                "metrics_csv": csv_path,
                "per_seed_metrics_csv": per_seed_csv_path,
                "plot_png": png_path,
            },
        )

        mid = len(agg_df) // 2
        print(f"spearman_rho={summary['spearman_rho_mean']:.12e}")
        print(f"run_id={run_id}")
        print(f"csv_path={csv_path.as_posix()}")
        print(f"png_path={png_path.as_posix()}")
        print(
            f"costmin={float(per_seed_df['costmin'].mean()):.12e} "
            f"cost0={float(per_seed_df['cost0'].mean()):.12e}"
        )
        sample_idx = np.linspace(0, len(agg_df) - 1, 5, dtype=int)
        for idx in sample_idx:
            print(
                f"sample b={float(agg_df.loc[idx, 'b_mean']):.12e} "
                f"lam={float(agg_df.loc[idx, 'lam_mean']):.12e}"
            )
        _ = mid
    else:
        seed = int(seed_list[0])
        df, info = run_one_seed(args, seed)
        df.to_csv(csv_path, index=False)

        b_values = df["b"].to_numpy(dtype=np.float64)
        lam_values = df["lam"].to_list()
        plot_single(b_values, lam_values, run_id, png_path)

        summary = {
            "spearman_rho": float(info["spearman_rho"]),
            "costmin": float(info["costmin"]),
            "cost0": float(info["cost0"]),
            "lam_min": float(info["lam_min"]),
            "lam_max": float(info["lam_max"]),
        }
        params = vars(args).copy()
        save_metadata(
            run_id=run_id,
            params=params,
            metrics=summary,
            paths={
                "run_dir": run_dir,
                "metrics_csv": csv_path,
                "plot_png": png_path,
            },
        )

        print(f"spearman_rho={float(info['spearman_rho']):.12e}")
        print("spearman_p=nan")
        print(f"run_id={run_id}")
        print(f"csv_path={csv_path.as_posix()}")
        print(f"png_path={png_path.as_posix()}")
        print(
            f"costmin={float(info['costmin']):.12e} "
            f"cost0={float(info['cost0']):.12e}"
        )

        sample_idx = np.linspace(0, len(df) - 1, 5, dtype=int)
        for idx in sample_idx:
            print(
                f"sample b={float(df.loc[idx, 'b']):.12e} "
                f"lam={float(df.loc[idx, 'lam']):.12e}"
            )


if __name__ == "__main__":
    main()
