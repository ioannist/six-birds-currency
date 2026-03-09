from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from currencymorphism.audits_pathkl import SigmaEstimate, sigma_T_empirical
from currencymorphism.generators import module_rings_chain, ring_drift_kernel
from currencymorphism.lens import block_partition, coarse_path
from currencymorphism.markov import simulate_many
from currencymorphism.runlog import new_run_id, save_metadata


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for DPI scan experiment."""
    parser = argparse.ArgumentParser(
        description="DPI scan via pushforward trajectories"
    )

    parser.add_argument("--kernel", choices=["module", "ring"], default="module")
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--traj-len", type=int, default=80)
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--tol", type=float, default=5e-3)
    parser.add_argument("--max-N", type=int, default=64000)
    parser.add_argument("--k-list", type=str, default=None)

    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--affinity-A", type=float, default=2.0)
    parser.add_argument("--stay-prob", type=float, default=0.05)

    parser.add_argument("--M", type=int, default=6)
    parser.add_argument("--L", type=int, default=10)
    parser.add_argument("--bridge-weight", type=float, default=0.2)
    parser.add_argument("--affinities", type=str, default=None)

    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated integer list."""
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return [int(v) for v in vals]


def parse_float_list(raw: str) -> list[float]:
    """Parse a comma-separated float list."""
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return [float(v) for v in vals]


def choose_k_values(n_states: int, raw_k_list: str | None) -> list[int]:
    """Build the k ladder for coarse partitions."""
    if raw_k_list is not None:
        k_vals = sorted(set(parse_int_list(raw_k_list)))
    else:
        cap = min(32, n_states)
        k_vals = []
        k = 2
        while k <= cap:
            k_vals.append(k)
            k *= 2
        if not k_vals:
            k_vals = [1]

    for k in k_vals:
        if not (1 <= k <= n_states):
            msg = f"Invalid k={k}; each k must satisfy 1 <= k <= {n_states}."
            raise ValueError(msg)
    return k_vals


def build_kernel(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Construct micro kernel and initial distribution from CLI args."""
    if args.kernel == "ring":
        P = ring_drift_kernel(args.n, args.affinity_A, args.stay_prob)
        rho0 = np.full(args.n, 1.0 / args.n, dtype=np.float64)
        return P, rho0

    if args.affinities is not None:
        affinities = parse_float_list(args.affinities)
        if len(affinities) != args.M:
            msg = f"Expected {args.M} affinities, got {len(affinities)}."
            raise ValueError(msg)
    else:
        affinities = np.linspace(1.0, -0.5, args.M, dtype=np.float64).tolist()

    P = module_rings_chain(args.M, args.L, affinities, args.bridge_weight)
    n_states = args.M * args.L
    rho0 = np.full(n_states, 1.0 / n_states, dtype=np.float64)
    return P, rho0


def evaluate_once(
    P: np.ndarray,
    rho0: np.ndarray,
    T: int,
    traj_len: int,
    N: int,
    seed: int,
    k_values: list[int],
    kernel_name: str,
) -> tuple[pd.DataFrame, SigmaEstimate]:
    """Run one simulation pass and return per-k DPI rows."""
    rng = np.random.default_rng(seed)
    paths_micro = simulate_many(P, x0_dist=rho0, T=traj_len - 1, N=N, rng=rng)

    sig_micro = sigma_T_empirical(paths_micro, T)

    rows = []
    n_states = P.shape[0]
    for k in k_values:
        part = block_partition(n_states, k)
        paths_coarse = coarse_path(paths_micro, part)
        sig_coarse = sigma_T_empirical(paths_coarse, T)

        rows.append(
            {
                "seed": int(seed),
                "k": int(k),
                "sigma_micro": float(sig_micro.value),
                "sigma_coarse": float(sig_coarse.value),
                "delta": float(sig_micro.value - sig_coarse.value),
                "inf_micro": float(sig_micro.infinite_risk_mass),
                "inf_coarse": float(sig_coarse.infinite_risk_mass),
                "nwin_micro": int(sig_micro.n_windows),
                "nwin_coarse": int(sig_coarse.n_windows),
                "N": int(N),
                "traj_len": int(traj_len),
                "T": int(T),
                "kernel": kernel_name,
            }
        )

    df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    return df, sig_micro


def run_one_seed(
    args: argparse.Namespace,
    P: np.ndarray,
    rho0: np.ndarray,
    k_values: list[int],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    """Run one seed with adaptive N while keeping the seed fixed."""
    current_N = int(args.N)
    while True:
        df, _ = evaluate_once(
            P=P,
            rho0=rho0,
            T=args.T,
            traj_len=args.traj_len,
            N=current_N,
            seed=seed,
            k_values=k_values,
            kernel_name=args.kernel,
        )
        violations = df[df["delta"] < -args.tol]
        if violations.empty:
            info = {
                "seed": int(seed),
                "final_N": int(current_N),
                "failed": False,
                "worst_k": -1,
                "worst_delta": float(df["delta"].min()),
            }
            break

        next_N = current_N * 2
        if next_N > args.max_N:
            worst = violations.sort_values("delta").iloc[0]
            info = {
                "seed": int(seed),
                "final_N": int(current_N),
                "failed": True,
                "worst_k": int(worst["k"]),
                "worst_delta": float(worst["delta"]),
            }
            break

        current_N = next_N

    df = df.copy()
    df["final_N"] = int(info["final_N"])
    return df, info


def pop_std(series: pd.Series) -> float:
    """Population standard deviation helper returning a scalar float."""
    return float(np.std(series.to_numpy(dtype=np.float64), ddof=0))


def aggregate_per_seed(per_seed_df: pd.DataFrame, n_seeds: int) -> pd.DataFrame:
    """Aggregate per-seed rows by k into mean/std summary table."""
    grouped = per_seed_df.groupby("k", sort=True)
    agg = pd.DataFrame(
        {
            "k": grouped["k"].first().astype(int),
            "sigma_micro_mean": grouped["sigma_micro"].mean().astype(float),
            "sigma_micro_std": grouped["sigma_micro"].apply(pop_std),
            "sigma_coarse_mean": grouped["sigma_coarse"].mean().astype(float),
            "sigma_coarse_std": grouped["sigma_coarse"].apply(pop_std),
            "delta_mean": grouped["delta"].mean().astype(float),
            "delta_std": grouped["delta"].apply(pop_std),
            "inf_micro_mean": grouped["inf_micro"].mean().astype(float),
            "inf_micro_std": grouped["inf_micro"].apply(pop_std),
            "inf_coarse_mean": grouped["inf_coarse"].mean().astype(float),
            "inf_coarse_std": grouped["inf_coarse"].apply(pop_std),
            "final_N_mean": grouped["final_N"].mean().astype(float),
            "final_N_std": grouped["final_N"].apply(pop_std),
            "n_seeds": int(n_seeds),
        }
    )
    return agg.reset_index(drop=True)


def plot_single(df: pd.DataFrame, png_path: Path) -> None:
    """Plot coarse sigma vs k with micro sigma reference line (single-seed mode)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["k"], df["sigma_coarse"], marker="o", label="coarse")
    ax.axhline(
        y=float(df["sigma_micro"].iloc[0]),
        linestyle="--",
        linewidth=1.2,
        color="black",
        label="micro",
    )
    ax.set_xlabel("k")
    ax.set_ylabel("Sigma_T")
    ax.set_title("DPI Scan: Coarse vs Micro")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def plot_aggregated(agg_df: pd.DataFrame, png_path: Path) -> None:
    """Plot aggregated delta mean with std error bars across k."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        agg_df["k"],
        agg_df["delta_mean"],
        yerr=agg_df["delta_std"],
        marker="o",
        capsize=3,
        linestyle="-",
    )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("k")
    ax.set_ylabel("delta_mean")
    ax.set_title("DPI Robustness: delta mean +/- std")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the DPI scan experiment and persist summary artifacts."""
    args = parse_args()

    if args.T < 1:
        raise ValueError(f"T must be >= 1, got {args.T}.")
    if args.traj_len < args.T + 1:
        msg = f"traj-len must be >= T+1={args.T + 1}, got {args.traj_len}."
        raise ValueError(msg)
    if args.N <= 0:
        raise ValueError(f"N must be positive, got {args.N}.")
    if args.max_N < args.N:
        raise ValueError(f"max-N must be >= N ({args.N}), got {args.max_N}.")

    P, rho0 = build_kernel(args)
    n_states = P.shape[0]
    k_values = choose_k_values(n_states, args.k_list)

    seed_list = args.seeds if args.seeds else [int(args.seed)]
    multi_seed = args.seeds is not None and len(args.seeds) > 0

    run_id = f"dpi_{new_run_id()}"
    run_dir = Path("results") / "dpi_scan" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if multi_seed:
        per_seed_frames: list[pd.DataFrame] = []
        seed_min_deltas: dict[int, float] = {}

        for seed in seed_list:
            df_seed, info = run_one_seed(args, P, rho0, k_values, int(seed))
            if bool(info["failed"]):
                msg = (
                    "DPI robustness failed: "
                    f"seed={int(info['seed'])}, "
                    f"worst_k={int(info['worst_k'])}, "
                    f"worst_delta={float(info['worst_delta']):.6e}, "
                    f"tol={args.tol:.6e}"
                )
                raise RuntimeError(msg)

            per_seed_frames.append(df_seed)
            seed_min_deltas[int(seed)] = float(df_seed["delta"].min())

        per_seed_df = pd.concat(per_seed_frames, ignore_index=True)
        agg_df = aggregate_per_seed(per_seed_df, n_seeds=len(seed_list))

        per_seed_csv_path = run_dir / "per_seed_metrics.csv"
        csv_path = run_dir / "metrics.csv"
        png_path = run_dir / "plot.png"
        per_seed_df.to_csv(per_seed_csv_path, index=False)
        agg_df.to_csv(csv_path, index=False)
        plot_aggregated(agg_df, png_path)

        worst_seed = min(seed_min_deltas, key=seed_min_deltas.get)
        worst_seed_min_delta = seed_min_deltas[worst_seed]
        idx = int(agg_df["delta_mean"].idxmin())
        summary = {
            "n_seeds": int(len(seed_list)),
            "min_delta_mean": float(agg_df.loc[idx, "delta_mean"]),
            "min_delta_std": float(agg_df.loc[idx, "delta_std"]),
            "worst_seed_min_delta": float(worst_seed_min_delta),
            "tol": float(args.tol),
        }
        params = vars(args).copy()
        params["n_states"] = int(n_states)
        params["k_values"] = k_values

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

        print(f"run_id={run_id}")
        print(f"csv_path={csv_path.as_posix()}")
        print(f"png_path={png_path.as_posix()}")
        print(
            f"min_delta={float(agg_df['delta_mean'].min()):.6e} "
            f"(tol={args.tol:.6e}) n_seeds={len(seed_list)}"
        )
    else:
        seed = int(seed_list[0])
        df, info = run_one_seed(args, P, rho0, k_values, seed)

        if bool(info["failed"]):
            print(
                "DPI_VIOLATION: "
                f"min_delta={float(info['worst_delta']):.6e} "
                f"at k={int(info['worst_k'])} "
                f"(tol={args.tol:.6e})"
            )

        csv_path = run_dir / "metrics.csv"
        png_path = run_dir / "plot.png"
        df.to_csv(csv_path, index=False)
        plot_single(df, png_path)

        summary = {
            "min_delta": float(df["delta"].min()),
            "max_delta": float(df["delta"].max()),
            "final_N": int(info["final_N"]),
            "tol": float(args.tol),
            "n_levels": int(df.shape[0]),
        }
        params = vars(args).copy()
        params["n_states"] = int(n_states)
        params["k_values"] = k_values

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

        print(f"run_id={run_id}")
        print(f"csv_path={csv_path.as_posix()}")
        print(f"png_path={png_path.as_posix()}")
        print(
            f"min_delta={float(df['delta'].min()):.6e} "
            f"(tol={args.tol:.6e}) final_N={int(info['final_N'])}"
        )


if __name__ == "__main__":
    main()
