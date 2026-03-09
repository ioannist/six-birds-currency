from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from currencymorphism.generators import module_rings_chain
from currencymorphism.lens import lift_dist
from currencymorphism.markov import normalize_rows, stationary_dist
from currencymorphism.maxcal_single import (
    expected_cost,
    maxent_kernel,
    solve_lambda_for_budget,
)
from currencymorphism.packaging import idempotence_defect
from currencymorphism.runlog import new_run_id, save_metadata


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for idempotence-vs-budget sweep experiment."""
    parser = argparse.ArgumentParser(
        description="Idempotence-defect curve under budget-controlled packaging"
    )

    parser.add_argument("--M", type=int, default=6)
    parser.add_argument("--L", type=int, default=10)
    parser.add_argument("--bridge-weight", type=float, default=0.2)
    parser.add_argument("--affinities", type=str, default=None)
    parser.add_argument("--tau", type=int, default=5)
    parser.add_argument(
        "--prototype",
        choices=["stationary", "uniform"],
        default="stationary",
    )

    parser.add_argument("--n-budgets", type=int, default=15)
    parser.add_argument("--b-min-frac", type=float, default=0.05)
    parser.add_argument("--b-max-mult", type=float, default=1.10)
    parser.add_argument("--tol", type=float, default=1e-10)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="results/idempotence_budget")
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    """Parse comma-separated float values."""
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return [float(v) for v in vals]


def build_base_kernel(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build base kernel and natural ring-collapse partition."""
    if args.affinities is None:
        affinities = np.linspace(1.0, -0.5, args.M, dtype=np.float64)
    else:
        affinities = np.asarray(parse_float_list(args.affinities), dtype=np.float64)
        if affinities.shape != (args.M,):
            msg = f"Expected {args.M} affinities, got {affinities.shape[0]}."
            raise ValueError(msg)

    P_base = module_rings_chain(args.M, args.L, affinities, args.bridge_weight)
    n_states = args.M * args.L
    part = np.arange(n_states, dtype=np.int64) // args.L
    return P_base, part, affinities


def boundary_cost_matrix(k: int) -> np.ndarray:
    """Return macro maintenance cost with zero on-diagonal and one off-diagonal."""
    u = np.ones((k, k), dtype=np.float64)
    np.fill_diagonal(u, 0.0)
    return u


def build_controlled_kernel(
    P_base: np.ndarray,
    part: np.ndarray,
    u_macro: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Apply lambda-weighted maintenance penalty and row-normalize."""
    edge_cost = u_macro[part[:, None], part[None, :]]
    W = P_base * np.exp(-lam * edge_cost)
    return normalize_rows(W, eps=0.0)


def compute_defect_stats(
    P_lam: np.ndarray,
    part: np.ndarray,
    tau: int,
    prototype: str,
) -> tuple[float, float, float]:
    """Compute mean/std/max idempotence defect over macro basis states."""
    k = int(np.max(part) + 1)
    defects: list[float] = []

    pi_lam: np.ndarray | None = None
    if prototype == "stationary":
        pi_lam = stationary_dist(P_lam, method="eigs")

    for a in range(k):
        e_a = np.zeros(k, dtype=np.float64)
        e_a[a] = 1.0

        if prototype == "uniform":
            mu_a = lift_dist(e_a, part, scheme="uniform")
        else:
            assert pi_lam is not None
            mu_a = lift_dist(e_a, part, scheme="stationary", pi_micro=pi_lam)

        d_a = idempotence_defect(
            mu_a,
            P_lam,
            tau,
            part,
            prototype=prototype,
            pi_micro=pi_lam,
        )
        defects.append(float(d_a))

    arr = np.asarray(defects, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0)), float(arr.max())


def plot_curve(df: pd.DataFrame, run_id: str, png_path: Path) -> None:
    """Plot defect mean vs lambda with std error bars."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        df["lam"],
        df["defect_mean"],
        yerr=df["defect_std"],
        marker="o",
        capsize=3,
        linestyle="-",
    )
    ax.set_xlabel("lambda")
    ax.set_ylabel("idempotence defect mean")
    ax.set_title(f"Idempotence vs Budget: {run_id}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run idempotence-defect budget sweep and persist artifacts."""
    args = parse_args()
    if args.tau < 0:
        raise ValueError(f"tau must be >= 0, got {args.tau}.")
    if args.n_budgets < 2:
        raise ValueError(f"n-budgets must be >= 2, got {args.n_budgets}.")

    P_base, part, affinities = build_base_kernel(args)
    k = int(np.max(part) + 1)

    u_macro = boundary_cost_matrix(k)
    q0 = maxent_kernel(u_macro, lam=0.0)
    cost0 = expected_cost(q0, u_macro)
    costmin = float(np.mean(np.min(u_macro, axis=1)))

    gap = cost0 - costmin
    b_min = costmin + args.b_min_frac * gap
    b_max = cost0 + (args.b_max_mult - 1.0) * gap
    b_values = np.linspace(b_min, b_max, args.n_budgets, dtype=np.float64)

    rows: list[dict[str, float | int | str]] = []
    lam_values: list[float] = []
    defect_values: list[float] = []

    for b in b_values:
        lam, _, achieved = solve_lambda_for_budget(
            u_macro,
            float(b),
            bracket=(0.0, 50.0),
            tol=args.tol,
            mu=None,
        )

        P_lam = build_controlled_kernel(P_base, part, u_macro, float(lam))
        defect_mean, defect_std, defect_max = compute_defect_stats(
            P_lam,
            part,
            tau=args.tau,
            prototype=args.prototype,
        )

        lam_values.append(float(lam))
        defect_values.append(float(defect_mean))

        rows.append(
            {
                "b": float(b),
                "lam": float(lam),
                "achieved": float(achieved),
                "defect_mean": float(defect_mean),
                "defect_std": float(defect_std),
                "defect_max": float(defect_max),
                "tau": int(args.tau),
                "prototype": args.prototype,
                "seed": int(args.seed),
            }
        )

    lam_arr = np.asarray(lam_values, dtype=np.float64)
    defect_arr = np.asarray(defect_values, dtype=np.float64)

    rho, _ = spearmanr(lam_arr, defect_arr)
    if not np.isfinite(rho):
        raise RuntimeError("Spearman correlation is non-finite.")

    idx_max_lam = int(np.argmax(lam_arr))
    idx_min_lam = int(np.argmin(lam_arr))
    defect_at_max_lam = float(defect_arr[idx_max_lam])
    defect_at_min_lam = float(defect_arr[idx_min_lam])

    if float(rho) >= -0.7:
        msg = f"Self-check failed: spearman_rho={float(rho):.6e} >= -0.7"
        raise RuntimeError(msg)
    if defect_at_max_lam >= defect_at_min_lam:
        msg = (
            "Self-check failed: defect at max lambda is not smaller than defect at "
            f"min lambda ({defect_at_max_lam:.6e} >= {defect_at_min_lam:.6e})."
        )
        raise RuntimeError(msg)

    run_id = f"idem_{new_run_id()}_s{int(args.seed)}"
    run_dir = Path(args.outdir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "metrics.csv"
    png_path = run_dir / "plot.png"

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    plot_curve(df, run_id, png_path)

    summary = {
        "spearman_rho": float(rho),
        "defect_mean_min": float(df["defect_mean"].min()),
        "defect_mean_max": float(df["defect_mean"].max()),
        "lam_min": float(df["lam"].min()),
        "lam_max": float(df["lam"].max()),
        "tau": int(args.tau),
        "prototype": args.prototype,
        "n_budgets": int(args.n_budgets),
        "costmin": float(costmin),
        "cost0": float(cost0),
    }

    params = vars(args).copy()
    params["affinities_used"] = np.asarray(affinities, dtype=np.float64)

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
    print(f"spearman_rho={float(rho):.12e}")

    sample_idx = np.linspace(0, len(df) - 1, 5, dtype=int)
    for idx in sample_idx:
        print(
            f"sample b={float(df.loc[idx, 'b']):.12e} "
            f"lam={float(df.loc[idx, 'lam']):.12e} "
            f"defect={float(df.loc[idx, 'defect_mean']):.12e}"
        )


if __name__ == "__main__":
    main()
