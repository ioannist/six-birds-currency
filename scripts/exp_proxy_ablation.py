from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from currencymorphism.generators import module_rings_chain
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
    """Parse CLI arguments for proxy-currency ablation experiment."""
    parser = argparse.ArgumentParser(description="Proxy-currency ablation")

    parser.add_argument("--M", type=int, default=6)
    parser.add_argument("--L", type=int, default=10)
    parser.add_argument("--bridge-weight", type=float, default=0.2)
    parser.add_argument("--affinities", type=str, default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--partition", choices=["random", "block"], default="random")
    parser.add_argument("--seed-cost", type=int, default=0)
    parser.add_argument("--proxy-seed", type=int, default=0)
    parser.add_argument("--N-cost", type=int, default=8000)
    parser.add_argument("--traj-len", type=int, default=80)

    parser.add_argument("--seed-list", type=str, default="0,1,2,3,4")
    parser.add_argument("--n-train-per-row", type=int, default=30)
    parser.add_argument("--n-test-per-row", type=int, default=5000)
    parser.add_argument("--true-budget-frac", type=float, default=0.35)

    parser.add_argument("--eval-eps", type=float, default=1e-12)
    parser.add_argument("--outdir", type=str, default="results/proxy_ablation")
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    """Parse comma-separated float list."""
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return [float(v) for v in items]


def parse_int_list(raw: str) -> list[int]:
    """Parse comma-separated integer list."""
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return [int(v) for v in items]


def build_u_good(
    args: argparse.Namespace,
    seed_cost: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Construct structured macro cost matrix from micro EP spending."""
    rng_cost = np.random.default_rng(seed_cost)

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

    if not (1 <= args.k <= n_states):
        msg = f"k must satisfy 1 <= k <= {n_states}, got {args.k}."
        raise ValueError(msg)

    if args.partition == "random":
        part = random_partition(n_states, args.k, rng_cost)
    else:
        part = block_partition(n_states, args.k)

    paths = simulate_many(
        P_micro,
        x0_dist=rho0,
        T=args.traj_len - 1,
        N=args.N_cost,
        rng=rng_cost,
    )

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
        msg = "Observed micro transitions with zero reverse probability."
        raise RuntimeError(msg)

    cost = np.maximum(ep, 0.0)

    k = args.k
    sum_u = np.zeros((k, k), dtype=np.float64)
    cnt = np.zeros((k, k), dtype=np.float64)
    np.add.at(sum_u, (a, b), cost)
    np.add.at(cnt, (a, b), 1.0)

    global_mean = float(sum_u.sum() / max(float(cnt.sum()), 1.0))
    alpha = 1.0
    u_good = (sum_u + alpha * global_mean) / (cnt + alpha)
    u_good = np.clip(u_good, 0.0, None)

    stats = {
        "u_min": float(u_good.min()),
        "u_max": float(u_good.max()),
        "u_mean": float(u_good.mean()),
        "zero_count": float(np.sum(cnt == 0.0)),
    }

    return u_good, part, stats


def make_u_bad(u_good: np.ndarray, seed: int) -> np.ndarray:
    """Build row-wise permuted proxy costs preserving row value multisets."""
    rng = np.random.default_rng(seed)
    k = u_good.shape[0]
    u_bad = np.empty_like(u_good)

    for i in range(k):
        vals = np.sort(u_good[i])
        perm = rng.permutation(k)
        u_bad[i, perm] = vals

    return u_bad


def sample_counts(
    q: np.ndarray,
    n_per_row: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample row-wise multinomial transition counts from kernel q."""
    rows = [rng.multinomial(n_per_row, q[i]) for i in range(q.shape[0])]
    return np.vstack(rows).astype(np.float64)


def empirical_kernel_for_eval(C: np.ndarray, eval_eps: float) -> np.ndarray:
    """Construct empirical row-normalized baseline kernel with eval-time clipping."""
    row_sum = C.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        msg = "Each training-count row must have positive mass for baseline model."
        raise ValueError(msg)

    Q = C / row_sum
    Q = np.clip(Q, eval_eps, None)
    Q = Q / Q.sum(axis=1, keepdims=True)
    return Q


def heldout_nll_per_transition(C_test: np.ndarray, Q: np.ndarray) -> float:
    """Compute held-out per-transition negative log-likelihood."""
    total = float(C_test.sum())
    if total <= 0.0:
        raise ValueError("C_test must have positive total mass.")
    if C_test.shape != Q.shape or np.any(Q < 0.0) or not np.isfinite(Q).all():
        raise ValueError("Invalid held-out kernel.")
    observed = C_test > 0.0
    if np.any(Q[observed] == 0.0):
        return float(np.inf)
    return float(-np.sum(C_test[observed] * np.log(Q[observed])) / total)


def evaluate_seed_list(
    seed_list: list[int],
    q_true: np.ndarray,
    u_good: np.ndarray,
    eval_eps: float,
    n_train_per_row: int,
    n_test_per_row: int,
    proxy_seed: int = 0,
) -> tuple[pd.DataFrame, list[float], list[float]]:
    """Evaluate baseline/good/bad models across a list of seeds."""
    rows = []
    lam_good_vals: list[float] = []
    lam_bad_vals: list[float] = []

    u_bad = make_u_bad(u_good, proxy_seed)
    for seed in seed_list:
        rng = np.random.default_rng(seed)

        C_train = sample_counts(q_true, n_train_per_row, rng)
        C_test = sample_counts(q_true, n_test_per_row, rng)

        Q_base = empirical_kernel_for_eval(C_train, eval_eps)
        nll_baseline = heldout_nll_per_transition(C_test, Q_base)

        lam_good, se_good = fit_lambda(C_train, u_good)
        Q_good = maxent_kernel(u_good, lam_good)
        nll_good = heldout_nll_per_transition(C_test, Q_good)

        lam_bad, se_bad = fit_lambda(C_train, u_bad)
        Q_bad = maxent_kernel(u_bad, lam_bad)
        nll_bad = heldout_nll_per_transition(C_test, Q_bad)

        lam_good_vals.append(float(lam_good))
        lam_bad_vals.append(float(lam_bad))

        rows.append(
            {
                "seed": int(seed),
                "lam_good": float(lam_good),
                "se_good": float(se_good),
                "lam_bad": float(lam_bad),
                "se_bad": float(se_bad),
                "nll_baseline": float(nll_baseline),
                "nll_good": float(nll_good),
                "nll_bad": float(nll_bad),
            }
        )

    return pd.DataFrame(rows), lam_good_vals, lam_bad_vals


def summarize(
    df: pd.DataFrame,
    lam_good: list[float],
    lam_bad: list[float],
) -> dict[str, float]:
    """Compute aggregate metrics used for the experiment self-check."""
    mean_nll_baseline = float(df["nll_baseline"].mean())
    mean_nll_good = float(df["nll_good"].mean())
    mean_nll_bad = float(df["nll_bad"].mean())

    std_lam_good = float(np.std(lam_good, ddof=0))
    std_lam_bad = float(np.std(lam_bad, ddof=0))

    rel_adv = (mean_nll_bad - mean_nll_good) / mean_nll_baseline
    return {
        "mean_nll_baseline": mean_nll_baseline,
        "mean_nll_good": mean_nll_good,
        "mean_nll_bad": mean_nll_bad,
        "std_lam_good": std_lam_good,
        "std_lam_bad": std_lam_bad,
        "rel_adv": rel_adv,
    }


def passes_self_check(metrics: dict[str, float]) -> bool:
    """Return whether required ablation criteria are satisfied."""
    cond1 = metrics["mean_nll_good"] < metrics["mean_nll_bad"]
    cond2 = metrics["rel_adv"] >= 0.02
    cond3 = metrics["std_lam_bad"] > metrics["std_lam_good"]
    return bool(cond1 and cond2 and cond3)


def plot_nll_by_seed(df: pd.DataFrame, png_path: Path) -> None:
    """Plot held-out NLL curves across seeds."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["seed"], df["nll_baseline"], marker="o", label="baseline")
    ax.plot(df["seed"], df["nll_good"], marker="o", label="good")
    ax.plot(df["seed"], df["nll_bad"], marker="o", label="bad")
    ax.set_xlabel("seed")
    ax.set_ylabel("held-out NLL")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run proxy-currency ablation with deterministic defaults."""
    args = parse_args()
    if args.traj_len < 2:
        raise ValueError(f"traj-len must be >= 2, got {args.traj_len}.")
    if args.n_train_per_row <= 0 or args.n_test_per_row <= 0:
        raise ValueError("Train/test counts per row must be positive.")

    seed_list = parse_int_list(args.seed_list)
    if len(seed_list) == 0:
        raise ValueError("seed-list must contain at least one seed.")

    if not 0.0 < args.true_budget_frac < 1.0:
        raise ValueError("true-budget-frac must lie strictly between 0 and 1.")
    if not 0.0 < args.eval_eps < 1.0:
        raise ValueError("eval-eps must lie strictly between 0 and 1.")

    # Fixed design and fixed proxy across sampling seeds. Report every outcome;
    # do not choose a new design after inspecting held-out results.
    selected_seed_cost = int(args.seed_cost)
    u_good, _, selected_stats = build_u_good(args, seed_cost=selected_seed_cost)
    q0 = maxent_kernel(u_good, lam=0.0)
    cost0 = expected_cost(q0, u_good)
    costmin = float(np.mean(np.min(u_good, axis=1)))
    selected_b_true = costmin + args.true_budget_frac * (cost0 - costmin)
    selected_lam_true, q_true, _ = solve_lambda_for_budget(u_good, selected_b_true)
    selected_df, selected_lam_good, selected_lam_bad = evaluate_seed_list(
        seed_list=seed_list,
        q_true=q_true,
        u_good=u_good,
        eval_eps=args.eval_eps,
        n_train_per_row=args.n_train_per_row,
        n_test_per_row=args.n_test_per_row,
        proxy_seed=args.proxy_seed,
    )

    metrics = summarize(selected_df, selected_lam_good, selected_lam_bad)

    assert selected_stats is not None
    print(
        f"u_good_min={selected_stats['u_min']:.12e} "
        f"u_good_max={selected_stats['u_max']:.12e} "
        f"mean={selected_stats['u_mean']:.12e} "
        f"zero_count={int(selected_stats['zero_count'])}"
    )
    print(f"lam_true={selected_lam_true:.12e}")
    print(f"b_true={selected_b_true:.12e}")

    run_id = f"proxy_{new_run_id()}"
    run_dir = Path(args.outdir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "metrics.csv"
    png_path = run_dir / "plot.png"

    selected_df.to_csv(csv_path, index=False)
    plot_nll_by_seed(selected_df, png_path)

    params = vars(args).copy()
    params["selected_seed_cost"] = selected_seed_cost
    save_metadata(
        run_id=run_id,
        params=params,
        metrics={
            "self_check_pass": passes_self_check(metrics),
            "design_selection": "prespecified_no_retries",
            "proxy_policy": "fixed_across_sampling_seeds",
            "mean_nll_baseline": metrics["mean_nll_baseline"],
            "mean_nll_good": metrics["mean_nll_good"],
            "mean_nll_bad": metrics["mean_nll_bad"],
            "rel_adv": metrics["rel_adv"],
            "std_lam_good": metrics["std_lam_good"],
            "std_lam_bad": metrics["std_lam_bad"],
        },
        paths={
            "run_dir": run_dir,
            "metrics_csv": csv_path,
            "plot_png": png_path,
        },
    )

    print(f"self_check_pass={passes_self_check(metrics)}")
    print(f"run_id={run_id}")
    print(f"csv_path={csv_path.as_posix()}")
    print(f"png_path={png_path.as_posix()}")
    print(f"mean_nll_baseline={metrics['mean_nll_baseline']:.12e}")
    print(f"mean_nll_good={metrics['mean_nll_good']:.12e}")
    print(f"mean_nll_bad={metrics['mean_nll_bad']:.12e}")
    print(f"rel_adv={metrics['rel_adv']:.12e}")
    print(f"lam_good_list={selected_lam_good}")
    print(f"lam_bad_list={selected_lam_bad}")


if __name__ == "__main__":
    main()
