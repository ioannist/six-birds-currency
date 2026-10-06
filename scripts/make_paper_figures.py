"""Build the manuscript figures from the corrected receipts in review/.

Every plotted value is read from review/*_checked.csv, except the per-level
rank of the affinity-family map in the ladder figure. That rank is a
deterministic linear-algebra computation on the ladder construction (the
fine-level value is the one recorded in review/numerical_receipt.json); it is
written next to the figures as ladder_family_rank.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from currencymorphism.audits_cycles import (  # noqa: E402
    cycle_affinities,
    cycle_basis,
    undirected_support_graph,
)
from currencymorphism.lens import lumped_kernel  # noqa: E402
from currencymorphism.markov import stationary_dist  # noqa: E402
from exp_currency_ladder import (  # noqa: E402
    build_ladder_module_kernel,
    make_partitions,
)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e4e3df"

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(-0.14, 1.04, text, transform=ax.transAxes, fontweight="bold", va="bottom")


def fig_dpi(review: Path, out: Path) -> None:
    df = pd.read_csv(review / "dpi_checked.csv")
    k = df["k"].to_numpy()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.4))
    a.errorbar(
        k, df.sigma_micro_mean, yerr=df.sigma_micro_std, color=BLUE, marker="o",
        capsize=2, label=r"micro paths (60 states)",
    )
    a.errorbar(
        k, df.sigma_coarse_mean, yerr=df.sigma_coarse_std, color=ORANGE, marker="s",
        capsize=2, label=r"observed paths ($k$ blocks)",
    )
    a.set_xscale("log", base=2)
    a.set_xticks(k, [str(v) for v in k])
    a.set_xlabel(r"number of observed blocks $k$")
    a.set_ylabel(r"regularized audit $\Sigma_{5,\varepsilon}$")
    a.legend(loc="center left", bbox_to_anchor=(0.0, 0.55))
    panel_label(a, "(a)")
    b.errorbar(
        k, df.delta_mean, yerr=df.delta_std, color=AQUA, marker="o", capsize=2
    )
    b.axhline(0.0, color=MUTED, linewidth=0.8, linestyle="--")
    b.set_xscale("log", base=2)
    b.set_xticks(k, [str(v) for v in k])
    b.set_xlabel(r"number of observed blocks $k$")
    b.set_ylabel(r"margin $\Sigma^{\rm micro}-\Sigma^{\rm obs}$")
    b.set_ylim(bottom=-0.002)
    panel_label(b, "(b)")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out / "fig_dpi_audit.pdf")
    plt.close(fig)


def fig_budget(review: Path, out: Path) -> None:
    df = pd.read_csv(review / "budget_checked.csv")
    alpha = df["alpha"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.axvspan(1.0, alpha.max() + 0.04, color=GRID, alpha=0.7, linewidth=0)
    ax.axvline(0.0, color=MUTED, linewidth=0.8, linestyle="--")
    ax.errorbar(
        alpha, df.lam_mean, yerr=df.lam_std, color=BLUE, marker="o", capsize=2,
        label="mean over 5 seeds (bars: s.d.)",
    )
    ax.set_xlim(-0.06, alpha.max() + 0.04)
    ax.set_xlabel(
        r"normalized budget $(b-c_{\min})/(c_0-c_{\min})$"
    )
    ax.set_ylabel(r"shadow price $\lambda^\star(b)$")
    top = float((df.lam_mean + df.lam_std).max())
    ax.text(0.012, 0.3 * top, r"$b\to c_{\min}$:" "\n" r"$\lambda^\star\to\infty$",
            color=MUTED, va="top", fontsize=8)
    ax.text(0.55, 0.55 * top, "active budget\n" r"unique $\lambda^\star>0$",
            color=INK, ha="center", fontsize=8)
    ax.text(1.06, 0.55 * top, "slack\n" r"$\lambda^\star=0$", color=INK,
            ha="center", fontsize=8)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fig_budget_price.pdf")
    plt.close(fig)


def family_ranks(M: int = 6, L: int = 10, bridge: float = 0.2) -> pd.DataFrame:
    """Rank of the map from ring affinities (A_1..A_M) to cycle affinities."""
    parts = make_partitions(M, L)
    rows = []
    for name in ["coarse", "mid", "fine"]:
        part = parts[name]
        P0 = build_ladder_module_kernel(M, L, np.zeros(M), bridge)
        Q0 = lumped_kernel(P0, part, pi=stationary_dist(P0, method="eigs"))
        cycles = cycle_basis(undirected_support_graph(Q0, thresh=0.0))
        cols = []
        for m in range(M):
            A = np.zeros(M)
            A[m] = 1.0
            P = build_ladder_module_kernel(M, L, A, bridge)
            Q = lumped_kernel(P, part, pi=stationary_dist(P, method="eigs"))
            cols.append(cycle_affinities(Q, cycles) if cycles else np.zeros(0))
        mat = np.array(cols).T if cycles else np.zeros((0, M))
        rank = int(np.linalg.matrix_rank(mat, tol=1e-10)) if mat.size else 0
        rows.append({"level": name, "cycle_basis_size": len(cycles),
                     "affinity_family_rank": rank})
    return pd.DataFrame(rows)


def fig_ladder(review: Path, out: Path) -> None:
    df = pd.read_csv(review / "ladder_checked.csv")
    fam = family_ranks()
    receipt = json.loads((review / "numerical_receipt.json").read_text())
    fine_rank = int(fam.loc[fam.level == "fine", "affinity_family_rank"].iloc[0])
    if fine_rank != int(receipt["fine_affinity_family_rank"]):
        raise RuntimeError("Fine affinity-family rank disagrees with the receipt.")
    fam.to_csv(out / "ladder_family_rank.csv", index=False)
    labels = ["coarse\n($k=6$)", "middle\n($k=12$)", "fine\n($k=60$)"]
    x = np.arange(3)
    w = 0.36
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(6.4, 2.5), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    beta = df["beta1_mean"].to_numpy()
    rank = fam["affinity_family_rank"].to_numpy()
    a.bar(x - w / 2 - 0.01, beta, w, color=BLUE,
          label=r"cycle-space dimension $\beta_1$")
    a.bar(x + w / 2 + 0.01, rank, w, color=ORANGE, label="rank of drive-parameter map")
    for xi, v in zip(x - w / 2 - 0.01, beta, strict=True):
        a.text(xi, v + 0.25, f"{v:.0f}", ha="center", fontsize=8, color=INK)
    for xi, v in zip(x + w / 2 + 0.01, rank, strict=True):
        a.text(xi, v + 0.25, f"{v:.0f}", ha="center", fontsize=8, color=INK)
    a.set_xticks(x, labels)
    a.set_ylim(0, 14)
    a.set_ylabel("dimension")
    a.legend(loc="upper left")
    a.grid(axis="x", visible=False)
    panel_label(a, "(a)")
    norms = df["aff_norm_mean"].to_numpy()
    b.bar(x, norms, 0.5, color=AQUA)
    for xi, v in zip(x, norms, strict=True):
        txt = "0" if v < 1e-12 else f"{v:.3f}"
        b.text(xi, v + 0.04, txt, ha="center", fontsize=8, color=INK)
    b.set_xticks(x, labels)
    b.set_ylim(0, 1.7)
    b.set_ylabel(r"affinity-vector norm $\|\mathcal{A}\|_2$")
    b.grid(axis="x", visible=False)
    panel_label(b, "(b)")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out / "fig_ladder.pdf")
    plt.close(fig)


def fig_proxy(review: Path, out: Path) -> None:
    df = pd.read_csv(review / "proxy_checked.csv")
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    s = df["seed"].to_numpy()
    ax.plot(s, df.nll_baseline, color=MUTED, marker="D", linestyle=":",
            label="row-wise empirical frequencies")
    ax.plot(s, df.nll_bad, color=ORANGE, marker="s",
            label="one-price model, proxy cost")
    ax.plot(s, df.nll_good, color=BLUE, marker="o",
            label="one-price model, ledger cost")
    ax.set_xticks(s, [str(v) for v in s])
    ax.set_xlabel("sampling seed (training and test counts)")
    ax.set_ylabel("held-out NLL per transition")
    ax.set_ylim(1.5, 6.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out / "fig_proxy.pdf")
    plt.close(fig)


def fig_packaging(review: Path, out: Path) -> None:
    df = pd.read_csv(review / "idempotence_checked.csv")
    df = df.sort_values("lam")
    lam = df["lam"].to_numpy()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    a.plot(lam, df.stationary_defect_bound.clip(upper=2.0), color=MUTED,
           linestyle="--", label=r"bound $\min(2,\,2\tau\varepsilon_\lambda)$")
    a.plot(lam, df.defect_max, color=ORANGE, marker="s", label="worst block")
    a.plot(lam, df.defect_mean, color=BLUE, marker="o", label="mean over blocks")
    a.set_yscale("log")
    a.set_ylim(5e-4, 5)
    a.set_xlabel(r"maintenance price $\lambda$")
    a.set_ylabel(r"idempotence defect $\delta$")
    a.legend(loc="lower left")
    panel_label(a, "(a)")
    b.plot(lam, df.calibration_cost, color=MUTED, marker="D", linestyle=":",
           label="auxiliary calibration cost")
    b.plot(lam, df.controlled_stationary_cost, color=AQUA, marker="o",
           label="actual stationary exit rate")
    b.set_yscale("log")
    b.set_ylim(3e-5, 3)
    b.set_xlabel(r"maintenance price $\lambda$")
    b.set_ylabel("cost per step")
    b.legend(loc="lower left")
    panel_label(b, "(b)")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out / "fig_packaging.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=Path("review"))
    parser.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for build in (fig_dpi, fig_budget, fig_ladder, fig_proxy, fig_packaging):
        build(args.review, args.out)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
