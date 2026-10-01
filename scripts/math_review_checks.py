"""Reproduce corrected mathematical evidence without changing the frozen pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from currencymorphism.audits_cycles import (
    cycle_affinities,
    cycle_basis,
    cycle_rank,
    undirected_support_graph,
)
from currencymorphism.generators import reversible_kernel
from currencymorphism.lens import lumped_kernel
from currencymorphism.markov import stationary_dist
from exp_currency_ladder import build_ladder_module_kernel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("review"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = {}
    commands = {}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env["MPLBACKEND"] = "Agg"
    with tempfile.TemporaryDirectory(prefix="currency_math_review_") as tmp:
        env["MPLCONFIGDIR"] = str(Path(tmp) / "mpl")
        for name, script, extra in [
            ("dpi", "exp_dpi_scan.py", ["--seeds", "0", "1", "2", "3", "4"]),
            ("budget", "exp_budget_sweep.py", ["--seeds", "0", "1", "2", "3", "4"]),
            ("ladder", "exp_currency_ladder.py", ["--seeds", "0", "1", "2", "3", "4"]),
            ("proxy", "exp_proxy_ablation.py", []),
            ("idempotence", "exp_idempotence_vs_budget.py", []),
        ]:
            command = [
                sys.executable,
                str(root / "scripts" / script),
                "--outdir",
                str(Path(tmp) / name),
                *extra,
            ]
            proc = subprocess.run(
                command, cwd=root, env=env, capture_output=True, text=True, check=True
            )
            csv = next(
                line.split("=", 1)[1]
                for line in proc.stdout.splitlines()
                if line.startswith("csv_path=")
            )
            df = pd.read_csv(csv)
            df.to_csv(output / f"{name}_checked.csv", index=False)
            frames[name] = df
            commands[name] = {
                "script": script,
                "arguments": extra,
                "exit_code": proc.returncode,
            }

    proxy, idem, dpi = (frames[n] for n in ["proxy", "idempotence", "dpi"])
    M, L = 6, 10
    P = build_ladder_module_kernel(M, L, np.zeros(M), 0.2)
    cycles = cycle_basis(undirected_support_graph(P))
    columns = []
    for m in range(M):
        affinities = np.zeros(M)
        affinities[m] = 1
        kernel = build_ladder_module_kernel(M, L, affinities, 0.2)
        columns.append(cycle_affinities(kernel, cycles))
    family_rank = int(np.linalg.matrix_rank(np.array(columns).T, tol=1e-10))
    Ptree = reversible_kernel(nx.path_graph(4))
    Q = lumped_kernel(Ptree, np.array([0, 1, 2, 0]), stationary_dist(Ptree))
    source_files = sorted(
        set(
            list((root / "src").rglob("*.py"))
            + list((root / "tests").glob("*.py"))
            + list((root / "scripts").glob("*.py"))
            + list((root / "lean" / "CurrencyMorphism").glob("*.lean"))
            + [
                root / "lean" / "CurrencyMorphism.lean",
                root / "lean" / "CheckAxioms.lean",
            ]
        )
    )
    receipt = {
        "checkpoint": "b4a5140",
        "commands": commands,
        "source_sha256": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files
        },
        "reverse_mixture": 0.01,
        "dpi_min_delta_mean": float(dpi.delta_mean.min()),
        "dpi_raw_micro_missing_reverse_mass": float(dpi.inf_micro_mean.iloc[0]),
        "proxy_design_seed": 0,
        "proxy_permutation_seed": 0,
        "proxy_no_retries": True,
        "proxy_good_mean_nll": float(proxy.nll_good.mean()),
        "proxy_bad_mean_nll": float(proxy.nll_bad.mean()),
        "proxy_good_price_std": float(proxy.lam_good.std(ddof=0)),
        "proxy_bad_price_std": float(proxy.lam_bad.std(ddof=0)),
        "proxy_prediction_advantage": bool(
            proxy.nll_good.mean() < proxy.nll_bad.mean()
        ),
        "proxy_old_price_stability_claim_supported": bool(
            proxy.lam_bad.std(ddof=0) > proxy.lam_good.std(ddof=0)
        ),
        "ladder_cycle_ranks": [int(v) for v in frames["ladder"].beta1_mean],
        "fine_affinity_family_rank": family_rank,
        "coarsening_counterexample_cycle_ranks": [
            cycle_rank(undirected_support_graph(Ptree)),
            cycle_rank(undirected_support_graph(Q)),
        ],
        "idempotence_min": float(idem.defect_mean.min()),
        "idempotence_max": float(idem.defect_mean.max()),
        "idempotence_all_exit_bounds_hold": bool(
            (idem.defect_mean <= idem.stationary_defect_bound + 1e-12).all()
        ),
        "idempotence_calibration_min": float(idem.calibration_cost.min()),
        "idempotence_controlled_stationary_cost_min": float(
            idem.controlled_stationary_cost.min()
        ),
        "idempotence_controlled_stationary_cost_max": float(
            idem.controlled_stationary_cost.max()
        ),
        "old_frozen_evidence": {
            "status": "historical; truncated DPI and selected proxy design",
            "unchanged": True,
        },
    }
    (output / "numerical_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"receipt_path={output / 'numerical_receipt.json'}")
    print(f"proxy_prediction_advantage={receipt['proxy_prediction_advantage']}")
    print(
        "proxy_old_price_stability_claim_supported="
        f"{receipt['proxy_old_price_stability_claim_supported']}"
    )


if __name__ == "__main__":
    main()
