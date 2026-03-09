from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class CmdRun:
    name: str
    command: str
    exit_code: int
    stdout: str
    stderr: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean-room reproducibility audit")
    parser.add_argument(
        "--manifest",
        type=str,
        default="docs/experiments/final/run_manifest.json",
    )
    parser.add_argument(
        "--tolerances",
        type=str,
        default="configs/final/tolerances.json",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="docs/experiments/final/repro_audit.json",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="docs/experiments/final/repro_audit.csv",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.resolve().as_posix()


def run_shell(command: str, cwd: Path, env: dict[str, str] | None = None) -> CmdRun:
    proc = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return CmdRun(
        name=command,
        command=command,
        exit_code=int(proc.returncode),
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def parse_key_values(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            out[key] = value
    return out


def ensure_exists(path: Path, desc: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {desc}: {path}")
    return path


def resolve_from_manifest(root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def representative_dpi(root: Path, manifest: dict[str, Any]) -> float:
    rel = manifest["summaries"]["dpi_scan"]["metrics_csv"]
    df = pd.read_csv(resolve_from_manifest(root, rel))
    if "k" in df.columns and (df["k"] == 16).any():
        row = df[df["k"] == 16].iloc[0]
    else:
        row = df.iloc[len(df) // 2]
    return float(row["delta_mean"])


def representative_budget(root: Path, manifest: dict[str, Any]) -> float:
    rel = manifest["summaries"]["budget_sweep"]["metrics_csv"]
    df = pd.read_csv(resolve_from_manifest(root, rel))
    row = df.iloc[len(df) // 2]
    return float(row["lam_mean"])


def representative_idempotence(root: Path, manifest: dict[str, Any]) -> float:
    rel = manifest["summaries"]["idempotence_budget"]["metrics_csv"]
    df = pd.read_csv(resolve_from_manifest(root, rel))
    row = df.iloc[len(df) // 2]
    return float(row["defect_mean"])


def relative_diff(old: float, new: float, eps: float = 1e-12) -> float:
    return abs(new - old) / max(abs(old), eps)


def check_docs_artifacts(root: Path) -> list[str]:
    required = [
        "docs/experiments/final/run_manifest.json",
        "docs/experiments/final/claim_ledger.csv",
        "docs/experiments/final/dpi_summary.csv",
        "docs/experiments/final/budget_summary.csv",
        "docs/experiments/final/ladder_summary.csv",
        "docs/experiments/final/proxy_summary.csv",
        "docs/experiments/final/idempotence_summary.csv",
        "docs/experiments/final/lean_summary.json",
        "docs/experiments/final/fig_dpi_scan.png",
        "docs/experiments/final/fig_budget_sweep.png",
        "docs/experiments/final/fig_currency_ladder.png",
        "docs/experiments/final/fig_proxy_ablation.png",
        "docs/experiments/final/fig_idempotence_budget.png",
    ]
    missing = [p for p in required if not (root / p).exists()]
    return missing


def main() -> None:
    args = parse_args()
    root = repo_root()

    baseline_manifest_path = ensure_exists(
        (root / args.manifest).resolve(),
        "baseline run manifest",
    )
    tolerances_path = ensure_exists(
        (root / args.tolerances).resolve(),
        "tolerances json",
    )

    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    tolerances = json.loads(tolerances_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="currency_repro_") as tmp:
        tmp_path = Path(tmp)
        venv_dir = tmp_path / "venv"
        env_mode = "venv"

        venv_create = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        env = os.environ.copy()
        if venv_create.returncode == 0:
            python_exec = venv_dir / "bin" / "python"
            env["VIRTUAL_ENV"] = str(venv_dir)
            env["PATH"] = f"{venv_dir / 'bin'}:{env.get('PATH', '')}"
        else:
            env_mode = "fallback_system_python"
            python_exec = Path(sys.executable)

        clean_install = run_shell(
            f"{python_exec} -m pip install -e \".[dev]\"",
            cwd=root,
            env=env,
        )
        pytest_run = run_shell("pytest -q", cwd=root, env=env)
        lint_run = run_shell("make lint", cwd=root, env=env)
        lean_run = run_shell("cd lean && lake build", cwd=root, env=env)
        pack_run = run_shell(
            f"{python_exec} scripts/final_evidence_pack.py",
            cwd=root,
            env=env,
        )

        pack_kv = parse_key_values("\n".join([pack_run.stdout, pack_run.stderr]))
        if "manifest_path" not in pack_kv:
            raise RuntimeError(
                "Could not parse manifest_path from final_evidence_pack output"
            )

        new_manifest_path = resolve_from_manifest(root, pack_kv["manifest_path"])
        ensure_exists(new_manifest_path, "new evidence manifest")
        new_manifest = json.loads(new_manifest_path.read_text(encoding="utf-8"))

        missing_docs = check_docs_artifacts(root)

        rows: list[dict[str, Any]] = []

        # dpi
        old_dpi = representative_dpi(root, baseline_manifest)
        new_dpi = representative_dpi(root, new_manifest)
        dpi_abs = abs(new_dpi - old_dpi)
        dpi_tol = float(tolerances["dpi"]["delta_rep_abs"])
        rows.append(
            {
                "experiment": "dpi",
                "metric": "delta_rep",
                "baseline": old_dpi,
                "new": new_dpi,
                "deviation": dpi_abs,
                "tolerance": dpi_tol,
                "pass": bool(dpi_abs <= dpi_tol),
                "mode": "abs",
            }
        )

        # budget
        old_budget = representative_budget(root, baseline_manifest)
        new_budget = representative_budget(root, new_manifest)
        budget_rel = relative_diff(old_budget, new_budget)
        budget_tol = float(tolerances["budget"]["lam_rep_rel"])
        rows.append(
            {
                "experiment": "budget",
                "metric": "lam_rep",
                "baseline": old_budget,
                "new": new_budget,
                "deviation": budget_rel,
                "tolerance": budget_tol,
                "pass": bool(budget_rel <= budget_tol),
                "mode": "rel",
            }
        )

        # ladder exact equality
        for metric in ["beta1_coarse_mean", "beta1_mid_mean", "beta1_fine_mean"]:
            old_v = float(baseline_manifest["summaries"]["currency_ladder"][metric])
            new_v = float(new_manifest["summaries"]["currency_ladder"][metric])
            rows.append(
                {
                    "experiment": "ladder",
                    "metric": metric,
                    "baseline": old_v,
                    "new": new_v,
                    "deviation": abs(new_v - old_v),
                    "tolerance": 0.0,
                    "pass": bool(new_v == old_v),
                    "mode": "exact",
                }
            )

        # proxy rel diffs
        proxy_tol = float(tolerances["proxy"]["mean_nll_rel"])
        for metric in ["mean_nll_baseline", "mean_nll_good", "mean_nll_bad"]:
            old_v = float(baseline_manifest["summaries"]["proxy_ablation"][metric])
            new_v = float(new_manifest["summaries"]["proxy_ablation"][metric])
            rel = relative_diff(old_v, new_v)
            rows.append(
                {
                    "experiment": "proxy",
                    "metric": metric,
                    "baseline": old_v,
                    "new": new_v,
                    "deviation": rel,
                    "tolerance": proxy_tol,
                    "pass": bool(rel <= proxy_tol),
                    "mode": "rel",
                }
            )

        # idempotence representative
        old_idem = representative_idempotence(root, baseline_manifest)
        new_idem = representative_idempotence(root, new_manifest)
        idem_rel = relative_diff(old_idem, new_idem)
        idem_tol = float(tolerances["idempotence"]["defect_rep_rel"])
        rows.append(
            {
                "experiment": "idempotence",
                "metric": "defect_rep",
                "baseline": old_idem,
                "new": new_idem,
                "deviation": idem_rel,
                "tolerance": idem_tol,
                "pass": bool(idem_rel <= idem_tol),
                "mode": "rel",
            }
        )

        # lean
        old_lean = bool(baseline_manifest["summaries"]["lean"]["build_success"])
        new_lean = bool(new_manifest["summaries"]["lean"]["build_success"])
        rows.append(
            {
                "experiment": "lean",
                "metric": "build_success",
                "baseline": old_lean,
                "new": new_lean,
                "deviation": 0 if old_lean == new_lean else 1,
                "tolerance": 0,
                "pass": bool(old_lean and new_lean),
                "mode": "exact",
            }
        )

        index_run = run_shell("python scripts/index_results.py", cwd=root, env=env)

        all_cmds_ok = all(
            x.exit_code == 0
            for x in [
                clean_install,
                pytest_run,
                lint_run,
                lean_run,
                pack_run,
                index_run,
            ]
        )
        all_metrics_ok = all(bool(row["pass"]) for row in rows)
        no_missing_docs = len(missing_docs) == 0

        out_json = (root / args.out_json).resolve()
        out_csv = (root / args.out_csv).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)

        audit = {
            "baseline_manifest": relpath(baseline_manifest_path, root),
            "new_manifest": relpath(new_manifest_path, root),
            "tolerances": json.loads(tolerances_path.read_text(encoding="utf-8")),
            "environment_mode": env_mode,
            "venv_create_exit_code": int(venv_create.returncode),
            "venv_create_stdout": venv_create.stdout,
            "command_exit_codes": {
                "clean_install": clean_install.exit_code,
                "pytest_q": pytest_run.exit_code,
                "make_lint": lint_run.exit_code,
                "lean_build": lean_run.exit_code,
                "final_evidence_pack": pack_run.exit_code,
                "index_results": index_run.exit_code,
            },
            "missing_docs_artifacts": missing_docs,
            "comparisons": rows,
            "all_commands_ok": all_cmds_ok,
            "all_metrics_within_tolerance": all_metrics_ok,
            "no_missing_docs_artifacts": no_missing_docs,
            "pass": bool(all_cmds_ok and all_metrics_ok and no_missing_docs),
        }

        out_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "experiment",
                    "metric",
                    "baseline",
                    "new",
                    "deviation",
                    "tolerance",
                    "mode",
                    "pass",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        if not audit["pass"]:
            raise RuntimeError(
                "Repro audit failed: "
                f"commands_ok={all_cmds_ok}, metrics_ok={all_metrics_ok}, "
                f"missing_docs={missing_docs}"
            )

        print(f"repro_audit_json={relpath(out_json, root)}")
        print(f"repro_audit_csv={relpath(out_csv, root)}")


if __name__ == "__main__":
    main()
