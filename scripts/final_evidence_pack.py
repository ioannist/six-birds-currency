from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from currencymorphism.runlog import new_run_id


@dataclass
class CommandResult:
    cmd_id: str
    command: str
    exit_code: int
    stdout: str
    stderr: str
    started_utc: str
    ended_utc: str
    parsed_kv: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run canonical evidence pack")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/final/canonical_runs.yaml",
        help="YAML config listing commands to execute",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/final_evidence",
        help="Base directory for frozen evidence packs",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in YAML config: {path}")
    return payload


def run_command(cmd_id: str, command: str, cwd: Path) -> CommandResult:
    started = datetime.now(timezone.utc).isoformat()
    proc = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    ended = datetime.now(timezone.utc).isoformat()

    combined = "\n".join([proc.stdout, proc.stderr])
    parsed = parse_key_values(combined)

    return CommandResult(
        cmd_id=cmd_id,
        command=command,
        exit_code=int(proc.returncode),
        stdout=proc.stdout,
        stderr=proc.stderr,
        started_utc=started,
        ended_utc=ended,
        parsed_kv=parsed,
    )


def parse_key_values(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.+)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.resolve().as_posix()


def git_short_hash(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        val = out.stdout.strip()
        return val if val else "nogit"
    except Exception:
        return "nogit"


def theorem_signature_line(root: Path) -> str:
    theorem_path = root / "lean" / "CurrencyMorphism" / "FiniteKL.lean"
    lines = theorem_path.read_text(encoding="utf-8").splitlines()

    start: int | None = None
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("theorem finiteKL_map_le"):
            start = idx
            break
    if start is None:
        return ""

    collected: list[str] = []
    for line in lines[start:]:
        collected.append(line.strip())
        if ":=" in line:
            break

    text = " ".join(collected).strip()
    if ":=" in text:
        text = text.split(":=", 1)[0].strip()
    return text


def require_path(root: Path, raw: str, desc: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"Missing {desc}: {p}")
    return p


def load_meta_and_metrics(
    root: Path,
    kv: dict[str, str],
) -> tuple[dict[str, Any], pd.DataFrame, Path, Path, Path | None]:
    csv_path = require_path(root, kv["csv_path"], "metrics csv")
    run_dir = csv_path.parent
    meta_path = require_path(root, str(run_dir / "meta.json"), "meta.json")
    metrics_df = pd.read_csv(csv_path)
    if "png_path" in kv:
        png_path = require_path(root, kv["png_path"], "plot png")
    else:
        png_path = None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta, metrics_df, run_dir, csv_path, png_path


def summarize_dpi(root: Path, result: CommandResult) -> dict[str, Any]:
    meta, metrics, run_dir, csv_path, png_path = load_meta_and_metrics(
        root,
        result.parsed_kv,
    )
    if "delta_mean" not in metrics.columns:
        raise ValueError("DPI metrics.csv missing delta_mean")

    summary = {
        "run_id": result.parsed_kv.get("run_id", ""),
        "run_dir": relpath(run_dir, root),
        "metrics_csv": relpath(csv_path, root),
        "plot_png": relpath(png_path, root) if png_path else "",
        "min_delta_mean": float(metrics["delta_mean"].min()),
        "max_delta_mean": float(metrics["delta_mean"].max()),
        "n_seeds": int(meta.get("summary_metrics", {}).get("n_seeds", 0)),
    }

    per_seed_rel = meta.get("paths", {}).get("per_seed_metrics_csv", "")
    if isinstance(per_seed_rel, str) and per_seed_rel:
        per_seed_path = require_path(root, per_seed_rel, "DPI per_seed_metrics.csv")
        per_seed = pd.read_csv(per_seed_path)
        tol = float(meta.get("summary_metrics", {}).get("tol", 0.0))
        summary["tol"] = tol
        summary["worst_seed_delta"] = float(per_seed["delta"].min())
        summary["dpi_seed_robust_pass"] = bool(float(per_seed["delta"].min()) >= -tol)
        summary["per_seed_metrics_csv"] = relpath(per_seed_path, root)
    else:
        summary["dpi_seed_robust_pass"] = bool(summary["min_delta_mean"] >= 0.0)

    return summary


def summarize_budget(root: Path, result: CommandResult) -> dict[str, Any]:
    meta, metrics, run_dir, csv_path, png_path = load_meta_and_metrics(
        root,
        result.parsed_kv,
    )
    sm = meta.get("summary_metrics", {})
    return {
        "run_id": result.parsed_kv.get("run_id", ""),
        "run_dir": relpath(run_dir, root),
        "metrics_csv": relpath(csv_path, root),
        "plot_png": relpath(png_path, root) if png_path else "",
        "spearman_rho_mean": float(sm.get("spearman_rho_mean", float("nan"))),
        "lam_mean_max": float(sm.get("lam_mean_max", float("nan"))),
        "n_tail_zero_lambda": int(sm.get("n_tail_zero_lambda", 0)),
        "budget_pass": bool(
            float(sm.get("spearman_rho_mean", 0.0)) < -0.9
            and int(sm.get("n_tail_zero_lambda", 0)) >= 1
        ),
    }


def summarize_ladder(root: Path, result: CommandResult) -> dict[str, Any]:
    meta, metrics, run_dir, csv_path, png_path = load_meta_and_metrics(
        root,
        result.parsed_kv,
    )
    sm = meta.get("summary_metrics", {})
    beta0 = float(sm.get("beta1_coarse_mean", float("nan")))
    beta1 = float(sm.get("beta1_mid_mean", float("nan")))
    beta2 = float(sm.get("beta1_fine_mean", float("nan")))

    json_path = ""
    if "json_path" in result.parsed_kv:
        jp = require_path(root, result.parsed_kv["json_path"], "ladder summary json")
        json_path = relpath(jp, root)

    return {
        "run_id": result.parsed_kv.get("run_id", ""),
        "run_dir": relpath(run_dir, root),
        "metrics_csv": relpath(csv_path, root),
        "plot_png": relpath(png_path, root) if png_path else "",
        "summary_json": json_path,
        "beta1_coarse_mean": beta0,
        "beta1_mid_mean": beta1,
        "beta1_fine_mean": beta2,
        "ladder_pass": bool(beta0 == 0.0 and beta1 > 0.0 and beta2 > beta1),
    }


def summarize_proxy(root: Path, result: CommandResult) -> dict[str, Any]:
    meta, metrics, run_dir, csv_path, png_path = load_meta_and_metrics(
        root,
        result.parsed_kv,
    )
    sm = meta.get("summary_metrics", {})
    good = float(sm.get("mean_nll_good", float("nan")))
    bad = float(sm.get("mean_nll_bad", float("nan")))
    rel_adv = float(sm.get("rel_adv", float("nan")))
    std_good = float(sm.get("std_lam_good", float("nan")))
    std_bad = float(sm.get("std_lam_bad", float("nan")))

    return {
        "run_id": result.parsed_kv.get("run_id", ""),
        "run_dir": relpath(run_dir, root),
        "metrics_csv": relpath(csv_path, root),
        "plot_png": relpath(png_path, root) if png_path else "",
        "mean_nll_baseline": float(sm.get("mean_nll_baseline", float("nan"))),
        "mean_nll_good": good,
        "mean_nll_bad": bad,
        "rel_adv": rel_adv,
        "std_lam_good": std_good,
        "std_lam_bad": std_bad,
        "proxy_pass": bool(good < bad and rel_adv >= 0.02 and std_bad > std_good),
    }


def summarize_idempotence(root: Path, result: CommandResult) -> dict[str, Any]:
    meta, metrics, run_dir, csv_path, png_path = load_meta_and_metrics(
        root,
        result.parsed_kv,
    )
    sm = meta.get("summary_metrics", {})

    lam_vals = metrics["lam"].to_numpy(dtype=float)
    defect_vals = metrics["defect_mean"].to_numpy(dtype=float)
    idx_max_lam = int(lam_vals.argmax())
    idx_min_lam = int(lam_vals.argmin())
    defect_decrease = float(defect_vals[idx_max_lam]) < float(defect_vals[idx_min_lam])

    rho = float(sm.get("spearman_rho", float("nan")))
    return {
        "run_id": result.parsed_kv.get("run_id", ""),
        "run_dir": relpath(run_dir, root),
        "metrics_csv": relpath(csv_path, root),
        "plot_png": relpath(png_path, root) if png_path else "",
        "spearman_rho": rho,
        "defect_mean_min": float(sm.get("defect_mean_min", float("nan"))),
        "defect_mean_max": float(sm.get("defect_mean_max", float("nan"))),
        "defect_end_lt_start": defect_decrease,
        "idempotence_pass": bool(rho < -0.7 and defect_decrease),
    }


def summarize_lean(root: Path, result: CommandResult) -> dict[str, Any]:
    sig = theorem_signature_line(root)
    return {
        "build_success": bool(result.exit_code == 0),
        "theorem_name": "CurrencyMorphism.finiteKL_map_le",
        "theorem_signature": sig,
        "proof_target": "fallback_finiteKL",
    }


def summarize_index(root: Path, result: CommandResult) -> dict[str, Any]:
    idx_path = result.parsed_kv.get("index_path", "results/index.csv")
    p = require_path(root, idx_path, "results index csv")
    rows = int(result.parsed_kv.get("n_rows", -1))
    if rows < 0:
        rows = len(pd.read_csv(p))
    return {
        "index_csv": relpath(p, root),
        "n_rows": rows,
    }


def build_manifest(
    root: Path,
    pack_id: str,
    config_path: Path,
    cmd_results: list[CommandResult],
) -> dict[str, Any]:
    by_id = {r.cmd_id: r for r in cmd_results}

    summaries: dict[str, Any] = {
        "dpi_scan": summarize_dpi(root, by_id["dpi_scan"]),
        "budget_sweep": summarize_budget(root, by_id["budget_sweep"]),
        "currency_ladder": summarize_ladder(root, by_id["currency_ladder"]),
        "proxy_ablation": summarize_proxy(root, by_id["proxy_ablation"]),
        "idempotence_budget": summarize_idempotence(root, by_id["idempotence_budget"]),
        "lean": summarize_lean(root, by_id["lean_build"]),
        "index_results": summarize_index(root, by_id["index_results"]),
    }

    threshold_pass = {
        "dpi": bool(summaries["dpi_scan"].get("dpi_seed_robust_pass", False)),
        "budget": bool(summaries["budget_sweep"].get("budget_pass", False)),
        "ladder": bool(summaries["currency_ladder"].get("ladder_pass", False)),
        "proxy": bool(summaries["proxy_ablation"].get("proxy_pass", False)),
        "idempotence": bool(
            summaries["idempotence_budget"].get("idempotence_pass", False)
        ),
        "lean": bool(summaries["lean"].get("build_success", False)),
    }

    cmd_payload = [
        {
            "id": r.cmd_id,
            "command": r.command,
            "exit_code": r.exit_code,
            "started_utc": r.started_utc,
            "ended_utc": r.ended_utc,
            "parsed": r.parsed_kv,
        }
        for r in cmd_results
    ]

    return {
        "pack_id": pack_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_short_hash": git_short_hash(root),
        "config_path": relpath(config_path, root),
        "commands": cmd_payload,
        "summaries": summaries,
        "threshold_checks": threshold_pass,
        "all_commands_exit_zero": all(r.exit_code == 0 for r in cmd_results),
        "all_thresholds_pass": all(threshold_pass.values()),
    }


def write_commands_log(path: Path, cmd_results: list[CommandResult]) -> None:
    lines: list[str] = []
    for r in cmd_results:
        lines.append(f"## {r.cmd_id}")
        lines.append(f"command: {r.command}")
        lines.append(f"exit_code: {r.exit_code}")
        lines.append(f"started_utc: {r.started_utc}")
        lines.append(f"ended_utc: {r.ended_utc}")
        lines.append("stdout:")
        lines.append(r.stdout.rstrip())
        lines.append("stderr:")
        lines.append(r.stderr.rstrip())
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = repo_root()

    config_path = (root / args.config).resolve()
    cfg = read_yaml(config_path)

    commands = cfg.get("commands", [])
    if not isinstance(commands, list) or not commands:
        raise ValueError("Config must contain a non-empty `commands` list.")

    pack_id = f"pack_{new_run_id()}"
    pack_dir = (root / args.outdir / pack_id).resolve()
    pack_dir.mkdir(parents=True, exist_ok=True)

    cmd_results: list[CommandResult] = []
    for entry in commands:
        if not isinstance(entry, dict):
            raise ValueError("Each command entry must be a dict.")
        cmd_id = str(entry.get("id", "")).strip()
        command = str(entry.get("cmd", "")).strip()
        if not cmd_id or not command:
            raise ValueError(f"Invalid command entry: {entry}")
        result = run_command(cmd_id=cmd_id, command=command, cwd=root)
        cmd_results.append(result)

    commands_log = pack_dir / "commands.log"
    write_commands_log(commands_log, cmd_results)

    manifest = build_manifest(root, pack_id, config_path, cmd_results)
    manifest_path = pack_dir / "manifest.json"
    summary_path = pack_dir / "summary.json"

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_payload = {
        "pack_id": pack_id,
        "summaries": manifest["summaries"],
        "threshold_checks": manifest["threshold_checks"],
        "all_commands_exit_zero": manifest["all_commands_exit_zero"],
        "all_thresholds_pass": manifest["all_thresholds_pass"],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    if not manifest["all_commands_exit_zero"]:
        failed = [r.cmd_id for r in cmd_results if r.exit_code != 0]
        raise RuntimeError(f"One or more commands failed: {failed}")

    # Verify artifact existence for all referenced paths.
    referenced_paths: list[str] = []
    for exp in [
        "dpi_scan",
        "budget_sweep",
        "currency_ladder",
        "proxy_ablation",
        "idempotence_budget",
    ]:
        exp_summary = manifest["summaries"][exp]
        for key in [
            "run_dir",
            "metrics_csv",
            "plot_png",
            "summary_json",
            "per_seed_metrics_csv",
        ]:
            val = exp_summary.get(key)
            if isinstance(val, str) and val:
                referenced_paths.append(val)

    for rel in referenced_paths:
        p = (root / rel).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Manifest references missing artifact: {rel}")

    print(f"pack_id={pack_id}")
    print(f"manifest_path={relpath(manifest_path, root)}")
    print(f"summary_path={relpath(summary_path, root)}")
    print(f"commands_log={relpath(commands_log, root)}")


if __name__ == "__main__":
    main()
