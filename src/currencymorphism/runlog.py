"""Utilities for standardized experiment run metadata logging."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def new_run_id() -> str:
    """Return sortable UTC timestamp plus git short hash (or `nogit`)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    git_hash = _git_short_hash()
    return f"{ts}_{git_hash}"


def save_metadata(
    run_id: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    paths: dict[str, Any],
) -> Path:
    """Write `<run_dir>/meta.json` for one run and return its path."""
    if "run_dir" not in paths:
        raise ValueError("paths must include `run_dir`.")

    run_dir = Path(paths["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "meta.json"

    created_utc = datetime.now(timezone.utc).isoformat()
    git_hash = _git_short_hash()

    payload = {
        "run_id": run_id,
        "created_utc": created_utc,
        "git_short_hash": git_hash,
        "params": _to_jsonable(params),
        "summary_metrics": _to_jsonable(metrics),
        "paths": _to_jsonable(_normalize_paths(paths)),
    }

    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return meta_path


def _git_short_hash() -> str:
    """Return git short hash or `nogit` when unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        val = out.stdout.strip()
        return val if val else "nogit"
    except Exception:
        return "nogit"


def _repo_root() -> Path:
    """Return repository root path when detectable, else current directory."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        root = out.stdout.strip()
        return Path(root) if root else Path.cwd()
    except Exception:
        return Path.cwd()


def _normalize_paths(paths: dict[str, Any]) -> dict[str, Any]:
    """Convert path-like values to POSIX strings relative to repo root."""
    root = _repo_root().resolve()
    out: dict[str, Any] = {}
    for key, value in paths.items():
        if isinstance(value, (str, Path)):
            p = Path(value)
            abs_p = p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()
            try:
                rel = abs_p.relative_to(root)
                out[key] = rel.as_posix()
            except ValueError:
                out[key] = abs_p.as_posix()
        else:
            out[key] = value
    return out


def _to_jsonable(value: Any) -> Any:
    """Recursively convert numpy/path objects into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
