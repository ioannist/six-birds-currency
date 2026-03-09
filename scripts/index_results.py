from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main() -> None:
    """Scan results tree and build a unified run index CSV."""
    results_root = Path("results")
    results_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    for exp_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        exp_name = exp_dir.name

        for run_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            meta_path = run_dir / "meta.json"
            if not meta_path.exists():
                continue

            row = build_modern_row(results_root, exp_name, run_dir, meta_path)
            rows.append(row)

        rows.extend(build_legacy_rows(results_root, exp_name, exp_dir))

    df = pd.DataFrame(rows)
    if not df.empty:
        df["_sort_key"] = df["created_utc"].map(created_utc_sort_key)
        df = df.sort_values(
            by=["_sort_key", "exp_name", "run_id"],
            ascending=[False, True, True],
        ).drop(columns=["_sort_key"])

    index_path = results_root / "index.csv"
    df.to_csv(index_path, index=False)

    print(f"index_path={index_path.as_posix()}")
    print(f"n_rows={len(df)}")


def build_modern_row(
    results_root: Path,
    exp_name: str,
    run_dir: Path,
    meta_path: Path,
) -> dict[str, object]:
    """Build one index row from a modern run directory with meta.json."""
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    run_id = str(payload.get("run_id", run_dir.name))
    created_utc = str(payload.get("created_utc", ""))

    paths_value = payload.get("paths")
    path_map = paths_value if isinstance(paths_value, dict) else {}
    metrics_csv = normalize_index_path(results_root, path_map.get("metrics_csv", ""))
    if metrics_csv == "":
        fallback_csv = run_dir / "metrics.csv"
        if fallback_csv.exists():
            metrics_csv = rel_posix(results_root, fallback_csv)

    artifacts = [
        rel_posix(results_root, p)
        for p in sorted(run_dir.glob("*"))
        if p.is_file()
    ]

    row: dict[str, object] = {
        "exp_name": exp_name,
        "run_id": run_id,
        "layout": "modern",
        "has_meta": True,
        "created_utc": created_utc,
        "meta_json": rel_posix(results_root, meta_path),
        "metrics_csv": metrics_csv,
        "artifacts": ";".join(artifacts),
        "n_artifacts": len(artifacts),
    }

    summary = payload.get("summary_metrics", {})
    if isinstance(summary, dict):
        for key, value in summary.items():
            if isinstance(value, (int, float, str, bool)):
                row[f"metric_{key}"] = value

    return row


def normalize_index_path(results_root: Path, value: object) -> str:
    """Normalize an indexed path to the same base as other artifact fields."""
    if not isinstance(value, str) or value.strip() == "":
        return ""

    raw = value.strip()
    p = Path(raw)
    if p.is_absolute():
        return rel_posix(results_root, p)

    parts = p.parts
    if len(parts) > 0 and parts[0] == "results":
        p = Path(*parts[1:])
    return p.as_posix()


def build_legacy_rows(
    results_root: Path,
    exp_name: str,
    exp_dir: Path,
) -> list[dict[str, object]]:
    """Build index rows from legacy flat artifacts under one experiment directory."""
    files = [p for p in exp_dir.iterdir() if p.is_file()]
    grouped: dict[str, list[Path]] = {}
    for file_path in files:
        stem = file_path.stem
        grouped.setdefault(stem, []).append(file_path)

    rows: list[dict[str, object]] = []
    for run_id, artifacts in grouped.items():
        # Ignore modern index files accidentally grouped at root-level experiment dirs.
        if run_id == "index":
            continue

        rel_artifacts = [rel_posix(results_root, p) for p in sorted(artifacts)]
        metrics_csv = ""
        for p in artifacts:
            if p.suffix.lower() == ".csv":
                metrics_csv = rel_posix(results_root, p)
                break

        mtime = max(p.stat().st_mtime for p in artifacts)
        created_utc = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

        row: dict[str, object] = {
            "exp_name": exp_name,
            "run_id": run_id,
            "layout": "legacy_flat",
            "has_meta": False,
            "created_utc": created_utc,
            "meta_json": "",
            "metrics_csv": metrics_csv,
            "artifacts": ";".join(rel_artifacts),
            "n_artifacts": len(rel_artifacts),
        }
        rows.append(row)

    return rows


def rel_posix(root: Path, path: Path) -> str:
    """Return path relative to root as POSIX string when possible."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.resolve().as_posix()


def created_utc_sort_key(value: object) -> float:
    """Parse ISO datetime string to timestamp for sorting."""
    if not isinstance(value, str) or value == "":
        return float("-inf")
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return float("-inf")


if __name__ == "__main__":
    main()
