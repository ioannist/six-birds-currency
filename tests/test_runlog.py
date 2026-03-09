import json
from pathlib import Path

from currencymorphism.runlog import new_run_id, save_metadata


def test_new_run_id_returns_nonempty_string() -> None:
    run_id = new_run_id()
    assert isinstance(run_id, str)
    assert len(run_id) >= 10
    assert "_" in run_id


def test_save_metadata_writes_meta_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_01"
    meta_path = save_metadata(
        run_id="test_run",
        params={"alpha": 1},
        metrics={"score": 0.5},
        paths={"run_dir": run_dir, "metrics_csv": run_dir / "metrics.csv"},
    )

    assert meta_path == run_dir / "meta.json"
    assert meta_path.exists()

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "test_run"
    assert "params" in payload
    assert "summary_metrics" in payload
    assert "paths" in payload
