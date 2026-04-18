import json
from pathlib import Path

from viewer.data import available_conditions, available_episodes, filter_rows, list_run_manifests, load_trace_rows, tail_jsonl


def test_trace_jsonl_loading_and_filtering(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    rows = [
        {"condition": "comm", "episode": 0, "step": 0, "done": False},
        {"condition": "comm", "episode": 0, "step": 1, "done": True},
        {"condition": "silent", "episode": 1, "step": 0, "done": True},
    ]
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    loaded = load_trace_rows(trace_path)
    assert len(loaded) == 3
    assert available_conditions(loaded) == ["comm", "silent"]
    assert available_episodes(loaded, "comm") == [0]
    assert len(filter_rows(loaded, condition="comm", episode=0)) == 2


def test_partial_tail_is_tolerant(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_bytes(b'{"step": 0}\n{"step": 1')
    rows, offset = tail_jsonl(trace_path, 0)
    assert rows == [{"step": 0}]
    assert offset > 0


def test_run_manifest_listing(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_test"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"run_id": "run_test", "started_at": "2026-04-18T00:00:00+00:00"}), encoding="utf-8")
    manifests = list_run_manifests(runs_dir)
    assert len(manifests) == 1
    assert manifests[0]["run_id"] == "run_test"
