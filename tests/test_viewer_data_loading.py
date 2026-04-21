import json
from pathlib import Path

from viewer.data import (
    available_conditions,
    available_episodes,
    build_convention_hints,
    compute_protocol_metrics,
    filter_rows,
    list_run_manifests,
    load_trace_rows,
    load_trace_tail_state,
    manifest_status_message,
    tail_jsonl,
)


def test_trace_jsonl_loading_and_filtering(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    rows = [
        {"condition": "comm", "episode": 0, "step": 0, "phase": "comm_only", "done": False},
        {"condition": "comm", "episode": 0, "step": 1, "phase": "act", "done": True},
        {"condition": "silent", "episode": 1, "step": 0, "phase": "act", "done": True},
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


def test_manifest_without_trace_still_loads(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_failed"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run_failed",
                "status": "failed",
                "trace_path": str(tmp_path / "missing.jsonl"),
                "last_error_message": "runner failed",
                "started_at": "2026-04-18T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    manifests = list_run_manifests(runs_dir)
    assert manifests[0]["run_id"] == "run_failed"
    assert manifest_status_message(manifests[0]) == "runner failed"
    tail_state = load_trace_tail_state(manifest=manifests[0], trace_offsets={})
    assert tail_state["new_rows"] == []


def test_protocol_metrics_and_hints_from_trace_rows() -> None:
    rows = [
        {
            "condition": "comm",
            "episode": 0,
            "step": 0,
            "phase": "comm_only",
            "glyph_a_sent": ["1111111"] * 7,
            "glyph_b_sent": ["0000000"] * 7,
            "glyph_a_reused_from_success": False,
            "glyph_b_reused_from_success": False,
            "target_a_after": "LEFT",
            "target_b_after": "LEFT",
            "target_a_changed": True,
            "target_b_changed": True,
            "value_left": 8,
            "value_right": 2,
        },
        {
            "condition": "comm",
            "episode": 0,
            "step": 1,
            "phase": "act",
            "done": True,
            "target_a": "LEFT",
            "target_b": "LEFT",
            "target_a_after": "LEFT",
            "target_b_after": "LEFT",
            "outcome": "high_value",
            "value_left": 8,
            "value_right": 2,
        },
    ]
    metrics = compute_protocol_metrics(rows)
    assert metrics["post_comm_agreement_rate"] == 1.0
    hints = build_convention_hints(rows)
    assert hints["recent_successes"]
