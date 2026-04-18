"""Data loading helpers for the Streamlit viewer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def list_run_manifests(runs_dir: str | Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for manifest_path in sorted(Path(runs_dir).glob("*/manifest.json")):
        manifest = load_manifest(manifest_path)
        if manifest:
            manifest["_manifest_path"] = str(manifest_path.resolve())
            manifests.append(manifest)
    manifests.sort(key=lambda item: item.get("started_at", ""), reverse=True)
    return manifests


def parse_jsonl_lines(lines: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError:
            continue
    return rows


def load_trace_rows(trace_path: str | Path) -> list[dict[str, Any]]:
    path = Path(trace_path)
    if not path.exists():
        return []
    return parse_jsonl_lines(path.read_text(encoding="utf-8").splitlines())


def load_trace_tail_state(
    *,
    manifest: dict[str, Any] | None,
    trace_offsets: dict[str, int],
) -> dict[str, Any]:
    if not manifest or not manifest.get("trace_path"):
        return {"new_rows": [], "trace_offsets": trace_offsets}
    trace_path = str(manifest["trace_path"])
    current_offset = int(trace_offsets.get(trace_path, 0))
    new_rows, next_offset = tail_jsonl(trace_path, current_offset)
    updated_offsets = dict(trace_offsets)
    updated_offsets[trace_path] = next_offset
    return {"new_rows": new_rows, "trace_offsets": updated_offsets}


def manifest_status_message(manifest: dict[str, Any]) -> str:
    status = str(manifest.get("status", "unknown"))
    error = str(manifest.get("last_error_message", "")).strip()
    if status == "failed":
        return error or "Run failed."
    if status == "starting":
        return f"Run {manifest.get('run_id', '')} is starting. Waiting for trace output."
    if status == "running":
        return f"Run {manifest.get('run_id', '')} is running. Waiting for first step or additional trace rows."
    if status == "completed":
        return f"Run {manifest.get('run_id', '')} completed."
    return f"Run status: {status}"


def tail_jsonl(trace_path: str | Path, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
    path = Path(trace_path)
    if not path.exists():
        return [], offset
    with path.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read()
        if not chunk:
            return [], offset
        next_offset = handle.tell()
    if not chunk.endswith(b"\n"):
        last_newline = chunk.rfind(b"\n")
        if last_newline == -1:
            return [], offset
        next_offset = offset + last_newline + 1
        chunk = chunk[: last_newline + 1]
    lines = chunk.decode("utf-8", errors="ignore").splitlines()
    return parse_jsonl_lines(lines), next_offset


def available_conditions(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("condition", "")) for row in rows if row.get("condition")})


def available_episodes(rows: list[dict[str, Any]], condition: str | None = None) -> list[int]:
    filtered = filter_rows(rows, condition=condition)
    return sorted({int(row["episode"]) for row in filtered if "episode" in row})


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    condition: str | None = None,
    episode: int | None = None,
) -> list[dict[str, Any]]:
    filtered = rows
    if condition:
        filtered = [row for row in filtered if row.get("condition") == condition]
    if episode is not None:
        filtered = [row for row in filtered if int(row.get("episode", -1)) == int(episode)]
    return filtered


def latest_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rows[-1] if rows else None


def final_episode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    finals = [row for row in rows if row.get("done")]
    finals.sort(key=lambda row: (int(row.get("episode", -1)), int(row.get("step", -1))))
    return finals


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finals = final_episode_rows(rows)
    if not finals:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "average_reward": 0.0,
            "target_agreement_rate": 0.0,
            "outcome_breakdown": {},
        }
    success_count = sum(1 for row in finals if row.get("outcome") == "high_value")
    reward_total = sum(float(row.get("cumulative_team_reward", row.get("team_reward", 0.0))) for row in finals)
    agreement_count = sum(
        1
        for row in finals
        if row.get("target_a") in {"LEFT", "RIGHT"} and row.get("target_a") == row.get("target_b")
    )
    breakdown: dict[str, int] = {}
    for row in finals:
        outcome = str(row.get("outcome", "unknown"))
        breakdown[outcome] = breakdown.get(outcome, 0) + 1
    total = len(finals)
    return {
        "episodes": total,
        "success_rate": success_count / total,
        "average_reward": reward_total / total,
        "target_agreement_rate": agreement_count / total,
        "outcome_breakdown": breakdown,
    }
