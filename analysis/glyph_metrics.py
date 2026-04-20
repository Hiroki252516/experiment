"""Trace-based glyph metrics for the frozen-LLM convention experiments."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def load_trace_rows_for_run_ids(run_ids: list[str], trace_dir: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_dir = Path(trace_dir)
    for run_id in sorted(set(run_ids)):
        trace_path = base_dir / f"{run_id}.jsonl"
        if not trace_path.exists():
            continue
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def _agent_rows(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in trace_rows:
        for agent_name, glyph_key, target_key, initial_key, changed_key, value_key in (
            ("agent_a", "glyph_a_sent", "target_a", "initial_target_a", "target_changed_a", "value_left"),
            ("agent_b", "glyph_b_sent", "target_b", "initial_target_b", "target_changed_b", "value_right"),
        ):
            glyph_rows = row.get(glyph_key, [])
            expanded.append(
                {
                    "run_id": row.get("run_id", ""),
                    "condition": row.get("condition", ""),
                    "episode": int(row.get("episode", -1)),
                    "step": int(row.get("step", -1)),
                    "phase": row.get("phase", ""),
                    "agent": agent_name,
                    "known_value": int(row.get(value_key, 0)),
                    "glyph": "".join(glyph_rows) if isinstance(glyph_rows, list) else "",
                    "target": row.get(target_key, "UNKNOWN"),
                    "initial_target": row.get(initial_key, "UNKNOWN"),
                    "target_changed": bool(row.get(changed_key, False)),
                    "done": bool(row.get("done", False)),
                    "outcome": row.get("outcome", ""),
                }
            )
    return expanded


def compute_glyph_reuse_consistency(trace_rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped_counts: dict[tuple[str, str, int, str], Counter[str]] = defaultdict(Counter)
    for row in _agent_rows(trace_rows):
        if row["target"] not in {"LEFT", "RIGHT"} or not row["glyph"]:
            continue
        key = (row["condition"], row["agent"], row["known_value"], row["target"])
        grouped_counts[key][row["glyph"]] += 1
    totals: dict[str, int] = defaultdict(int)
    matches: dict[str, int] = defaultdict(int)
    for (condition, _agent, _known_value, _target), counter in grouped_counts.items():
        count_total = sum(counter.values())
        totals[condition] += count_total
        matches[condition] += max(counter.values())
    return {
        condition: (matches[condition] / totals[condition] if totals[condition] else 0.0)
        for condition in sorted(set(totals) | set(matches))
    }


def compute_glyph_target_association(trace_rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in _agent_rows(trace_rows):
        if row["target"] not in {"LEFT", "RIGHT"} or not row["glyph"]:
            continue
        key = (row["condition"], row["agent"], row["glyph"])
        grouped_counts[key][row["target"]] += 1
    totals: dict[str, int] = defaultdict(int)
    matches: dict[str, int] = defaultdict(int)
    for (condition, _agent, _glyph), counter in grouped_counts.items():
        count_total = sum(counter.values())
        totals[condition] += count_total
        matches[condition] += max(counter.values())
    return {
        condition: (matches[condition] / totals[condition] if totals[condition] else 0.0)
        for condition in sorted(set(totals) | set(matches))
    }


def compute_target_flip_rate(trace_rows: list[dict[str, Any]]) -> dict[str, float]:
    rows = [row for row in _agent_rows(trace_rows) if row["done"]]
    if not rows:
        return {}
    dataframe = pd.DataFrame(rows)
    summary = dataframe.groupby("condition")["target_changed"].mean()
    return {str(condition): float(value) for condition, value in summary.items()}


def compute_convention_hints(
    trace_rows: list[dict[str, Any]],
    *,
    condition: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    rows = _agent_rows(trace_rows)
    if condition:
        rows = [row for row in rows if row["condition"] == condition]
    counter: Counter[str] = Counter(row["glyph"] for row in rows if row["glyph"])
    hints: list[dict[str, Any]] = []
    for glyph, count in counter.most_common(limit):
        glyph_rows = [glyph[index : index + 7] for index in range(0, min(len(glyph), 49), 7)]
        glyph_rows = glyph_rows[:7] if glyph_rows else []
        matching_rows = [row for row in rows if row["glyph"] == glyph]
        target_counter = Counter(row["target"] for row in matching_rows if row["target"] in {"LEFT", "RIGHT"})
        outcome_counter = Counter(row["outcome"] for row in matching_rows if row["done"])
        hints.append(
            {
                "glyph": glyph,
                "glyph_rows": glyph_rows,
                "count": count,
                "dominant_target": target_counter.most_common(1)[0][0] if target_counter else "UNKNOWN",
                "success_count": int(outcome_counter.get("high_value", 0)),
            }
        )
    return hints

