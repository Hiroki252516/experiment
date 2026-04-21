"""Data loading and protocol-metric helpers for the Streamlit viewer."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

AGENT_NAMES = ("agent_a", "agent_b")
GLYPH_SIDE = 7
ZERO_GLYPH_ROWS = ["0" * GLYPH_SIDE for _ in range(GLYPH_SIDE)]


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


def coerce_glyph_rows(glyph_rows: Any) -> list[str]:
    if not isinstance(glyph_rows, list):
        return list(ZERO_GLYPH_ROWS)
    rows: list[str] = []
    for row in glyph_rows[:GLYPH_SIDE]:
        row_text = str(row)
        if len(row_text) < GLYPH_SIDE:
            row_text = row_text.ljust(GLYPH_SIDE, "0")
        rows.append("".join("1" if bit == "1" else "0" for bit in row_text[:GLYPH_SIDE]))
    while len(rows) < GLYPH_SIDE:
        rows.append("0" * GLYPH_SIDE)
    return rows


def glyph_hash(glyph_rows: Any) -> str:
    return glyph_key(coerce_glyph_rows(glyph_rows))


def _glyph_exchange_label(changed_a: bool, changed_b: bool) -> str:
    left = "A->B updated" if changed_a else "A->B unchanged"
    right = "B->A updated" if changed_b else "B->A unchanged"
    return f"{left}, {right}"


def glyph_is_zero(glyph_rows: Any) -> bool:
    rows = coerce_glyph_rows(glyph_rows)
    return all(set(row) <= {"0"} for row in rows)


def glyph_delta_pixels(current_rows: Any, previous_rows: Any) -> int:
    current = coerce_glyph_rows(current_rows)
    previous = coerce_glyph_rows(previous_rows)
    if not previous_rows:
        return 0
    return sum(
        1
        for current_row, previous_row in zip(current, previous, strict=True)
        for current_bit, previous_bit in zip(current_row, previous_row, strict=True)
        if current_bit != previous_bit
    )


def augment_trace_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous_sent_by_episode: dict[tuple[str, int], dict[str, list[str]]] = {}
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        context = (str(enriched.get("condition", "")), int(enriched.get("episode", -1)))
        sent_a = coerce_glyph_rows(enriched.get("glyph_a_sent"))
        sent_b = coerce_glyph_rows(enriched.get("glyph_b_sent"))
        received_a = coerce_glyph_rows(enriched.get("glyph_a_received"))
        received_b = coerce_glyph_rows(enriched.get("glyph_b_received"))

        enriched["glyph_a_sent"] = sent_a
        enriched["glyph_b_sent"] = sent_b
        enriched["glyph_a_received"] = received_a
        enriched["glyph_b_received"] = received_b
        enriched.setdefault("glyph_a_hash", glyph_hash(sent_a))
        enriched.setdefault("glyph_b_hash", glyph_hash(sent_b))
        enriched.setdefault("glyph_a_received_hash", glyph_hash(received_a))
        enriched.setdefault("glyph_b_received_hash", glyph_hash(received_b))

        previous = previous_sent_by_episode.get(context)
        default_changed = False if previous is None else sent_a != previous["agent_a"]
        enriched.setdefault("glyph_a_changed", default_changed)
        default_changed = False if previous is None else sent_b != previous["agent_b"]
        enriched.setdefault("glyph_b_changed", default_changed)
        enriched.setdefault("glyph_a_zero", glyph_is_zero(sent_a))
        enriched.setdefault("glyph_b_zero", glyph_is_zero(sent_b))
        enriched.setdefault("glyph_a_delta_pixels", glyph_delta_pixels(sent_a, previous["agent_a"] if previous else None))
        enriched.setdefault("glyph_b_delta_pixels", glyph_delta_pixels(sent_b, previous["agent_b"] if previous else None))
        enriched.setdefault(
            "glyph_a_same_streak",
            previous["agent_a_same_streak"] + 1 if previous is not None and sent_a == previous["agent_a"] else 1,
        )
        enriched.setdefault(
            "glyph_b_same_streak",
            previous["agent_b_same_streak"] + 1 if previous is not None and sent_b == previous["agent_b"] else 1,
        )
        enriched.setdefault(
            "glyph_event",
            bool(enriched.get("glyph_a_changed", False) or enriched.get("glyph_b_changed", False)),
        )
        enriched.setdefault(
            "glyph_exchange_label",
            _glyph_exchange_label(
                bool(enriched.get("glyph_a_changed", False)),
                bool(enriched.get("glyph_b_changed", False)),
            ),
        )

        previous_sent_by_episode[context] = {
            "agent_a": sent_a,
            "agent_b": sent_b,
            "agent_a_same_streak": int(enriched.get("glyph_a_same_streak", 1)),
            "agent_b_same_streak": int(enriched.get("glyph_b_same_streak", 1)),
        }
        enriched_rows.append(enriched)
    return enriched_rows


def load_trace_rows(trace_path: str | Path) -> list[dict[str, Any]]:
    path = Path(trace_path)
    if not path.exists():
        return []
    return augment_trace_rows(parse_jsonl_lines(path.read_text(encoding="utf-8").splitlines()))


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
    return augment_trace_rows(parse_jsonl_lines(lines)), next_offset


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


def available_conditions(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("condition", "")) for row in rows if row.get("condition")})


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


def available_episodes(rows: list[dict[str, Any]], condition: str | None = None) -> list[int]:
    filtered = filter_rows(rows, condition=condition)
    return sorted({int(row["episode"]) for row in filtered if "episode" in row})


def latest_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rows[-1] if rows else None


def final_episode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    finals = [row for row in rows if row.get("done")]
    finals.sort(key=lambda row: (str(row.get("condition", "")), int(row.get("episode", -1)), int(row.get("step", -1))))
    return finals


def glyph_key(glyph_rows: list[str] | Any) -> str:
    rows = coerce_glyph_rows(glyph_rows)
    return "/".join(str(row) for row in rows)


def glyph_event_steps(rows: list[dict[str, Any]]) -> list[int]:
    return [int(row.get("step", 0)) for row in rows if bool(row.get("glyph_event", False))]


def glyph_history_rows(rows: list[dict[str, Any]], current_step: int, limit: int = 8) -> list[dict[str, Any]]:
    if not rows:
        return []
    target_index = len(rows) - 1
    for index, row in enumerate(rows):
        if int(row.get("step", 0)) == int(current_step):
            target_index = index
            break
    start_index = max(0, target_index - max(limit - 1, 0))
    return rows[start_index : target_index + 1]


def previous_episode_row(rows: list[dict[str, Any]], current_step: int) -> dict[str, Any] | None:
    previous: dict[str, Any] | None = None
    for row in rows:
        step = int(row.get("step", 0))
        if step >= int(current_step):
            break
        previous = row
    return previous


def adjacent_glyph_event_step(rows: list[dict[str, Any]], current_step: int, direction: int) -> int | None:
    event_steps = glyph_event_steps(rows)
    if direction < 0:
        candidates = [step for step in event_steps if step < current_step]
        return candidates[-1] if candidates else None
    candidates = [step for step in event_steps if step > current_step]
    return candidates[0] if candidates else None


def representative_protocol_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    episodes: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        condition = str(row.get("condition", ""))
        episode = int(row.get("episode", -1))
        if condition and episode >= 0:
            episodes[(condition, episode)].append(row)

    entries: list[dict[str, Any]] = []
    for (condition, episode), episode_rows in episodes.items():
        episode_rows.sort(key=lambda row: int(row.get("step", -1)))
        final_row = episode_rows[-1]
        comm_rows = [row for row in episode_rows if row.get("phase") == "comm_only"]
        for agent_name in AGENT_NAMES:
            agent_suffix = "a" if agent_name == "agent_a" else "b"
            sent_key = f"glyph_{agent_suffix}_sent"
            target_key = f"target_{agent_suffix}_after"
            fallback_target_key = f"target_{agent_suffix}"
            sent_row = comm_rows[-1] if comm_rows else None
            sent_glyph = sent_row.get(sent_key, []) if sent_row else []
            target_value = final_row.get(target_key, final_row.get(fallback_target_key, "UNKNOWN"))
            known_value = int(final_row.get("value_left", 0)) if agent_name == "agent_a" else int(final_row.get("value_right", 0))
            entries.append(
                {
                    "condition": condition,
                    "episode": episode,
                    "agent_name": agent_name,
                    "my_known_value": known_value,
                    "final_target": target_value,
                    "outcome": final_row.get("outcome", ""),
                    "agreement": final_row.get("target_a") in {"LEFT", "RIGHT"}
                    and final_row.get("target_a") == final_row.get("target_b"),
                    "glyph_key": glyph_key(sent_glyph) if sent_glyph else "",
                    "glyph_rows": sent_glyph,
                }
            )
    return entries


def _weighted_consistency(entries: list[dict[str, Any]]) -> float:
    grouped: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    for entry in entries:
        if not entry["glyph_key"] or entry["final_target"] not in {"LEFT", "RIGHT"}:
            continue
        context = (entry["agent_name"], int(entry["my_known_value"]), str(entry["final_target"]))
        grouped[context].append(str(entry["glyph_key"]))
    total_weight = 0
    weighted_sum = 0.0
    for glyphs in grouped.values():
        counts = Counter(glyphs)
        total = len(glyphs)
        weighted_sum += max(counts.values()) / total * total
        total_weight += total
    return weighted_sum / total_weight if total_weight else 0.0


def _success_failure_divergence(entries: list[dict[str, Any]]) -> float:
    grouped: dict[tuple[str, int, str], dict[str, list[str]]] = defaultdict(lambda: {"success": [], "failure": []})
    for entry in entries:
        if not entry["glyph_key"] or entry["final_target"] not in {"LEFT", "RIGHT"}:
            continue
        context = (entry["agent_name"], int(entry["my_known_value"]), str(entry["final_target"]))
        bucket = "success" if entry["outcome"] == "high_value" else "failure"
        grouped[context][bucket].append(str(entry["glyph_key"]))

    comparable = 0
    divergent = 0
    for buckets in grouped.values():
        if not buckets["success"] or not buckets["failure"]:
            continue
        comparable += 1
        dominant_success = Counter(buckets["success"]).most_common(1)[0][0]
        dominant_failure = Counter(buckets["failure"]).most_common(1)[0][0]
        if dominant_success != dominant_failure:
            divergent += 1
    return divergent / comparable if comparable else 0.0


def _convention_persistence(entries: list[dict[str, Any]]) -> float:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        if entry["outcome"] == "high_value" and entry["glyph_key"] and entry["final_target"] in {"LEFT", "RIGHT"}:
            context = (entry["agent_name"], int(entry["my_known_value"]), str(entry["final_target"]))
            grouped[context].append(entry)

    total = 0
    matches = 0
    for episode_entries in grouped.values():
        episode_entries.sort(key=lambda item: int(item["episode"]))
        first_glyph = str(episode_entries[0]["glyph_key"])
        for entry in episode_entries:
            total += 1
            if str(entry["glyph_key"]) == first_glyph:
                matches += 1
    return matches / total if total else 0.0


def _post_comm_agreement_rate(rows: list[dict[str, Any]]) -> float:
    episodes: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("phase") == "comm_only":
            condition = str(row.get("condition", ""))
            episode = int(row.get("episode", -1))
            if condition and episode >= 0:
                episodes[(condition, episode)].append(row)
    if not episodes:
        return 0.0
    agreed = 0
    for episode_rows in episodes.values():
        episode_rows.sort(key=lambda row: int(row.get("step", -1)))
        last_row = episode_rows[-1]
        target_a = last_row.get("target_a_after", last_row.get("target_a", "UNKNOWN"))
        target_b = last_row.get("target_b_after", last_row.get("target_b", "UNKNOWN"))
        if target_a in {"LEFT", "RIGHT"} and target_a == target_b:
            agreed += 1
    return agreed / len(episodes)


def _mean_bool_from_rows(rows: list[dict[str, Any]], key_a: str, key_b: str, *, phase: str | None = None) -> float:
    values: list[bool] = []
    for row in rows:
        if phase and row.get("phase") != phase:
            continue
        values.append(bool(row.get(key_a, False)))
        values.append(bool(row.get(key_b, False)))
    return sum(1 for value in values if value) / len(values) if values else 0.0


def compute_protocol_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    entries = representative_protocol_entries(rows)
    return {
        "glyph_reuse_rate": _mean_bool_from_rows(
            rows,
            "glyph_a_reused_from_success",
            "glyph_b_reused_from_success",
            phase="comm_only",
        ),
        "same_context_glyph_consistency": _weighted_consistency(entries),
        "success_failure_glyph_divergence": _success_failure_divergence(entries),
        "convention_persistence": _convention_persistence(entries),
        "target_switch_after_glyph_rate": _mean_bool_from_rows(
            rows,
            "target_a_changed",
            "target_b_changed",
            phase="comm_only",
        ),
        "post_comm_agreement_rate": _post_comm_agreement_rate(rows),
    }


def build_convention_hints(rows: list[dict[str, Any]], limit: int = 5) -> dict[str, Any]:
    entries = representative_protocol_entries(rows)
    successful = [entry for entry in entries if entry["outcome"] == "high_value" and entry["glyph_key"]]
    successful.sort(key=lambda item: (str(item["condition"]), int(item["episode"])), reverse=True)
    recent_successes = successful[:limit]

    grouped: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    for entry in entries:
        if entry["glyph_key"] and entry["final_target"] in {"LEFT", "RIGHT"}:
            context = (entry["agent_name"], int(entry["my_known_value"]), str(entry["final_target"]))
            grouped[context].append(str(entry["glyph_key"]))

    frequent_contexts: list[dict[str, Any]] = []
    for context, glyphs in grouped.items():
        counts = Counter(glyphs)
        dominant_glyph, dominant_count = counts.most_common(1)[0]
        frequent_contexts.append(
            {
                "context": context,
                "dominant_glyph": dominant_glyph,
                "dominant_rows": dominant_glyph.split("/"),
                "dominant_share": dominant_count / len(glyphs),
                "samples": len(glyphs),
            }
        )
    frequent_contexts.sort(key=lambda item: (item["dominant_share"], item["samples"]), reverse=True)

    return {
        "recent_successes": recent_successes,
        "frequent_contexts": frequent_contexts[:limit],
        "protocol_metrics": compute_protocol_metrics(rows),
    }


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finals = final_episode_rows(rows)
    if not finals:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "average_reward": 0.0,
            "target_agreement_rate": 0.0,
            "outcome_breakdown": {},
            **compute_protocol_metrics(rows),
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
        **compute_protocol_metrics(rows),
    }
