"""Run ScoreG-style local LLM coordination experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.ollama_agent import (  # noqa: E402
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    AgentExecutionError,
    OllamaAgent,
    agent_known_value,
    apply_decision_guard,
    build_memory_summary,
    check_ollama_setup,
)
from envs.scoreg_env import GLYPH_SIDE, MOVE_TO_INDEX, ScoreGParallelEnv, glyph_matrix_to_rows  # noqa: E402

CONDITIONS = ("comm", "silent", "random")
AGENT_NAMES = ("agent_a", "agent_b")
ZERO_GLYPH_ROWS = ["0" * GLYPH_SIDE for _ in range(GLYPH_SIDE)]
PROMPT_VERSION = "frozen_llm_glyph_emergence_v1"


@dataclass
class EpisodeRecord:
    run_id: str
    seed: int
    condition: str
    episode_id: int
    value_left: int
    value_right: int
    best_item: str
    outcome: str
    team_reward: float
    target_a: str
    target_b: str
    error_message: str = ""


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def default_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{timestamp}_{uuid.uuid4().hex[:8]}"


def glyph_rows_from_matrix(glyph: np.ndarray) -> list[str]:
    return glyph_matrix_to_rows(glyph)


def zero_trace_row() -> list[str]:
    return list(ZERO_GLYPH_ROWS)


def observation_step(observations: dict[str, dict[str, np.ndarray]]) -> int:
    return int(observations["agent_a"]["step_count"][0])


def int_list(value: np.ndarray) -> list[int]:
    return [int(item) for item in value.tolist()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ScoreG local LLM coordination experiments.")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per condition.")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name.")
    parser.add_argument("--base-url", default=DEFAULT_OLLAMA_BASE_URL, help="Ollama base URL.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--comm-phase-steps", type=int, default=0, help="Number of opening glyph-only steps.")
    parser.add_argument(
        "--randomize-positions",
        action="store_true",
        help="Randomize agent and item positions each episode.",
    )
    parser.add_argument(
        "--hard-split-prob",
        type=float,
        default=0.0,
        help="Probability of sampling a large value gap between left and right items.",
    )
    parser.add_argument(
        "--memory-budget",
        type=int,
        default=6,
        help="Maximum number of private memory summaries retained per agent.",
    )
    parser.add_argument("--output-csv", default="logs/results.csv")
    parser.add_argument("--output-jsonl", default="logs/episodes.jsonl")
    parser.add_argument("--run-id", default="", help="Optional explicit run id for viewer integration.")
    parser.add_argument("--trace-dir", default="logs/traces", help="Directory for step-level trace JSONL.")
    parser.add_argument("--runs-dir", default="logs/runs", help="Directory for per-run manifests.")
    return parser.parse_args(argv)


def override_glyph(condition: str, glyph: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if condition == "comm":
        return glyph
    if condition == "silent":
        return np.zeros((GLYPH_SIDE, GLYPH_SIDE), dtype=np.int8)
    return rng.integers(0, 2, size=(GLYPH_SIDE, GLYPH_SIDE), dtype=np.int8)


def ensure_ollama_ready(model: str, base_url: str) -> None:
    status = check_ollama_setup(model=model, base_url=base_url)
    if not status.ok:
        raise SystemExit(status.guidance(model))


def resolve_run_paths(args: argparse.Namespace) -> dict[str, Path]:
    run_id = args.run_id or default_run_id()
    trace_dir = Path(args.trace_dir)
    runs_dir = Path(args.runs_dir)
    run_dir = runs_dir / run_id
    return {
        "run_id": Path(run_id),
        "trace_dir": trace_dir,
        "runs_dir": runs_dir,
        "run_dir": run_dir,
        "trace_path": trace_dir / f"{run_id}.jsonl",
        "manifest_path": run_dir / "manifest.json",
        "results_csv_path": Path(args.output_csv),
        "episodes_jsonl_path": Path(args.output_jsonl),
    }


def build_manifest(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "run_id": paths["run_id"].name,
        "model": args.model,
        "conditions": list(args.conditions),
        "episodes_per_condition": args.episodes,
        "base_seed": args.seed,
        "grid_size": args.grid_size,
        "max_steps": args.max_steps,
        "comm_phase_steps": args.comm_phase_steps,
        "randomize_positions": args.randomize_positions,
        "hard_split_prob": args.hard_split_prob,
        "memory_budget": args.memory_budget,
        "prompt_version": PROMPT_VERSION,
        "status": "starting",
        "started_at": utc_now_iso(),
        "completed_at": "",
        "trace_path": str(paths["trace_path"].resolve()),
        "results_csv_path": str(paths["results_csv_path"].resolve()),
        "episodes_jsonl_path": str(paths["episodes_jsonl_path"].resolve()),
        "launcher_log_path": str((paths["run_dir"] / "launcher.log").resolve()),
        "pid": -1,
        "current_condition": "",
        "current_episode": -1,
        "last_step": -1,
        "last_error_message": "",
    }


def write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def trace_target(initial_target: str, final_target: str) -> tuple[str, str, bool]:
    initial = initial_target if initial_target in {"LEFT", "RIGHT"} else "UNKNOWN"
    final = final_target if final_target in {"LEFT", "RIGHT"} else "UNKNOWN"
    changed = initial in {"LEFT", "RIGHT"} and final in {"LEFT", "RIGHT"} and initial != final
    return initial, final, changed


def build_trace_row(
    *,
    run_id: str,
    condition: str,
    episode_id: int,
    step: int,
    phase: str,
    movement_enabled: bool,
    env: ScoreGParallelEnv,
    sent_rows: dict[str, list[str]],
    received_rows: dict[str, list[str]],
    moves: dict[str, str],
    targets: dict[str, str],
    initial_targets: dict[str, str],
    guard_applied: dict[str, bool],
    guard_reasons: dict[str, str],
    raw_outputs: dict[str, str],
    rewards: dict[str, float],
    cumulative_team_reward: float,
    done: bool,
    outcome: str,
    error_message: str,
) -> dict[str, Any]:
    initial_target_a, final_target_a, target_changed_a = trace_target(initial_targets["agent_a"], targets["agent_a"])
    initial_target_b, final_target_b, target_changed_b = trace_target(initial_targets["agent_b"], targets["agent_b"])
    return {
        "run_id": run_id,
        "timestamp": utc_now_iso(),
        "condition": condition,
        "episode": episode_id,
        "step": step,
        "phase": phase,
        "movement_enabled": movement_enabled,
        "agent_a_pos": int_list(env.agent_positions["agent_a"]),
        "agent_b_pos": int_list(env.agent_positions["agent_b"]),
        "left_item_pos": int_list(env.item_positions["LEFT"]),
        "right_item_pos": int_list(env.item_positions["RIGHT"]),
        "value_left": env.left_value,
        "value_right": env.right_value,
        "best_item": env.best_item,
        "glyph_a_sent": sent_rows["agent_a"],
        "glyph_b_sent": sent_rows["agent_b"],
        "glyph_a_received": received_rows["agent_a"],
        "glyph_b_received": received_rows["agent_b"],
        "move_a": moves["agent_a"],
        "move_b": moves["agent_b"],
        "target_a": targets["agent_a"],
        "target_b": targets["agent_b"],
        "initial_target_a": initial_target_a,
        "initial_target_b": initial_target_b,
        "final_target_a": final_target_a,
        "final_target_b": final_target_b,
        "target_changed_a": target_changed_a,
        "target_changed_b": target_changed_b,
        "guard_a_applied": guard_applied["agent_a"],
        "guard_b_applied": guard_applied["agent_b"],
        "guard_a_reason": guard_reasons["agent_a"],
        "guard_b_reason": guard_reasons["agent_b"],
        "raw_a": raw_outputs["agent_a"],
        "raw_b": raw_outputs["agent_b"],
        "reward_a": round(rewards["agent_a"], 4),
        "reward_b": round(rewards["agent_b"], 4),
        "team_reward": round(rewards["agent_a"], 4),
        "cumulative_team_reward": round(cumulative_team_reward, 4),
        "done": done,
        "outcome": outcome,
        "error_message": error_message,
    }


def write_outputs(
    records: list[EpisodeRecord],
    details: list[dict[str, object]],
    csv_path: Path,
    jsonl_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame([asdict(record) for record in records])
    dataframe.to_csv(csv_path, index=False)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_episode(
    *,
    env: ScoreGParallelEnv,
    agents: dict[str, OllamaAgent],
    condition: str,
    episode_id: int,
    seed: int,
    rng: np.random.Generator,
    run_id: str,
    trace_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> tuple[EpisodeRecord, dict[str, object]]:
    observations, _ = env.reset(seed=seed)
    cumulative_team_reward = 0.0
    last_targets = {agent: "UNKNOWN" for agent in AGENT_NAMES}
    initial_targets = {agent: "UNKNOWN" for agent in AGENT_NAMES}
    last_raw_outputs = {agent: "" for agent in AGENT_NAMES}
    last_sent_rows = {agent: zero_trace_row() for agent in AGENT_NAMES}
    last_received_rows = {agent: zero_trace_row() for agent in AGENT_NAMES}
    last_observations = {agent: observations[agent] for agent in AGENT_NAMES}
    error_message = ""
    outcome = "max_steps"

    manifest["current_condition"] = condition
    manifest["current_episode"] = episode_id
    manifest["last_step"] = 0
    write_manifest(manifest_path, manifest)

    while env.agents:
        step = observation_step(observations)
        movement_enabled = env.movement_enabled
        phase = env.current_phase
        sent_rows = {agent: zero_trace_row() for agent in AGENT_NAMES}
        received_rows = {
            "agent_a": glyph_rows_from_matrix(observations["agent_a"]["other_last_glyph"]),
            "agent_b": glyph_rows_from_matrix(observations["agent_b"]["other_last_glyph"]),
        }
        moves = {agent: "" for agent in AGENT_NAMES}
        targets = {agent: last_targets[agent] for agent in AGENT_NAMES}
        guard_applied = {agent: False for agent in AGENT_NAMES}
        guard_reasons = {agent: "" for agent in AGENT_NAMES}
        raw_outputs = {agent: "" for agent in AGENT_NAMES}
        rewards = {agent: 0.0 for agent in AGENT_NAMES}
        done = False

        try:
            actions: dict[str, dict[str, object]] = {}
            for agent_name in env.agents:
                turn = agents[agent_name].act(
                    observations[agent_name],
                    previous_target=last_targets[agent_name],
                )
                guarded = apply_decision_guard(
                    agent_name=agent_name,
                    observation=observations[agent_name],
                    decision=turn.decision,
                    previous_target=last_targets[agent_name],
                    grid_size=env.grid_size,
                )
                executed_move = guarded.move
                guard_reason = guarded.reason
                guard_used = guarded.applied
                if not movement_enabled:
                    executed_move = "STAY"
                    if guarded.move != "STAY":
                        guard_reason = ",".join(filter(None, [guard_reason, "comm_phase_forced_stay"]))
                        guard_used = True

                glyph_matrix = override_glyph(condition, turn.decision.glyph_matrix(), rng)
                glyph_rows = glyph_rows_from_matrix(glyph_matrix)
                actions[agent_name] = {
                    "move": MOVE_TO_INDEX[executed_move],
                    "glyph": glyph_matrix,
                }
                sent_rows[agent_name] = glyph_rows
                moves[agent_name] = executed_move
                targets[agent_name] = guarded.target
                guard_applied[agent_name] = guard_used
                guard_reasons[agent_name] = guard_reason
                raw_outputs[agent_name] = turn.raw_response_text
                last_targets[agent_name] = guarded.target
                last_raw_outputs[agent_name] = turn.raw_response_text
                last_sent_rows[agent_name] = glyph_rows
                last_received_rows[agent_name] = received_rows[agent_name]
                last_observations[agent_name] = observations[agent_name]
                if initial_targets[agent_name] == "UNKNOWN" and guarded.target in {"LEFT", "RIGHT"}:
                    initial_targets[agent_name] = guarded.target

            observations, rewards, terminations, truncations, _ = env.step(actions)
            cumulative_team_reward += rewards["agent_a"]
            done = not env.agents or all(terminations.values()) or all(truncations.values())
            outcome = env.last_outcome if done else "not_finished"
        except AgentExecutionError as exc:
            outcome = "agent_error"
            error_message = str(exc)
            done = True
            manifest["last_error_message"] = error_message

        trace_row = build_trace_row(
            run_id=run_id,
            condition=condition,
            episode_id=episode_id,
            step=step,
            phase=phase,
            movement_enabled=movement_enabled,
            env=env,
            sent_rows=sent_rows,
            received_rows=received_rows,
            moves=moves,
            targets=targets,
            initial_targets=initial_targets,
            guard_applied=guard_applied,
            guard_reasons=guard_reasons,
            raw_outputs=raw_outputs,
            rewards=rewards,
            cumulative_team_reward=cumulative_team_reward,
            done=done,
            outcome=outcome,
            error_message=error_message,
        )
        append_jsonl(trace_path, trace_row)
        manifest["last_step"] = step
        write_manifest(manifest_path, manifest)

        if done:
            break

    agreed = last_targets["agent_a"] == last_targets["agent_b"] and last_targets["agent_a"] in {"LEFT", "RIGHT"}
    agents["agent_a"].record_episode_summary(
        build_memory_summary(
            episode_id=episode_id,
            condition=condition,
            agent_name="agent_a",
            known_value=agent_known_value("agent_a", last_observations["agent_a"]),
            sent_glyph_rows=last_sent_rows["agent_a"],
            received_glyph_rows=last_received_rows["agent_a"],
            target=last_targets["agent_a"],
            agreement=agreed,
            team_reward=cumulative_team_reward,
            outcome=outcome,
        )
    )
    agents["agent_b"].record_episode_summary(
        build_memory_summary(
            episode_id=episode_id,
            condition=condition,
            agent_name="agent_b",
            known_value=agent_known_value("agent_b", last_observations["agent_b"]),
            sent_glyph_rows=last_sent_rows["agent_b"],
            received_glyph_rows=last_received_rows["agent_b"],
            target=last_targets["agent_b"],
            agreement=agreed,
            team_reward=cumulative_team_reward,
            outcome=outcome,
        )
    )

    record = EpisodeRecord(
        run_id=run_id,
        seed=seed,
        condition=condition,
        episode_id=episode_id,
        value_left=env.left_value,
        value_right=env.right_value,
        best_item=env.best_item,
        outcome=outcome,
        team_reward=round(cumulative_team_reward, 4),
        target_a=last_targets["agent_a"],
        target_b=last_targets["agent_b"],
        error_message=error_message,
    )
    details = {
        "run_id": run_id,
        "seed": seed,
        "condition": condition,
        "episode_id": episode_id,
        "outcome": outcome,
        "team_reward": round(cumulative_team_reward, 4),
        "value_left": env.left_value,
        "value_right": env.right_value,
        "best_item": env.best_item,
        "target_a": last_targets["agent_a"],
        "target_b": last_targets["agent_b"],
        "initial_target_a": initial_targets["agent_a"],
        "initial_target_b": initial_targets["agent_b"],
        "final_raw_output_a": last_raw_outputs["agent_a"],
        "final_raw_output_b": last_raw_outputs["agent_b"],
        "error_message": error_message,
    }
    return record, details


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_run_paths(args)
    run_id = paths["run_id"].name
    manifest = build_manifest(args, paths)
    manifest["pid"] = os.getpid()
    write_manifest(paths["manifest_path"], manifest)

    records: list[EpisodeRecord] = []
    details: list[dict[str, object]] = []

    try:
        manifest["status"] = "running"
        write_manifest(paths["manifest_path"], manifest)
        ensure_ollama_ready(model=args.model, base_url=args.base_url)

        agents = {
            "agent_a": OllamaAgent(
                agent_name="agent_a",
                model=args.model,
                base_url=args.base_url,
                memory_limit=args.memory_budget,
            ),
            "agent_b": OllamaAgent(
                agent_name="agent_b",
                model=args.model,
                base_url=args.base_url,
                memory_limit=args.memory_budget,
            ),
        }

        global_episode_id = 0
        for condition_index, condition in enumerate(args.conditions):
            for episode_offset in range(args.episodes):
                seed = args.seed + condition_index * 10_000 + episode_offset
                rng = np.random.default_rng(seed + 99)
                env = ScoreGParallelEnv(
                    grid_size=args.grid_size,
                    max_steps=args.max_steps,
                    comm_phase_steps=args.comm_phase_steps,
                    randomize_positions=args.randomize_positions,
                    hard_split_prob=args.hard_split_prob,
                )
                record, detail = run_episode(
                    env=env,
                    agents=agents,
                    condition=condition,
                    episode_id=global_episode_id,
                    seed=seed,
                    rng=rng,
                    run_id=run_id,
                    trace_path=paths["trace_path"],
                    manifest=manifest,
                    manifest_path=paths["manifest_path"],
                )
                records.append(record)
                details.append(detail)
                global_episode_id += 1

        write_outputs(
            records=records,
            details=details,
            csv_path=paths["results_csv_path"],
            jsonl_path=paths["episodes_jsonl_path"],
        )
        manifest["status"] = "completed"
        manifest["completed_at"] = utc_now_iso()
        write_manifest(paths["manifest_path"], manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["completed_at"] = utc_now_iso()
        manifest["last_error_message"] = str(exc)
        write_manifest(paths["manifest_path"], manifest)
        raise

    print(f"Wrote {len(records)} episode records to {paths['results_csv_path']}")
    print(f"Run id: {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
