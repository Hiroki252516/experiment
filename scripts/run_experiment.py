"""Run ScoreG-style coordination experiments across communication conditions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.ollama_agent import (  # noqa: E402
    DEFAULT_AGENT_TIMEOUT_S,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    AgentExecutionError,
    AgentNotebookEntry,
    OllamaAgent,
    apply_decision_guard,
    check_ollama_setup,
)
from envs.scoreg_env import (  # noqa: E402
    AGENT_NAMES,
    GLYPH_SIDE,
    ITEM_LABELS,
    ScoreGParallelEnv,
    glyph_matrix_to_rows,
)

CONDITIONS = ("comm", "silent", "random")
DEFAULT_OUTPUT_CSV = "logs/results.csv"
DEFAULT_OUTPUT_JSONL = "logs/episodes.jsonl"
DEFAULT_TRACE_DIR = "logs/traces"
DEFAULT_RUNS_DIR = "logs/runs"
ZERO_GLYPH_ROWS = ["0" * GLYPH_SIDE for _ in range(GLYPH_SIDE)]


@dataclass
class RunPaths:
    output_csv: Path
    output_jsonl: Path
    trace_path: Path
    runs_dir: Path
    run_dir: Path
    manifest_path: Path
    launcher_log_path: Path


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
    target_agreement: bool
    final_raw_a: str
    final_raw_b: str


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def default_run_id() -> str:
    return f"run_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ScoreG experiments with local Ollama agents.")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per condition.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
        help="Experiment conditions to run.",
    )
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name.")
    parser.add_argument("--base-url", default=DEFAULT_OLLAMA_BASE_URL, help="Ollama base URL.")
    parser.add_argument(
        "--agent-timeout",
        type=float,
        default=DEFAULT_AGENT_TIMEOUT_S,
        help="Per-request Ollama read timeout in seconds.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--grid-size", type=int, default=5, help="Grid size.")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum act-phase steps.")
    parser.add_argument(
        "--comm-only-turns",
        type=int,
        default=2,
        help="Communication-only turns before movement begins.",
    )
    parser.add_argument(
        "--random-start-min-distance",
        type=int,
        default=2,
        help="Minimum Manhattan distance between agent starts.",
    )
    parser.add_argument(
        "--fixed-layout",
        action="store_true",
        help="Disable layout randomization and use the default symmetric layout.",
    )
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="Path to results CSV.")
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL, help="Path to episode JSONL.")
    parser.add_argument("--run-id", default=default_run_id(), help="Explicit run identifier.")
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR, help="Directory for step trace JSONL.")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR, help="Directory for run manifests.")
    return parser.parse_args(argv)


def resolve_run_paths(args: argparse.Namespace) -> RunPaths:
    output_csv = Path(args.output_csv)
    output_jsonl = Path(args.output_jsonl)
    trace_dir = Path(args.trace_dir)
    runs_dir = Path(args.runs_dir)
    run_dir = runs_dir / args.run_id
    return RunPaths(
        output_csv=output_csv,
        output_jsonl=output_jsonl,
        trace_path=trace_dir / f"{args.run_id}.jsonl",
        runs_dir=runs_dir,
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        launcher_log_path=run_dir / "launcher.log",
    )


def ensure_parent_dirs(paths: RunPaths) -> None:
    paths.output_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    paths.trace_path.parent.mkdir(parents=True, exist_ok=True)
    paths.run_dir.mkdir(parents=True, exist_ok=True)


def build_manifest(args: argparse.Namespace, paths: RunPaths) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "model": args.model,
        "agent_timeout_s": float(args.agent_timeout),
        "conditions": list(args.conditions),
        "episodes_per_condition": int(args.episodes),
        "base_seed": int(args.seed),
        "grid_size": int(args.grid_size),
        "max_steps": int(args.max_steps),
        "comm_only_turns": int(args.comm_only_turns),
        "randomize_layout": not bool(args.fixed_layout),
        "random_start_min_distance": int(args.random_start_min_distance),
        "status": "starting",
        "started_at": utc_now_iso(),
        "completed_at": "",
        "trace_path": str(paths.trace_path),
        "results_csv_path": str(paths.output_csv),
        "episodes_jsonl_path": str(paths.output_jsonl),
        "launcher_log_path": str(paths.launcher_log_path),
        "pid": os.getpid(),
        "current_condition": "",
        "current_episode": -1,
        "current_phase": "",
        "last_step": -1,
        "last_error_message": "",
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        handle.flush()


def phase_name_from_observation(observation: dict[str, np.ndarray]) -> str:
    phase_value = np.asarray(observation["phase"], dtype=np.int64).reshape(-1)
    return "comm_only" if int(phase_value[0]) == 0 else "act"


def observation_scalar(observation: dict[str, np.ndarray], key: str) -> int:
    values = np.asarray(observation[key], dtype=np.int64).reshape(-1)
    return int(values[0])


def known_value_for_agent(agent_name: str, observation: dict[str, np.ndarray]) -> int:
    private_value = np.asarray(observation["private_value"], dtype=np.int64).reshape(2)
    if agent_name == "agent_a":
        return int(private_value[0])
    return int(private_value[1])


def parse_target_rows(glyph_rows: list[str]) -> list[str]:
    return [str(row) for row in glyph_rows]


def glyph_rows_hash(glyph_rows: list[str]) -> str:
    return "/".join(str(row) for row in glyph_rows)


def glyph_exchange_label(*, changed_a: bool, changed_b: bool) -> str:
    a_label = "A->B updated" if changed_a else "A->B unchanged"
    b_label = "B->A updated" if changed_b else "B->A unchanged"
    return f"{a_label}, {b_label}"


def choose_condition_glyph(
    condition: str,
    generated_rows: list[str],
    rng: random.Random,
) -> list[str]:
    if condition == "silent":
        return list(ZERO_GLYPH_ROWS)
    if condition == "random":
        return [
            "".join(str(rng.randint(0, 1)) for _ in range(GLYPH_SIDE))
            for _ in range(GLYPH_SIDE)
        ]
    return list(generated_rows)


def build_trace_row(
    *,
    run_id: str,
    condition: str,
    episode_id: int,
    step: int,
    env: ScoreGParallelEnv,
    sent_rows: dict[str, list[str]],
    received_rows: dict[str, list[str]],
    moves: dict[str, str],
    targets: dict[str, str],
    guard_applied: dict[str, bool],
    guard_reasons: dict[str, str],
    raw_outputs: dict[str, str],
    rewards: dict[str, float],
    cumulative_team_reward: float,
    done: bool,
    outcome: str,
    error_message: str,
    target_before: dict[str, str] | None = None,
    target_after: dict[str, str] | None = None,
    target_changed: dict[str, bool] | None = None,
    glyph_reused_from_success: dict[str, bool] | None = None,
    phase: str | None = None,
    phase_turn_index: int | None = None,
    act_step: int | None = None,
    comm_only_turns: int | None = None,
    previous_sent_rows: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    target_before = target_before or dict(targets)
    target_after = target_after or dict(targets)
    target_changed = target_changed or {agent: False for agent in AGENT_NAMES}
    glyph_reused_from_success = glyph_reused_from_success or {agent: False for agent in AGENT_NAMES}
    previous_sent_rows = previous_sent_rows or {agent: [] for agent in AGENT_NAMES}
    current_phase = phase or getattr(env, "phase", "act")
    current_phase_turn = phase_turn_index if phase_turn_index is not None else int(getattr(env, "phase_turn_index", 0))
    current_act_step = act_step if act_step is not None else int(getattr(env, "act_step_count", step))
    current_comm_only_turns = (
        int(comm_only_turns)
        if comm_only_turns is not None
        else int(getattr(env, "comm_only_turns", 0))
    )
    glyph_a_hash = glyph_rows_hash(sent_rows["agent_a"])
    glyph_b_hash = glyph_rows_hash(sent_rows["agent_b"])
    glyph_a_received_hash = glyph_rows_hash(received_rows["agent_a"])
    glyph_b_received_hash = glyph_rows_hash(received_rows["agent_b"])
    glyph_a_changed = bool(previous_sent_rows.get("agent_a")) and sent_rows["agent_a"] != previous_sent_rows["agent_a"]
    glyph_b_changed = bool(previous_sent_rows.get("agent_b")) and sent_rows["agent_b"] != previous_sent_rows["agent_b"]
    row = {
        "run_id": run_id,
        "timestamp": utc_now_iso(),
        "condition": condition,
        "episode": int(episode_id),
        "step": int(step),
        "phase": current_phase,
        "phase_turn_index": int(current_phase_turn),
        "act_step": int(current_act_step),
        "comm_only_turns": int(current_comm_only_turns),
        "agent_a_pos": env.agent_positions["agent_a"].astype(int).tolist(),
        "agent_b_pos": env.agent_positions["agent_b"].astype(int).tolist(),
        "left_item_pos": env.item_positions["LEFT"].astype(int).tolist(),
        "right_item_pos": env.item_positions["RIGHT"].astype(int).tolist(),
        "value_left": int(env.left_value),
        "value_right": int(env.right_value),
        "best_item": str(env.best_item),
        "glyph_a_sent": list(sent_rows["agent_a"]),
        "glyph_b_sent": list(sent_rows["agent_b"]),
        "glyph_a_received": list(received_rows["agent_a"]),
        "glyph_b_received": list(received_rows["agent_b"]),
        "glyph_a_hash": glyph_a_hash,
        "glyph_b_hash": glyph_b_hash,
        "glyph_a_received_hash": glyph_a_received_hash,
        "glyph_b_received_hash": glyph_b_received_hash,
        "glyph_a_changed": bool(glyph_a_changed),
        "glyph_b_changed": bool(glyph_b_changed),
        "glyph_event": bool(glyph_a_changed or glyph_b_changed),
        "glyph_exchange_label": glyph_exchange_label(
            changed_a=bool(glyph_a_changed),
            changed_b=bool(glyph_b_changed),
        ),
        "move_a": moves["agent_a"],
        "move_b": moves["agent_b"],
        "target_a": target_after["agent_a"],
        "target_b": target_after["agent_b"],
        "target_a_before": target_before["agent_a"],
        "target_b_before": target_before["agent_b"],
        "target_a_after": target_after["agent_a"],
        "target_b_after": target_after["agent_b"],
        "target_a_changed": bool(target_changed["agent_a"]),
        "target_b_changed": bool(target_changed["agent_b"]),
        "glyph_a_reused_from_success": bool(glyph_reused_from_success["agent_a"]),
        "glyph_b_reused_from_success": bool(glyph_reused_from_success["agent_b"]),
        "guard_a_applied": bool(guard_applied["agent_a"]),
        "guard_b_applied": bool(guard_applied["agent_b"]),
        "guard_a_reason": guard_reasons["agent_a"],
        "guard_b_reason": guard_reasons["agent_b"],
        "raw_a": raw_outputs["agent_a"],
        "raw_b": raw_outputs["agent_b"],
        "reward_a": float(rewards["agent_a"]),
        "reward_b": float(rewards["agent_b"]),
        "team_reward": float((float(rewards["agent_a"]) + float(rewards["agent_b"])) / 2.0),
        "cumulative_team_reward": float(cumulative_team_reward),
        "done": bool(done),
        "outcome": outcome,
        "error_message": error_message,
    }
    return row


def record_episode_memory(
    *,
    agents: dict[str, OllamaAgent],
    condition: str,
    episode_id: int,
    env: ScoreGParallelEnv,
    final_targets: dict[str, str],
    agreement: bool,
    outcome: str,
    team_reward: float,
    last_comm_sent_rows: dict[str, list[str]],
    last_comm_received_rows: dict[str, list[str]],
) -> None:
    for agent_name, agent in agents.items():
        known_value = int(env.left_value) if agent_name == "agent_a" else int(env.right_value)
        final_target = final_targets[agent_name]
        if outcome == "high_value":
            glyph_helped_note = f"success after converging on {final_target}"
        elif agreement:
            glyph_helped_note = f"agreed on {final_target} but outcome={outcome}"
        else:
            glyph_helped_note = "targets diverged or failed to converge"
        agent.record_episode_entry(
            AgentNotebookEntry(
                episode_id=episode_id,
                condition=condition,
                my_known_value=known_value,
                comm_sent_glyph=list(last_comm_sent_rows[agent_name]),
                comm_received_glyph=list(last_comm_received_rows[agent_name]),
                final_target=final_target,
                agreement=agreement,
                outcome=outcome,
                team_reward=team_reward,
                glyph_helped_note=glyph_helped_note,
            )
        )


def final_episode_detail(
    *,
    record: EpisodeRecord,
    comm_sent_rows: dict[str, list[str]],
    comm_received_rows: dict[str, list[str]],
) -> dict[str, Any]:
    payload = asdict(record)
    payload.update(
        {
            "comm_sent_glyph_a": list(comm_sent_rows["agent_a"]),
            "comm_sent_glyph_b": list(comm_sent_rows["agent_b"]),
            "comm_received_glyph_a": list(comm_received_rows["agent_a"]),
            "comm_received_glyph_b": list(comm_received_rows["agent_b"]),
        }
    )
    return payload


def initialize_condition_agents(args: argparse.Namespace, condition: str) -> dict[str, OllamaAgent]:
    prompt_map = {
        "agent_a": PROJECT_ROOT / "prompts" / "agent_a_system.txt",
        "agent_b": PROJECT_ROOT / "prompts" / "agent_b_system.txt",
    }
    return {
        agent_name: OllamaAgent(
            agent_name=agent_name,
            model=args.model,
            system_prompt_path=prompt_map[agent_name],
            base_url=args.base_url,
            timeout_s=float(args.agent_timeout),
        )
        for agent_name in AGENT_NAMES
    }


def run_episode(
    *,
    args: argparse.Namespace,
    condition: str,
    episode_id: int,
    episode_seed: int,
    agents: dict[str, OllamaAgent],
    trace_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> tuple[EpisodeRecord, dict[str, Any]]:
    env = ScoreGParallelEnv(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        comm_only_turns=args.comm_only_turns,
        randomize_layout=not args.fixed_layout,
        random_start_min_distance=args.random_start_min_distance,
    )
    observations, _ = env.reset(seed=episode_seed)
    manifest["current_episode"] = episode_id
    manifest["current_phase"] = env.phase
    write_manifest(manifest_path, manifest)

    condition_rng = random.Random(episode_seed * 9973 + episode_id)
    step_index = 0
    cumulative_team_reward = 0.0
    last_targets = {agent: "UNKNOWN" for agent in AGENT_NAMES}
    last_raw_outputs = {agent: "" for agent in AGENT_NAMES}
    last_comm_sent_rows = {agent: list(ZERO_GLYPH_ROWS) for agent in AGENT_NAMES}
    last_comm_received_rows = {agent: list(ZERO_GLYPH_ROWS) for agent in AGENT_NAMES}
    previous_step_sent_rows = {agent: [] for agent in AGENT_NAMES}

    while True:
        phase = phase_name_from_observation(observations["agent_a"])
        phase_turn_index = observation_scalar(observations["agent_a"], "phase_turn_index")
        act_step = observation_scalar(observations["agent_a"], "act_step_count")

        raw_outputs: dict[str, str] = {}
        sent_rows: dict[str, list[str]] = {}
        received_rows: dict[str, list[str]] = {}
        moves: dict[str, str] = {}
        guard_applied: dict[str, bool] = {}
        guard_reasons: dict[str, str] = {}
        target_before = dict(last_targets)
        target_after: dict[str, str] = {}
        target_changed: dict[str, bool] = {}
        glyph_reused_from_success: dict[str, bool] = {}

        try:
            for agent_name in AGENT_NAMES:
                observation = observations[agent_name]
                received = glyph_matrix_to_rows(observation["other_last_glyph"])
                received_rows[agent_name] = received
                if phase == "comm_only":
                    last_comm_received_rows[agent_name] = list(received)

                turn = agents[agent_name].act(
                    observation,
                    previous_target=last_targets[agent_name],
                )
                raw_outputs[agent_name] = turn.raw_response_text
                guarded = apply_decision_guard(
                    agent_name=agent_name,
                    observation=observation,
                    decision=turn.decision,
                    previous_target=last_targets[agent_name],
                    grid_size=args.grid_size,
                )
                rows = choose_condition_glyph(
                    condition,
                    parse_target_rows(turn.decision.glyph),
                    condition_rng,
                )
                sent_rows[agent_name] = rows
                if phase == "comm_only":
                    last_comm_sent_rows[agent_name] = list(rows)

                prior_success = agents[agent_name].prior_success_glyph(
                    known_value=known_value_for_agent(agent_name, observation),
                    target=guarded.target,
                )
                glyph_reused_from_success[agent_name] = bool(prior_success and rows == prior_success)
                moves[agent_name] = guarded.move
                guard_applied[agent_name] = guarded.applied
                guard_reasons[agent_name] = guarded.reason
                target_after[agent_name] = guarded.target
                target_changed[agent_name] = (
                    target_before[agent_name] in ITEM_LABELS
                    and target_before[agent_name] != guarded.target
                )
                last_targets[agent_name] = guarded.target
                last_raw_outputs[agent_name] = turn.raw_response_text
        except AgentExecutionError as exc:
            error_message = str(exc)
            rewards = {agent: 0.0 for agent in AGENT_NAMES}
            row = build_trace_row(
                run_id=args.run_id,
                condition=condition,
                episode_id=episode_id,
                step=step_index,
                env=env,
                sent_rows={agent: sent_rows.get(agent, list(ZERO_GLYPH_ROWS)) for agent in AGENT_NAMES},
                received_rows={agent: received_rows.get(agent, list(ZERO_GLYPH_ROWS)) for agent in AGENT_NAMES},
                moves={agent: moves.get(agent, "STAY") for agent in AGENT_NAMES},
                targets={agent: target_after.get(agent, target_before.get(agent, "UNKNOWN")) for agent in AGENT_NAMES},
                guard_applied={agent: guard_applied.get(agent, False) for agent in AGENT_NAMES},
                guard_reasons={agent: guard_reasons.get(agent, "") for agent in AGENT_NAMES},
                raw_outputs={agent: raw_outputs.get(agent, "") for agent in AGENT_NAMES},
                rewards=rewards,
                cumulative_team_reward=cumulative_team_reward,
                done=True,
                outcome="agent_error",
                error_message=error_message,
                target_before=target_before,
                target_after={agent: target_after.get(agent, target_before.get(agent, "UNKNOWN")) for agent in AGENT_NAMES},
                target_changed={agent: target_changed.get(agent, False) for agent in AGENT_NAMES},
                glyph_reused_from_success={agent: glyph_reused_from_success.get(agent, False) for agent in AGENT_NAMES},
                phase=phase,
                phase_turn_index=phase_turn_index,
                act_step=act_step,
                comm_only_turns=args.comm_only_turns,
                previous_sent_rows=previous_step_sent_rows,
            )
            append_jsonl(trace_path, row)
            manifest["last_step"] = step_index
            manifest["current_phase"] = phase
            write_manifest(manifest_path, manifest)
            record = EpisodeRecord(
                run_id=args.run_id,
                seed=episode_seed,
                condition=condition,
                episode_id=episode_id,
                value_left=int(env.left_value),
                value_right=int(env.right_value),
                best_item=str(env.best_item),
                outcome="agent_error",
                team_reward=float(cumulative_team_reward),
                target_a=target_after.get("agent_a", target_before.get("agent_a", "UNKNOWN")),
                target_b=target_after.get("agent_b", target_before.get("agent_b", "UNKNOWN")),
                target_agreement=False,
                final_raw_a=raw_outputs.get("agent_a", ""),
                final_raw_b=raw_outputs.get("agent_b", ""),
            )
            detail = final_episode_detail(
                record=record,
                comm_sent_rows=last_comm_sent_rows,
                comm_received_rows=last_comm_received_rows,
            )
            return record, detail

        env_actions = {
            agent_name: {"move": moves[agent_name], "glyph": sent_rows[agent_name]}
            for agent_name in AGENT_NAMES
        }
        next_observations, rewards, terminations, truncations, _ = env.step(env_actions)
        step_reward = (float(rewards["agent_a"]) + float(rewards["agent_b"])) / 2.0
        cumulative_team_reward += step_reward
        done = bool(any(terminations.values()) or any(truncations.values()))
        outcome = str(env.last_outcome or "")

        row = build_trace_row(
            run_id=args.run_id,
            condition=condition,
            episode_id=episode_id,
            step=step_index,
            env=env,
            sent_rows=sent_rows,
            received_rows=received_rows,
            moves=moves,
            targets=target_after,
            guard_applied=guard_applied,
            guard_reasons=guard_reasons,
            raw_outputs=raw_outputs,
            rewards=rewards,
            cumulative_team_reward=cumulative_team_reward,
            done=done,
            outcome=outcome,
            error_message="",
            target_before=target_before,
            target_after=target_after,
            target_changed=target_changed,
            glyph_reused_from_success=glyph_reused_from_success,
            phase=phase,
            phase_turn_index=phase_turn_index,
            act_step=act_step,
            comm_only_turns=args.comm_only_turns,
            previous_sent_rows=previous_step_sent_rows,
        )
        append_jsonl(trace_path, row)
        previous_step_sent_rows = {agent: list(sent_rows[agent]) for agent in AGENT_NAMES}

        manifest["last_step"] = step_index
        manifest["current_phase"] = env.phase
        write_manifest(manifest_path, manifest)

        observations = next_observations
        if phase == "comm_only":
            for agent_name in AGENT_NAMES:
                last_comm_received_rows[agent_name] = glyph_matrix_to_rows(observations[agent_name]["other_last_glyph"])

        if done:
            final_targets = dict(target_after)
            agreement = (
                final_targets["agent_a"] in ITEM_LABELS
                and final_targets["agent_a"] == final_targets["agent_b"]
            )
            record_episode_memory(
                agents=agents,
                condition=condition,
                episode_id=episode_id,
                env=env,
                final_targets=final_targets,
                agreement=agreement,
                outcome=outcome,
                team_reward=cumulative_team_reward,
                last_comm_sent_rows=last_comm_sent_rows,
                last_comm_received_rows=last_comm_received_rows,
            )
            record = EpisodeRecord(
                run_id=args.run_id,
                seed=episode_seed,
                condition=condition,
                episode_id=episode_id,
                value_left=int(env.left_value),
                value_right=int(env.right_value),
                best_item=str(env.best_item),
                outcome=outcome,
                team_reward=float(cumulative_team_reward),
                target_a=final_targets["agent_a"],
                target_b=final_targets["agent_b"],
                target_agreement=agreement,
                final_raw_a=last_raw_outputs["agent_a"],
                final_raw_b=last_raw_outputs["agent_b"],
            )
            detail = final_episode_detail(
                record=record,
                comm_sent_rows=last_comm_sent_rows,
                comm_received_rows=last_comm_received_rows,
            )
            return record, detail

        step_index += 1


def write_outputs(
    *,
    paths: RunPaths,
    records: list[EpisodeRecord],
    details: list[dict[str, Any]],
) -> None:
    if records:
        fieldnames = list(asdict(records[0]).keys())
        with paths.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))
    else:
        with paths.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(list(EpisodeRecord.__annotations__.keys()))

    with paths.output_jsonl.open("w", encoding="utf-8") as handle:
        for detail in details:
            handle.write(json.dumps(detail, ensure_ascii=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_run_paths(args)
    ensure_parent_dirs(paths)
    paths.trace_path.write_text("", encoding="utf-8")

    manifest = build_manifest(args, paths)
    write_manifest(paths.manifest_path, manifest)

    status = check_ollama_setup(model=args.model, base_url=args.base_url)
    if not status.ok:
        manifest["status"] = "failed"
        manifest["completed_at"] = utc_now_iso()
        manifest["last_error_message"] = status.guidance(args.model)
        write_manifest(paths.manifest_path, manifest)
        print(status.guidance(args.model), file=sys.stderr)
        return 1

    records: list[EpisodeRecord] = []
    details: list[dict[str, Any]] = []
    manifest["status"] = "running"
    write_manifest(paths.manifest_path, manifest)

    try:
        for condition_index, condition in enumerate(args.conditions):
            manifest["current_condition"] = condition
            manifest["current_episode"] = -1
            manifest["current_phase"] = "comm_only" if args.comm_only_turns > 0 else "act"
            write_manifest(paths.manifest_path, manifest)

            agents = initialize_condition_agents(args, condition)
            for episode_id in range(int(args.episodes)):
                episode_seed = int(args.seed) + condition_index * 100000 + episode_id
                record, detail = run_episode(
                    args=args,
                    condition=condition,
                    episode_id=episode_id,
                    episode_seed=episode_seed,
                    agents=agents,
                    trace_path=paths.trace_path,
                    manifest=manifest,
                    manifest_path=paths.manifest_path,
                )
                records.append(record)
                details.append(detail)
                print(
                    f"[{condition}] episode={episode_id} outcome={record.outcome} "
                    f"team_reward={record.team_reward:.3f} targets=({record.target_a},{record.target_b})"
                )
    except Exception as exc:  # pragma: no cover - defensive top-level failure path
        manifest["status"] = "failed"
        manifest["completed_at"] = utc_now_iso()
        manifest["last_error_message"] = str(exc)
        write_manifest(paths.manifest_path, manifest)
        raise

    write_outputs(paths=paths, records=records, details=details)
    manifest["status"] = "completed"
    manifest["completed_at"] = utc_now_iso()
    manifest["current_phase"] = ""
    manifest["last_error_message"] = ""
    write_manifest(paths.manifest_path, manifest)
    print(
        f"Saved {len(records)} episode summaries to {paths.output_csv} and step trace to {paths.trace_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
