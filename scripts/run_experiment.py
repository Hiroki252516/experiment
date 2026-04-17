"""Run ScoreG-style local LLM coordination experiments."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

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
    check_ollama_setup,
)
from envs.scoreg_env import GLYPH_SIDE, MOVE_TO_INDEX, ScoreGParallelEnv  # noqa: E402

CONDITIONS = ("comm", "silent", "random")


@dataclass
class EpisodeRecord:
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ScoreG local LLM coordination experiments.")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per condition.")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name.")
    parser.add_argument("--base-url", default=DEFAULT_OLLAMA_BASE_URL, help="Ollama base URL.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--output-csv", default="logs/results.csv")
    parser.add_argument("--output-jsonl", default="logs/episodes.jsonl")
    return parser.parse_args(argv)


def override_glyph(condition: str, glyph: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if condition == "comm":
        return glyph
    if condition == "silent":
        return np.zeros((GLYPH_SIDE, GLYPH_SIDE), dtype=np.int8)
    return rng.integers(0, 2, size=(GLYPH_SIDE, GLYPH_SIDE), dtype=np.int8)


def build_agent_summary(agent_name: str, target: str, outcome: str, agreed: bool) -> str:
    return f"episode outcome={outcome}; self_target={target}; agreement={agreed}"


def run_episode(
    env: ScoreGParallelEnv,
    agents: dict[str, OllamaAgent],
    condition: str,
    episode_id: int,
    seed: int,
    rng: np.random.Generator,
) -> tuple[EpisodeRecord, dict[str, object]]:
    observations, _ = env.reset(seed=seed)
    team_reward = 0.0
    last_targets = {"agent_a": "UNKNOWN", "agent_b": "UNKNOWN"}
    last_raw_outputs = {"agent_a": "", "agent_b": ""}
    error_message = ""
    outcome = "max_steps"

    while env.agents:
        actions: dict[str, dict[str, object]] = {}
        try:
            for agent_name in env.agents:
                turn = agents[agent_name].act(observations[agent_name])
                glyph = override_glyph(condition, turn.decision.glyph_matrix(), rng)
                actions[agent_name] = {
                    "move": MOVE_TO_INDEX[turn.decision.move],
                    "glyph": glyph,
                }
                last_targets[agent_name] = turn.decision.target
                last_raw_outputs[agent_name] = turn.raw_response_text
        except AgentExecutionError as exc:
            outcome = "agent_error"
            error_message = str(exc)
            break

        observations, rewards, terminations, truncations, _ = env.step(actions)
        team_reward += rewards.get("agent_a", 0.0)
        if not env.agents or all(terminations.values()) or all(truncations.values()):
            outcome = env.last_outcome
            break

    agreed = last_targets["agent_a"] == last_targets["agent_b"] and last_targets["agent_a"] in {"LEFT", "RIGHT"}
    agents["agent_a"].record_episode_summary(
        build_agent_summary("agent_a", last_targets["agent_a"], outcome, agreed)
    )
    agents["agent_b"].record_episode_summary(
        build_agent_summary("agent_b", last_targets["agent_b"], outcome, agreed)
    )

    record = EpisodeRecord(
        seed=seed,
        condition=condition,
        episode_id=episode_id,
        value_left=env.left_value,
        value_right=env.right_value,
        best_item=env.best_item,
        outcome=outcome,
        team_reward=round(team_reward, 4),
        target_a=last_targets["agent_a"],
        target_b=last_targets["agent_b"],
        error_message=error_message,
    )
    details = {
        "seed": seed,
        "condition": condition,
        "episode_id": episode_id,
        "outcome": outcome,
        "team_reward": round(team_reward, 4),
        "value_left": env.left_value,
        "value_right": env.right_value,
        "best_item": env.best_item,
        "target_a": last_targets["agent_a"],
        "target_b": last_targets["agent_b"],
        "final_raw_output_a": last_raw_outputs["agent_a"],
        "final_raw_output_b": last_raw_outputs["agent_b"],
        "error_message": error_message,
    }
    return record, details


def ensure_ollama_ready(model: str, base_url: str) -> None:
    status = check_ollama_setup(model=model, base_url=base_url)
    if not status.ok:
        raise SystemExit(status.guidance(model))


def write_outputs(records: list[EpisodeRecord], details: list[dict[str, object]], csv_path: Path, jsonl_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame([asdict(record) for record in records])
    dataframe.to_csv(csv_path, index=False)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_ollama_ready(model=args.model, base_url=args.base_url)

    agents = {
        "agent_a": OllamaAgent(agent_name="agent_a", model=args.model, base_url=args.base_url),
        "agent_b": OllamaAgent(agent_name="agent_b", model=args.model, base_url=args.base_url),
    }

    records: list[EpisodeRecord] = []
    details: list[dict[str, object]] = []
    global_episode_id = 0

    for condition_index, condition in enumerate(args.conditions):
        for episode_offset in range(args.episodes):
            seed = args.seed + condition_index * 10_000 + episode_offset
            rng = np.random.default_rng(seed + 99)
            env = ScoreGParallelEnv(grid_size=args.grid_size, max_steps=args.max_steps)
            record, detail = run_episode(
                env=env,
                agents=agents,
                condition=condition,
                episode_id=global_episode_id,
                seed=seed,
                rng=rng,
            )
            records.append(record)
            details.append(detail)
            global_episode_id += 1

    write_outputs(
        records=records,
        details=details,
        csv_path=Path(args.output_csv),
        jsonl_path=Path(args.output_jsonl),
    )
    print(f"Wrote {len(records)} episode records to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
