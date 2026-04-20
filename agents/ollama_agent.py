"""Ollama-backed agent for the ScoreG coordination experiment."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import requests
from pydantic import BaseModel, Field, StringConstraints, ValidationError, field_validator

from envs.scoreg_env import GLYPH_SIDE, glyph_matrix_to_rows, rows_to_glyph_matrix

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
OFFICIAL_OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"
MOVE_CHOICES = ("UP", "DOWN", "LEFT", "RIGHT", "STAY", "PICK")
TARGET_CHOICES = ("LEFT", "RIGHT", "UNKNOWN")
GlyphRow = Annotated[str, StringConstraints(pattern=rf"^[01]{{{GLYPH_SIDE}}}$")]


@dataclass
class OllamaStatus:
    """Availability status for a local Ollama setup."""

    cli_available: bool
    api_reachable: bool
    model_available: bool
    available_models: list[str] = field(default_factory=list)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.cli_available and self.api_reachable and self.model_available

    def guidance(self, model: str) -> str:
        lines = [
            "Ollama の公式 macOS セットアップが必要です。",
            f"1. {OFFICIAL_OLLAMA_DOWNLOAD_URL} から公式インストーラを取得してインストールしてください。",
            "2. Ollama アプリを起動し、ローカル API を有効にしてください。",
            f"3. `ollama pull {model}` でモデルを取得してください。",
            "4. `python scripts/check_ollama.py --model <model>` を再実行してください。",
        ]
        if self.detail:
            lines.append(f"詳細: {self.detail}")
        return "\n".join(lines)


class AgentDecision(BaseModel):
    """Structured action produced by the LLM."""

    glyph: list[GlyphRow] = Field(min_length=GLYPH_SIDE, max_length=GLYPH_SIDE)
    move: Literal["UP", "DOWN", "LEFT", "RIGHT", "STAY", "PICK"]
    target: Literal["LEFT", "RIGHT", "UNKNOWN"]

    @field_validator("glyph")
    @classmethod
    def validate_glyph(cls, glyph: list[str]) -> list[str]:
        if len(glyph) != GLYPH_SIDE:
            raise ValueError("glyph must contain exactly 7 rows")
        return glyph

    def glyph_matrix(self) -> np.ndarray:
        return rows_to_glyph_matrix(self.glyph)


@dataclass
class AgentTurn:
    decision: AgentDecision
    raw_response_text: str


@dataclass
class DecisionGuardResult:
    target: str
    move: str
    applied: bool
    reason: str = ""


class AgentExecutionError(RuntimeError):
    """Raised when the model cannot produce a valid structured action."""


def load_system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _as_int_list(value: np.ndarray | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.tolist()]
    return [int(item) for item in value]


def build_agent_user_prompt(
    agent_name: str,
    observation: dict[str, np.ndarray],
    memory: list[str] | None,
    schema_json: str,
    current_hypothesis_target: str | None = None,
) -> str:
    """Build a prompt with only public state and the agent's own private value."""

    private_value = _as_int_list(observation["private_value"])
    self_position = _as_int_list(observation["self_position"])
    other_position = _as_int_list(observation["other_position"])
    item_positions = observation["item_positions"].tolist()
    glyph_rows = glyph_matrix_to_rows(observation["other_last_glyph"])
    step_count = int(_as_int_list(observation["step_count"])[0])
    movement_enabled = bool(_as_int_list(observation["movement_enabled"])[0])
    phase_label = "move" if movement_enabled else "comm"

    if agent_name == "agent_a":
        known_label = f"left_item={private_value[0]}"
    else:
        known_label = f"right_item={private_value[1]}"

    hypothesis_target = current_hypothesis_target or resolve_hypothesis_target(agent_name, None)
    memory_lines = memory or ["none"]
    memory_block = "\n".join(f"- {line}" for line in memory_lines)

    return "\n".join(
        [
            "Public state:",
            f"- step={step_count}",
            f"- phase={phase_label}",
            f"- movement_enabled={int(movement_enabled)}",
            f"- self_position={self_position}",
            f"- other_position={other_position}",
            f"- left_item_position={item_positions[0]}",
            f"- right_item_position={item_positions[1]}",
            f"- other_last_glyph={glyph_rows}",
            "",
            "Private observation:",
            f"- known_value={known_label}",
            "- The other item's value is hidden. Do not invent or state it.",
            f"- current_hypothesis_target={hypothesis_target}",
            "",
            "Private memory notes:",
            memory_block,
            "",
            "Decision policy:",
            "- Maximize team reward rather than playing passively.",
            "- Glyphs are the only channel for communicating hidden value information.",
            "- Try to keep a LEFT or RIGHT hypothesis target on every turn.",
            "- Reuse previously successful glyphs when the known value and coordination situation are similar.",
            "- If your known value is low, consider converging on the partner side instead of stubbornly staying on your own side.",
            "- Matching the partner's target matters more than acting alone.",
            "- Use UNKNOWN only at step 0 or when you truly cannot decide.",
            "- Use STAY only for a clear tactical reason, such as waiting on the target cell.",
            "- If your target is not reached yet, prefer moving toward it.",
            "- During communication-only phase, use the glyph to make the target easier to infer later.",
            "",
            "Return JSON only. Follow this schema exactly:",
            schema_json,
            "",
            "Constraints:",
            "- glyph must be 7 strings of 7 binary digits",
            "- move must be one of UP, DOWN, LEFT, RIGHT, STAY, PICK",
            "- target must be LEFT, RIGHT, or UNKNOWN",
            "- do not add explanations outside JSON",
        ]
    )


def default_target_for_agent(agent_name: str) -> str:
    return "LEFT" if agent_name == "agent_a" else "RIGHT"


def resolve_hypothesis_target(agent_name: str, previous_target: str | None) -> str:
    if previous_target in {"LEFT", "RIGHT"}:
        return previous_target
    return default_target_for_agent(agent_name)


def agent_known_value(agent_name: str, observation: dict[str, np.ndarray]) -> int:
    private_value = _as_int_list(observation["private_value"])
    return int(private_value[0] if agent_name == "agent_a" else private_value[1])


def glyph_signature_from_rows(rows: list[str]) -> str:
    return "".join(rows)


def build_memory_summary(
    *,
    episode_id: int,
    condition: str,
    agent_name: str,
    known_value: int,
    sent_glyph_rows: list[str],
    received_glyph_rows: list[str],
    target: str,
    agreement: bool,
    team_reward: float,
    outcome: str,
) -> str:
    return (
        f"episode={episode_id} "
        f"condition={condition} "
        f"agent={agent_name} "
        f"my_known_value={known_value} "
        f"my_sent_glyph={glyph_signature_from_rows(sent_glyph_rows)} "
        f"my_received_glyph={glyph_signature_from_rows(received_glyph_rows)} "
        f"my_target={target} "
        f"agreement={agreement} "
        f"team_reward={team_reward:.2f} "
        f"outcome={outcome}"
    )


def _target_position_from_observation(observation: dict[str, np.ndarray], target: str) -> np.ndarray:
    item_positions = np.asarray(observation["item_positions"], dtype=np.int64)
    target_index = 0 if target == "LEFT" else 1
    return item_positions[target_index]


def greedy_move_toward_target(
    self_position: np.ndarray | list[int],
    target_position: np.ndarray | list[int],
) -> str:
    current = np.asarray(self_position, dtype=np.int64)
    target = np.asarray(target_position, dtype=np.int64)
    row_delta = int(target[0] - current[0])
    col_delta = int(target[1] - current[1])
    if row_delta < 0:
        return "UP"
    if row_delta > 0:
        return "DOWN"
    if col_delta < 0:
        return "LEFT"
    if col_delta > 0:
        return "RIGHT"
    return "STAY"


def _next_position_for_move(
    self_position: np.ndarray | list[int],
    move: str,
    grid_size: int,
) -> np.ndarray:
    current = np.asarray(self_position, dtype=np.int64)
    deltas = {
        "UP": np.array([-1, 0], dtype=np.int64),
        "DOWN": np.array([1, 0], dtype=np.int64),
        "LEFT": np.array([0, -1], dtype=np.int64),
        "RIGHT": np.array([0, 1], dtype=np.int64),
        "STAY": np.array([0, 0], dtype=np.int64),
        "PICK": np.array([0, 0], dtype=np.int64),
    }
    return np.clip(current + deltas[move], 0, grid_size - 1)


def _manhattan_distance(position_a: np.ndarray | list[int], position_b: np.ndarray | list[int]) -> int:
    first = np.asarray(position_a, dtype=np.int64)
    second = np.asarray(position_b, dtype=np.int64)
    return int(np.abs(first - second).sum())


def apply_decision_guard(
    agent_name: str,
    observation: dict[str, np.ndarray],
    decision: AgentDecision,
    previous_target: str | None,
    grid_size: int = 5,
) -> DecisionGuardResult:
    step_count = int(_as_int_list(observation["step_count"])[0])
    effective_target = decision.target
    effective_move = decision.move
    reasons: list[str] = []

    if effective_target == "UNKNOWN" and step_count >= 1:
        effective_target = resolve_hypothesis_target(agent_name, previous_target)
        reasons.append("target_unknown_after_step0")

    if effective_target in {"LEFT", "RIGHT"}:
        self_position = np.asarray(observation["self_position"], dtype=np.int64)
        other_position = np.asarray(observation["other_position"], dtype=np.int64)
        target_position = _target_position_from_observation(observation, effective_target)
        on_target = bool(np.array_equal(self_position, target_position))
        other_on_target = bool(np.array_equal(other_position, target_position))

        if on_target:
            if other_on_target and effective_move != "PICK":
                effective_move = "PICK"
                reasons.append("on_target_with_partner_to_pick")
            elif not other_on_target and effective_move in {"UP", "DOWN", "LEFT", "RIGHT", "PICK"}:
                effective_move = "STAY"
                reasons.append("leave_target_to_stay")
        elif effective_move == "STAY":
            effective_move = greedy_move_toward_target(self_position, target_position)
            reasons.append("stay_to_greedy_move")
        elif effective_move == "PICK" and not on_target:
            effective_move = greedy_move_toward_target(self_position, target_position)
            reasons.append("invalid_pick_to_greedy_move")
        elif effective_move in {"UP", "DOWN", "LEFT", "RIGHT"} and not on_target:
            next_position = _next_position_for_move(self_position, effective_move, grid_size)
            current_distance = _manhattan_distance(self_position, target_position)
            next_distance = _manhattan_distance(next_position, target_position)
            if next_distance >= current_distance:
                effective_move = greedy_move_toward_target(self_position, target_position)
                reasons.append("nonprogress_move_to_greedy_move")

    return DecisionGuardResult(
        target=effective_target,
        move=effective_move,
        applied=bool(reasons),
        reason=",".join(reasons),
    )


def check_ollama_setup(
    model: str,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout_s: float = 5.0,
) -> OllamaStatus:
    """Check whether the local Ollama CLI, API, and model are available."""

    cli_available = shutil.which("ollama") is not None
    if not cli_available:
        return OllamaStatus(
            cli_available=False,
            api_reachable=False,
            model_available=False,
            detail="`ollama` コマンドが見つかりませんでした。",
        )

    tags_url = f"{base_url.rstrip('/')}/api/tags"
    try:
        response = requests.get(tags_url, timeout=timeout_s)
        response.raise_for_status()
    except requests.RequestException as exc:
        return OllamaStatus(
            cli_available=True,
            api_reachable=False,
            model_available=False,
            detail=f"`{tags_url}` に接続できませんでした: {exc}",
        )

    payload = response.json()
    available_models: list[str] = []
    for item in payload.get("models", []):
        for key in ("model", "name"):
            if key in item and item[key]:
                available_models.append(str(item[key]))
                break
    model_available = model in available_models
    detail = ""
    if not model_available:
        detail = f"モデル `{model}` が見つかりませんでした。利用可能: {', '.join(available_models) or 'なし'}"
    return OllamaStatus(
        cli_available=True,
        api_reachable=True,
        model_available=model_available,
        available_models=available_models,
        detail=detail,
    )


def _extract_json_candidate(raw_text: str) -> str:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return raw_text
    return raw_text[start : end + 1]


class OllamaAgent:
    """Agent wrapper that queries the local Ollama chat API with a JSON schema."""

    def __init__(
        self,
        agent_name: str,
        model: str = DEFAULT_OLLAMA_MODEL,
        system_prompt_path: str | Path | None = None,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        memory_limit: int = 6,
    ) -> None:
        self.agent_name = agent_name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.memory_limit = memory_limit
        default_prompt = Path(__file__).resolve().parents[1] / "prompts" / f"{agent_name}_system.txt"
        self.system_prompt_path = Path(system_prompt_path) if system_prompt_path else default_prompt
        self.system_prompt = load_system_prompt(self.system_prompt_path)
        self.memory: list[str] = []

    def act(self, observation: dict[str, np.ndarray], previous_target: str | None = None) -> AgentTurn:
        schema_json = json.dumps(AgentDecision.model_json_schema(), ensure_ascii=True)
        user_prompt = build_agent_user_prompt(
            agent_name=self.agent_name,
            observation=observation,
            memory=self.memory,
            schema_json=schema_json,
            current_hypothesis_target=resolve_hypothesis_target(self.agent_name, previous_target),
        )
        last_error = "unknown error"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "format": AgentDecision.model_json_schema(),
                        "options": {"temperature": 0},
                    },
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                payload = response.json()
                raw_content = payload["message"]["content"]
                try:
                    decision = AgentDecision.model_validate_json(raw_content)
                except ValidationError:
                    decision = AgentDecision.model_validate_json(_extract_json_candidate(raw_content))
                return AgentTurn(decision=decision, raw_response_text=raw_content)
            except (requests.RequestException, KeyError, ValidationError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"{user_prompt}\n\n"
                            f"Previous reply was invalid on attempt {attempt}: {last_error}\n"
                            "Return only valid JSON that matches the schema."
                        ),
                    },
                ]

        raise AgentExecutionError(
            f"{self.agent_name} failed to produce valid structured output after {self.max_retries} attempts: {last_error}"
        )

    def record_episode_summary(self, summary: str) -> None:
        self.memory.append(summary)
        self.memory = self.memory[-self.memory_limit :]
