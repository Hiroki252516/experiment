"""PettingZoo ParallelEnv for a ScoreG-style coordination task."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

GLYPH_SIDE = 7
GLYPH_BITS = GLYPH_SIDE * GLYPH_SIDE
MOVE_NAMES = ("UP", "DOWN", "LEFT", "RIGHT", "STAY", "PICK")
MOVE_TO_INDEX = {name: index for index, name in enumerate(MOVE_NAMES)}
INDEX_TO_MOVE = {index: name for name, index in MOVE_TO_INDEX.items()}
AGENT_NAMES = ("agent_a", "agent_b")
ITEM_LABELS = ("LEFT", "RIGHT")
ITEM_POSITIONS = {
    "LEFT": np.array([0, 1], dtype=np.int64),
    "RIGHT": np.array([0, 3], dtype=np.int64),
}
START_POSITIONS = {
    "agent_a": np.array([4, 1], dtype=np.int64),
    "agent_b": np.array([4, 3], dtype=np.int64),
}
MOVE_DELTAS = {
    "UP": np.array([-1, 0], dtype=np.int64),
    "DOWN": np.array([1, 0], dtype=np.int64),
    "LEFT": np.array([0, -1], dtype=np.int64),
    "RIGHT": np.array([0, 1], dtype=np.int64),
    "STAY": np.array([0, 0], dtype=np.int64),
    "PICK": np.array([0, 0], dtype=np.int64),
}


def flatten_glyph(glyph: np.ndarray | list[list[int]] | list[int]) -> list[int]:
    array = np.asarray(glyph, dtype=np.int8)
    if array.size != GLYPH_BITS:
        raise ValueError("glyph must contain 49 binary values")
    flattened = array.reshape(GLYPH_BITS)
    if np.any((flattened != 0) & (flattened != 1)):
        raise ValueError("glyph must be binary")
    return [int(item) for item in flattened.tolist()]


def unflatten_glyph(flat: list[int] | np.ndarray) -> np.ndarray:
    values = np.asarray(flat, dtype=np.int8).reshape(GLYPH_BITS)
    if values.size != GLYPH_BITS:
        raise ValueError("flat glyph must contain 49 binary values")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("glyph must be binary")
    return values.reshape((GLYPH_SIDE, GLYPH_SIDE))


def glyph_matrix_to_rows(glyph: np.ndarray | list[list[int]]) -> list[str]:
    matrix = np.asarray(glyph, dtype=np.int8).reshape((GLYPH_SIDE, GLYPH_SIDE))
    if np.any((matrix != 0) & (matrix != 1)):
        raise ValueError("glyph must be binary")
    return ["".join(str(int(bit)) for bit in row.tolist()) for row in matrix]


def rows_to_glyph_matrix(rows: list[str]) -> np.ndarray:
    if len(rows) != GLYPH_SIDE:
        raise ValueError("glyph rows must contain 7 strings")
    matrix: list[list[int]] = []
    for row in rows:
        if len(row) != GLYPH_SIDE or any(char not in {"0", "1"} for char in row):
            raise ValueError("each glyph row must be 7 binary digits")
        matrix.append([int(char) for char in row])
    return np.asarray(matrix, dtype=np.int8)


def _normalize_glyph(glyph: Any) -> np.ndarray:
    if isinstance(glyph, np.ndarray):
        matrix = glyph.astype(np.int8).reshape((GLYPH_SIDE, GLYPH_SIDE))
        if np.any((matrix != 0) & (matrix != 1)):
            raise ValueError("glyph must be binary")
        return matrix
    if isinstance(glyph, list):
        if glyph and isinstance(glyph[0], str):
            return rows_to_glyph_matrix(glyph)
        return unflatten_glyph(glyph)
    raise TypeError("unsupported glyph payload")


class ScoreGParallelEnv(ParallelEnv):
    """Two-agent partially observed gridworld with glyph-based communication."""

    metadata = {"name": "scoreg_v0", "render_modes": ["human"], "is_parallelizable": True}

    def __init__(self, grid_size: int = 5, max_steps: int = 10) -> None:
        if grid_size < 5:
            raise ValueError("grid_size must be at least 5")
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.possible_agents = list(AGENT_NAMES)
        self.agents: list[str] = []
        self._rng = np.random.default_rng()
        self.agent_positions: dict[str, np.ndarray] = {}
        self.item_positions = {
            "LEFT": ITEM_POSITIONS["LEFT"].copy(),
            "RIGHT": ITEM_POSITIONS["RIGHT"].copy(),
        }
        self.left_value = 1
        self.right_value = 1
        self.step_count = 0
        self.last_glyphs = {agent: np.zeros((GLYPH_SIDE, GLYPH_SIDE), dtype=np.int8) for agent in AGENT_NAMES}
        self.last_outcome = "not_finished"

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Dict:
        return spaces.Dict(
            {
                "self_position": spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=np.int64),
                "other_position": spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=np.int64),
                "item_positions": spaces.Box(0, self.grid_size - 1, shape=(2, 2), dtype=np.int64),
                "private_value": spaces.Box(0, 9, shape=(2,), dtype=np.int64),
                "other_last_glyph": spaces.MultiBinary((GLYPH_SIDE, GLYPH_SIDE)),
                "step_count": spaces.Box(0, self.max_steps, shape=(1,), dtype=np.int64),
            }
        )

    @lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Dict:
        return spaces.Dict(
            {
                "move": spaces.Discrete(len(MOVE_NAMES)),
                "glyph": spaces.MultiBinary((GLYPH_SIDE, GLYPH_SIDE)),
            }
        )

    @property
    def best_item(self) -> str:
        if self.left_value == self.right_value:
            return "TIE"
        return "LEFT" if self.left_value > self.right_value else "RIGHT"

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        del options
        self._rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.agent_positions = {name: START_POSITIONS[name].copy() for name in AGENT_NAMES}
        self.left_value = int(self._rng.integers(1, 10))
        self.right_value = int(self._rng.integers(1, 10))
        self.step_count = 0
        self.last_outcome = "not_finished"
        self.last_glyphs = {agent: np.zeros((GLYPH_SIDE, GLYPH_SIDE), dtype=np.int8) for agent in AGENT_NAMES}
        observations = {agent: self._observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, dict[str, Any]]
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}

        current_agents = self.agents[:]
        rewards = {agent: -0.01 for agent in current_agents}
        terminations = {agent: False for agent in current_agents}
        truncations = {agent: False for agent in current_agents}
        infos = {agent: {} for agent in current_agents}

        normalized_actions: dict[str, dict[str, Any]] = {}
        for agent in current_agents:
            action = actions[agent]
            move_raw = action["move"]
            move_name = move_raw if isinstance(move_raw, str) else INDEX_TO_MOVE[int(move_raw)]
            normalized_actions[agent] = {
                "move": move_name,
                "glyph": _normalize_glyph(action["glyph"]),
            }

        for agent, payload in normalized_actions.items():
            move_name = payload["move"]
            if move_name != "PICK":
                delta = MOVE_DELTAS[move_name]
                next_position = self.agent_positions[agent] + delta
                self.agent_positions[agent] = np.clip(next_position, 0, self.grid_size - 1)

        picked_targets: dict[str, str] = {}
        for agent, payload in normalized_actions.items():
            move_name = payload["move"]
            if move_name != "PICK":
                continue
            for item_label in ITEM_LABELS:
                if np.array_equal(self.agent_positions[agent], self.item_positions[item_label]):
                    picked_targets[agent] = item_label
                    break

        for agent, payload in normalized_actions.items():
            self.last_glyphs[agent] = payload["glyph"]

        self.step_count += 1
        if len(picked_targets) == 2:
            first_target = picked_targets["agent_a"]
            second_target = picked_targets["agent_b"]
            if first_target == second_target:
                if self.best_item == "TIE" or first_target == self.best_item:
                    shared_reward = 1.0 - 0.01
                    self.last_outcome = "high_value"
                else:
                    shared_reward = 0.2 - 0.01
                    self.last_outcome = "low_value"
            else:
                shared_reward = -0.2 - 0.01
                self.last_outcome = "split_pick"
            rewards = {agent: shared_reward for agent in current_agents}
            terminations = {agent: True for agent in current_agents}
            self.agents = []
        elif self.step_count >= self.max_steps:
            self.last_outcome = "max_steps"
            truncations = {agent: True for agent in current_agents}
            self.agents = []

        observation_agents = self.agents if self.agents else current_agents
        observations = {agent: self._observe(agent) for agent in observation_agents}
        return observations, rewards, terminations, truncations, infos

    def _observe(self, agent: str) -> dict[str, np.ndarray]:
        other_agent = "agent_b" if agent == "agent_a" else "agent_a"
        private_value = np.array(
            [self.left_value, 0] if agent == "agent_a" else [0, self.right_value],
            dtype=np.int64,
        )
        return {
            "self_position": self.agent_positions[agent].copy(),
            "other_position": self.agent_positions[other_agent].copy(),
            "item_positions": np.stack(
                [self.item_positions["LEFT"], self.item_positions["RIGHT"]]
            ).astype(np.int64),
            "private_value": private_value,
            "other_last_glyph": self.last_glyphs[other_agent].copy(),
            "step_count": np.array([self.step_count], dtype=np.int64),
        }

    def render(self) -> str:
        return (
            f"step={self.step_count} "
            f"a={self.agent_positions['agent_a'].tolist()} "
            f"b={self.agent_positions['agent_b'].tolist()} "
            f"left={self.left_value} right={self.right_value}"
        )

    def close(self) -> None:
        self.agents = []
