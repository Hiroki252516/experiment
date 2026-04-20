import json

import numpy as np

from agents.ollama_agent import AgentDecision, build_agent_user_prompt


def make_observation(private_value: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        "self_position": np.array([4, 1], dtype=np.int64),
        "other_position": np.array([4, 3], dtype=np.int64),
        "item_positions": np.array([[0, 1], [0, 3]], dtype=np.int64),
        "private_value": np.array(private_value, dtype=np.int64),
        "other_last_glyph": np.zeros((7, 7), dtype=np.int8),
        "step_count": np.array([0], dtype=np.int64),
        "movement_enabled": np.array([0], dtype=np.int64),
    }


def test_agent_a_prompt_does_not_include_right_value() -> None:
    prompt = build_agent_user_prompt(
        agent_name="agent_a",
        observation=make_observation((7, 0)),
        memory=[],
        schema_json=json.dumps(AgentDecision.model_json_schema(), ensure_ascii=True),
        current_hypothesis_target="LEFT",
    )
    assert "known_value=left_item=7" in prompt
    assert "right_item=9" not in prompt
    assert "Maximize team reward" in prompt
    assert "current_hypothesis_target=LEFT" in prompt
    assert "phase=comm" in prompt
    assert "Use UNKNOWN only at step 0" in prompt
    assert "Use STAY only for a clear tactical reason" in prompt
    assert "Glyphs are the only channel" in prompt
    assert "Reuse previously successful glyphs" in prompt


def test_agent_b_prompt_does_not_include_left_value() -> None:
    prompt = build_agent_user_prompt(
        agent_name="agent_b",
        observation=make_observation((0, 9)),
        memory=[],
        schema_json=json.dumps(AgentDecision.model_json_schema(), ensure_ascii=True),
        current_hypothesis_target="RIGHT",
    )
    assert "known_value=right_item=9" in prompt
    assert "left_item=7" not in prompt
    assert "Maximize team reward" in prompt
    assert "current_hypothesis_target=RIGHT" in prompt
    assert "consider converging on the partner side" in prompt
