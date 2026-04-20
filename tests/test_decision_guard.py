import numpy as np

from agents.ollama_agent import AgentDecision, apply_decision_guard


def make_observation(
    *,
    self_position: tuple[int, int] = (4, 1),
    other_position: tuple[int, int] = (4, 3),
    private_value: tuple[int, int] = (7, 0),
    step: int = 1,
) -> dict[str, np.ndarray]:
    return {
        "self_position": np.array(self_position, dtype=np.int64),
        "other_position": np.array(other_position, dtype=np.int64),
        "item_positions": np.array([[0, 1], [0, 3]], dtype=np.int64),
        "private_value": np.array(private_value, dtype=np.int64),
        "other_last_glyph": np.zeros((7, 7), dtype=np.int8),
        "step_count": np.array([step], dtype=np.int64),
    }


def make_decision(move: str = "STAY", target: str = "UNKNOWN") -> AgentDecision:
    return AgentDecision(
        glyph=["0" * 7 for _ in range(7)],
        move=move,
        target=target,
    )


def test_unknown_after_step0_uses_previous_target() -> None:
    result = apply_decision_guard(
        "agent_a",
        make_observation(step=2),
        make_decision(move="RIGHT", target="UNKNOWN"),
        previous_target="RIGHT",
    )
    assert result.target == "RIGHT"
    assert result.applied is True
    assert "target_unknown_after_step0" in result.reason


def test_unknown_after_step0_falls_back_to_own_side() -> None:
    result = apply_decision_guard(
        "agent_b",
        make_observation(step=3, private_value=(0, 9)),
        make_decision(move="LEFT", target="UNKNOWN"),
        previous_target="UNKNOWN",
    )
    assert result.target == "RIGHT"
    assert result.applied is True


def test_stay_off_target_becomes_greedy_move() -> None:
    result = apply_decision_guard(
        "agent_a",
        make_observation(self_position=(4, 1), other_position=(4, 3)),
        make_decision(move="STAY", target="LEFT"),
        previous_target="LEFT",
    )
    assert result.move == "UP"
    assert "stay_to_greedy_move" in result.reason


def test_stay_on_target_waiting_is_preserved() -> None:
    result = apply_decision_guard(
        "agent_a",
        make_observation(self_position=(0, 1), other_position=(4, 3)),
        make_decision(move="STAY", target="LEFT"),
        previous_target="LEFT",
    )
    assert result.move == "STAY"
    assert result.applied is False


def test_stay_on_shared_target_becomes_pick() -> None:
    result = apply_decision_guard(
        "agent_a",
        make_observation(self_position=(0, 1), other_position=(0, 1)),
        make_decision(move="STAY", target="LEFT"),
        previous_target="LEFT",
    )
    assert result.move == "PICK"
    assert "on_target_with_partner_to_pick" in result.reason


def test_invalid_pick_off_target_becomes_greedy_move() -> None:
    result = apply_decision_guard(
        "agent_b",
        make_observation(self_position=(4, 3), other_position=(4, 1), private_value=(0, 8)),
        make_decision(move="PICK", target="RIGHT"),
        previous_target="RIGHT",
    )
    assert result.move == "UP"
    assert "invalid_pick_to_greedy_move" in result.reason


def test_guard_does_not_depend_on_hidden_value_slot() -> None:
    observation_a = make_observation(private_value=(7, 0), step=2)
    observation_b = make_observation(private_value=(7, 9), step=2)
    decision = make_decision(move="STAY", target="UNKNOWN")
    result_a = apply_decision_guard("agent_a", observation_a, decision, previous_target="UNKNOWN")
    result_b = apply_decision_guard("agent_a", observation_b, decision, previous_target="UNKNOWN")
    assert result_a.target == result_b.target
    assert result_a.move == result_b.move


def test_nonprogress_move_becomes_greedy_move() -> None:
    result = apply_decision_guard(
        "agent_a",
        make_observation(self_position=(4, 0), other_position=(4, 4)),
        make_decision(move="LEFT", target="LEFT"),
        previous_target="LEFT",
    )
    assert result.move == "UP"
    assert "nonprogress_move_to_greedy_move" in result.reason


def test_leaving_target_becomes_stay_until_partner_arrives() -> None:
    result = apply_decision_guard(
        "agent_a",
        make_observation(self_position=(0, 1), other_position=(0, 3)),
        make_decision(move="RIGHT", target="LEFT"),
        previous_target="LEFT",
    )
    assert result.move == "STAY"
    assert "leave_target_to_stay" in result.reason
