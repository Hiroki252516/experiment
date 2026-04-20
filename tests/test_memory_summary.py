import numpy as np

from agents.ollama_agent import agent_known_value, build_memory_summary


def test_memory_summary_contains_expected_fields_without_hidden_leakage() -> None:
    summary = build_memory_summary(
        episode_id=17,
        condition="comm",
        agent_name="agent_a",
        known_value=3,
        sent_glyph_rows=["1000000"] * 7,
        received_glyph_rows=["0000001"] * 7,
        target="RIGHT",
        agreement=True,
        team_reward=1.0,
        outcome="high_value",
    )
    assert "episode=17" in summary
    assert "condition=comm" in summary
    assert "my_known_value=3" in summary
    assert "my_sent_glyph=" in summary
    assert "my_received_glyph=" in summary
    assert "my_target=RIGHT" in summary
    assert "team_reward=1.00" in summary
    assert "outcome=high_value" in summary
    assert "right_item=9" not in summary


def test_agent_known_value_uses_only_own_slot() -> None:
    observation = {
        "private_value": np.array([4, 9], dtype=np.int64),
    }
    assert agent_known_value("agent_a", observation) == 4
    assert agent_known_value("agent_b", observation) == 9
