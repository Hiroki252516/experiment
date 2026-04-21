from pathlib import Path

from agents.ollama_agent import AgentNotebookEntry, OllamaAgent, select_prompt_memory_lines


def make_entry(*, episode_id: int, outcome: str, note: str) -> AgentNotebookEntry:
    return AgentNotebookEntry(
        episode_id=episode_id,
        condition="comm",
        my_known_value=7,
        comm_sent_glyph=["1010101"] * 7,
        comm_received_glyph=["0101010"] * 7,
        final_target="LEFT",
        agreement=True,
        outcome=outcome,
        team_reward=1.0 if outcome == "high_value" else -0.2,
        glyph_helped_note=note,
    )


def test_prompt_memory_prioritizes_successes() -> None:
    notebook = [
        make_entry(episode_id=0, outcome="split_pick", note="failed"),
        make_entry(episode_id=1, outcome="high_value", note="success-1"),
        make_entry(episode_id=2, outcome="high_value", note="success-2"),
    ]
    lines = select_prompt_memory_lines(notebook, prompt_limit=2)
    assert len(lines) == 2
    assert all("outcome=high_value" in line for line in lines)
    assert all("right_item" not in line for line in lines)


def test_prior_success_glyph_uses_same_context() -> None:
    agent = OllamaAgent(
        agent_name="agent_a",
        model="dummy",
        system_prompt_path=Path("prompts/agent_a_system.txt"),
        base_url="http://localhost:11434",
    )
    agent.record_episode_entry(make_entry(episode_id=3, outcome="high_value", note="same-context"))
    assert agent.prior_success_glyph(known_value=7, target="LEFT") == ["1010101"] * 7
    assert agent.prior_success_glyph(known_value=6, target="LEFT") is None
