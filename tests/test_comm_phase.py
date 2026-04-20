import numpy as np

from envs.scoreg_env import MOVE_TO_INDEX, ScoreGParallelEnv


def make_action(move: str, bit: int) -> dict[str, object]:
    glyph = np.full((7, 7), bit, dtype=np.int8)
    return {"move": MOVE_TO_INDEX[move], "glyph": glyph}


def test_comm_phase_blocks_movement_but_updates_glyphs() -> None:
    env = ScoreGParallelEnv(comm_phase_steps=2)
    observations, _ = env.reset(seed=7)
    before_a = observations["agent_a"]["self_position"].copy()
    before_b = observations["agent_b"]["self_position"].copy()
    observations, *_ = env.step(
        {
            "agent_a": make_action("UP", 1),
            "agent_b": make_action("LEFT", 0),
        }
    )
    assert observations["agent_a"]["other_last_glyph"].sum() == 0
    assert observations["agent_b"]["other_last_glyph"].sum() == 49
    assert np.array_equal(env.agent_positions["agent_a"], before_a)
    assert np.array_equal(env.agent_positions["agent_b"], before_b)
    assert env.current_phase == "comm"


def test_movement_resumes_after_comm_phase() -> None:
    env = ScoreGParallelEnv(comm_phase_steps=1)
    env.reset(seed=11)
    env.step(
        {
            "agent_a": make_action("UP", 1),
            "agent_b": make_action("UP", 1),
        }
    )
    before = env.agent_positions["agent_a"].copy()
    env.step(
        {
            "agent_a": make_action("UP", 1),
            "agent_b": make_action("UP", 1),
        }
    )
    assert not np.array_equal(env.agent_positions["agent_a"], before)
    assert env.current_phase == "move"
