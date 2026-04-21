from pettingzoo.test import parallel_api_test

from envs.scoreg_env import ScoreGParallelEnv


def test_parallel_env_api() -> None:
    parallel_api_test(ScoreGParallelEnv(), num_cycles=20)


def test_randomized_layout_avoids_overlap() -> None:
    env = ScoreGParallelEnv(randomize_layout=True, random_start_min_distance=2)
    env.reset(seed=7)
    agent_a = tuple(env.agent_positions["agent_a"].tolist())
    agent_b = tuple(env.agent_positions["agent_b"].tolist())
    left_item = tuple(env.item_positions["LEFT"].tolist())
    right_item = tuple(env.item_positions["RIGHT"].tolist())
    assert agent_a != agent_b
    assert left_item != right_item
    assert agent_a not in {left_item, right_item}
    assert agent_b not in {left_item, right_item}
    manhattan = abs(agent_a[0] - agent_b[0]) + abs(agent_a[1] - agent_b[1])
    assert manhattan >= 2


def test_comm_only_phase_holds_positions_then_switches() -> None:
    env = ScoreGParallelEnv(comm_only_turns=2, randomize_layout=False)
    observations, _ = env.reset(seed=1)
    start_a = tuple(observations["agent_a"]["self_position"].tolist())
    start_b = tuple(observations["agent_b"]["self_position"].tolist())
    actions = {
        "agent_a": {"move": "UP", "glyph": ["1" * 7 for _ in range(7)]},
        "agent_b": {"move": "LEFT", "glyph": ["0" * 7 for _ in range(7)]},
    }
    observations, _, _, _, _ = env.step(actions)
    assert tuple(observations["agent_a"]["self_position"].tolist()) == start_a
    assert tuple(observations["agent_b"]["self_position"].tolist()) == start_b
    assert env.phase == "comm_only"
    observations, _, _, _, _ = env.step(actions)
    assert env.phase == "act"
    assert tuple(observations["agent_a"]["self_position"].tolist()) == start_a
