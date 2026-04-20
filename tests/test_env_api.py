from pettingzoo.test import parallel_api_test

from envs.scoreg_env import ScoreGParallelEnv


def test_parallel_env_api() -> None:
    parallel_api_test(ScoreGParallelEnv(), num_cycles=20)


def test_parallel_env_api_with_comm_phase_and_randomization() -> None:
    parallel_api_test(
        ScoreGParallelEnv(comm_phase_steps=2, randomize_positions=True, hard_split_prob=0.5),
        num_cycles=20,
    )
