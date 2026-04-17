from pettingzoo.test import parallel_api_test

from envs.scoreg_env import ScoreGParallelEnv


def test_parallel_env_api() -> None:
    parallel_api_test(ScoreGParallelEnv(), num_cycles=20)
