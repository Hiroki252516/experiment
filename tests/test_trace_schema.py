from scripts.run_experiment import build_manifest, build_trace_row, default_run_id, parse_args, resolve_run_paths
from envs.scoreg_env import ScoreGParallelEnv


def test_trace_row_has_required_keys() -> None:
    env = ScoreGParallelEnv()
    env.reset(seed=1)
    row = build_trace_row(
        run_id=default_run_id(),
        condition="comm",
        episode_id=0,
        step=0,
        env=env,
        sent_rows={"agent_a": ["0" * 7 for _ in range(7)], "agent_b": ["0" * 7 for _ in range(7)]},
        received_rows={"agent_a": ["0" * 7 for _ in range(7)], "agent_b": ["0" * 7 for _ in range(7)]},
        moves={"agent_a": "STAY", "agent_b": "STAY"},
        targets={"agent_a": "UNKNOWN", "agent_b": "UNKNOWN"},
        raw_outputs={"agent_a": "{}", "agent_b": "{}"},
        rewards={"agent_a": -0.01, "agent_b": -0.01},
        cumulative_team_reward=-0.01,
        done=True,
        outcome="max_steps",
        error_message="",
    )
    required_keys = {
        "run_id",
        "timestamp",
        "condition",
        "episode",
        "step",
        "agent_a_pos",
        "agent_b_pos",
        "left_item_pos",
        "right_item_pos",
        "value_left",
        "value_right",
        "best_item",
        "glyph_a_sent",
        "glyph_b_sent",
        "glyph_a_received",
        "glyph_b_received",
        "move_a",
        "move_b",
        "target_a",
        "target_b",
        "raw_a",
        "raw_b",
        "reward_a",
        "reward_b",
        "team_reward",
        "done",
        "outcome",
        "error_message",
    }
    assert required_keys.issubset(row.keys())
    assert row["done"] is True
    assert row["outcome"] == "max_steps"


def test_manifest_has_required_keys() -> None:
    args = parse_args(["--episodes", "1", "--conditions", "comm", "--run-id", "run_test_manifest"])
    paths = resolve_run_paths(args)
    manifest = build_manifest(args, paths)
    required_keys = {
        "run_id",
        "model",
        "conditions",
        "episodes_per_condition",
        "base_seed",
        "grid_size",
        "max_steps",
        "status",
        "started_at",
        "completed_at",
        "trace_path",
        "results_csv_path",
        "episodes_jsonl_path",
        "launcher_log_path",
        "pid",
        "current_condition",
        "current_episode",
        "last_step",
        "last_error_message",
    }
    assert required_keys.issubset(manifest.keys())
