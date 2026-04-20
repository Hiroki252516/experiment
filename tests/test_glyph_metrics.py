from analysis.glyph_metrics import (
    compute_glyph_reuse_consistency,
    compute_glyph_target_association,
    compute_target_flip_rate,
)


def sample_trace_rows() -> list[dict[str, object]]:
    return [
        {
            "run_id": "run_x",
            "condition": "comm",
            "episode": 0,
            "step": 0,
            "phase": "comm",
            "glyph_a_sent": ["1000000"] * 7,
            "glyph_b_sent": ["0100000"] * 7,
            "target_a": "LEFT",
            "target_b": "RIGHT",
            "initial_target_a": "LEFT",
            "initial_target_b": "RIGHT",
            "target_changed_a": False,
            "target_changed_b": False,
            "value_left": 9,
            "value_right": 1,
            "done": False,
            "outcome": "not_finished",
        },
        {
            "run_id": "run_x",
            "condition": "comm",
            "episode": 0,
            "step": 1,
            "phase": "move",
            "glyph_a_sent": ["1000000"] * 7,
            "glyph_b_sent": ["0100000"] * 7,
            "target_a": "LEFT",
            "target_b": "RIGHT",
            "initial_target_a": "LEFT",
            "initial_target_b": "RIGHT",
            "target_changed_a": False,
            "target_changed_b": False,
            "value_left": 9,
            "value_right": 1,
            "done": True,
            "outcome": "high_value",
        },
        {
            "run_id": "run_x",
            "condition": "comm",
            "episode": 1,
            "step": 1,
            "phase": "move",
            "glyph_a_sent": ["1000000"] * 7,
            "glyph_b_sent": ["1111111"] * 7,
            "target_a": "LEFT",
            "target_b": "LEFT",
            "initial_target_a": "RIGHT",
            "initial_target_b": "RIGHT",
            "target_changed_a": True,
            "target_changed_b": True,
            "value_left": 9,
            "value_right": 2,
            "done": True,
            "outcome": "low_value",
        },
    ]


def test_glyph_reuse_consistency_is_computable() -> None:
    metrics = compute_glyph_reuse_consistency(sample_trace_rows())
    assert "comm" in metrics
    assert 0.0 <= metrics["comm"] <= 1.0


def test_glyph_target_association_is_computable() -> None:
    metrics = compute_glyph_target_association(sample_trace_rows())
    assert "comm" in metrics
    assert 0.0 <= metrics["comm"] <= 1.0


def test_target_flip_rate_is_computable() -> None:
    metrics = compute_target_flip_rate(sample_trace_rows())
    assert metrics["comm"] > 0.0
