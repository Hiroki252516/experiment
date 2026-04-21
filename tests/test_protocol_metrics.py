from viewer.data import compute_protocol_metrics


def test_protocol_metrics_capture_reuse_and_persistence() -> None:
    rows = [
        {
            "condition": "comm",
            "episode": 0,
            "step": 0,
            "phase": "comm_only",
            "glyph_a_sent": ["1111111"] * 7,
            "glyph_b_sent": ["0000000"] * 7,
            "glyph_a_reused_from_success": False,
            "glyph_b_reused_from_success": False,
            "target_a_after": "LEFT",
            "target_b_after": "LEFT",
            "target_a_changed": True,
            "target_b_changed": True,
            "value_left": 8,
            "value_right": 2,
        },
        {
            "condition": "comm",
            "episode": 0,
            "step": 1,
            "phase": "act",
            "done": True,
            "target_a": "LEFT",
            "target_b": "LEFT",
            "target_a_after": "LEFT",
            "target_b_after": "LEFT",
            "outcome": "high_value",
            "value_left": 8,
            "value_right": 2,
        },
        {
            "condition": "comm",
            "episode": 1,
            "step": 0,
            "phase": "comm_only",
            "glyph_a_sent": ["1111111"] * 7,
            "glyph_b_sent": ["0000000"] * 7,
            "glyph_a_reused_from_success": True,
            "glyph_b_reused_from_success": True,
            "target_a_after": "LEFT",
            "target_b_after": "LEFT",
            "target_a_changed": False,
            "target_b_changed": False,
            "value_left": 8,
            "value_right": 2,
        },
        {
            "condition": "comm",
            "episode": 1,
            "step": 1,
            "phase": "act",
            "done": True,
            "target_a": "LEFT",
            "target_b": "LEFT",
            "target_a_after": "LEFT",
            "target_b_after": "LEFT",
            "outcome": "high_value",
            "value_left": 8,
            "value_right": 2,
        },
    ]
    metrics = compute_protocol_metrics(rows)
    assert metrics["glyph_reuse_rate"] == 0.5
    assert metrics["same_context_glyph_consistency"] == 1.0
    assert metrics["convention_persistence"] == 1.0
    assert metrics["post_comm_agreement_rate"] == 1.0
