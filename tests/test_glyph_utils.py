import numpy as np

from envs.scoreg_env import flatten_glyph, glyph_matrix_to_rows, rows_to_glyph_matrix, unflatten_glyph


def test_flatten_unflatten_round_trip() -> None:
    glyph = np.array(
        [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 1, 0],
        ],
        dtype=np.int8,
    )
    flattened = flatten_glyph(glyph)
    restored = unflatten_glyph(flattened)
    assert flattened == flatten_glyph(restored)
    assert np.array_equal(glyph, restored)


def test_rows_conversion_round_trip() -> None:
    glyph = np.zeros((7, 7), dtype=np.int8)
    glyph[0, 0] = 1
    glyph[6, 6] = 1
    rows = glyph_matrix_to_rows(glyph)
    restored = rows_to_glyph_matrix(rows)
    assert rows[0] == "1000000"
    assert rows[-1] == "0000001"
    assert np.array_equal(glyph, restored)
