import numpy as np

from viewer.render import glyph_rows_to_array


def test_glyph_rows_to_array_shape_and_values() -> None:
    rows = [
        "1000000",
        "0100000",
        "0010000",
        "0001000",
        "0000100",
        "0000010",
        "0000001",
    ]
    image = glyph_rows_to_array(rows, scale=10, role="a_sent")
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.shape[0] == image.shape[1]
    unique_colors = np.unique(image.reshape(-1, 3), axis=0)
    assert len(unique_colors) >= 3


def test_zero_glyph_still_has_visible_grid() -> None:
    image = glyph_rows_to_array(["0" * 7 for _ in range(7)], scale=10, role="b_received")
    unique_colors = np.unique(image.reshape(-1, 3), axis=0)
    assert len(unique_colors) >= 2
