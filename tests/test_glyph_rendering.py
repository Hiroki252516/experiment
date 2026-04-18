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
    image = glyph_rows_to_array(rows, scale=10)
    assert image.shape == (70, 70)
    assert int(image[0, 0]) == 0
    assert int(image[-1, -1]) == 0
    assert int(image[0, 10]) == 255
