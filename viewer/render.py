"""Rendering helpers for the Streamlit viewer."""

from __future__ import annotations

import html
from typing import Any

import numpy as np

GRID_BACKGROUND = "#f8fafc"
AGENT_A_COLOR = "#2563eb"
AGENT_B_COLOR = "#ea580c"
LEFT_ITEM_COLOR = "#16a34a"
RIGHT_ITEM_COLOR = "#7c3aed"
HIGHLIGHT_BORDER = "#111827"


def glyph_rows_to_array(rows: list[str], scale: int = 16) -> np.ndarray:
    matrix = np.asarray([[int(bit) for bit in row] for row in rows], dtype=np.uint8)
    grayscale = np.where(matrix == 1, 0, 255).astype(np.uint8)
    return np.repeat(np.repeat(grayscale, scale, axis=0), scale, axis=1)


def _cell_entities(frame: dict[str, Any], row_index: int, col_index: int) -> tuple[list[str], str, bool]:
    labels: list[str] = []
    accents: list[str] = []
    highlight = False
    if frame.get("agent_a_pos") == [row_index, col_index]:
        labels.append("A")
        accents.append(AGENT_A_COLOR)
    if frame.get("agent_b_pos") == [row_index, col_index]:
        labels.append("B")
        accents.append(AGENT_B_COLOR)
    if frame.get("left_item_pos") == [row_index, col_index]:
        left_label = f"L:{frame.get('value_left', '?')}"
        labels.append(left_label)
        accents.append(LEFT_ITEM_COLOR)
        highlight = highlight or frame.get("best_item") == "LEFT"
    if frame.get("right_item_pos") == [row_index, col_index]:
        right_label = f"R:{frame.get('value_right', '?')}"
        labels.append(right_label)
        accents.append(RIGHT_ITEM_COLOR)
        highlight = highlight or frame.get("best_item") == "RIGHT"
    accent = accents[0] if accents else GRID_BACKGROUND
    return labels, accent, highlight


def build_grid_html(frame: dict[str, Any], grid_size: int = 5) -> str:
    rows_html: list[str] = []
    for row_index in range(grid_size):
        cells: list[str] = []
        for col_index in range(grid_size):
            labels, accent, highlight = _cell_entities(frame, row_index, col_index)
            label_html = "<br>".join(html.escape(label) for label in labels) or "&nbsp;"
            border = f"3px solid {HIGHLIGHT_BORDER}" if highlight else "1px solid #cbd5e1"
            background = "#ffffff" if labels else GRID_BACKGROUND
            cell_html = (
                f"<td style='width:72px;height:72px;border:{border};"
                f"background:{background};vertical-align:middle;text-align:center;"
                f"font-family:monospace;font-size:14px;color:{accent};font-weight:700;'>"
                f"{label_html}</td>"
            )
            cells.append(cell_html)
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table style='border-collapse:collapse;margin:0 auto;'>"
        + "".join(rows_html)
        + "</table>"
    )
