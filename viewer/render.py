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
GLYPH_SIDE = 7
GLYPH_GRID = np.array([203, 213, 225], dtype=np.uint8)

GLYPH_STYLES: dict[str, dict[str, tuple[int, int, int]]] = {
    "a_sent": {"on": (37, 99, 235), "off": (239, 246, 255), "border": (29, 78, 216)},
    "a_received": {"on": (59, 130, 246), "off": (239, 246, 255), "border": (96, 165, 250)},
    "b_sent": {"on": (234, 88, 12), "off": (255, 247, 237), "border": (194, 65, 12)},
    "b_received": {"on": (249, 115, 22), "off": (255, 247, 237), "border": (251, 146, 60)},
    "hint": {"on": (17, 24, 39), "off": (248, 250, 252), "border": (148, 163, 184)},
}


def normalize_glyph_rows(rows: list[str] | Any) -> list[str]:
    if not isinstance(rows, list):
        return ["0" * GLYPH_SIDE for _ in range(GLYPH_SIDE)]
    normalized: list[str] = []
    for row in rows[:GLYPH_SIDE]:
        row_text = str(row)
        normalized.append("".join("1" if bit == "1" else "0" for bit in row_text[:GLYPH_SIDE].ljust(GLYPH_SIDE, "0")))
    while len(normalized) < GLYPH_SIDE:
        normalized.append("0" * GLYPH_SIDE)
    return normalized


def _glyph_mask(rows: list[str] | Any) -> np.ndarray:
    glyph_rows = normalize_glyph_rows(rows)
    return np.asarray([[1 if bit == "1" else 0 for bit in row] for row in glyph_rows], dtype=np.uint8)


def _lighten(color: np.ndarray, amount: float) -> np.ndarray:
    return np.clip(color + (255 - color) * amount, 0, 255).astype(np.uint8)


def glyph_rows_to_array(
    rows: list[str],
    scale: int = 16,
    role: str = "hint",
    *,
    mode: str = "current",
    previous_rows: list[str] | None = None,
) -> np.ndarray:
    glyph_rows = normalize_glyph_rows(rows)
    current_mask = _glyph_mask(glyph_rows)
    previous_mask = _glyph_mask(previous_rows) if previous_rows is not None else np.zeros_like(current_mask)
    changed_mask = current_mask != previous_mask
    style = GLYPH_STYLES.get(role, GLYPH_STYLES["hint"])
    cell_size = max(6, int(scale))
    line_size = max(1, cell_size // 6)
    outer_padding = max(2, cell_size // 4)
    canvas_size = GLYPH_SIDE * cell_size + (GLYPH_SIDE + 1) * line_size + outer_padding * 2
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    canvas[:, :] = np.asarray(style["border"], dtype=np.uint8)

    inner_top = outer_padding
    inner_left = outer_padding
    inner_size = GLYPH_SIDE * cell_size + (GLYPH_SIDE + 1) * line_size
    canvas[inner_top : inner_top + inner_size, inner_left : inner_left + inner_size] = GLYPH_GRID

    on_color = np.asarray(style["on"], dtype=np.uint8)
    off_color = np.asarray(style["off"], dtype=np.uint8)
    ghost_on = _lighten(on_color, 0.55)
    muted_off = _lighten(off_color, 0.12)
    diff_on = on_color
    diff_off = _lighten(off_color, 0.35)
    for row_index in range(GLYPH_SIDE):
        for col_index in range(GLYPH_SIDE):
            top = inner_top + line_size + row_index * (cell_size + line_size)
            left = inner_left + line_size + col_index * (cell_size + line_size)
            if mode == "previous":
                color = ghost_on if previous_mask[row_index, col_index] else muted_off
            elif mode == "diff":
                if changed_mask[row_index, col_index]:
                    color = diff_on if current_mask[row_index, col_index] else np.asarray(style["border"], dtype=np.uint8)
                else:
                    color = diff_off
            else:
                color = on_color if current_mask[row_index, col_index] else off_color
            canvas[top : top + cell_size, left : left + cell_size] = color
    return canvas


def glyph_rows_text(rows: list[str] | Any) -> str:
    return "\n".join(normalize_glyph_rows(rows))


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
