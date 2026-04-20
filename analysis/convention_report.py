"""Human-readable summaries of glyph convention hints."""

from __future__ import annotations

from typing import Any


def format_convention_hints(hints: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for hint in hints:
        lines.append(
            "glyph_count={count} dominant_target={dominant_target} success_count={success_count}".format(
                count=hint.get("count", 0),
                dominant_target=hint.get("dominant_target", "UNKNOWN"),
                success_count=hint.get("success_count", 0),
            )
        )
    return lines
