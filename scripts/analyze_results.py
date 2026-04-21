"""Analyze experiment logs for the ScoreG coordination study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from viewer.data import compute_protocol_metrics, load_trace_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ScoreG experiment outputs.")
    parser.add_argument("--input", default="logs/results.csv", help="Path to results.csv")
    parser.add_argument("--trace", default="", help="Optional step-trace JSONL path")
    parser.add_argument("--figure", default="logs/summary.png", help="Output figure path")
    return parser.parse_args()


def infer_trace_path(dataframe: pd.DataFrame, trace_arg: str) -> Path | None:
    if trace_arg:
        return Path(trace_arg)
    if "run_id" not in dataframe.columns or dataframe["run_id"].nunique() != 1:
        return None
    run_id = str(dataframe["run_id"].iloc[0])
    candidate = PROJECT_ROOT / "logs" / "traces" / f"{run_id}.jsonl"
    return candidate if candidate.exists() else None


def build_condition_protocol_summary(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "condition",
                "glyph_reuse_rate",
                "same_context_glyph_consistency",
                "success_failure_glyph_divergence",
                "convention_persistence",
                "target_switch_after_glyph_rate",
                "post_comm_agreement_rate",
            ]
        )
    summaries: list[dict[str, object]] = []
    conditions = sorted({str(row.get("condition", "")) for row in rows if row.get("condition")})
    for condition in conditions:
        filtered = [row for row in rows if row.get("condition") == condition]
        metrics = compute_protocol_metrics(filtered)
        summaries.append({"condition": condition, **metrics})
    return pd.DataFrame(summaries)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    figure_path = Path(args.figure)
    if not input_path.exists():
        raise SystemExit(f"Results file not found: {input_path}")

    dataframe = pd.read_csv(input_path)
    if dataframe.empty:
        raise SystemExit("Results file is empty.")

    summary = (
        dataframe.groupby("condition")
        .agg(
            avg_team_reward=("team_reward", "mean"),
            success_rate=("outcome", lambda col: float((col == "high_value").mean())),
        )
        .reset_index()
    )
    agreement_rows = dataframe[
        dataframe["target_a"].isin(["LEFT", "RIGHT"]) & dataframe["target_b"].isin(["LEFT", "RIGHT"])
    ].copy()
    if agreement_rows.empty:
        agreement = {}
    else:
        agreement_rows["agreed"] = agreement_rows["target_a"] == agreement_rows["target_b"]
        agreement = agreement_rows.groupby("condition")["agreed"].mean().to_dict()
    summary["target_agreement_rate"] = summary["condition"].map(agreement).fillna(0.0)

    trace_path = infer_trace_path(dataframe, args.trace)
    if trace_path and trace_path.exists():
        protocol_summary = build_condition_protocol_summary(load_trace_rows(trace_path))
        if not protocol_summary.empty:
            summary = summary.merge(protocol_summary, on="condition", how="left")

    print(summary.to_string(index=False))

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    metric_specs = [
        ("avg_team_reward", "Average Team Reward"),
        ("success_rate", "Success Rate"),
        ("target_agreement_rate", "Target Agreement Rate"),
        ("glyph_reuse_rate", "Glyph Reuse Rate"),
        ("same_context_glyph_consistency", "Context Consistency"),
        ("convention_persistence", "Convention Persistence"),
    ]
    for axis, (column, title) in zip(axes.flatten(), metric_specs):
        if column not in summary.columns:
            axis.set_axis_off()
            continue
        axis.bar(summary["condition"], summary[column].fillna(0.0), color=["#0f766e", "#6b7280", "#b45309"])
        axis.set_title(title)
        axis.set_ylim(
            bottom=min(0.0, float(summary[column].fillna(0.0).min()) - 0.05),
            top=max(1.05, float(summary[column].fillna(0.0).max()) + 0.05),
        )
    plt.tight_layout()
    fig.savefig(figure_path)
    print(f"Saved figure to {figure_path}")
    if trace_path:
        print(f"Analyzed trace metrics from {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
