"""Analyze experiment logs for the ScoreG coordination study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis.glyph_metrics import (
    compute_glyph_reuse_consistency,
    compute_glyph_target_association,
    compute_target_flip_rate,
    load_trace_rows_for_run_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ScoreG experiment outputs.")
    parser.add_argument("--input", default="logs/results.csv", help="Path to results.csv")
    parser.add_argument("--figure", default="logs/summary.png", help="Output figure path")
    parser.add_argument("--trace-dir", default="logs/traces", help="Directory containing per-run trace JSONL files")
    return parser.parse_args()


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

    trace_rows: list[dict[str, object]] = []
    if "run_id" in dataframe.columns:
        trace_rows = load_trace_rows_for_run_ids(
            [str(run_id) for run_id in dataframe["run_id"].dropna().unique().tolist()],
            args.trace_dir,
        )
    reuse = compute_glyph_reuse_consistency(trace_rows)
    association = compute_glyph_target_association(trace_rows)
    target_flip = compute_target_flip_rate(trace_rows)
    summary["glyph_reuse_consistency"] = summary["condition"].map(reuse).fillna(0.0)
    summary["glyph_target_association"] = summary["condition"].map(association).fillna(0.0)
    summary["target_flip_rate"] = summary["condition"].map(target_flip).fillna(0.0)
    baseline_reward = (
        float(summary.loc[summary["condition"].isin(["silent", "random"]), "avg_team_reward"].max())
        if summary["condition"].isin(["silent", "random"]).any()
        else 0.0
    )
    summary["communication_gain"] = summary.apply(
        lambda row: float(row["avg_team_reward"] - baseline_reward) if row["condition"] == "comm" else 0.0,
        axis=1,
    )
    print(summary.to_string(index=False))

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    metrics = [
        ("avg_team_reward", "Average Team Reward"),
        ("success_rate", "Success Rate"),
        ("target_agreement_rate", "Target Agreement Rate"),
        ("glyph_reuse_consistency", "Glyph Reuse Consistency"),
        ("glyph_target_association", "Glyph-Target Association"),
        ("target_flip_rate", "Target Flip Rate"),
    ]
    for axis, (column, title) in zip(axes.flatten(), metrics):
        axis.bar(summary["condition"], summary[column], color=["#0f766e", "#6b7280", "#b45309"])
        axis.set_title(title)
        axis.set_ylim(bottom=min(0.0, float(summary[column].min()) - 0.05), top=max(1.05, float(summary[column].max()) + 0.05))
    plt.tight_layout()
    fig.savefig(figure_path)
    print(f"Saved figure to {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
