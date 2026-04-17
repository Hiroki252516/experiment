"""Analyze experiment logs for the ScoreG coordination study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ScoreG experiment outputs.")
    parser.add_argument("--input", default="logs/results.csv", help="Path to results.csv")
    parser.add_argument("--figure", default="logs/summary.png", help="Output figure path")
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
    print(summary.to_string(index=False))

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = [
        ("avg_team_reward", "Average Team Reward"),
        ("success_rate", "Success Rate"),
        ("target_agreement_rate", "Target Agreement Rate"),
    ]
    for axis, (column, title) in zip(axes, metrics):
        axis.bar(summary["condition"], summary[column], color=["#0f766e", "#6b7280", "#b45309"])
        axis.set_title(title)
        axis.set_ylim(bottom=min(0.0, float(summary[column].min()) - 0.05), top=max(1.05, float(summary[column].max()) + 0.05))
    plt.tight_layout()
    fig.savefig(figure_path)
    print(f"Saved figure to {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
