"""Run a short multi-condition smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment import main as run_experiment_main  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short smoke test across all conditions.")
    parser.add_argument("--model", required=True, help="Ollama model to use.")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_experiment_main(
        [
            "--episodes",
            "3",
            "--conditions",
            "comm",
            "silent",
            "random",
            "--model",
            args.model,
            "--base-url",
            args.base_url,
            "--output-csv",
            "logs/results.csv",
            "--output-jsonl",
            "logs/episodes.jsonl",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
