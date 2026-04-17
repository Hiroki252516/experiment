"""Check whether the local Ollama setup is ready for experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.ollama_agent import (  # noqa: E402
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    check_ollama_setup,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Ollama CLI, API, and model availability.")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Model name to verify.")
    parser.add_argument("--base-url", default=DEFAULT_OLLAMA_BASE_URL, help="Ollama base URL.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status = check_ollama_setup(model=args.model, base_url=args.base_url)

    print(f"CLI available : {status.cli_available}")
    print(f"API reachable : {status.api_reachable}")
    print(f"Model present : {status.model_available}")
    if status.available_models:
        print(f"Available models: {', '.join(status.available_models)}")

    if status.ok:
        print("Ollama is ready.")
        return 0

    print(status.guidance(args.model))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
