"""Miscellaneous helpers for the Streamlit viewer."""

from __future__ import annotations

import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def generate_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{timestamp}_{uuid.uuid4().hex[:8]}"


def format_timestamp(value: str) -> str:
    if not value:
        return "-"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def format_event_line(row: dict[str, Any]) -> str:
    phase = row.get("phase", "act")
    target_a = row.get("target_a_after", row.get("target_a"))
    target_b = row.get("target_b_after", row.get("target_b"))
    return (
        f"step {row.get('step')} [{phase}]: "
        f"A move={row.get('move_a')}, target={target_a} | "
        f"B move={row.get('move_b')}, target={target_b} | "
        f"reward={row.get('team_reward')} | outcome={row.get('outcome')}"
    )


def resolve_python_executable(project_root: Path) -> str:
    candidate = project_root / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def launcher_log_path(project_root: Path, run_id: str) -> Path:
    return project_root / "logs" / "runs" / run_id / "launcher.log"


def read_log_tail(log_path: str | Path, max_chars: int = 4000) -> str:
    path = Path(log_path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[-max_chars:]


def start_experiment_process(
    *,
    project_root: Path,
    model: str,
    episodes: int,
    conditions: list[str],
    base_url: str,
    seed: int,
    run_id: str,
) -> tuple[subprocess.Popen[Any], Path]:
    command = [
        resolve_python_executable(project_root),
        str(project_root / "scripts" / "run_experiment.py"),
        "--episodes",
        str(episodes),
        "--conditions",
        *conditions,
        "--model",
        model,
        "--base-url",
        base_url,
        "--seed",
        str(seed),
        "--run-id",
        run_id,
    ]
    log_path = launcher_log_path(project_root, run_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=project_root,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_handle.close()
    return process, log_path


def process_running(process: Any) -> bool:
    return process is not None and process.poll() is None
