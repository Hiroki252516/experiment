from pathlib import Path

from viewer.data import manifest_status_message
from viewer.utils import launcher_log_path, read_log_tail


def resolve_selected_run_id(run_ids: list[str], selected_run_id: str, pending_run_id: str) -> str:
    if pending_run_id and pending_run_id not in run_ids:
        return pending_run_id
    if pending_run_id and pending_run_id in run_ids:
        return pending_run_id
    if run_ids and selected_run_id not in run_ids:
        return run_ids[0]
    return selected_run_id if run_ids else pending_run_id


def test_pending_run_is_not_replaced_before_manifest_exists() -> None:
    resolved = resolve_selected_run_id(
        run_ids=["run_old"],
        selected_run_id="run_new",
        pending_run_id="run_new",
    )
    assert resolved == "run_new"


def test_pending_run_is_selected_when_manifest_appears() -> None:
    resolved = resolve_selected_run_id(
        run_ids=["run_old", "run_new"],
        selected_run_id="run_old",
        pending_run_id="run_new",
    )
    assert resolved == "run_new"


def test_launcher_log_tail_and_failed_message(tmp_path: Path) -> None:
    log_path = launcher_log_path(tmp_path, "run_x")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("line1\nline2\nfailure details\n", encoding="utf-8")
    assert "failure details" in read_log_tail(log_path)
    assert manifest_status_message({"status": "failed", "last_error_message": "boom"}) == "boom"
