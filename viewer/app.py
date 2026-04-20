"""Streamlit app for realtime and replay visualization of ScoreG experiments."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from viewer.controls import (
    advance_playback,
    init_session_state,
    reset_run_filters,
    resolve_option_index,
    sync_selected_run,
)
from viewer.data import (
    available_conditions,
    available_episodes,
    compute_metrics,
    convention_hints,
    filter_rows,
    list_run_manifests,
    load_trace_rows,
    load_trace_tail_state,
    manifest_status_message,
    recent_glyph_history,
    recent_received_history,
)
from viewer.render import build_grid_html, glyph_history_arrays, glyph_rows_to_array
from viewer.utils import (
    format_event_line,
    format_timestamp,
    generate_run_id,
    process_running,
    read_log_tail,
    start_experiment_process,
)

RUNS_DIR = PROJECT_ROOT / "logs" / "runs"


def selected_manifest(manifests: list[dict[str, Any]]) -> dict[str, Any] | None:
    run_id = st.session_state.selected_run_id
    for manifest in manifests:
        if manifest.get("run_id") == run_id:
            return manifest
    return manifests[0] if manifests else None


def load_selected_rows(manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not manifest:
        return []
    trace_path = manifest.get("trace_path", "")
    if not trace_path:
        return []
    return load_trace_rows(trace_path)


def render_header(manifest: dict[str, Any] | None) -> None:
    run_id = manifest.get("run_id", st.session_state.pending_run_id or "-") if manifest else st.session_state.pending_run_id or "-"
    model = manifest.get("model", "-") if manifest else "-"
    st.title("ScoreG Realtime Viewer")
    st.caption(f"mode={st.session_state.mode} | run_id={run_id} | model={model}")


def render_launch_status(manifest: dict[str, Any] | None) -> None:
    if st.session_state.launch_error_message:
        st.error(st.session_state.launch_error_message)
    process_is_running = process_running(st.session_state.active_process)
    if process_is_running:
        st.info(f"Active subprocess is running for run_id={st.session_state.active_process_run_id}")
    elif st.session_state.active_process_run_id:
        return_code = st.session_state.active_process.poll() if st.session_state.active_process else None
        st.caption(
            f"Last launched run_id={st.session_state.active_process_run_id}, returncode={return_code}"
        )
    if manifest:
        status_message = manifest_status_message(manifest)
        if manifest.get("status") == "failed":
            st.error(status_message)
        elif manifest.get("status") in {"starting", "running"}:
            st.info(status_message)
        elif manifest.get("status") == "completed":
            st.success(status_message)
    if st.session_state.active_launcher_log_path:
        tail = read_log_tail(st.session_state.active_launcher_log_path)
        st.session_state.launch_output_tail = tail
        if tail:
            st.caption("Launcher output tail")
            st.code(tail, language="text")


def render_launch_panel(manifests: list[dict[str, Any]]) -> None:
    with st.expander("Launch View", expanded=False):
        process_is_running = process_running(st.session_state.active_process)
        with st.form("launch_form"):
            model = st.text_input("Model name", value="gemma3:1b")
            episodes = st.number_input("Episodes per condition", min_value=1, max_value=1000, value=5)
            comm_phase_steps = st.number_input("Comm-only steps", min_value=0, max_value=10, value=2)
            conditions = st.multiselect(
                "Conditions",
                options=["comm", "silent", "random"],
                default=["comm", "silent", "random"],
            )
            base_url = st.text_input("Base URL", value="http://localhost:11434")
            seed = st.number_input("Seed", min_value=0, value=42)
            randomize_positions = st.checkbox("Randomize positions", value=True)
            hard_split_prob = st.slider("Hard split probability", min_value=0.0, max_value=1.0, value=0.5)
            memory_budget = st.number_input("Memory budget", min_value=1, max_value=200, value=20)
            submitted = st.form_submit_button("Run experiment", disabled=process_is_running or not conditions)
        if submitted:
            run_id = generate_run_id()
            try:
                process, log_path = start_experiment_process(
                    project_root=PROJECT_ROOT,
                    model=model,
                    episodes=int(episodes),
                    conditions=list(conditions),
                    base_url=base_url,
                    seed=int(seed),
                    run_id=run_id,
                    comm_phase_steps=int(comm_phase_steps),
                    randomize_positions=bool(randomize_positions),
                    hard_split_prob=float(hard_split_prob),
                    memory_budget=int(memory_budget),
                )
            except OSError as exc:
                st.session_state.launch_error_message = f"Failed to start runner: {exc}"
            else:
                st.session_state.active_process = process
                st.session_state.active_process_run_id = run_id
                st.session_state.active_launcher_log_path = str(log_path)
                st.session_state.pending_run_id = run_id
                st.session_state.selected_run_id = run_id
                st.session_state.launch_error_message = ""
                st.session_state.mode = "live"
                reset_run_filters()
                sync_selected_run(manifests)
                st.rerun()

        manifest = selected_manifest(manifests)
        if st.session_state.pending_run_id and (not manifest or manifest.get("run_id") != st.session_state.pending_run_id):
            st.info(f"Pending run_id={st.session_state.pending_run_id}. Waiting for manifest to appear.")
        render_launch_status(manifest)


def render_control_panel(manifests: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    sync_selected_run(manifests)
    selected = selected_manifest(manifests)
    run_options = [manifest.get("run_id", "") for manifest in manifests]
    condition_options = available_conditions(rows)
    if condition_options and st.session_state.selected_condition not in condition_options:
        st.session_state.selected_condition = condition_options[0]
    if not condition_options:
        st.session_state.selected_condition = ""
    episode_options = available_episodes(rows, st.session_state.selected_condition or None)
    if episode_options and st.session_state.selected_episode not in episode_options:
        st.session_state.selected_episode = episode_options[0]
    if not episode_options:
        st.session_state.selected_episode = 0

    col1, col2, col3, col4 = st.columns([1.2, 1.3, 1.3, 1.2])
    with col1:
        st.session_state.mode = st.radio("Mode", options=["live", "replay"], horizontal=True)
    with col2:
        if run_options:
            next_run = st.selectbox(
                "Run",
                options=run_options,
                index=run_options.index(selected.get("run_id")) if selected else 0,
            )
            if next_run != st.session_state.selected_run_id:
                st.session_state.selected_run_id = next_run
                st.session_state.pending_run_id = ""
                reset_run_filters()
                st.rerun()
        else:
            st.selectbox("Run", options=[""], disabled=True)
    with col3:
        if condition_options:
            st.session_state.selected_condition = st.selectbox(
                "Condition",
                options=condition_options,
                index=resolve_option_index(condition_options, st.session_state.selected_condition),
            )
        else:
            st.selectbox("Condition", options=[""], disabled=True)
    with col4:
        if episode_options:
            st.session_state.selected_episode = st.selectbox(
                "Episode",
                options=episode_options,
                index=resolve_option_index(episode_options, st.session_state.selected_episode),
            )
        else:
            st.selectbox("Episode", options=[0], disabled=True)

    if st.session_state.mode == "replay" and episode_options:
        episode_rows = filter_rows(
            rows,
            condition=st.session_state.selected_condition or None,
            episode=st.session_state.selected_episode,
        )
        max_step = max(int(row.get("step", 0)) for row in episode_rows) if episode_rows else 0
        st.session_state.selected_step = st.slider(
            "Step",
            min_value=0,
            max_value=max_step,
            value=min(st.session_state.selected_step, max_step),
        )
        left, middle, right, speed_col = st.columns([1, 1, 1, 1.2])
        with left:
            if st.button("Prev"):
                st.session_state.selected_step = max(0, st.session_state.selected_step - 1)
        with middle:
            if st.button("Play/Pause"):
                st.session_state.playing = not st.session_state.playing
                st.session_state.last_playback_tick = 0.0
        with right:
            if st.button("Next"):
                st.session_state.selected_step = min(max_step, st.session_state.selected_step + 1)
        with speed_col:
            st.session_state.playback_speed = st.select_slider(
                "Speed",
                options=[0.5, 1.0, 2.0, 4.0],
                value=float(st.session_state.playback_speed),
            )
        advance_playback(max_step)

    render_launch_panel(manifests)


def render_metrics(metrics: dict[str, Any]) -> None:
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Episodes", metrics["episodes"])
    col2.metric("Success rate", f"{metrics['success_rate']:.2%}")
    col3.metric("Average reward", f"{metrics['average_reward']:.3f}")
    col4.metric("Target agreement", f"{metrics['target_agreement_rate']:.2%}")
    col5.metric("Glyph reuse", f"{metrics['glyph_reuse_consistency']:.2%}")
    col6.metric("Glyph-target assoc", f"{metrics['glyph_target_association']:.2%}")
    st.caption(f"Target flip rate: {metrics['target_flip_rate']:.2%}")
    st.caption(f"Outcome breakdown: {metrics['outcome_breakdown']}")


def render_agent_panel(
    agent_name: str,
    frame: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    sent_key: str,
    received_key: str,
    move_key: str,
    target_key: str,
    raw_key: str,
    guard_applied_key: str,
    guard_reason_key: str,
    private_value: int,
) -> None:
    st.subheader(agent_name)
    guard_applied = bool(frame.get(guard_applied_key, False))
    guard_reason = frame.get(guard_reason_key, "")
    initial_target_key = "initial_target_a" if agent_name == "agent_a" else "initial_target_b"
    final_target_key = "final_target_a" if agent_name == "agent_a" else "final_target_b"
    target_changed_key = "target_changed_a" if agent_name == "agent_a" else "target_changed_b"
    st.write(
        {
            "phase": frame.get("phase", ""),
            "position": frame.get("agent_a_pos" if agent_name == "agent_a" else "agent_b_pos"),
            "move": frame.get(move_key, ""),
            "target": frame.get(target_key, ""),
            "initial_target": frame.get(initial_target_key, "UNKNOWN"),
            "final_target": frame.get(final_target_key, frame.get(target_key, "")),
            "target_changed": bool(frame.get(target_changed_key, False)),
            "private_value": private_value,
            "guard_applied": guard_applied,
            "guard_reason": guard_reason or "",
            "error": frame.get("error_message", ""),
        }
    )
    sent_col, received_col = st.columns(2)
    with sent_col:
        st.caption("Sent glyph")
        st.image(glyph_rows_to_array(frame.get(sent_key, ["0" * 7] * 7)), clamp=True)
    with received_col:
        st.caption("Received glyph")
        st.image(glyph_rows_to_array(frame.get(received_key, ["0" * 7] * 7)), clamp=True)
    sent_history = glyph_history_arrays(recent_glyph_history(rows, agent_name=agent_name))
    received_history = glyph_history_arrays(recent_received_history(rows, agent_name=agent_name))
    if sent_history:
        st.caption("Sent glyph history")
        history_cols = st.columns(len(sent_history))
        for column, image in zip(history_cols, sent_history):
            column.image(image, clamp=True)
    if received_history:
        st.caption("Received glyph history")
        history_cols = st.columns(len(received_history))
        for column, image in zip(history_cols, received_history):
            column.image(image, clamp=True)
    st.caption("Raw JSON output")
    st.code(frame.get(raw_key, ""), language="json")


def render_panels(frame: dict[str, Any], rows: list[dict[str, Any]], manifest: dict[str, Any] | None) -> None:
    status_col, grid_col = st.columns([1.2, 1.3])
    with status_col:
        st.subheader("Live / Replay Status")
        st.write(
            {
                "condition": frame.get("condition", manifest.get("current_condition", "-") if manifest else "-"),
                "episode": frame.get("episode", manifest.get("current_episode", "-") if manifest else "-"),
                "step": frame.get("step", manifest.get("last_step", "-") if manifest else "-"),
                "phase": frame.get("phase", "-"),
                "movement_enabled": frame.get("movement_enabled", "-"),
                "cumulative_reward": frame.get("cumulative_team_reward", frame.get("team_reward", 0.0)),
                "outcome": frame.get("outcome", "-"),
                "run_status": manifest.get("status", "-") if manifest else "-",
                "started_at": format_timestamp(manifest.get("started_at", "")) if manifest else "-",
                "completed_at": format_timestamp(manifest.get("completed_at", "")) if manifest else "-",
            }
        )
        render_metrics(compute_metrics(rows))
    with grid_col:
        st.subheader("Gridworld Panel")
        st.markdown(build_grid_html(frame), unsafe_allow_html=True)

    agent_a_col, agent_b_col = st.columns(2)
    with agent_a_col:
        render_agent_panel(
            "agent_a",
            frame,
            rows,
            sent_key="glyph_a_sent",
            received_key="glyph_a_received",
            move_key="move_a",
            target_key="target_a",
            raw_key="raw_a",
            guard_applied_key="guard_a_applied",
            guard_reason_key="guard_a_reason",
            private_value=int(frame.get("value_left", 0)),
        )
    with agent_b_col:
        render_agent_panel(
            "agent_b",
            frame,
            rows,
            sent_key="glyph_b_sent",
            received_key="glyph_b_received",
            move_key="move_b",
            target_key="target_b",
            raw_key="raw_b",
            guard_applied_key="guard_b_applied",
            guard_reason_key="guard_b_reason",
            private_value=int(frame.get("value_right", 0)),
        )

    st.subheader("Convention Hints Panel")
    hints = convention_hints(rows, condition=str(frame.get("condition", "")) or None, limit=3)
    if hints:
        for hint in hints:
            hint_cols = st.columns([1, 3])
            with hint_cols[0]:
                if hint.get("glyph_rows"):
                    st.image(glyph_rows_to_array(hint["glyph_rows"], scale=10), clamp=True)
            with hint_cols[1]:
                st.write(
                    {
                        "count": hint.get("count", 0),
                        "dominant_target": hint.get("dominant_target", "UNKNOWN"),
                        "success_count": hint.get("success_count", 0),
                    }
                )
    else:
        st.caption("No convention hints available yet.")

    st.subheader("Timeline / Event Log Panel")
    for row in rows[-10:]:
        st.text(format_event_line(row))


def render_live_section() -> None:
    @st.fragment(run_every="1s")
    def live_fragment() -> None:
        manifests = list_run_manifests(RUNS_DIR)
        sync_selected_run(manifests)
        manifest = selected_manifest(manifests)
        rows = load_selected_rows(manifest)
        tail_state = load_trace_tail_state(
            manifest=manifest,
            trace_offsets=dict(st.session_state.trace_offsets),
        )
        st.session_state.trace_offsets = tail_state["trace_offsets"]

        if st.session_state.selected_condition:
            rows = filter_rows(rows, condition=st.session_state.selected_condition)
        if st.session_state.selected_episode in available_episodes(rows):
            rows = filter_rows(rows, episode=st.session_state.selected_episode)

        frame = rows[-1] if rows else {}
        if not frame:
            if manifest:
                status_message = manifest_status_message(manifest)
                if manifest.get("status") == "failed":
                    st.error(status_message)
                elif manifest.get("status") == "completed":
                    st.warning(status_message)
                else:
                    st.info(status_message)
            else:
                st.info("No live trace available yet. Start a run from Launch View or wait for logs.")
            render_launch_status(manifest)
            return
        render_panels(frame, rows, manifest)
        render_launch_status(manifest)

    live_fragment()


def render_replay_section(manifest: dict[str, Any] | None, rows: list[dict[str, Any]]) -> None:
    filtered = filter_rows(
        rows,
        condition=st.session_state.selected_condition or None,
        episode=st.session_state.selected_episode,
    )
    if not filtered:
        if manifest:
            st.info(manifest_status_message(manifest))
        else:
            st.info("No replay frames available for the selected run and episode.")
        return
    step_map = {int(row.get("step", 0)): row for row in filtered}
    frame = step_map.get(st.session_state.selected_step, filtered[-1])
    render_panels(frame, filtered, manifest)
    render_launch_status(manifest)


def main() -> None:
    st.set_page_config(page_title="ScoreG Realtime Viewer", layout="wide")
    init_session_state()
    manifests = list_run_manifests(RUNS_DIR)
    sync_selected_run(manifests)
    manifest = selected_manifest(manifests)
    rows = load_selected_rows(manifest)

    render_header(manifest)
    render_control_panel(manifests, rows)

    manifest = selected_manifest(list_run_manifests(RUNS_DIR))
    rows = load_selected_rows(manifest)
    if st.session_state.mode == "live":
        render_live_section()
    else:
        render_replay_section(manifest, rows)


if __name__ == "__main__":
    main()
