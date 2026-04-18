"""Streamlit app for realtime and replay visualization of ScoreG experiments."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from viewer.controls import advance_playback, init_session_state, sync_selected_run
from viewer.data import (
    available_conditions,
    available_episodes,
    compute_metrics,
    filter_rows,
    list_run_manifests,
    load_trace_rows,
    tail_jsonl,
)
from viewer.render import build_grid_html, glyph_rows_to_array
from viewer.utils import format_event_line, format_timestamp, generate_run_id, process_running, start_experiment_process

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
    run_id = manifest.get("run_id", "-") if manifest else "-"
    model = manifest.get("model", "-") if manifest else "-"
    st.title("ScoreG Realtime Viewer")
    st.caption(
        f"mode={st.session_state.mode} | run_id={run_id} | model={model}"
    )


def render_launch_panel() -> None:
    with st.expander("Launch View", expanded=False):
        process_is_running = process_running(st.session_state.active_process)
        with st.form("launch_form"):
            model = st.text_input("Model name", value="gemma3:1b")
            episodes = st.number_input("Episodes per condition", min_value=1, max_value=1000, value=5)
            conditions = st.multiselect(
                "Conditions",
                options=["comm", "silent", "random"],
                default=["comm", "silent", "random"],
            )
            base_url = st.text_input("Base URL", value="http://localhost:11434")
            seed = st.number_input("Seed", min_value=0, value=42)
            submitted = st.form_submit_button("Run experiment", disabled=process_is_running or not conditions)
        if submitted:
            run_id = generate_run_id()
            process = start_experiment_process(
                project_root=PROJECT_ROOT,
                model=model,
                episodes=int(episodes),
                conditions=list(conditions),
                base_url=base_url,
                seed=int(seed),
                run_id=run_id,
            )
            st.session_state.active_process = process
            st.session_state.active_process_run_id = run_id
            st.session_state.selected_run_id = run_id
            st.session_state.mode = "live"
            st.rerun()
        if process_is_running:
            st.info(f"Active subprocess is running for run_id={st.session_state.active_process_run_id}")
        elif st.session_state.active_process_run_id:
            return_code = st.session_state.active_process.poll()
            st.caption(
                f"Last launched run_id={st.session_state.active_process_run_id}, returncode={return_code}"
            )


def render_control_panel(manifests: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    sync_selected_run(manifests)
    selected = selected_manifest(manifests)
    run_options = [manifest.get("run_id", "") for manifest in manifests]
    condition_options = available_conditions(rows)
    if condition_options and st.session_state.selected_condition not in condition_options:
        st.session_state.selected_condition = condition_options[0]
    episode_options = available_episodes(rows, st.session_state.selected_condition or None)
    if episode_options and st.session_state.selected_episode not in episode_options:
        st.session_state.selected_episode = episode_options[0]

    col1, col2, col3, col4 = st.columns([1.2, 1.3, 1.3, 1.2])
    with col1:
        st.session_state.mode = st.radio("Mode", options=["live", "replay"], horizontal=True)
    with col2:
        if run_options:
            st.session_state.selected_run_id = st.selectbox(
                "Run",
                options=run_options,
                index=run_options.index(selected.get("run_id")) if selected else 0,
            )
        else:
            st.selectbox("Run", options=[""], disabled=True)
    with col3:
        if condition_options:
            st.session_state.selected_condition = st.selectbox(
                "Condition",
                options=condition_options,
                index=condition_options.index(st.session_state.selected_condition),
            )
        else:
            st.selectbox("Condition", options=[""], disabled=True)
    with col4:
        if episode_options:
            st.session_state.selected_episode = st.selectbox(
                "Episode",
                options=episode_options,
                index=episode_options.index(st.session_state.selected_episode),
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
        st.session_state.selected_step = st.slider("Step", min_value=0, max_value=max_step, value=min(st.session_state.selected_step, max_step))
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
            st.session_state.playback_speed = st.select_slider("Speed", options=[0.5, 1.0, 2.0, 4.0], value=float(st.session_state.playback_speed))
        advance_playback(max_step)

    render_launch_panel()


def render_metrics(metrics: dict[str, Any]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Episodes", metrics["episodes"])
    col2.metric("Success rate", f"{metrics['success_rate']:.2%}")
    col3.metric("Average reward", f"{metrics['average_reward']:.3f}")
    col4.metric("Target agreement", f"{metrics['target_agreement_rate']:.2%}")
    st.caption(f"Outcome breakdown: {metrics['outcome_breakdown']}")


def render_agent_panel(agent_name: str, frame: dict[str, Any], *, sent_key: str, received_key: str, move_key: str, target_key: str, raw_key: str, private_value: int) -> None:
    st.subheader(agent_name)
    st.write(
        {
            "position": frame.get("agent_a_pos" if agent_name == "agent_a" else "agent_b_pos"),
            "move": frame.get(move_key, ""),
            "target": frame.get(target_key, ""),
            "private_value": private_value,
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
    st.caption("Raw JSON output")
    st.code(frame.get(raw_key, ""), language="json")


def render_panels(frame: dict[str, Any], rows: list[dict[str, Any]], manifest: dict[str, Any] | None) -> None:
    status_col, grid_col = st.columns([1.2, 1.3])
    with status_col:
        st.subheader("Live / Replay Status")
        st.write(
            {
                "condition": frame.get("condition", "-"),
                "episode": frame.get("episode", "-"),
                "step": frame.get("step", "-"),
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
            sent_key="glyph_a_sent",
            received_key="glyph_a_received",
            move_key="move_a",
            target_key="target_a",
            raw_key="raw_a",
            private_value=int(frame.get("value_left", 0)),
        )
    with agent_b_col:
        render_agent_panel(
            "agent_b",
            frame,
            sent_key="glyph_b_sent",
            received_key="glyph_b_received",
            move_key="move_b",
            target_key="target_b",
            raw_key="raw_b",
            private_value=int(frame.get("value_right", 0)),
        )

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
        if st.session_state.selected_condition:
            rows = filter_rows(rows, condition=st.session_state.selected_condition)
        if st.session_state.selected_episode in available_episodes(rows):
            rows = filter_rows(rows, episode=st.session_state.selected_episode)
        if manifest and manifest.get("trace_path"):
            trace_path = manifest["trace_path"]
            offset_map = dict(st.session_state.trace_offsets)
            trace_offset = int(offset_map.get(trace_path, 0))
            _, next_offset = tail_jsonl(trace_path, trace_offset)
            offset_map[trace_path] = next_offset
            st.session_state.trace_offsets = offset_map
        frame = rows[-1] if rows else {}
        if not frame:
            st.info("No live trace available yet. Start a run from Launch View or wait for logs.")
            return
        render_panels(frame, rows, manifest)

    live_fragment()


def render_replay_section(manifest: dict[str, Any] | None, rows: list[dict[str, Any]]) -> None:
    filtered = filter_rows(
        rows,
        condition=st.session_state.selected_condition or None,
        episode=st.session_state.selected_episode,
    )
    if not filtered:
        st.info("No replay frames available for the selected run and episode.")
        return
    step_map = {int(row.get("step", 0)): row for row in filtered}
    frame = step_map.get(st.session_state.selected_step, filtered[-1])
    render_panels(frame, filtered, manifest)


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
