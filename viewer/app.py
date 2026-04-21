"""Streamlit app for glyph-first realtime and replay visualization."""

from __future__ import annotations

import sys
from pathlib import Path
from time import monotonic
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
    adjacent_glyph_event_step,
    available_conditions,
    available_episodes,
    build_convention_hints,
    compute_metrics,
    filter_rows,
    glyph_history_rows,
    list_run_manifests,
    load_trace_rows,
    load_trace_tail_state,
    manifest_status_message,
    previous_episode_row,
)
from viewer.render import build_grid_html, glyph_rows_text, glyph_rows_to_array
from viewer.utils import (
    format_event_line,
    format_timestamp,
    generate_run_id,
    process_running,
    read_log_tail,
    start_experiment_process,
)

RUNS_DIR = PROJECT_ROOT / "logs" / "runs"
ZERO_GLYPH_ROWS = ["0" * 7 for _ in range(7)]
ANIMATION_INTERVAL_S = 0.4
ANIMATION_TOTAL_S = 1.2


def selected_manifest(manifests: list[dict[str, Any]]) -> dict[str, Any] | None:
    run_id = st.session_state.selected_run_id
    for manifest in manifests:
        if manifest.get("run_id") == run_id:
            return manifest
    return manifests[0] if manifests else None


def load_selected_rows(manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not manifest or not manifest.get("trace_path"):
        return []
    return load_trace_rows(manifest["trace_path"])


def render_header(manifest: dict[str, Any] | None) -> None:
    run_id = manifest.get("run_id", st.session_state.pending_run_id or "-") if manifest else st.session_state.pending_run_id or "-"
    model = manifest.get("model", "-") if manifest else "-"
    st.title("ScoreG Realtime Viewer")
    st.caption(f"mode={st.session_state.mode} | run_id={run_id} | model={model}")


def animation_key_for_frame(frame: dict[str, Any]) -> str:
    return ":".join(
        [
            str(frame.get("run_id", "")),
            str(frame.get("condition", "")),
            str(frame.get("episode", "")),
            str(frame.get("step", "")),
            str(frame.get("glyph_a_hash", "")),
            str(frame.get("glyph_b_hash", "")),
        ]
    )


def current_animation_mode(frame: dict[str, Any]) -> str:
    if not frame:
        return "current"
    key = animation_key_for_frame(frame)
    now = monotonic()
    if st.session_state.glyph_animation_key != key:
        st.session_state.glyph_animation_key = key
        st.session_state.glyph_animation_started_at = now
        return "previous" if bool(frame.get("glyph_event", False)) else "current"
    if not bool(frame.get("glyph_event", False)):
        return "current"
    elapsed = now - float(st.session_state.glyph_animation_started_at or 0.0)
    if elapsed < ANIMATION_INTERVAL_S:
        return "previous"
    if elapsed < ANIMATION_INTERVAL_S * 2:
        return "diff"
    return "current"


def render_launch_status(manifest: dict[str, Any] | None) -> None:
    if st.session_state.launch_error_message:
        st.error(st.session_state.launch_error_message)
    process_is_running = process_running(st.session_state.active_process)
    if process_is_running:
        st.info(f"Active subprocess is running for run_id={st.session_state.active_process_run_id}")
    elif st.session_state.active_process_run_id:
        return_code = st.session_state.active_process.poll() if st.session_state.active_process else None
        st.caption(f"Last launched run_id={st.session_state.active_process_run_id}, returncode={return_code}")
    if manifest:
        status_message = manifest_status_message(manifest)
        status = manifest.get("status")
        if status == "failed":
            st.error(status_message)
        elif status in {"starting", "running"}:
            st.info(status_message)
        elif status == "completed":
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
            conditions = st.multiselect(
                "Conditions",
                options=["comm", "silent", "random"],
                default=["comm", "silent", "random"],
            )
            base_url = st.text_input("Base URL", value="http://localhost:11434")
            agent_timeout = st.number_input(
                "Agent timeout (s)",
                min_value=10.0,
                max_value=600.0,
                value=120.0,
                step=10.0,
            )
            seed = st.number_input("Seed", min_value=0, value=42)
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
                    agent_timeout=float(agent_timeout),
                    seed=int(seed),
                    run_id=run_id,
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


def phase_label(phase: str) -> str:
    if phase == "comm_only":
        return "COMMUNICATION PHASE"
    return "ACTION PHASE"


def agent_suffix(agent_name: str) -> str:
    return "a" if agent_name == "agent_a" else "b"


def target_transition(frame: dict[str, Any], agent_name: str) -> str:
    suffix = agent_suffix(agent_name)
    before = str(frame.get(f"target_{suffix}_before", frame.get(f"target_{suffix}", "UNKNOWN")))
    after = str(frame.get(f"target_{suffix}_after", frame.get(f"target_{suffix}", "UNKNOWN")))
    return f"{before} -> {after}" if before != after else after


def render_glyph_card(
    *,
    title: str,
    rows: list[str],
    role: str,
    width: int,
    subtitle: str = "",
    detail_lines: list[str] | None = None,
    show_rows: bool = False,
    mode: str = "current",
    previous_rows: list[str] | None = None,
    badges: list[str] | None = None,
) -> None:
    st.caption(title)
    if badges:
        st.caption(" | ".join(badges))
    st.image(
        glyph_rows_to_array(
            rows,
            scale=20 if width >= 220 else 10,
            role=role,
            mode=mode,
            previous_rows=previous_rows,
        ),
        width=width,
        clamp=True,
    )
    if subtitle:
        st.caption(subtitle)
    for line in detail_lines or []:
        st.caption(line)
    if show_rows:
        st.code(glyph_rows_text(rows), language="text")


def glyph_badges(frame: dict[str, Any], suffix: str) -> list[str]:
    badges: list[str] = []
    if bool(frame.get(f"glyph_{suffix}_zero", False)):
        badges.append("zero glyph")
    if bool(frame.get(f"glyph_{suffix}_reused_from_success", False)):
        badges.append("reused success")
    if bool(frame.get(f"glyph_{suffix}_changed", False)):
        badges.append("glyph event")
    same_streak = int(frame.get(f"glyph_{suffix}_same_streak", 1))
    delta_pixels = int(frame.get(f"glyph_{suffix}_delta_pixels", 0))
    badges.append(f"delta={delta_pixels}")
    badges.append(f"same x{same_streak}")
    return badges


def render_target_switches(frame: dict[str, Any]) -> None:
    switches: list[str] = []
    if frame.get("target_a_changed"):
        switches.append(f"agent_a target switched {target_transition(frame, 'agent_a')}")
    if frame.get("target_b_changed"):
        switches.append(f"agent_b target switched {target_transition(frame, 'agent_b')}")
    if switches:
        for switch in switches:
            st.warning(switch)


def render_glyph_theater(frame: dict[str, Any], previous_frame: dict[str, Any] | None) -> None:
    st.subheader("Glyph Theater")
    phase = str(frame.get("phase", "act"))
    animation_mode = current_animation_mode(frame)
    previous_frame = previous_frame or {}
    left, right = st.columns(2)
    with left:
        render_glyph_card(
            title="agent_a sent",
            rows=frame.get("glyph_a_sent", ZERO_GLYPH_ROWS),
            role="a_sent",
            width=260,
            subtitle=f"step {frame.get('step', '-')} | {phase_label(phase)}",
            detail_lines=[
                f"move={frame.get('move_a', '-')}",
                f"target={target_transition(frame, 'agent_a')}",
                f"guard={frame.get('guard_a_reason', '') or 'none'}",
            ],
            mode=animation_mode,
            previous_rows=previous_frame.get("glyph_a_sent"),
            badges=glyph_badges(frame, "a"),
        )
    with right:
        render_glyph_card(
            title="agent_b received",
            rows=frame.get("glyph_b_received", ZERO_GLYPH_ROWS),
            role="b_received",
            width=260,
            subtitle=f"agent_b sees A's glyph | {frame.get('glyph_exchange_label', '')}",
            detail_lines=[
                f"move={frame.get('move_b', '-')}",
                f"target={target_transition(frame, 'agent_b')}",
                f"outcome={frame.get('outcome', '-')}",
            ],
            mode=animation_mode,
            previous_rows=previous_frame.get("glyph_b_received"),
            badges=glyph_badges(frame, "a"),
        )
    left, right = st.columns(2)
    with left:
        render_glyph_card(
            title="agent_b sent",
            rows=frame.get("glyph_b_sent", ZERO_GLYPH_ROWS),
            role="b_sent",
            width=260,
            subtitle=f"step {frame.get('step', '-')} | {frame.get('glyph_exchange_label', '')}",
            detail_lines=[
                f"move={frame.get('move_b', '-')}",
                f"target={target_transition(frame, 'agent_b')}",
                f"reused_success={bool(frame.get('glyph_b_reused_from_success', False))}",
            ],
            mode=animation_mode,
            previous_rows=previous_frame.get("glyph_b_sent"),
            badges=glyph_badges(frame, "b"),
        )
    with right:
        render_glyph_card(
            title="agent_a received",
            rows=frame.get("glyph_a_received", ZERO_GLYPH_ROWS),
            role="a_received",
            width=260,
            subtitle="agent_a sees B's glyph",
            detail_lines=[
                f"move={frame.get('move_a', '-')}",
                f"target={target_transition(frame, 'agent_a')}",
                f"reused_success={bool(frame.get('glyph_a_reused_from_success', False))}",
            ],
            mode=animation_mode,
            previous_rows=previous_frame.get("glyph_a_received"),
            badges=glyph_badges(frame, "b"),
        )
    render_target_switches(frame)


def render_glyph_history_strip(rows: list[dict[str, Any]], frame: dict[str, Any]) -> None:
    st.subheader("Glyph History Strip")
    history = glyph_history_rows(rows, int(frame.get("step", 0)), limit=8)
    if not history:
        st.info("waiting for first glyph frame")
        return
    columns = st.columns(len(history))
    current_frame_step = int(frame.get("step", -1))
    for column, history_row in zip(columns, history):
        with column:
            step = int(history_row.get("step", 0))
            label = f"step {step}"
            if step == current_frame_step:
                label += " | current"
            st.caption(label)
            if history_row.get("glyph_event"):
                st.caption("glyph event")
            st.caption(
                f"A delta={int(history_row.get('glyph_a_delta_pixels', 0))} | same x{int(history_row.get('glyph_a_same_streak', 1))}"
            )
            st.image(
                glyph_rows_to_array(history_row.get("glyph_a_sent", ZERO_GLYPH_ROWS), scale=8, role="a_sent"),
                width=88,
                clamp=True,
            )
            st.caption(
                f"B delta={int(history_row.get('glyph_b_delta_pixels', 0))} | same x{int(history_row.get('glyph_b_same_streak', 1))}"
            )
            st.image(
                glyph_rows_to_array(history_row.get("glyph_b_sent", ZERO_GLYPH_ROWS), scale=8, role="b_sent"),
                width=88,
                clamp=True,
            )
            st.caption(f"A {history_row.get('target_a_after', history_row.get('target_a', '-'))}")
            st.caption(f"B {history_row.get('target_b_after', history_row.get('target_b', '-'))}")
            if bool(history_row.get("glyph_a_zero", False)) and bool(history_row.get("glyph_b_zero", False)):
                st.caption("all-zero exchange")


def render_communication_timeline(rows: list[dict[str, Any]]) -> None:
    st.subheader("Communication Timeline")
    if not rows:
        st.info("No trace rows available yet.")
        return
    for row in rows[-12:]:
        st.text(format_event_line(row))


def render_metrics(metrics: dict[str, Any]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Episodes", metrics["episodes"])
    col2.metric("Success rate", f"{metrics['success_rate']:.2%}")
    col3.metric("Average reward", f"{metrics['average_reward']:.3f}")
    col4.metric("Target agreement", f"{metrics['target_agreement_rate']:.2%}")
    st.caption(f"Outcome breakdown: {metrics['outcome_breakdown']}")
    proto_col1, proto_col2, proto_col3 = st.columns(3)
    proto_col1.metric("Glyph reuse", f"{metrics['glyph_reuse_rate']:.2%}")
    proto_col2.metric("Context consistency", f"{metrics['same_context_glyph_consistency']:.2%}")
    proto_col3.metric("Convention persistence", f"{metrics['convention_persistence']:.2%}")
    st.caption(
        "Proto-language metrics: "
        f"post_comm_agreement={metrics['post_comm_agreement_rate']:.2%}, "
        f"target_switch_after_glyph={metrics['target_switch_after_glyph_rate']:.2%}, "
        f"success_failure_divergence={metrics['success_failure_glyph_divergence']:.2%}"
    )


def render_status_summary(frame: dict[str, Any], metrics: dict[str, Any], manifest: dict[str, Any] | None) -> None:
    st.subheader("Run Summary")
    st.write(
        {
            "condition": frame.get("condition", manifest.get("current_condition", "-") if manifest else "-"),
            "episode": frame.get("episode", manifest.get("current_episode", "-") if manifest else "-"),
            "step": frame.get("step", manifest.get("last_step", "-") if manifest else "-"),
            "phase": frame.get("phase", manifest.get("current_phase", "-") if manifest else "-"),
            "phase_turn_index": frame.get("phase_turn_index", "-"),
            "act_step": frame.get("act_step", "-"),
            "team_reward": frame.get("team_reward", 0.0),
            "cumulative_team_reward": frame.get("cumulative_team_reward", 0.0),
            "run_status": manifest.get("status", "-") if manifest else "-",
            "started_at": format_timestamp(manifest.get("started_at", "")) if manifest else "-",
            "completed_at": format_timestamp(manifest.get("completed_at", "")) if manifest else "-",
        }
    )
    render_metrics(metrics)


def render_agent_panel(agent_name: str, frame: dict[str, Any]) -> None:
    suffix = agent_suffix(agent_name)
    st.subheader(agent_name)
    private_value = int(frame.get("value_left", 0)) if agent_name == "agent_a" else int(frame.get("value_right", 0))
    st.write(
        {
            "position": frame.get(f"agent_{suffix}_pos"),
            "phase": frame.get("phase", "act"),
            "move": frame.get(f"move_{suffix}", ""),
            "target": target_transition(frame, agent_name),
            "target_changed": bool(frame.get(f"target_{suffix}_changed", False)),
            "private_value": private_value,
            "glyph_reused_from_success": bool(frame.get(f"glyph_{suffix}_reused_from_success", False)),
            "glyph_changed": bool(frame.get(f"glyph_{suffix}_changed", False)),
            "glyph_zero": bool(frame.get(f"glyph_{suffix}_zero", False)),
            "glyph_delta_pixels": int(frame.get(f"glyph_{suffix}_delta_pixels", 0)),
            "glyph_same_streak": int(frame.get(f"glyph_{suffix}_same_streak", 1)),
            "guard_applied": bool(frame.get(f"guard_{suffix}_applied", False)),
            "guard_reason": frame.get(f"guard_{suffix}_reason", "") or "",
            "error": frame.get("error_message", ""),
        }
    )
    sent_col, received_col = st.columns(2)
    with sent_col:
        render_glyph_card(
            title="Sent glyph",
            rows=frame.get(f"glyph_{suffix}_sent", ZERO_GLYPH_ROWS),
            role=f"{suffix}_sent",
            width=112,
            show_rows=False,
        )
    with received_col:
        peer_suffix = "b" if suffix == "a" else "a"
        render_glyph_card(
            title="Received glyph",
            rows=frame.get(f"glyph_{suffix}_received", ZERO_GLYPH_ROWS),
            role=f"{peer_suffix}_received",
            width=112,
            show_rows=False,
        )
    st.caption("Raw JSON output")
    st.code(frame.get(f"raw_{suffix}", ""), language="json")


def render_convention_hints(rows: list[dict[str, Any]]) -> None:
    hints = build_convention_hints(rows)
    st.subheader("Convention Hints")
    left, right = st.columns(2)
    with left:
        st.caption("Recent successful glyphs")
        if not hints["recent_successes"]:
            st.info("No successful communication episodes yet.")
        for entry in hints["recent_successes"]:
            cols = st.columns([0.55, 1.45])
            with cols[0]:
                render_glyph_card(
                    title=f"{entry['agent_name']} episode {entry['episode']}",
                    rows=entry["glyph_rows"],
                    role="hint",
                    width=92,
                )
            with cols[1]:
                st.caption(
                    f"condition={entry['condition']} | known_value={entry['my_known_value']} | final_target={entry['final_target']}"
                )
    with right:
        st.caption("Frequent glyph by context")
        if not hints["frequent_contexts"]:
            st.info("No repeated communication context yet.")
        for item in hints["frequent_contexts"]:
            agent_name, known_value, final_target = item["context"]
            cols = st.columns([0.55, 1.45])
            with cols[0]:
                render_glyph_card(
                    title=agent_name,
                    rows=item["dominant_rows"],
                    role="hint",
                    width=92,
                )
            with cols[1]:
                st.caption(
                    f"known_value={known_value} | final_target={final_target} | dominant_share={item['dominant_share']:.2%} | samples={item['samples']}"
                )


def render_run_notes(frame: dict[str, Any]) -> None:
    condition = str(frame.get("condition", ""))
    both_zero = bool(frame.get("glyph_a_zero", False)) and bool(frame.get("glyph_b_zero", False))
    if condition == "silent":
        st.info("silent condition: all-zero glyph is expected and intentional.")
    elif condition == "comm" and both_zero:
        st.warning("comm condition is currently showing zero-signal collapse: both agents emitted all-zero glyphs.")


def render_main_panels(frame: dict[str, Any], rows: list[dict[str, Any]], manifest: dict[str, Any] | None) -> None:
    previous_frame = previous_episode_row(rows, int(frame.get("step", 0)))
    render_glyph_theater(frame, previous_frame)
    render_run_notes(frame)
    render_glyph_history_strip(rows, frame)
    render_communication_timeline(rows)

    status_col, agent_a_col, agent_b_col = st.columns([1.0, 1.1, 1.1])
    with status_col:
        render_status_summary(frame, compute_metrics(rows), manifest)
        st.subheader("Gridworld Panel")
        st.markdown(build_grid_html(frame), unsafe_allow_html=True)
    with agent_a_col:
        render_agent_panel("agent_a", frame)
    with agent_b_col:
        render_agent_panel("agent_b", frame)
    render_convention_hints(rows)


def render_live_section() -> None:
    @st.fragment(run_every="400ms")
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
        episode_options = available_episodes(rows, st.session_state.selected_condition or None)
        if episode_options and st.session_state.selected_episode in episode_options:
            rows = filter_rows(rows, episode=st.session_state.selected_episode)

        frame = rows[-1] if rows else {}
        if not frame:
            if manifest:
                status = manifest.get("status")
                if status == "failed":
                    st.error(manifest_status_message(manifest))
                elif status == "completed":
                    st.warning(manifest_status_message(manifest))
                else:
                    st.info("waiting for first glyph frame")
                    if manifest.get("current_phase") == "comm_only":
                        st.caption("movement is paused, glyph exchange is active")
            else:
                st.info("No live trace available yet. Start a run from Launch View or wait for logs.")
            render_launch_status(manifest)
            return

        render_main_panels(frame, rows, manifest)
        render_launch_status(manifest)

    live_fragment()


def render_replay_section(manifest: dict[str, Any] | None, rows: list[dict[str, Any]]) -> None:
    @st.fragment(run_every="400ms")
    def replay_fragment() -> None:
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
        render_main_panels(frame, filtered, manifest)
        render_launch_status(manifest)

    replay_fragment()


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

    col1, col2, col3, col4 = st.columns([1.1, 1.4, 1.2, 1.1])
    with col1:
        st.session_state.mode = st.radio("Mode", options=["live", "replay"], horizontal=True)
    with col2:
        if run_options:
            next_run = st.selectbox(
                "Run",
                options=run_options,
                index=resolve_option_index(
                    run_options,
                    selected.get("run_id") if selected else st.session_state.selected_run_id,
                ),
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
        left, middle, right, prev_event_col, next_event_col, speed_col = st.columns([1, 1, 1, 1.1, 1.1, 1.2])
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
        with prev_event_col:
            if st.button("Prev glyph event"):
                event_step = adjacent_glyph_event_step(episode_rows, st.session_state.selected_step, -1)
                if event_step is not None:
                    st.session_state.selected_step = event_step
        with next_event_col:
            if st.button("Next glyph event"):
                event_step = adjacent_glyph_event_step(episode_rows, st.session_state.selected_step, 1)
                if event_step is not None:
                    st.session_state.selected_step = event_step
        with speed_col:
            st.session_state.playback_speed = st.select_slider(
                "Speed",
                options=[0.5, 1.0, 2.0, 4.0],
                value=float(st.session_state.playback_speed),
            )
        advance_playback(max_step)

    render_launch_panel(manifests)


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
