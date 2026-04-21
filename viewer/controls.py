"""Session-state helpers for the Streamlit viewer."""

from __future__ import annotations

from time import monotonic
from typing import Any

import streamlit as st


DEFAULT_STATE: dict[str, Any] = {
    "mode": "live",
    "selected_run_id": "",
    "pending_run_id": "",
    "selected_condition": "",
    "selected_episode": 0,
    "selected_step": 0,
    "playing": False,
    "playback_speed": 1.0,
    "active_process": None,
    "active_process_run_id": "",
    "active_launcher_log_path": "",
    "launch_error_message": "",
    "launch_output_tail": "",
    "last_playback_tick": 0.0,
    "trace_offsets": {},
    "glyph_animation_key": "",
    "glyph_animation_started_at": 0.0,
}


def init_session_state() -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sync_selected_run(manifests: list[dict[str, Any]]) -> None:
    run_ids = [manifest.get("run_id", "") for manifest in manifests]
    pending_run_id = st.session_state.pending_run_id
    if pending_run_id and pending_run_id not in run_ids:
        st.session_state.selected_run_id = pending_run_id
        return
    if pending_run_id and pending_run_id in run_ids:
        st.session_state.selected_run_id = pending_run_id
        st.session_state.pending_run_id = ""
        return
    if run_ids and st.session_state.selected_run_id not in run_ids:
        st.session_state.selected_run_id = run_ids[0]
    if not run_ids:
        st.session_state.selected_run_id = pending_run_id or ""


def reset_run_filters() -> None:
    st.session_state.selected_condition = ""
    st.session_state.selected_episode = 0
    st.session_state.selected_step = 0
    st.session_state.playing = False
    st.session_state.last_playback_tick = 0.0
    st.session_state.glyph_animation_key = ""
    st.session_state.glyph_animation_started_at = 0.0


def resolve_option_index(options: list[Any], value: Any, default: int = 0) -> int:
    if not options:
        return default
    try:
        return options.index(value)
    except ValueError:
        return default


def advance_playback(max_index: int) -> None:
    if not st.session_state.playing:
        return
    now = monotonic()
    last_tick = float(st.session_state.last_playback_tick or 0.0)
    if last_tick == 0.0:
        st.session_state.last_playback_tick = now
        return
    interval = max(0.2, 1.0 / max(float(st.session_state.playback_speed), 0.25))
    if now - last_tick < interval:
        return
    st.session_state.last_playback_tick = now
    if st.session_state.selected_step < max_index:
        st.session_state.selected_step += 1
    else:
        st.session_state.playing = False
