"""Session-state helpers for the Streamlit viewer."""

from __future__ import annotations

from time import monotonic
from typing import Any

import streamlit as st


DEFAULT_STATE: dict[str, Any] = {
    "mode": "live",
    "selected_run_id": "",
    "selected_condition": "",
    "selected_episode": 0,
    "selected_step": 0,
    "playing": False,
    "playback_speed": 1.0,
    "active_process": None,
    "active_process_run_id": "",
    "last_playback_tick": 0.0,
    "trace_offsets": {},
}


def init_session_state() -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sync_selected_run(manifests: list[dict[str, Any]]) -> None:
    run_ids = [manifest.get("run_id", "") for manifest in manifests]
    if run_ids and st.session_state.selected_run_id not in run_ids:
        st.session_state.selected_run_id = run_ids[0]
    if not run_ids:
        st.session_state.selected_run_id = ""


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
