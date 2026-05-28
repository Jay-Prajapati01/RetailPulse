from __future__ import annotations

import streamlit as st

from src.dashboard.components import render_page_shell, render_section_header
from src.dashboard.config import APP_TITLE, ARTIFACT_PATHS


def render_settings_page() -> None:
    render_page_shell("Settings", "Dashboard controls, environment notes, and refresh guidance.")
    render_section_header("Dashboard Settings")
    st.checkbox("Enable auto-refresh placeholder", value=False, help="Reserved for future live data refresh integration.")
    st.selectbox("Theme mode", ["Dark Enterprise", "Dark High-Contrast", "Dark Minimal"], index=0)
    st.number_input("Refresh interval (minutes)", min_value=5, max_value=120, value=30, step=5)

    render_section_header("Environment Notes")
    st.write(f"Application: {APP_TITLE}")
    st.write("The dashboard consumes files from `processed/`, `monitoring/drift_reports/`, and `graphify-out/`.")
    st.write("For development, keep the dashboard as a thin presentation layer over the reusable `src/` modules.")
    st.write(ARTIFACT_PATHS)
