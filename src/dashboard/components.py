from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard.config import APP_SUBTITLE, APP_TITLE, PAGE_INDEX, PAGES
from src.dashboard.data_access import load_monitoring_summary


def apply_branding() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        .rp-surface {
            background: linear-gradient(135deg, rgba(56,189,248,0.10), rgba(52,211,153,0.06));
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }
        .rp-card {
            background: rgba(17,24,39,0.95);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 12px 30px rgba(0,0,0,0.18);
        }
        .rp-muted { color: #9ca3af; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(active_page: str) -> str:
    st.sidebar.markdown(f"## {APP_TITLE}")
    st.sidebar.caption(APP_SUBTITLE)
    # show light monitoring status in the sidebar
    try:
        mon = load_monitoring_summary()
        mon_metrics = mon.get("metrics", {}) if isinstance(mon, dict) else {}
        data_drift = float(mon_metrics.get("data_drift_score", 0.0))
        prediction_drift = float(mon_metrics.get("prediction_drift_score", 0.0))
        severity = max(data_drift, prediction_drift)
        if severity >= 0.6:
            status_md = "<span style='color:#ef4444'>&#9679;</span> Monitoring: Critical"
        elif severity >= 0.3:
            status_md = "<span style='color:#f59e0b'>&#9679;</span> Monitoring: Warning"
        else:
            status_md = "<span style='color:#10b981'>&#9679;</span> Monitoring: Healthy"
        st.sidebar.markdown(status_md, unsafe_allow_html=True)
    except Exception:
        st.sidebar.markdown("<span style='color:#9ca3af'>&#9679;</span> Monitoring: Unknown", unsafe_allow_html=True)

    labels = [f"{page.icon}  {page.title}" for page in PAGES]
    selected = st.sidebar.radio("Navigation", labels, index=max(next((i for i, page in enumerate(PAGES) if page.key == active_page), 0), 0))
    st.sidebar.markdown("---")
    st.sidebar.caption("Graphify-backed repository intelligence is the onboarding layer for this dashboard.")
    # provide a quick refresh control to clear cached artifacts and rerun
    if st.sidebar.button("Refresh artifacts"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.rerun()
    selected_index = labels.index(selected)
    return PAGES[selected_index].key


def render_page_shell(title: str, subtitle: str, help_text: str | None = None) -> None:
    """Render a consistent page shell with title, subtitle and optional help text."""
    help_md = f"<div style='margin-top:6px;color:#9ca3af;font-size:0.92rem'>{help_text}</div>" if help_text else ""
    st.markdown(
        f'<div class="rp-surface"><div style="display:flex;justify-content:space-between;align-items:center"><div><h1 style="margin-bottom:0;">{title}</h1><div class="rp-muted">{subtitle}</div>{help_md}</div></div></div>',
        unsafe_allow_html=True,
    )


def render_kpis(metrics: dict[str, float], items: list[tuple[str, str, str | None]] | None = None) -> None:
    if items is None:
        items = [(key.replace("_", " ").title(), f"{value:,.2f}", None) for key, value in metrics.items()]
    # Render KPI cards in a consistent enterprise style
    columns = st.columns(min(4, max(len(items), 1)))
    for idx, (label, value, delta) in enumerate(items):
        with columns[idx % len(columns)]:
            st.markdown("<div class='rp-card'>", unsafe_allow_html=True)
            st.metric(label, value, delta)
            st.markdown("</div>", unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def render_loading_state(message: str) -> None:
    st.info(message)


def render_skeleton(width: int = 12, lines: int = 4) -> None:
    """Render a simple skeleton placeholder for loading content."""
    for _ in range(lines):
        st.markdown("<div style='background:rgba(255,255,255,0.03);height:14px;margin-bottom:8px;border-radius:4px'></div>", unsafe_allow_html=True)


def render_alert_banner(message: str, level: str = "info") -> None:
    """Render a horizontal alert banner at the top of a page.

    level: 'info'|'warning'|'error' maps to colors
    """
    color = "#0ea5a4" if level == "info" else ("#f59e0b" if level == "warning" else "#ef4444")
    st.markdown(
        f"<div style='background:{color};padding:10px;border-radius:8px;color:#0b1220;margin-bottom:12px'><strong>{message}</strong></div>",
        unsafe_allow_html=True,
    )


def render_error_state(message: str) -> None:
    st.error(message)


def render_download_frame(frame: pd.DataFrame, file_name: str, label: str) -> None:
    csv_data = frame.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv_data, file_name=file_name, mime="text/csv")


def render_artifact_links(artifacts: dict[str, Path]) -> None:
    for label, path in artifacts.items():
        if path.exists():
            st.write(f"- {label}: `{path}`")


def get_chart_width_kwargs() -> dict[str, any]:
    """Resolve chart width arguments dynamically based on Streamlit version to avoid deprecation warnings."""
    try:
        parts = []
        for p in st.__version__.split('.'):
            digits = ''.join(c for c in p if c.isdigit())
            if digits:
                parts.append(int(digits))
            else:
                parts.append(0)
        if parts >= [1, 50, 0]:
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}

