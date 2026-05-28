from __future__ import annotations

import streamlit as st

from src.dashboard.components import apply_branding, render_sidebar, render_error_state
from src.dashboard.config import PAGE_INDEX
from src.dashboard.pages import (
    render_churn_page,
    render_forecasting_page,
    render_home_page,
    render_inventory_page,
    render_monitoring_page,
    render_reports_page,
    render_segmentation_page,
    render_settings_page,
)


PAGE_RENDERERS = {
    "home": render_home_page,
    "forecasting": render_forecasting_page,
    "segmentation": render_segmentation_page,
    "churn": render_churn_page,
    "inventory": render_inventory_page,
    "monitoring": render_monitoring_page,
    "reports": render_reports_page,
    "settings": render_settings_page,
}


def run_app() -> None:
    apply_branding()

    initial_page = st.session_state.get("active_dashboard_page", "home")
    active_page = render_sidebar(initial_page)
    st.session_state["active_dashboard_page"] = active_page

    renderer = PAGE_RENDERERS.get(active_page, render_home_page)
    try:
        renderer()
    except Exception as exc:  # top-level protection so the dashboard never crashes
        render_error_state(f"Failed to render page '{active_page}': {exc}")


if __name__ == "__main__":
    run_app()
