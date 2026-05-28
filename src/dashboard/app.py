from __future__ import annotations

import importlib
import streamlit as st

from src.dashboard.components import apply_branding, render_sidebar, render_error_state
from src.dashboard.config import PAGE_INDEX


PAGE_MODULES: dict[str, str] = {
    "home": "src.dashboard.pages.home",
    "forecasting": "src.dashboard.pages.forecasting",
    "segmentation": "src.dashboard.pages.segmentation",
    "churn": "src.dashboard.pages.churn",
    "inventory": "src.dashboard.pages.inventory",
    "monitoring": "src.dashboard.pages.monitoring",
    "reports": "src.dashboard.pages.reports",
    "settings": "src.dashboard.pages.settings",
}


def _get_renderer_for(page_key: str):
    """Dynamically import the page module and return its renderer function.

    This avoids importing all page modules at startup, keeping Streamlit
    startup fast for lightweight deployments.
    """
    module_name = PAGE_MODULES.get(page_key, PAGE_MODULES.get("home"))
    try:
        module = importlib.import_module(module_name)
        func_name = f"render_{page_key}_page"
        return getattr(module, func_name)
    except Exception:
        # Fallback to importing the home renderer directly if something fails
        module = importlib.import_module(PAGE_MODULES["home"])
        return getattr(module, "render_home_page")


def run_app() -> None:
    apply_branding()

    initial_page = st.session_state.get("active_dashboard_page", "home")
    active_page = render_sidebar(initial_page)
    st.session_state["active_dashboard_page"] = active_page
    renderer = _get_renderer_for(active_page)
    try:
        renderer()
    except Exception as exc:  # top-level protection so the dashboard never crashes
        render_error_state(f"Failed to render page '{active_page}': {exc}")


if __name__ == "__main__":
    run_app()
