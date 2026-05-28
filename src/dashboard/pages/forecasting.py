from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.artifact_bootstrap import ensure_artifacts
from src.dashboard.components import render_download_frame, render_page_shell, render_section_header, get_chart_width_kwargs
from src.dashboard.data_access import load_forecast_comparison, load_future_forecast
from src.dashboard.forecasting_simulation import run_what_if
from src.dashboard.visuals import forecast_band_figure, forecast_comparison_figure, product_level_forecast_figure, what_if_chart
from src.dashboard.session import ensure_defaults


def render_forecasting_page() -> None:
    render_page_shell("Demand Forecasting", "Interactive forecast intelligence with historical comparison and what-if simulation.")

    with st.spinner("Preparing forecast artifacts..."):
        ensure_artifacts("forecasting")

    with st.spinner("Loading forecast artifacts..."):
        forecast = load_forecast_comparison()
        future = load_future_forecast()

    ensure_defaults({
        "sales_growth_pct": 10,
        "seasonality_multiplier": 1.0,
        "inventory_capacity_multiplier": 1.0,
        "demand_shift_pct": 0,
    })

    if forecast.empty or future.empty:
        st.error("Forecast artifacts could not be generated automatically. Check the pipeline dependencies.")
        return

    # monitoring alert banner
    try:
        from src.dashboard.data_access import load_monitoring_summary
        from src.dashboard.components import render_alert_banner

        mon = load_monitoring_summary()
        mon_metrics = mon.get("metrics", {}) if isinstance(mon, dict) else {}
        max_score = max(float(mon_metrics.get("data_drift_score", 0.0)), float(mon_metrics.get("prediction_drift_score", 0.0)))
        if max_score >= 0.6:
            render_alert_banner("Monitoring critical: significant drift detected. Check Monitoring page.", level="error")
        elif max_score >= 0.3:
            render_alert_banner("Monitoring warning: emerging drift signals.", level="warning")
    except Exception:
        pass

    render_section_header("Forecast Controls", "Use the sliders to simulate business scenarios.")
    controls = st.columns(4)
    with controls[0]:
        sales_growth_pct = st.slider("Sales growth %", -30, 60, key="sales_growth_pct", step=1)
    with controls[1]:
        seasonality_multiplier = st.slider("Seasonality multiplier", 0.5, 2.5, key="seasonality_multiplier", step=0.05)
    with controls[2]:
        inventory_capacity_multiplier = st.slider("Inventory capacity multiplier", 0.5, 3.0, key="inventory_capacity_multiplier", step=0.05)
    with controls[3]:
        demand_shift_pct = st.slider("Demand shift %", -25, 50, key="demand_shift_pct", step=1)

    scenario_frame = run_what_if(future, sales_growth_pct, seasonality_multiplier, inventory_capacity_multiplier, demand_shift_pct)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Forecast rows", f"{len(forecast):,}")
    metric_cols[1].metric("Future horizon", f"{len(future):,} days")
    metric_cols[2].metric("Scenario demand", f"{scenario_frame['what_if_quantity'].sum():,.0f}")
    metric_cols[3].metric("Inventory pressure", f"{scenario_frame['inventory_pressure'].mean():.2f}")

    width_kwargs = get_chart_width_kwargs()
    st.plotly_chart(forecast_comparison_figure(forecast), **width_kwargs)
    st.plotly_chart(forecast_band_figure(future), **width_kwargs)

    render_section_header("What-if Simulation", "Interactive scenario output across the forecast horizon.")
    st.plotly_chart(what_if_chart(scenario_frame), **width_kwargs)

    render_section_header("Product-Level Forecast View", "Forecast horizon summarized for executive review.")
    st.plotly_chart(product_level_forecast_figure(future), **width_kwargs)

    render_section_header("Forecast Export")
    render_download_frame(scenario_frame, "retailpulse_forecast_scenario.csv", "Download scenario forecast CSV")
