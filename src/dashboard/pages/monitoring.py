from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from src.dashboard.components import render_page_shell, render_section_header, get_chart_width_kwargs
from src.dashboard.data_access import load_monitoring_summary, load_mlflow_summary
from src.dashboard.visuals import drift_breakdown_chart, drift_summary_chart, health_gauge


def render_monitoring_page() -> None:
    render_page_shell("Monitoring", "Drift detection, model reliability, and behavior-shift visibility.")

    with st.spinner("Loading monitoring artifacts..."):
        mon_summary = load_monitoring_summary()
        summary = mon_summary.get("metrics", {}) if isinstance(mon_summary, dict) else {}
        drift_summary = mon_summary.get("drift_summary", {}) if isinstance(mon_summary, dict) else {}

    if not summary:
        st.error("Monitoring outputs could not be generated automatically. Check the pipeline dependencies.")
        return

    data_drift = float(summary.get("data_drift_score", 0.0))
    prediction_drift = float(summary.get("prediction_drift_score", 0.0))
    reliability = float(summary.get("model_reliability_score", 0.0))
    risk = float(summary.get("forecasting_degradation_risk", 0.0))
    behavior_shift = float(summary.get("behavior_shift_proxy", 0.0))

    banner_cols = st.columns(4)
    banner_cols[0].metric("Data drift", f"{data_drift:.3f}")
    banner_cols[1].metric("Prediction drift", f"{prediction_drift:.3f}")
    banner_cols[2].metric("Reliability", f"{reliability:.3f}")
    banner_cols[3].metric("Risk", f"{risk:.3f}")

    width_kwargs = get_chart_width_kwargs()
    health_cols = st.columns(3)
    health_cols[0].plotly_chart(health_gauge(data_drift, "Data drift"), **width_kwargs)
    health_cols[1].plotly_chart(health_gauge(prediction_drift, "Prediction drift"), **width_kwargs)
    health_cols[2].plotly_chart(health_gauge(risk, "Degradation risk"), **width_kwargs)

    if risk >= 0.5 or data_drift >= 0.5:
        st.error("Monitoring indicates elevated degradation risk. Review drift and pipeline health immediately.")
    elif risk >= 0.25 or data_drift >= 0.25:
        st.warning("Monitoring shows emerging drift patterns. Retain watchlist status.")
    else:
        st.success("Monitoring health is within acceptable operating bounds.")

    render_section_header("Drift Summary")
    tab1, tab2, tab3 = st.tabs(["Health Score", "Feature Shifts", "Pipeline Status"])
    with tab1:
        st.plotly_chart(drift_summary_chart(summary), **width_kwargs)
    with tab2:
        st.plotly_chart(drift_breakdown_chart(drift_summary), **width_kwargs)
    with tab3:
        status_frame = st.columns(4)
        status_frame[0].metric("Data drift", f"{data_drift:.3f}")
        status_frame[1].metric("Prediction drift", f"{prediction_drift:.3f}")
        status_frame[2].metric("Behavior shift", f"{behavior_shift:.3f}")
        status_frame[3].metric("Reliability", f"{reliability:.3f}")

        mlflow_summary = load_mlflow_summary()
        if mlflow_summary:
            st.text_area("MLflow health summary", mlflow_summary[:5000], height=280)

    render_section_header("Monitoring Dashboard")
    tabs_html = st.tabs(["Unified dashboard", "Data drift report", "Prediction drift report"])
    with tabs_html[0]:
        from src.dashboard.data_access import load_monitoring_reports
        reports = load_monitoring_reports()
        dashboard_html = reports.get("html")
        if dashboard_html:
            components.html(dashboard_html, height=700, scrolling=True)
        else:
            st.info("Monitoring HTML dashboard not found.")
    with tabs_html[1]:
        from src.dashboard.data_access import load_monitoring_reports
        reports = load_monitoring_reports()
        data_drift_html = reports.get("data_drift_html")
        if data_drift_html:
            components.html(data_drift_html, height=700, scrolling=True)
    with tabs_html[2]:
        from src.dashboard.data_access import load_monitoring_reports
        reports = load_monitoring_reports()
        pred_drift_html = reports.get("prediction_drift_html")
        if pred_drift_html:
            components.html(pred_drift_html, height=700, scrolling=True)
