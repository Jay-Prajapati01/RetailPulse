from __future__ import annotations

import streamlit as st

from src.dashboard.components import render_kpis, render_page_shell, render_section_header, get_chart_width_kwargs
from src.dashboard.config import ARTIFACT_PATHS
from src.dashboard.data_access import load_dashboard_summary, load_mlflow_summary, load_ensemble_metrics, load_monitoring_summary
from src.dashboard.reporting import build_pdf_report
from src.dashboard.visuals import health_gauge


def render_home_page() -> None:
    summary = load_dashboard_summary()
    render_page_shell("Executive Overview", "Unified operating view across forecasting, churn, inventory, monitoring, and experiment tracking.")

    # Quick monitoring banner and mini health panel
    try:
        mon = load_monitoring_summary()
        mon_metrics = mon.get("metrics", {}) if isinstance(mon, dict) else {}
        data_drift = float(mon_metrics.get("data_drift_score", 0.0))
        prediction_drift = float(mon_metrics.get("prediction_drift_score", 0.0))
        reliability = float(mon_metrics.get("model_reliability_score", 0.0))
        max_score = max(data_drift, prediction_drift)
        if max_score >= 0.6:
            st.error("Monitoring alert: Elevated drift detected. Review the Monitoring page for details.")
        elif max_score >= 0.3:
            st.warning("Monitoring notice: Emerging drift signals detected.")
        else:
            st.success("Monitoring OK: No significant drift detected.")

        width_kwargs = get_chart_width_kwargs()
        cols = st.columns(3)
        cols[0].plotly_chart(health_gauge(data_drift, "Data drift"), **width_kwargs)
        cols[1].plotly_chart(health_gauge(prediction_drift, "Prediction drift"), **width_kwargs)
        cols[2].plotly_chart(health_gauge(reliability, "Model reliability"), **width_kwargs)
        # Lazy-load monitoring dashboard HTML only on button click (don't block startup)
        if st.button("View monitoring dashboard (loads on demand)"):
            st.session_state["show_monitoring_html"] = True
        if st.session_state.get("show_monitoring_html", False):
            with st.spinner("Loading monitoring dashboard..."):
                try:
                    from src.dashboard.data_access import load_monitoring_reports

                    reports = load_monitoring_reports()
                    html = reports.get("html")
                    if html:
                        import streamlit.components.v1 as components

                        components.html(html, height=420, scrolling=True)
                    else:
                        st.info("Monitoring HTML dashboard not found.")
                except Exception as e:
                    st.error(f"Failed to load dashboard: {e}")
    except Exception:
        st.info("Monitoring status unavailable.")
    render_kpis(
        summary,
        [
            ("Ensemble RMSE", f"{summary.get('ensemble_rmse', 0.0):,.2f}", None),
            ("Churn ROC AUC", f"{summary.get('churn_roc_auc', 0.0):.3f}", None),
            ("Inventory Alerts", f"{summary.get('alert_count', 0)}", None),
            ("Data Drift", f"{summary.get('data_drift_score', 0.0):.3f}", None),
        ],
    )

    col1, col2 = st.columns([1.3, 0.9])
    with col1:
        render_section_header("Platform Summary", "The dashboard is wired to the latest generated RetailPulse artifacts.")
        st.markdown(
            """
            <div class='rp-card'>
            <ul>
                <li>Forecasting, churn, inventory, and monitoring outputs are loaded from processed artifacts.</li>
                <li>Graphify-backed onboarding is the first reference layer for the project.</li>
                <li>Use the page navigation to jump straight to business-specific views.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        render_section_header("Latest MLflow Summary")
        summary_text = load_mlflow_summary()
        if summary_text:
            st.text_area("Tracking summary", summary_text[:4000], height=260)
        else:
            st.info("MLflow summary artifact not found yet.")

    render_section_header("Operational Shortcuts")
    st.caption("Fast links to the most important artifacts.")
    st.write(ARTIFACT_PATHS)

    # Executive PDF export
    if st.button("Download Executive Summary PDF"):
        ensemble = load_ensemble_metrics()
        summary = load_dashboard_summary()
        lines = [f"Ensemble RMSE: {summary.get('ensemble_rmse', 0.0):,.2f}", f"Churn ROC AUC: {summary.get('churn_roc_auc', 0.0):.3f}", f"Inventory Alerts: {summary.get('alert_count', 0)}"]
        tables = [("Ensemble metrics (sample)", ensemble)] if not ensemble.empty else None
        pdf_bytes = build_pdf_report("RetailPulse Executive Summary", lines, tables=tables)
        st.download_button("Download PDF", data=pdf_bytes, file_name="retailpulse_executive_summary.pdf", mime="application/pdf")
