from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.artifact_bootstrap import ensure_artifacts
from src.dashboard.components import render_download_frame, render_page_shell, render_section_header, get_chart_width_kwargs
from src.dashboard.data_access import load_churn_outputs
from src.dashboard.visuals import churn_feature_profile_chart, churn_probability_profile_chart, churn_risk_chart, customer_explanation_table, top_shap_features_chart
from src.dashboard.session import ensure_defaults


def render_churn_page() -> None:
    render_page_shell("Churn Analytics", "Retention risk, customer-level signals, and explainable drivers.")
    with st.spinner("Preparing churn artifacts..."):
        ensure_artifacts("churn")

    with st.spinner("Loading churn outputs..."):
        outputs = load_churn_outputs()
        predictions = outputs["predictions"]
        metrics = outputs["metrics"]
        shap_ranking = outputs["shap_ranking"]

    ensure_defaults({"selected_risk_band": "All", "actual_filter": "All"})

    if predictions.empty:
        st.error("Churn outputs could not be generated automatically. Check the pipeline dependencies.")
        return

    metric_map = metrics.iloc[0].to_dict() if not metrics.empty else {}
    cols = st.columns(4)
    cols[0].metric("ROC AUC", f"{metric_map.get('roc_auc', 0.0):.3f}")
    cols[1].metric("Precision", f"{metric_map.get('precision', 0.0):.3f}")
    cols[2].metric("Recall", f"{metric_map.get('recall', 0.0):.3f}")
    cols[3].metric("F1-score", f"{metric_map.get('f1_score', 0.0):.3f}")

    churn_probability = float(predictions["churn_probability"].mean()) if "churn_probability" in predictions.columns else 0.0
    retention_rate = 1.0 - churn_probability
    high_risk_share = float((predictions["risk_band"].astype(str).str.lower() == "high").mean()) if "risk_band" in predictions.columns else 0.0
    customer_value = float(predictions["total_revenue"].mean()) if "total_revenue" in predictions.columns else 0.0
    active_base = float((predictions["predicted_churn"] == 0).mean()) if "predicted_churn" in predictions.columns else 0.0
    insight_cols = st.columns(4)
    insight_cols[0].metric("Retention rate", f"{retention_rate:.2%}")
    insight_cols[1].metric("Churn risk", f"{churn_probability:.2%}")
    insight_cols[2].metric("Customer value", f"{customer_value:,.2f}")
    insight_cols[3].metric("Healthy customer share", f"{active_base:.2%}")

    if high_risk_share >= 0.4:
        st.error(f"High-risk customers make up {high_risk_share:.1%} of the base. Retention intervention recommended.")
    elif high_risk_share >= 0.2:
        st.warning(f"High-risk customers make up {high_risk_share:.1%} of the base. Watchlist should be reviewed.")
    else:
        st.success(f"High-risk customers remain contained at {high_risk_share:.1%} of the base.")

    risk_band_options = ["All"] + sorted(predictions["risk_band"].astype(str).dropna().unique().tolist()) if "risk_band" in predictions.columns else ["All"]
    filter_cols = st.columns(3)
    with filter_cols[0]:
        selected_risk_band = st.selectbox("Risk band", risk_band_options, key="selected_risk_band")
    with filter_cols[1]:
        probability_min, probability_max = st.slider("Churn probability range", 0.0, 1.0, (0.0, 1.0), 0.01, key="churn_probability_range")
    with filter_cols[2]:
        actual_filter = st.selectbox("Actual churn", ["All", "Churned", "Retained"], key="actual_filter")

    filtered = predictions.copy()
    if selected_risk_band != "All" and "risk_band" in filtered.columns:
        filtered = filtered[filtered["risk_band"].astype(str) == selected_risk_band]
    if "churn_probability" in filtered.columns:
        filtered = filtered[(filtered["churn_probability"] >= probability_min) & (filtered["churn_probability"] <= probability_max)]
    if actual_filter != "All" and "actual_churn" in filtered.columns:
        desired = 1 if actual_filter == "Churned" else 0
        filtered = filtered[filtered["actual_churn"] == desired]

    render_section_header("Churn Risk Distribution")
    width_kwargs = get_chart_width_kwargs()
    tab_risk, tab_profile, tab_shap = st.tabs(["Risk Distribution", "Trend Analysis", "Explainability"])
    with tab_risk:
        st.plotly_chart(churn_risk_chart(filtered), **width_kwargs)
    with tab_profile:
        st.plotly_chart(churn_probability_profile_chart(filtered), **width_kwargs)
    with tab_shap:
        st.plotly_chart(top_shap_features_chart(shap_ranking), **width_kwargs)

    render_section_header("Customer Drill-Down", "Select a row to inspect explainability and customer signals.")
    selectable = filtered.reset_index(drop=True)
    if selectable.empty:
        st.info("No customers match the selected filters.")
        return

    try:
        from src.dashboard.data_access import load_monitoring_summary
        from src.dashboard.components import render_alert_banner

        mon = load_monitoring_summary()
        mon_metrics = mon.get("metrics", {}) if isinstance(mon, dict) else {}
        max_score = max(float(mon_metrics.get("data_drift_score", 0.0)), float(mon_metrics.get("prediction_drift_score", 0.0)))
        if max_score >= 0.6:
            render_alert_banner("Monitoring critical: significant drift detected. Validation recommended.", level="error")
        elif max_score >= 0.3:
            render_alert_banner("Monitoring notice: drift signals present.", level="warning")
    except Exception:
        pass

    selected_index = st.selectbox("Customer row", options=list(range(len(selectable))), format_func=lambda idx: f"Customer {idx + 1}")
    selected_row = selectable.iloc[selected_index]
    drilldown_cols = st.columns(3)
    drilldown_cols[0].metric("Risk band", str(selected_row.get("risk_band", "Unknown")))
    drilldown_cols[1].metric("Churn probability", f"{float(selected_row.get('churn_probability', 0.0)):.2%}")
    drilldown_cols[2].metric("Customer lifetime value", f"{float(selected_row.get('customer_lifetime_value', selected_row.get('total_revenue', 0.0))):,.2f}")

    customer_features = [column for column in ["recency_days", "frequency", "total_revenue", "avg_unit_price", "avg_basket_size", "orders_per_day", "revenue_per_order", "units_per_order", "tenure_days", "purchase_span_days", "inactivity_duration", "avg_purchase_interval", "order_consistency", "customer_lifetime_value", "purchase_velocity", "monetary_per_day", "basket_value_stability"] if column in selected_row.index]
    if customer_features:
        st.plotly_chart(churn_feature_profile_chart(selected_row, customer_features), **width_kwargs)
    explanation = customer_explanation_table(selected_row, shap_ranking)
    st.dataframe(explanation, use_container_width=True)

    render_section_header("High-Risk Customers")
    high_risk = filtered.sort_values("churn_probability", ascending=False).head(20)
    st.dataframe(high_risk, use_container_width=True)
    render_download_frame(high_risk, "customer_churn_high_risk.csv", "Download high-risk customers")
