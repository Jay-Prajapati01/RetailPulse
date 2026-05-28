from __future__ import annotations

import streamlit as st

from src.dashboard.components import render_download_frame, render_page_shell, render_section_header, get_chart_width_kwargs
from src.dashboard.data_access import load_inventory_outputs
from src.dashboard.visuals import inventory_heatmap_chart, inventory_risk_chart, reorder_timeline_chart
from src.dashboard.session import ensure_defaults


def render_inventory_page() -> None:
    render_page_shell("Inventory Optimization", "Reorder intelligence and stock health for retail operations.")

    with st.spinner("Loading inventory outputs..."):
        outputs = load_inventory_outputs()
        recommendations = outputs["recommendations"]
        alerts = outputs["alerts"]
        metrics = outputs["metrics"]
    if recommendations.empty:
        st.error("Inventory outputs could not be generated automatically. Check the pipeline dependencies.")
        return

    metric_row = metrics.iloc[0].to_dict() if not metrics.empty else {}
    cols = st.columns(4)
    cols[0].metric("SKUs", f"{len(recommendations):,}")
    cols[1].metric("Alerts", f"{len(alerts):,}")
    cols[2].metric("Avg days of cover", f"{metric_row.get('avg_days_of_cover', 0.0):.2f}")
    cols[3].metric("Avg order qty", f"{metric_row.get('avg_order_qty', 0.0):.2f}")

    operational_cols = st.columns(4)
    understock_count = int(metric_row.get("understock_count", 0))
    overstock_count = int(metric_row.get("overstock_count", 0))
    reorder_urgency = understock_count / max(len(recommendations), 1)
    turnover_proxy = float((recommendations["forecast_units_30d"] / recommendations["current_stock"].replace(0, 1)).mean()) if {"forecast_units_30d", "current_stock"}.issubset(recommendations.columns) else 0.0
    operational_cols[0].metric("Understock risk", f"{understock_count}")
    operational_cols[1].metric("Overstock risk", f"{overstock_count}")
    operational_cols[2].metric("Reorder urgency", f"{reorder_urgency:.2%}")
    operational_cols[3].metric("Inventory turnover", f"{turnover_proxy:.2f}")

    if understock_count > 0:
        st.warning(f"{understock_count} items are below reorder policy and require attention.")
    if overstock_count > 0:
        st.info(f"{overstock_count} items are flagged for possible overstock review.")

    ensure_defaults({"selected_product": "All", "selected_risk": "All", "cover_range": (0.0, float(recommendations["days_of_cover"].max()) if "days_of_cover" in recommendations.columns and not recommendations.empty else 0.0), "stock_search": ""})

    product_options = ["All"] + sorted(recommendations["StockCode"].astype(str).unique().tolist()) if "StockCode" in recommendations.columns else ["All"]
    risk_options = ["All"] + sorted(recommendations["risk_flag"].astype(str).dropna().unique().tolist()) if "risk_flag" in recommendations.columns else ["All"]
    filter_cols = st.columns(4)
    with filter_cols[0]:
        selected_product = st.selectbox("Product", product_options, key="selected_product")
    with filter_cols[1]:
        selected_risk = st.selectbox("Risk flag", risk_options, key="selected_risk")
    with filter_cols[2]:
        max_cover = float(recommendations["days_of_cover"].max()) if "days_of_cover" in recommendations.columns and not recommendations.empty else 0.0
        cover_min, cover_max = st.slider("Days of cover", 0.0, max_cover, (0.0, max_cover), 0.5, key="cover_range")
    with filter_cols[3]:
        stock_search = st.text_input("Search description", "", key="stock_search")

    filtered = recommendations.copy()
    if selected_product != "All" and "StockCode" in filtered.columns:
        filtered = filtered[filtered["StockCode"].astype(str) == selected_product]
    if selected_risk != "All" and "risk_flag" in filtered.columns:
        filtered = filtered[filtered["risk_flag"].astype(str) == selected_risk]
    if "days_of_cover" in filtered.columns:
        filtered = filtered[(filtered["days_of_cover"] >= cover_min) & (filtered["days_of_cover"] <= cover_max)]
    if stock_search and "Description" in filtered.columns:
        filtered = filtered[filtered["Description"].astype(str).str.contains(stock_search, case=False, na=False)]

    if filtered.empty:
        st.info("No inventory items match the selected filters.")
        return

    try:
        from src.dashboard.data_access import load_monitoring_summary
        from src.dashboard.components import render_alert_banner

        mon = load_monitoring_summary()
        mon_metrics = mon.get("metrics", {}) if isinstance(mon, dict) else {}
        max_score = max(float(mon_metrics.get("data_drift_score", 0.0)), float(mon_metrics.get("prediction_drift_score", 0.0)))
        if max_score >= 0.6:
            render_alert_banner("Monitoring critical: significant drift detected. Check model inputs.", level="error")
        elif max_score >= 0.3:
            render_alert_banner("Monitoring notice: emerging drift signals.", level="warning")
    except Exception:
        pass

    render_section_header("Stock Risk Overview")
    tabs = st.tabs(["Risk View", "Heatmap", "Replenishment Timeline"])
    width_kwargs = get_chart_width_kwargs()
    with tabs[0]:
        st.plotly_chart(inventory_risk_chart(filtered), **width_kwargs)
    with tabs[1]:
        st.plotly_chart(inventory_heatmap_chart(filtered), **width_kwargs)
    with tabs[2]:
        st.plotly_chart(reorder_timeline_chart(filtered), **width_kwargs)

    render_section_header("Reorder Recommendations")
    st.dataframe(filtered, use_container_width=True)
    render_download_frame(filtered, "inventory_recommendations.csv", "Download reorder recommendations")

    render_section_header("Alerts")
    st.dataframe(alerts, use_container_width=True)
