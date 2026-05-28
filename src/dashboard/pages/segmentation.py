from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.components import render_download_frame, render_page_shell, render_section_header, get_chart_width_kwargs
from src.dashboard.data_access import load_segmentation_outputs, load_customer_intelligence
from src.dashboard.visuals import cluster_distribution_chart, segment_profile_chart, segmentation_embedding_chart
from src.dashboard.session import ensure_defaults


def render_segmentation_page() -> None:
    render_page_shell("Customer Segmentation", "Business personas and cluster structure for customer intelligence.")
    with st.spinner("Loading segmentation artifacts..."):
        outputs = load_segmentation_outputs()
        labels = outputs["labels"]
        personas = outputs["personas"]
        rfm = outputs["rfm"]
        intelligence = load_customer_intelligence()

    if labels.empty:
        st.warning("Segmentation outputs are not available yet. Run the segmentation pipeline first.")
        return

    try:
        from src.dashboard.data_access import load_monitoring_summary
        from src.dashboard.components import render_alert_banner

        mon = load_monitoring_summary()
        mon_metrics = mon.get("metrics", {}) if isinstance(mon, dict) else {}
        max_score = max(float(mon_metrics.get("data_drift_score", 0.0)), float(mon_metrics.get("prediction_drift_score", 0.0)))
        if max_score >= 0.6:
            render_alert_banner("Monitoring critical: significant drift detected. Review monitoring.", level="error")
        elif max_score >= 0.3:
            render_alert_banner("Monitoring notice: drift signals present.", level="warning")
    except Exception:
        pass

    customer_frame = intelligence if not intelligence.empty else labels
    ensure_defaults({"selected_country": "All", "selected_persona": "All", "selected_cluster": "All", "pca_or_tsne": "PCA"})

    country_options = ["All"] + sorted([str(value) for value in customer_frame["Country"].dropna().astype(str).unique()]) if "Country" in customer_frame.columns else ["All"]
    persona_options = ["All"] + sorted([str(value) for value in customer_frame.get("persona", pd.Series(dtype=str)).dropna().astype(str).unique()]) if "persona" in customer_frame.columns else ["All"]
    cluster_column = "kmeans_cluster" if "kmeans_cluster" in customer_frame.columns else ("cluster" if "cluster" in customer_frame.columns else None)
    cluster_options = ["All"] + sorted([str(value) for value in customer_frame[cluster_column].dropna().unique()]) if cluster_column else ["All"]

    filter_cols = st.columns(4)
    with filter_cols[0]:
        selected_country = st.selectbox("Country", country_options, key="selected_country")
    with filter_cols[1]:
        selected_persona = st.selectbox("Persona", persona_options, key="selected_persona")
    with filter_cols[2]:
        selected_cluster = st.selectbox("Cluster", cluster_options, key="selected_cluster")
    with filter_cols[3]:
        pca_or_tsne = st.selectbox("Embedding", ["PCA", "t-SNE"], key="pca_or_tsne")

    filtered = customer_frame.copy()
    if selected_country != "All" and "Country" in filtered.columns:
        filtered = filtered[filtered["Country"].astype(str) == selected_country]
    if selected_persona != "All" and "persona" in filtered.columns:
        filtered = filtered[filtered["persona"].astype(str) == selected_persona]
    if selected_cluster != "All" and cluster_column:
        filtered = filtered[filtered[cluster_column].astype(str) == selected_cluster]

    cols = st.columns(4)
    cols[0].metric("Customers", f"{len(filtered):,}")
    cols[1].metric("Personas", f"{len(personas):,}" if not personas.empty else "0")
    cols[2].metric("RFM rows", f"{len(rfm):,}" if not rfm.empty else "0")
    cols[3].metric("Clusters", f"{labels['kmeans_cluster'].nunique() if 'kmeans_cluster' in labels.columns else labels.iloc[:, -1].nunique()}")

    insight_cols = st.columns(4)
    retention_proxy = 1.0 - min(1.0, float(filtered["RFM_Total"].mean() / max(labels["RFM_Total"].max(), 1))) if "RFM_Total" in filtered.columns and "RFM_Total" in labels.columns else 0.0
    recent_active_share = float((filtered["Recency"] <= 30).mean()) if "Recency" in filtered.columns and not filtered.empty else 0.0
    avg_customer_value = float(filtered["Monetary"].mean()) if "Monetary" in filtered.columns and not filtered.empty else 0.0
    customer_growth_proxy = float(filtered["CustomerID"].nunique() / max(labels["CustomerID"].nunique(), 1)) if "CustomerID" in filtered.columns else 0.0
    insight_cols[0].metric("Retention proxy", f"{retention_proxy:.2%}")
    insight_cols[1].metric("Recent active share", f"{recent_active_share:.2%}")
    insight_cols[2].metric("Average customer value", f"{avg_customer_value:,.2f}")
    insight_cols[3].metric("Customer growth proxy", f"{customer_growth_proxy:.2%}")

    render_section_header("Cluster Distribution")
    tab_a, tab_b = st.tabs(["Cluster Distribution", "Embedding Views"])
    width_kwargs = get_chart_width_kwargs()
    with tab_a:
        st.altair_chart(cluster_distribution_chart(filtered if not filtered.empty else labels), **width_kwargs)
        if "persona" in filtered.columns:
            st.plotly_chart(segment_profile_chart(filtered, "persona"), **width_kwargs)
    with tab_b:
        st.info("Embedding computation is expensive. Click below to generate on demand.")
        if st.button("Generate embedding visualization"):
            st.session_state["show_embedding"] = True
        if st.session_state.get("show_embedding", False):
            with st.spinner("Computing embedding (this may take 30-60 seconds)..."):
                try:
                    st.plotly_chart(segmentation_embedding_chart(filtered if not filtered.empty else labels, method="pca" if pca_or_tsne == "PCA" else "tsne", color_col="persona"), **width_kwargs)
                except Exception as e:
                    st.error(f"Failed to compute embedding: {e}")

    render_section_header("Segment Analytics", "Cluster behavior and persona concentration.")
    analytics_cols = st.columns(2)
    with analytics_cols[0]:
        if "kmeans_cluster" in filtered.columns:
            st.dataframe(filtered.groupby(["kmeans_cluster"]).agg({"Monetary": "mean", "Frequency": "mean", "Recency": "mean"}).reset_index(), use_container_width=True)
        else:
            st.dataframe(filtered.head(20), use_container_width=True)
    with analytics_cols[1]:
        if not outputs["kmeans_profile"].empty:
            st.dataframe(outputs["kmeans_profile"], use_container_width=True)
        if not outputs["dbscan_profile"].empty:
            st.dataframe(outputs["dbscan_profile"], use_container_width=True)

    render_section_header("Persona Table")
    persona_view = filtered if not filtered.empty else personas
    st.dataframe(persona_view.head(25), use_container_width=True)
    render_download_frame(persona_view, "customer_persona_view.csv", "Download persona view")

    render_section_header("RFM Sample")
    st.dataframe(rfm.head(20), use_container_width=True)
