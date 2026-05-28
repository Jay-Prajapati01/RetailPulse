from __future__ import annotations

import json

import streamlit as st

from src.dashboard.components import render_page_shell, render_section_header
from src.dashboard.config import ARTIFACT_PATHS
from src.dashboard.data_access import load_churn_outputs, load_forecast_comparison, load_inventory_outputs, load_mlflow_summary, load_monitoring_outputs, load_segmentation_outputs
from src.dashboard.reporting import build_pdf_report, dataframe_to_csv_bytes


def render_reports_page() -> None:
    render_page_shell("Reports", "Artifact index and exported business reports.")

    render_section_header("Artifact Index")
    for name, path in ARTIFACT_PATHS.items():
        st.write(f"- **{name}**: `{path}`")

    render_section_header("CSV Exports")
    export_cols = st.columns(3)
    forecast = load_forecast_comparison()
    segmentation = load_segmentation_outputs()["labels"]
    churn = load_churn_outputs()["predictions"]
    inventory = load_inventory_outputs()["recommendations"]
    monitoring = load_monitoring_outputs()

    with export_cols[0]:
        if not forecast.empty:
            st.download_button("Download forecast CSV", dataframe_to_csv_bytes(forecast), "retailpulse_forecast.csv", "text/csv")
        if not segmentation.empty:
            st.download_button("Download segmentation CSV", dataframe_to_csv_bytes(segmentation), "retailpulse_segmentation.csv", "text/csv")
    with export_cols[1]:
        if not churn.empty:
            st.download_button("Download churn CSV", dataframe_to_csv_bytes(churn), "retailpulse_churn.csv", "text/csv")
        if not inventory.empty:
            st.download_button("Download inventory CSV", dataframe_to_csv_bytes(inventory), "retailpulse_inventory.csv", "text/csv")
    with export_cols[2]:
        if monitoring.get("summary"):
            st.download_button("Download monitoring JSON", json.dumps(monitoring["summary"], indent=2).encode("utf-8"), "retailpulse_monitoring_summary.json", "application/json")

    render_section_header("MLflow Tracking Summary")
    summary = load_mlflow_summary()
    if summary:
        st.text_area("MLflow summary", summary, height=320)
    else:
        st.info("No MLflow summary was found.")

    render_section_header("Executive PDF Reports")
    overview_lines = [
        f"Forecast rows: {len(forecast):,}",
        f"Customer rows: {len(segmentation):,}",
        f"Churn rows: {len(churn):,}",
        f"Inventory rows: {len(inventory):,}",
        f"Monitoring status: {monitoring.get('summary', {}).get('metrics', {}).get('model_reliability_score', 0.0):.3f}",
    ]
    if st.button("Generate executive PDF report"):
        pdf_bytes = build_pdf_report(
            "RetailPulse Executive Report",
            overview_lines,
            tables=[
                ("Forecast preview", forecast.head(10)),
                ("Segmentation preview", segmentation.head(10)),
                ("Churn preview", churn.head(10)),
                ("Inventory preview", inventory.head(10)),
            ],
        )
        st.download_button("Download executive PDF", pdf_bytes, "retailpulse_executive_report.pdf", "application/pdf")

