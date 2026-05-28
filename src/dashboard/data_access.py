from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.dashboard.artifact_bootstrap import ensure_artifacts
from src.dashboard.config import ARTIFACT_PATHS, FIGURES_DIR, MODELS_DIR, MONITORING_DIR, PROCESSED_DIR

LOGGER = logging.getLogger("retailpulse.data_access")

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def read_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        LOGGER.debug("CSV not found, returning empty frame: %s", path)
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception as exc:
        LOGGER.error("Failed to read CSV %s: %s", path, exc)
        return pd.DataFrame()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        LOGGER.debug("JSON not found, returning empty dict: %s", path)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.error("Failed to read JSON %s: %s", path, exc)
        return {}


def read_text(path: Path) -> str:
    if not path.exists():
        LOGGER.debug("Text file not found, returning empty string: %s", path)
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        LOGGER.error("Failed to read text file %s: %s", path, exc)
        return ""


def _validate_frame(frame: pd.DataFrame, name: str, required_columns: list[str] | None = None) -> pd.DataFrame:
    """Log a warning if the frame is empty or missing expected columns."""
    if frame.empty:
        LOGGER.warning("Loaded frame '%s' is empty", name)
        return frame
    if required_columns:
        missing = [c for c in required_columns if c not in frame.columns]
        if missing:
            LOGGER.warning("Frame '%s' is missing expected columns: %s", name, missing)
    return frame


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def load_forecast_comparison() -> pd.DataFrame:
    ensure_artifacts("forecasting")
    frame = read_csv(ARTIFACT_PATHS["forecast_eval"], parse_dates=["date"])
    _validate_frame(frame, "forecast_comparison", required_columns=["date"])
    if frame.empty:
        return frame
    return frame.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=600)
def load_future_forecast() -> pd.DataFrame:
    ensure_artifacts("forecasting")
    frame = read_csv(ARTIFACT_PATHS["forecast_data"], parse_dates=["date"])
    _validate_frame(frame, "future_forecast", required_columns=["date"])
    if frame.empty:
        return frame
    return frame.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=600)
def load_segmentation_outputs() -> dict[str, pd.DataFrame]:
    ensure_artifacts("segmentation")
    outputs = {
        "labels": read_csv(PROCESSED_DIR / "customer_cluster_labels.csv"),
        "kmeans_profile": read_csv(PROCESSED_DIR / "cluster_profile_kmeans.csv"),
        "dbscan_profile": read_csv(PROCESSED_DIR / "cluster_profile_dbscan.csv"),
        "personas": read_csv(PROCESSED_DIR / "customer_personas.csv"),
        "rfm": read_csv(PROCESSED_DIR / "customer_rfm.csv"),
    }
    _validate_frame(outputs["labels"], "segmentation/labels", required_columns=["CustomerID"])
    return outputs


def load_customer_intelligence() -> pd.DataFrame:
    outputs = load_segmentation_outputs()
    labels = outputs["labels"]
    personas = outputs["personas"]
    rfm = outputs["rfm"]
    if labels.empty:
        return labels
    base = labels.copy()
    if not personas.empty:
        persona_columns = [
            c for c in ["CustomerID", "persona", "Country", "kmeans_cluster", "dbscan_cluster", "pca_1", "pca_2"]
            if c in personas.columns
        ]
        persona_frame = (
            personas[persona_columns].rename(columns={"persona": "persona_label"})
            if persona_columns
            else personas
        )
        join_keys = [c for c in ["CustomerID"] if c in base.columns and c in persona_frame.columns]
        if join_keys:
            base = base.merge(persona_frame, on=join_keys, how="left", suffixes=("", "_persona"))
    if not rfm.empty and "CustomerID" in base.columns and "CustomerID" in rfm.columns:
        base = base.merge(rfm, on="CustomerID", how="left", suffixes=("", "_rfm"))
    return base


@st.cache_data(ttl=600)
def load_churn_outputs() -> dict[str, pd.DataFrame | dict[str, Any]]:
    ensure_artifacts("churn")
    predictions = read_csv(ARTIFACT_PATHS["churn_predictions"])
    _validate_frame(predictions, "churn/predictions", required_columns=["CustomerID"])
    return {
        "predictions": predictions,
        "metrics": read_csv(PROCESSED_DIR / "customer_churn_metrics.csv"),
        "shap_ranking": read_csv(PROCESSED_DIR / "customer_churn_shap_ranking.csv"),
        "config": read_json(PROCESSED_DIR / "churn_config.json"),
    }


@st.cache_data(ttl=600)
def load_inventory_outputs() -> dict[str, pd.DataFrame | dict[str, Any]]:
    ensure_artifacts("inventory")
    recommendations = read_csv(ARTIFACT_PATHS["inventory_recommendations"])
    _validate_frame(recommendations, "inventory/recommendations", required_columns=["StockCode"])
    return {
        "recommendations": recommendations,
        "alerts": read_csv(PROCESSED_DIR / "inventory_alerts.csv"),
        "metrics": read_csv(PROCESSED_DIR / "inventory_metrics.csv"),
        "summary": read_json(PROCESSED_DIR / "inventory_streamlit_summary.json"),
    }


@st.cache_data(ttl=300)
def load_monitoring_summary() -> dict[str, Any]:
    ensure_artifacts("monitoring")
    return read_json(ARTIFACT_PATHS["monitoring_summary"])


@st.cache_data(ttl=300)
def load_monitoring_reports() -> dict[str, str]:
    ensure_artifacts("monitoring")
    return {
        "html": read_text(ARTIFACT_PATHS["monitoring_dashboard"]),
        "data_drift_html": read_text(MONITORING_DIR / "data_drift_report.html"),
        "prediction_drift_html": read_text(MONITORING_DIR / "prediction_drift_report.html"),
    }


def load_monitoring_outputs() -> dict[str, Any]:
    summary = load_monitoring_summary()
    reports = load_monitoring_reports()
    return {
        "summary": summary,
        "html": reports["html"],
        "data_drift_html": reports["data_drift_html"],
        "prediction_drift_html": reports["prediction_drift_html"],
    }


@st.cache_data(ttl=300)
def load_mlflow_summary() -> str:
    return read_text(ARTIFACT_PATHS["mlflow_summary"])


@st.cache_data(ttl=600)
def load_ensemble_metrics() -> pd.DataFrame:
    return read_csv(PROCESSED_DIR / "ensemble_metrics.csv")


def frame_to_metric_map(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    if {"metric", "value"}.issubset(frame.columns):
        return {str(row["metric"]): float(row["value"]) for _, row in frame.iterrows()}
    first_row = frame.iloc[0].to_dict()
    return {str(k): float(v) for k, v in first_row.items() if pd.notna(v)}


@st.cache_data(ttl=600)
def load_optuna_outputs() -> dict[str, pd.DataFrame]:
    return {
        "churn_trials": read_csv(PROCESSED_DIR / "optuna" / "churn_optuna_trial_metrics.csv"),
        "lstm_trials": read_csv(PROCESSED_DIR / "optuna" / "lstm_optuna_trial_metrics.csv"),
        "ensemble_trials": read_csv(PROCESSED_DIR / "optuna" / "ensemble_optuna_trial_metrics.csv"),
    }


@st.cache_data(ttl=600)
def load_dashboard_summary() -> dict[str, Any]:
    ensure_artifacts("forecasting", "churn", "inventory", "segmentation", "monitoring")
    churn = load_churn_outputs()
    inventory = load_inventory_outputs()
    monitoring = load_monitoring_summary()
    forecast = load_forecast_comparison()
    future = load_future_forecast()

    summary: dict[str, Any] = {
        "forecast_rows": int(len(forecast)),
        "future_rows": int(len(future)),
        "churn_rows": int(len(churn["predictions"])) if isinstance(churn["predictions"], pd.DataFrame) else 0,
        "inventory_rows": int(len(inventory["recommendations"])) if isinstance(inventory["recommendations"], pd.DataFrame) else 0,
        "alert_count": int(len(inventory["alerts"])) if isinstance(inventory["alerts"], pd.DataFrame) else 0,
        "data_drift_score": monitoring.get("metrics", {}).get("data_drift_score", 0.0),
        "prediction_drift_score": monitoring.get("metrics", {}).get("prediction_drift_score", 0.0),
    }

    if isinstance(churn["metrics"], pd.DataFrame) and not churn["metrics"].empty:
        metric_map = frame_to_metric_map(churn["metrics"])
        summary["churn_roc_auc"] = float(metric_map.get("roc_auc", 0.0))
    else:
        summary["churn_roc_auc"] = 0.0

    if isinstance(inventory["metrics"], pd.DataFrame) and not inventory["metrics"].empty:
        inventory_map = frame_to_metric_map(inventory["metrics"])
        summary["inventory_avg_days_of_cover"] = float(inventory_map.get("avg_days_of_cover", 0.0))
    else:
        summary["inventory_avg_days_of_cover"] = 0.0

    ensemble_metrics = load_ensemble_metrics()
    if not ensemble_metrics.empty:
        if {"model", "RMSE"}.issubset(ensemble_metrics.columns):
            ensemble_row = ensemble_metrics.loc[ensemble_metrics["model"].astype(str).str.lower() == "ensemble"]
            if not ensemble_row.empty:
                summary["ensemble_rmse"] = float(ensemble_row.iloc[0]["RMSE"])
            else:
                summary["ensemble_rmse"] = float(ensemble_metrics.iloc[0]["RMSE"])
        elif "RMSE" in ensemble_metrics.columns:
            summary["ensemble_rmse"] = float(ensemble_metrics.iloc[0]["RMSE"])
        else:
            summary["ensemble_rmse"] = float(frame_to_metric_map(ensemble_metrics).get("RMSE", 0.0))
    else:
        summary["ensemble_rmse"] = 0.0

    LOGGER.info(
        "Dashboard summary loaded — forecast_rows=%d churn_rows=%d inventory_rows=%d",
        summary["forecast_rows"],
        summary["churn_rows"],
        summary["inventory_rows"],
    )
    return summary
