from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# Allow environment-variable overrides so Streamlit Cloud / Docker deployments
# can redirect artifact directories without code changes.
PROCESSED_DIR = Path(os.environ.get("RETAILPULSE_PROCESSED_DIR", ROOT_DIR / "processed"))
FIGURES_DIR = Path(os.environ.get("RETAILPULSE_FIGURES_DIR", PROCESSED_DIR / "figures"))
MODELS_DIR = Path(os.environ.get("RETAILPULSE_MODELS_DIR", PROCESSED_DIR / "models"))
MONITORING_DIR = Path(os.environ.get("RETAILPULSE_MONITORING_DIR", ROOT_DIR / "monitoring" / "drift_reports"))


@dataclass(frozen=True)
class DashboardPage:
    key: str
    title: str
    icon: str
    subtitle: str


APP_TITLE = "RetailPulse"
APP_SUBTITLE = "AI-powered retail intelligence platform"

PAGES: list[DashboardPage] = [
    DashboardPage("home", "Home", "house", "Executive overview"),
    DashboardPage("forecasting", "Forecasting", "graph-up-arrow", "Demand intelligence and what-if analysis"),
    DashboardPage("segmentation", "Customer Segmentation", "people", "Persona and cluster insights"),
    DashboardPage("churn", "Churn Analytics", "exclamation-triangle", "Retention and risk analysis"),
    DashboardPage("inventory", "Inventory Optimization", "boxes", "Reorder and stock risk control"),
    DashboardPage("monitoring", "Monitoring", "activity", "Drift and reliability monitoring"),
    DashboardPage("reports", "Reports", "file-earmark-text", "Artifacts and exports"),
    DashboardPage("settings", "Settings", "gear", "Controls and refresh options"),
]

PAGE_INDEX = {page.key: page for page in PAGES}

ARTIFACT_PATHS = {
    "forecast_report": PROCESSED_DIR / "ensemble_forecasting_report.md",
    "forecast_data": PROCESSED_DIR / "ensemble_future_30d_forecast.csv",
    "forecast_eval": PROCESSED_DIR / "ensemble_evaluation_forecast.csv",
    "churn_report": PROCESSED_DIR / "churn_prediction_report.md",
    "churn_predictions": PROCESSED_DIR / "customer_churn_predictions.csv",
    "inventory_report": PROCESSED_DIR / "inventory_optimization_report.md",
    "inventory_recommendations": PROCESSED_DIR / "inventory_recommendations.csv",
    "monitoring_dashboard": MONITORING_DIR / "monitoring_dashboard.html",
    "monitoring_summary": MONITORING_DIR / "drift_monitoring_summary.json",
    "mlflow_summary": PROCESSED_DIR / "mlflow_tracking_summary.md",
}
