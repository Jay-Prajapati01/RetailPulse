from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "processed"
FIGURES_DIR = PROCESSED_DIR / "figures"
MODELS_DIR = PROCESSED_DIR / "models"
MONITORING_REPORTS_DIR = ROOT_DIR / "monitoring" / "drift_reports"

ARTIFACT_INDEX = {
    "ensemble_report": PROCESSED_DIR / "ensemble_forecasting_report.md",
    "ensemble_forecast": PROCESSED_DIR / "ensemble_future_30d_forecast.csv",
    "churn_report": PROCESSED_DIR / "churn_prediction_report.md",
    "inventory_report": PROCESSED_DIR / "inventory_optimization_report.md",
    "monitoring_dashboard": MONITORING_REPORTS_DIR / "monitoring_dashboard.html",
}
