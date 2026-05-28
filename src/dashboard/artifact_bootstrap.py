from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from src.dashboard.config import ARTIFACT_PATHS, MONITORING_DIR, PROCESSED_DIR


PipelineRunner = Callable[[], object]


def _forecast_outputs() -> list[Path]:
    return [ARTIFACT_PATHS["forecast_data"], ARTIFACT_PATHS["forecast_eval"]]


def _churn_outputs() -> list[Path]:
    return [ARTIFACT_PATHS["churn_predictions"], PROCESSED_DIR / "customer_churn_metrics.csv", PROCESSED_DIR / "customer_churn_shap_ranking.csv", ARTIFACT_PATHS["churn_report"]]


def _inventory_outputs() -> list[Path]:
    return [ARTIFACT_PATHS["inventory_recommendations"], PROCESSED_DIR / "inventory_alerts.csv", PROCESSED_DIR / "inventory_metrics.csv", ARTIFACT_PATHS["inventory_report"]]


def _segmentation_outputs() -> list[Path]:
    return [
        PROCESSED_DIR / "customer_cluster_labels.csv",
        PROCESSED_DIR / "customer_personas.csv",
        PROCESSED_DIR / "cluster_profile_kmeans.csv",
        PROCESSED_DIR / "cluster_profile_dbscan.csv",
        PROCESSED_DIR / "customer_segmentation_report.md",
    ]


def _monitoring_outputs() -> list[Path]:
    return [ARTIFACT_PATHS["monitoring_dashboard"], ARTIFACT_PATHS["monitoring_summary"], MONITORING_DIR / "data_drift_report.html", MONITORING_DIR / "prediction_drift_report.html"]


PIPELINE_TARGETS: dict[str, tuple[Callable[[], list[Path]], PipelineRunner]] = {
    "forecasting": (_forecast_outputs, lambda: _run_forecasting()),
    "churn": (_churn_outputs, lambda: _run_churn()),
    "inventory": (_inventory_outputs, lambda: _run_inventory()),
    "segmentation": (_segmentation_outputs, lambda: _run_segmentation()),
    "monitoring": (_monitoring_outputs, lambda: _run_monitoring()),
}


def _run_forecasting() -> object:
    from retailpulse_forecasting_pipeline import run_forecasting_pipeline

    return run_forecasting_pipeline()


def _run_churn() -> object:
    from src.churn.churn_pipeline import run_churn_prediction_pipeline

    return run_churn_prediction_pipeline()


def _run_inventory() -> object:
    from src.inventory.inventory_optimization import run_inventory_optimization_pipeline

    return run_inventory_optimization_pipeline()


def _run_segmentation() -> object:
    from retailpulse_customer_segmentation import run_customer_segmentation_pipeline

    return run_customer_segmentation_pipeline()


def _run_monitoring() -> object:
    from src.monitoring.drift_monitoring import run_drift_monitoring_pipeline

    return run_drift_monitoring_pipeline()


def ensure_artifacts(*categories: str) -> list[str]:
    """Generate missing artifacts on demand for the requested dashboard categories."""

    triggered: list[str] = []
    for category in categories:
        if category not in PIPELINE_TARGETS:
            continue
        required_paths, runner = PIPELINE_TARGETS[category]
        missing = [path for path in required_paths() if not path.exists()]
        if not missing:
            continue
        runner()
        triggered.append(category)
    return triggered
