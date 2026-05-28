"""Artifact bootstrap module for RetailPulse dashboard.

NOTE: This module is currently DISABLED for visualization-only deployment.
All artifacts are pre-generated locally and committed to the repository.
The ensure_artifacts() function is no longer called from dashboard pages.

This module is preserved for local development and future production orchestration.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import streamlit as st

from src.dashboard.config import ARTIFACT_PATHS, MONITORING_DIR, PROCESSED_DIR

LOGGER = logging.getLogger("retailpulse.bootstrap")

PipelineRunner = Callable[[], object]


def _forecast_outputs() -> list[Path]:
    return [ARTIFACT_PATHS["forecast_data"], ARTIFACT_PATHS["forecast_eval"]]


def _churn_outputs() -> list[Path]:
    return [
        ARTIFACT_PATHS["churn_predictions"],
        PROCESSED_DIR / "customer_churn_metrics.csv",
        PROCESSED_DIR / "customer_churn_shap_ranking.csv",
        ARTIFACT_PATHS["churn_report"],
    ]


def _inventory_outputs() -> list[Path]:
    return [
        ARTIFACT_PATHS["inventory_recommendations"],
        PROCESSED_DIR / "inventory_alerts.csv",
        PROCESSED_DIR / "inventory_metrics.csv",
        ARTIFACT_PATHS["inventory_report"],
    ]


def _segmentation_outputs() -> list[Path]:
    return [
        PROCESSED_DIR / "customer_cluster_labels.csv",
        PROCESSED_DIR / "customer_personas.csv",
        PROCESSED_DIR / "cluster_profile_kmeans.csv",
        PROCESSED_DIR / "cluster_profile_dbscan.csv",
        PROCESSED_DIR / "customer_segmentation_report.md",
    ]


def _monitoring_outputs() -> list[Path]:
    return [
        ARTIFACT_PATHS["monitoring_dashboard"],
        ARTIFACT_PATHS["monitoring_summary"],
        MONITORING_DIR / "data_drift_report.html",
        MONITORING_DIR / "prediction_drift_report.html",
    ]


def _run_forecasting() -> object:
    from retailpulse_forecasting_pipeline import run_forecasting_pipeline

    return run_forecasting_pipeline()


def _run_churn() -> object:
    from src.churn.churn_pipeline import run_churn_prediction_pipeline

    return run_churn_prediction_pipeline()


def _run_inventory() -> object:
    # Inventory depends on forecasting outputs — delegate to orchestrator
    # so the dependency is always satisfied in the correct order.
    from src.dashboard.orchestrator import ensure_inventory
    ensure_inventory()
    return None


def _run_segmentation() -> object:
    from retailpulse_customer_segmentation import run_customer_segmentation_pipeline

    return run_customer_segmentation_pipeline()


def _run_monitoring() -> object:
    # Monitoring depends on forecasting outputs — delegate to orchestrator.
    from src.dashboard.orchestrator import ensure_monitoring
    ensure_monitoring()
    return None


PIPELINE_TARGETS: dict[str, tuple[Callable[[], list[Path]], PipelineRunner]] = {
    "forecasting": (_forecast_outputs, _run_forecasting),
    "churn": (_churn_outputs, _run_churn),
    "inventory": (_inventory_outputs, _run_inventory),
    "segmentation": (_segmentation_outputs, _run_segmentation),
    "monitoring": (_monitoring_outputs, _run_monitoring),
}


def ensure_artifacts(*categories: str) -> list[str]:
    """Generate missing artifacts on demand for the requested dashboard categories.

    Returns the list of pipeline categories that were actually triggered.
    Logs timing and surfaces errors via st.warning without crashing the dashboard.
    """
    triggered: list[str] = []

    for category in categories:
        if category not in PIPELINE_TARGETS:
            LOGGER.warning("Unknown pipeline category requested: %s", category)
            continue

        required_paths_fn, runner = PIPELINE_TARGETS[category]
        missing = [p for p in required_paths_fn() if not p.exists()]
        if not missing:
            continue

        LOGGER.info(
            "Artifact bootstrap triggered for '%s' — %d missing file(s): %s",
            category,
            len(missing),
            [p.name for p in missing],
        )

        start = time.perf_counter()
        try:
            runner()
            elapsed = time.perf_counter() - start
            LOGGER.info("Pipeline '%s' completed in %.2fs", category, elapsed)
            triggered.append(category)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            LOGGER.error(
                "Pipeline '%s' failed after %.2fs: %s",
                category,
                elapsed,
                exc,
                exc_info=True,
            )
            st.warning(
                f"⚠️ Could not generate '{category}' artifacts: {exc}. "
                "Some data may be unavailable.",
                icon="⚠️",
            )

    return triggered
