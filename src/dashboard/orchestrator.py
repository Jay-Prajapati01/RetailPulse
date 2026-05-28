"""Centralized pipeline orchestrator for RetailPulse.

Handles dependency-aware artifact generation so that downstream pipelines
(e.g. inventory) always have their upstream dependencies (e.g. forecasting)
satisfied before they run.

Dependency order:
  preprocessing → forecasting → inventory
                              → monitoring
  preprocessing → churn
  preprocessing → segmentation

Usage:
    from src.dashboard.orchestrator import ensure_forecasting, ensure_inventory, ensure_all
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from src.dashboard.config import ARTIFACT_PATHS, PROCESSED_DIR

LOGGER = logging.getLogger("retailpulse.orchestrator")

# ---------------------------------------------------------------------------
# Artifact presence checks
# ---------------------------------------------------------------------------

def _preprocessing_ready() -> bool:
    """Core processed CSVs that all downstream pipelines depend on."""
    required = [
        PROCESSED_DIR / "daily_sales_forecasting.csv",
        PROCESSED_DIR / "customer_base_features.csv",
        PROCESSED_DIR / "product_daily_demand.csv",
    ]
    return all(p.exists() for p in required)

def _forecast_ready() -> bool:
    return (
        ARTIFACT_PATHS["forecast_data"].exists()
        and ARTIFACT_PATHS["forecast_eval"].exists()
    )


def _churn_ready() -> bool:
    return ARTIFACT_PATHS["churn_predictions"].exists()


def _inventory_ready() -> bool:
    return ARTIFACT_PATHS["inventory_recommendations"].exists()


def _segmentation_ready() -> bool:
    return (PROCESSED_DIR / "customer_cluster_labels.csv").exists()


def _monitoring_ready() -> bool:
    return ARTIFACT_PATHS["monitoring_summary"].exists()


# ---------------------------------------------------------------------------
# Individual pipeline runners (lazy imports to keep startup fast)
# ---------------------------------------------------------------------------

def _run_forecasting_pipeline() -> None:
    from retailpulse_forecasting_pipeline import run_forecasting_pipeline
    run_forecasting_pipeline()


def _run_churn_pipeline() -> None:
    from src.churn.churn_pipeline import run_churn_prediction_pipeline
    run_churn_prediction_pipeline()


def _run_inventory_pipeline() -> None:
    from src.inventory.inventory_optimization import run_inventory_optimization_pipeline
    run_inventory_optimization_pipeline()


def _run_segmentation_pipeline() -> None:
    from retailpulse_customer_segmentation import run_customer_segmentation_pipeline
    run_customer_segmentation_pipeline()


def _run_monitoring_pipeline() -> None:
    from src.monitoring.drift_monitoring import run_drift_monitoring_pipeline
    run_drift_monitoring_pipeline()


def _run_preprocessing_pipeline() -> None:
    from retailpulse_feature_pipeline import run_and_save_pipeline, DATA_FILE, ROOT_DIR as FP_ROOT
    from pathlib import Path

    # Prefer CSV if available (faster, no openpyxl overhead)
    csv_path = FP_ROOT / "data" / "online_retail_II.csv"
    xlsx_path = DATA_FILE  # online_retail_II.xlsx at root

    if csv_path.exists():
        LOGGER.info("Preprocessing: using CSV source %s", csv_path)
        run_and_save_pipeline(data_file=csv_path)
    elif xlsx_path.exists():
        LOGGER.info("Preprocessing: using XLSX source %s", xlsx_path)
        run_and_save_pipeline(data_file=xlsx_path)
    else:
        raise FileNotFoundError(
            f"No source data file found. Expected one of:\n"
            f"  {csv_path}\n  {xlsx_path}\n"
            "Please add the data file to the repository."
        )


# ---------------------------------------------------------------------------
# Public orchestration API
# ---------------------------------------------------------------------------

def _run_with_timing(name: str, runner) -> None:
    """Run a pipeline, logging timing and re-raising on failure."""
    LOGGER.info("Orchestrator: starting '%s' pipeline", name)
    start = time.perf_counter()
    try:
        runner()
        LOGGER.info("Orchestrator: '%s' completed in %.2fs", name, time.perf_counter() - start)
    except Exception as exc:
        LOGGER.error(
            "Orchestrator: '%s' failed after %.2fs: %s",
            name, time.perf_counter() - start, exc,
            exc_info=True,
        )
        raise


def ensure_preprocessing() -> None:
    """Ensure core processed CSVs exist, running the feature pipeline if missing."""
    if not _preprocessing_ready():
        _run_with_timing("preprocessing", _run_preprocessing_pipeline)


def ensure_forecasting() -> None:
    """Ensure forecasting artifacts exist, generating them if missing."""
    ensure_preprocessing()  # forecasting needs daily_sales_forecasting.csv
    if not _forecast_ready():
        _run_with_timing("forecasting", _run_forecasting_pipeline)


def ensure_churn() -> None:
    """Ensure churn artifacts exist, generating them if missing."""
    ensure_preprocessing()  # churn needs customer_base_features.csv
    if not _churn_ready():
        _run_with_timing("churn", _run_churn_pipeline)


def ensure_inventory() -> None:
    """Ensure inventory artifacts exist.

    Inventory depends on forecasting outputs — forecasting is guaranteed
    to run first if its artifacts are missing.
    """
    ensure_forecasting()  # also ensures preprocessing
    if not _inventory_ready():
        _run_with_timing("inventory", _run_inventory_pipeline)


def ensure_segmentation() -> None:
    """Ensure segmentation artifacts exist, generating them if missing."""
    ensure_preprocessing()  # segmentation needs customer_base_features.csv
    if not _segmentation_ready():
        _run_with_timing("segmentation", _run_segmentation_pipeline)


def ensure_monitoring() -> None:
    """Ensure monitoring artifacts exist.

    Monitoring depends on forecasting outputs being present.
    """
    ensure_forecasting()  # also ensures preprocessing
    if not _monitoring_ready():
        _run_with_timing("monitoring", _run_monitoring_pipeline)


def ensure_all() -> None:
    """Generate all artifacts in dependency order."""
    ensure_preprocessing()
    ensure_forecasting()
    ensure_churn()
    ensure_inventory()
    ensure_segmentation()
    ensure_monitoring()
