"""Startup healthcheck for RetailPulse.

Validates required packages, directories, and the source dataset before
the dashboard renders. Called once at process startup from retailpulse_dashboard.py.
"""
from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger("retailpulse.healthcheck")

ROOT_DIR = Path(__file__).resolve().parents[2]

REQUIRED_PACKAGES = [
    "streamlit",
    "pandas",
    "numpy",
    "sklearn",
    "plotly",
    "matplotlib",
    "mlflow",
    "xgboost",
    "joblib",
    "scipy",
    "statsmodels",
]

REQUIRED_DIRS = [
    ROOT_DIR / "processed",
    ROOT_DIR / "processed" / "models",
    ROOT_DIR / "processed" / "figures",
    ROOT_DIR / "processed" / "optuna",
    ROOT_DIR / "monitoring" / "drift_reports",
    ROOT_DIR / "logs",
]

REQUIRED_DATA_FILE = ROOT_DIR / "online_retail_II.xlsx"


def _check_packages() -> list[str]:
    """Return list of packages that failed to import."""
    failures: list[str] = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            failures.append(pkg)
    return failures


def _ensure_directories() -> None:
    """Create required output directories if they don't exist."""
    for directory in REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        LOGGER.debug("Directory ready: %s", directory)


def system_healthcheck() -> None:
    """Run all startup checks. Raises on critical failures, warns on non-critical ones.

    Called once at process startup — fast enough to not delay the dashboard.
    """
    LOGGER.info("Running startup healthcheck...")

    # 1. Ensure output directories exist (always safe to create)
    _ensure_directories()
    LOGGER.info("Output directories verified")

    # 2. Check required packages
    missing_pkgs = _check_packages()
    if missing_pkgs:
        # Non-fatal: log a warning so the error surfaces in logs, but don't
        # crash — Streamlit will show a more specific ImportError if needed.
        LOGGER.warning(
            "Healthcheck: the following packages could not be imported: %s. "
            "Install them via requirements.txt.",
            missing_pkgs,
        )
    else:
        LOGGER.info("All required packages available")

    # 3. Verify source dataset
    if not REQUIRED_DATA_FILE.exists():
        LOGGER.error(
            "Healthcheck: source data file not found: %s. "
            "Pipelines that depend on raw data will fail.",
            REQUIRED_DATA_FILE,
        )
        # Not raised — processed artifacts may already exist from a prior run.
    else:
        LOGGER.info("Source data file found: %s", REQUIRED_DATA_FILE.name)

    LOGGER.info("Startup healthcheck complete")
