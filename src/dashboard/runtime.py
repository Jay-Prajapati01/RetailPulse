from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any

import joblib
import streamlit as st

LOGGER = logging.getLogger("retailpulse.dashboard")


def configure_logging(level: int | None = None) -> None:
    """Configure the root RetailPulse logger.

    Reads LOG_LEVEL from the environment so Streamlit Cloud deployments can
    control verbosity without code changes. Defaults to INFO.
    """
    if LOGGER.handlers:
        return

    if level is None:
        env_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)
    LOGGER.info("RetailPulse logging initialised at level %s", logging.getLevelName(level))


@st.cache_resource(show_spinner=False)
def load_model_artifact(model_path: str) -> Any:
    """Load a serialised model from disk, cached for the lifetime of the session."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Model artifact is empty (0 bytes): {path}")

    LOGGER.info("Loading model artifact: %s", path.name)
    start = time.perf_counter()
    try:
        suffix = path.suffix.lower()
        if suffix == ".joblib":
            model = joblib.load(path)
        elif suffix in {".pkl", ".pickle"}:
            with path.open("rb") as handle:
                model = pickle.load(handle)
        else:
            raise ValueError(f"Unsupported model artifact format: {path.suffix}")
        LOGGER.info("Model '%s' loaded in %.2fs", path.name, time.perf_counter() - start)
        return model
    except Exception as exc:
        LOGGER.error("Failed to load model '%s': %s", path.name, exc, exc_info=True)
        raise


def clear_dashboard_cache() -> None:
    """Clear all Streamlit caches and artifact bootstrap session flags."""
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    for key in list(st.session_state.keys()):
        if key.startswith("rp_artifacts_bootstrapped_"):
            del st.session_state[key]
    LOGGER.info("Dashboard cache cleared")
