"""RetailPulse Dashboard – Streamlit entry point.

Visualization-only dashboard that loads pre-generated ML artifacts.
All analytics are generated locally and committed to the repository.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that `src.dashboard` imports work
# regardless of how Streamlit is launched (Docker, local, IDE).
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.runtime import configure_logging  # noqa: E402
from src.dashboard.healthcheck import system_healthcheck  # noqa: E402
from src.dashboard.app import run_app  # noqa: E402

configure_logging()
system_healthcheck()
run_app()