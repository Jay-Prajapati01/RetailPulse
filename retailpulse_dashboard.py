"""RetailPulse Dashboard – Streamlit entry point.

This file is the single entry point that Docker and `streamlit run` execute.
It delegates entirely to the multi-page application defined in src/dashboard/app.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that `src.dashboard` imports work
# regardless of how Streamlit is launched (Docker, local, IDE).
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.app import run_app  # noqa: E402

run_app()