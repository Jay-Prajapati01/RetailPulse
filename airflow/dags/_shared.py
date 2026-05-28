from __future__ import annotations

from datetime import timedelta
from pathlib import Path

DEFAULT_ARGS = {
    "owner": "RetailPulse",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def module_doc(module_name: str, callable_name: str) -> str:
    return f"{module_name}:{callable_name}"
