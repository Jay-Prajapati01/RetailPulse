from __future__ import annotations

from typing import Any, Mapping

import streamlit as st


def ensure_defaults(defaults: Mapping[str, Any]) -> None:
    """Ensure keys exist in `st.session_state` with provided defaults."""
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_state(key: str, value: Any) -> None:
    st.session_state[key] = value


def get_state(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def clear_keys(keys: list[str]) -> None:
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]
