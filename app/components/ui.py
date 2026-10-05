"""Shared page pieces: workspace guard, data-freshness banner, advisory note."""

from __future__ import annotations

import streamlit as st

from app.components.auth import current_user
from src.serving.permissions import WORKSPACES, AccessDenied, require
from src.serving.queries import snapshot_info


def require_workspace(workspace: str) -> str:
    """Stop the page unless the signed-in user's role may open this workspace; returns the role."""
    user = current_user()
    if not user:
        st.warning("Please sign in first.")
        st.stop()
    try:
        require(user["role"], workspace)
    except AccessDenied:
        st.error(f"Your role cannot open the {WORKSPACES[workspace]} workspace.")
        st.stop()
    return user["role"]


def freshness_banner() -> None:
    info = snapshot_info()
    st.caption(f"📦 **Precomputed snapshot** — pipeline run `{info.get('pipeline_run_id', '?')}`, published "
               f"{info.get('published_at', '?')} from generation run `{info.get('generation_run_id', '?')}`. "
               "The app reads published files; no pipeline is running live.")


def advisory_note() -> None:
    st.info("**Advisory only.** Risk tiers and suggested quantities support a manager's decision; "
            "nothing here places an order or changes stock.", icon="ℹ️")
