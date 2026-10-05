"""Home page after sign-in: who you are, what you can open, what data you are looking at."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components.auth import current_user  # noqa: E402
from app.components.ui import freshness_banner  # noqa: E402
from src.serving.permissions import ROLE_LABELS, WORKSPACES, allowed_workspaces  # noqa: E402
from src.serving.queries import snapshot_info  # noqa: E402

user = current_user() or {}
st.title("Quick-Commerce Intelligence")
st.markdown(f"Welcome, **{user.get('name', '')}** — {ROLE_LABELS.get(user.get('role', ''), '')}.")
freshness_banner()

st.subheader("Your workspaces")
built = {"inventory"}
for ws in allowed_workspaces(user.get("role", "")):
    status = "open it from the sidebar" if ws in built else "coming in a later phase"
    st.markdown(f"- **{WORKSPACES[ws]}** — {status}")

info = snapshot_info()
st.subheader("About this data")
c1, c2, c3 = st.columns(3)
c1.metric("Tables in snapshot", info.get("tables", "?"))
c2.metric("Rows", f"{int(info.get('rows', 0)):,}")
c3.metric("Snapshot size", f"{info.get('size_mb', '?')} MB")
st.markdown(
    "Synthetic quick-commerce data (12 dark stores, 3 cities, Apr–Sep 2025) processed through a PySpark "
    "Medallion pipeline: **Bronze** (raw) → **Silver** (cleaned, validated, quarantine) → **Gold** "
    "(business tables) → **ML** (forecasts, stockout risk). Every number on these pages comes from that pipeline.")
