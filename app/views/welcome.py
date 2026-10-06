"""Admin overview (the admin's landing page): every workspace with its use cases and a link, plus the data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components.auth import current_user  # noqa: E402
from app.components.personas import PERSONAS  # noqa: E402
from app.components.ui import inject_css  # noqa: E402
from src.serving.permissions import allowed_workspaces  # noqa: E402
from src.serving.queries import snapshot_info  # noqa: E402

inject_css()
user = current_user() or {}
st.markdown("<div class='qc-hero'><div class='qc-icon'>🛒</div><div><h1>Quick-Commerce Intelligence</h1>"
            f"<p>Welcome, {user.get('name', '')}. Open any workspace below.</p></div></div>", unsafe_allow_html=True)

mine = allowed_workspaces(user.get("role", ""))
cols = st.columns(2)
for i, ws in enumerate(mine):
    p = PERSONAS[ws]
    with cols[i % 2].container(border=True):
        st.markdown(f"#### {p['icon']} {p['title']}")
        st.caption(p["tagline"])
        st.markdown("\n".join(f"- {uc}" for uc in p["use_cases"]))
        if p.get("on_hold"):
            st.caption("🚧 On hold — coming in a later phase")
        st.page_link(p["page"], label=f"Open {p['title']}", icon="➡️")

info = snapshot_info()
st.subheader("About this data")
c1, c2, c3 = st.columns(3)
c1.metric("Tables in snapshot", info.get("tables", "?"))
c2.metric("Rows", f"{int(info.get('rows', 0)):,}")
c3.metric("Snapshot size", f"{info.get('size_mb', '?')} MB")
st.markdown(
    "Synthetic quick-commerce data (12 dark stores, 3 cities, Apr–Sep 2025) processed through a PySpark "
    "Medallion pipeline: **Bronze** (raw) → **Silver** (cleaned, validated, quarantine) → **Gold** "
    "(business tables) → **ML** (forecasts, stockout risk, anomalies, segments, recommendations). "
    "Every number on these pages comes from that pipeline.")
