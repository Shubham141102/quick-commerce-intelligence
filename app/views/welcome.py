"""Admin overview (the admin's landing page): snapshot facts and a grid of every workspace with its use cases."""

from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components.auth import current_user  # noqa: E402
from app.components.personas import PERSONAS  # noqa: E402
from app.components.ui import APP_NAME, DATA_PERIOD, inject_css, label  # noqa: E402
from src.serving.permissions import allowed_workspaces  # noqa: E402
from src.serving.queries import snapshot_info  # noqa: E402

inject_css()
user = current_user() or {}
info = snapshot_info()
st.markdown(
    "<div class='qc-head'><div>"
    f"<div class='qc-crumb'>{html.escape(APP_NAME)} &nbsp;›&nbsp; <b>Overview</b></div>"
    "<h1>Overview</h1>"
    f"<p>Welcome, {html.escape(user.get('name', ''))}. All workspaces and the data behind them.</p></div>"
    "<div class='qc-meta'><span class='qc-status'><span class='qc-dot'></span>Snapshot current</span><br>"
    f"Data period {DATA_PERIOD} · {html.escape(info.get('profile', '?'))} profile</div></div>",
    unsafe_allow_html=True)

st.write("")
c = st.columns(4)
c[0].metric("Workspaces live", sum(1 for w in allowed_workspaces(user.get("role", "")) if not PERSONAS[w].get("on_hold")))
c[1].metric("Tables in snapshot", info.get("tables", "?"))
c[2].metric("Rows", f"{int(info.get('rows', 0)):,}")
c[3].metric("Snapshot size", f"{info.get('size_mb', '?')} MB")

st.write("")
label("Workspaces")
mine = allowed_workspaces(user.get("role", ""))
cols = st.columns(2, gap="medium")
for i, ws in enumerate(mine):
    p = PERSONAS[ws]
    with cols[i % 2].container(border=True):
        status = ":orange-badge[On hold]" if p.get("on_hold") else ":green-badge[Live]"
        st.markdown(f"**{p['title']}** &nbsp; {status}")
        st.caption(p["tagline"])
        st.markdown("\n".join(f"- {uc}" for uc in p["use_cases"]))
        target = st.session_state.get("_workspace_landing", {}).get(ws)
        if target is not None:
            st.page_link(target, label=f"Open {p['title']}", icon=":material/arrow_forward:")

st.write("")
label("About the data")
st.markdown(
    "Synthetic quick-commerce data (12 dark stores, 3 cities, Apr–Sep 2025) processed through a PySpark "
    "Medallion pipeline: **Bronze** (raw) → **Silver** (cleaned, validated, quarantine) → **Gold** "
    "(business tables) → **ML** (forecasts, stockout risk, anomalies, segments, recommendations). "
    "Every number in these workspaces comes from that pipeline.")
