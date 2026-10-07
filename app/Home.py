"""Quick-Commerce Intelligence — Streamlit entry point.

    streamlit run app/Home.py

Before sign-in only the login page exists. After sign-in the left sidebar is built from the user's role: each
workspace is a sidebar group whose sections are pages (e.g. Inventory & Supply Chain → Overview, Demand
forecasting, Stockout risk & replenishment, Inventory explorer). A persona lands directly on the first section
of its own workspace; the admin lands on an overview and sees every workspace group.
The app reads the published snapshot in data/demo/ through DuckDB; it never runs Spark or the pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components.auth import current_user, login_page  # noqa: E402
from app.components.personas import PERSONAS  # noqa: E402
from app.components.ui import sidebar_profile  # noqa: E402
from app.views import business, inventory, marketing  # noqa: E402
from src.serving.permissions import allowed_workspaces  # noqa: E402

st.set_page_config(page_title="Quick-Commerce Intelligence", page_icon=str(ROOT / "app" / "assets" / "mark.svg"),
                   layout="wide")

VIEWS = ROOT / "app" / "views"
ASSETS = ROOT / "app" / "assets"
SECTION_PAGES = {"inventory": inventory.PAGES, "business": business.PAGES, "marketing": marketing.PAGES}


def workspace_pages(ws: str, landing: bool = False) -> list:
    """The sidebar pages of one workspace; with `landing`, its first section is the app's default page."""
    p = PERSONAS[ws]
    if ws not in SECTION_PAGES:     # single-page workspace (Data Engineer, on hold)
        return [st.Page(ROOT / "app" / p["page"], title=p["title"], icon=p["icon"], url_path=ws, default=landing)]
    return [st.Page(render, title=title, icon=icon, url_path=f"{ws}-{key}", default=landing and i == 0)
            for i, (key, title, icon, render) in enumerate(SECTION_PAGES[ws])]


user = current_user()
if not user:
    page = st.navigation([st.Page(login_page, title="Sign in", icon=":material/login:")], position="hidden")
else:
    mine = allowed_workspaces(user["role"])
    per_ws = {ws: workspace_pages(ws, landing=len(mine) == 1) for ws in mine}
    groups = {PERSONAS[ws]["title"]: pages for ws, pages in per_ws.items()}
    if len(mine) > 1:   # admin: an overview first, then every workspace group
        overview = st.Page(VIEWS / "welcome.py", title="Overview", icon=":material/dashboard:", default=True)
        groups = {"": [overview], **groups}
    st.session_state["_workspace_landing"] = {ws: pages[0] for ws, pages in per_ws.items()}   # overview links
    page = st.navigation(groups, position="sidebar", expanded=True)
    st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "mark.svg"), size="large")
    sidebar_profile()
page.run()
