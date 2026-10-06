"""Quick-Commerce Intelligence — Streamlit entry point.

    streamlit run app/Home.py

Before sign-in only the login page exists. After sign-in the menu is built from the user's role and the
user's own workspace is the default page, so each persona lands directly in its workspace:
inventory → Inventory & Supply Chain, business → Business & Revenue, marketing → Customer Growth & Marketing,
engineer → Data Engineer (on hold), admin → an overview linking every workspace.
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
from src.serving.permissions import allowed_workspaces  # noqa: E402

st.set_page_config(page_title="Quick-Commerce Intelligence", page_icon="🛒", layout="wide")

VIEWS = ROOT / "app" / "views"


def workspace_page(ws: str, default: bool = False) -> st.Page:
    p = PERSONAS[ws]
    return st.Page(ROOT / "app" / p["page"], title=p["title"], icon=p["icon"], url_path=ws, default=default)


user = current_user()
if not user:
    page = st.navigation([st.Page(login_page, title="Sign in", icon="🔐")], position="hidden")
else:
    mine = allowed_workspaces(user["role"])
    if len(mine) == 1:      # a persona: its workspace is the landing page, no menu needed
        page = st.navigation([workspace_page(mine[0], default=True)], position="hidden")
    else:                   # admin: overview first, then every workspace
        overview = st.Page(VIEWS / "welcome.py", title="Overview", icon="🏠", default=True)
        page = st.navigation({"": [overview], "Workspaces": [workspace_page(w) for w in mine]})
    sidebar_profile()
page.run()
