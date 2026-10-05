"""Quick-Commerce Intelligence — Streamlit entry point.

    streamlit run app/Home.py

Shows the login page until a demo user signs in, then only the workspaces that user's role may open.
The app reads the published snapshot in data/demo/ through DuckDB; it never runs Spark or the pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components.auth import current_user, login_page, logout  # noqa: E402
from src.serving.permissions import ROLE_LABELS, allowed_workspaces  # noqa: E402

st.set_page_config(page_title="Quick-Commerce Intelligence", page_icon="🛒", layout="wide")

VIEWS = ROOT / "app" / "views"
WORKSPACE_PAGES = {
    "inventory": st.Page(VIEWS / "inventory.py", title="Inventory & Supply Chain", icon="📦", url_path="inventory"),
}

user = current_user()
if not user:
    page = st.navigation([st.Page(login_page, title="Sign in", icon="🔐")])
else:
    home = st.Page(VIEWS / "welcome.py", title="Home", icon="🏠", default=True)
    workspaces = [WORKSPACE_PAGES[w] for w in allowed_workspaces(user["role"]) if w in WORKSPACE_PAGES]
    page = st.navigation({"": [home], "Workspaces": workspaces} if workspaces else [home])
    with st.sidebar:
        st.markdown(f"**{user['name']}**  \n{ROLE_LABELS.get(user['role'], user['role'])}")
        if st.button("Sign out"):
            logout()
page.run()
