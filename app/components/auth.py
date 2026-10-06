"""Demo login (Project_Plan_v2.md §10.1, §16.2): users and bcrypt hashes come from Streamlit secrets.

Authentication = who is logged in (here). Authorization = what that role may open; it is enforced again
inside every data function (src/serving/queries.py), not only by hiding pages.
"""

from __future__ import annotations

import bcrypt
import streamlit as st

from src.serving.permissions import ROLE_LABELS


def configured_users() -> dict:
    try:
        return {name: dict(user) for name, user in st.secrets["auth"]["users"].items()}
    except Exception:  # noqa: BLE001  (no secrets file / no [auth] section)
        return {}


def current_user() -> dict | None:
    return st.session_state.get("user")


def verify(username: str, password: str) -> dict | None:
    user = configured_users().get(username.strip().lower())
    if user and bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        return {"username": username.strip().lower(), "name": user["name"], "role": user["role"]}
    return None


def login_page() -> None:
    """Two columns: what the platform is (left), the sign-in card (right). After sign-in the user lands
    directly in their own workspace (app/Home.py makes it the default page)."""
    from app.components.personas import PERSONAS
    from app.components.ui import inject_css

    inject_css()
    users = configured_users()
    left, gap, right = st.columns([1.25, 0.1, 1])
    with right:
        with st.container(border=True):
            st.markdown("### 🔐 Sign in")
            st.caption("Use your persona's demo account. You will land directly in your workspace.")
            if not users:
                st.error("No demo users are configured.")
                st.code("python -m scripts.create_demo_secrets", language="bash")
                st.caption("This writes .streamlit/secrets.toml (git-ignored). On Streamlit Cloud, paste its "
                           "contents into the app's Secrets settings.")
            else:
                with st.form("login", border=False):
                    username = st.text_input("Username", placeholder="e.g. inventory")
                    password = st.text_input("Password", type="password")
                    submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
                if submitted:
                    user = verify(username, password)
                    if user:
                        st.session_state["user"] = user
                        st.rerun()
                    st.error("Wrong username or password.")
        if users:
            with st.expander("Demo accounts"):
                st.markdown("\n".join(f"- **{name}** → {ROLE_LABELS.get(u['role'], u['role'])}"
                                      for name, u in users.items()))
                st.caption("Passwords are printed by `python -m scripts.create_demo_secrets` "
                           "(see docs/07_inventory_workspace.md). Demo-grade login, not production security.")
    with left:
        st.markdown("## 🛒 Quick-Commerce Intelligence")
        st.markdown("Decision support for a dark-store grocery business: **12 stores, 3 cities, April–September "
                    "2025**. Every number comes from a PySpark Medallion pipeline (Bronze → Silver → Gold → ML) "
                    "over synthetic data.")
        for p in PERSONAS.values():
            with st.container(border=True):
                status = " · *coming soon*" if p.get("on_hold") else ""
                st.markdown(f"**{p['icon']} {p['title']}**{status}  \n{p['tagline']}")
                st.caption(" · ".join(p["use_cases"]))


def logout() -> None:
    st.session_state.pop("user", None)
    st.rerun()
