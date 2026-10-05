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
    st.title("Quick-Commerce Intelligence")
    st.caption("Sign in with a demo account to open your workspace.")
    users = configured_users()
    if not users:
        st.error("No demo users are configured.")
        st.code("python -m scripts.create_demo_secrets", language="bash")
        st.caption("This writes .streamlit/secrets.toml (git-ignored). On Streamlit Cloud, paste its contents "
                   "into the app's Secrets settings.")
        return
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        user = verify(username, password)
        if user:
            st.session_state["user"] = user
            st.rerun()
        st.error("Wrong username or password.")
    with st.expander("Demo accounts"):
        st.markdown("\n".join(f"- **{name}** → {ROLE_LABELS.get(u['role'], u['role'])}" for name, u in users.items()))
        st.caption("Passwords are printed by `python -m scripts.create_demo_secrets` (see docs/07_inventory_workspace.md). "
                   "Demo-grade login, not production security.")


def logout() -> None:
    st.session_state.pop("user", None)
    st.rerun()
