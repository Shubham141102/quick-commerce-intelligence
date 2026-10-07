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


_LOGIN_CSS = """
<style>
.qc-brandpanel { background: #0F172A; border-radius: 12px; padding: 36px 36px 30px 36px; color: #CBD5E1;
  min-height: 560px; display: flex; flex-direction: column; }
.qc-brandpanel .qc-mark { width: 40px; height: 40px; border-radius: 8px; background: #1D4ED8; color: #FFFFFF;
  font-weight: 700; display: flex; align-items: center; justify-content: center; font-size: 0.95rem; }
.qc-brandpanel h2 { color: #F8FAFC; font-size: 1.7rem; font-weight: 700; letter-spacing: -0.01em;
  margin: 22px 0 8px 0; padding: 0; line-height: 1.25; }
.qc-brandpanel .qc-lede { color: #94A3B8; font-size: 0.95rem; line-height: 1.55; margin: 0 0 24px 0; }
.qc-facts { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 26px; }
.qc-fact { border: 1px solid #1E293B; border-radius: 8px; padding: 10px 12px; }
.qc-fact b { display: block; color: #F8FAFC; font-size: 1.2rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.qc-fact small { color: #94A3B8; font-size: 0.72rem; letter-spacing: 0.04em; text-transform: uppercase; }
.qc-ws { display: flex; gap: 12px; align-items: flex-start; padding: 11px 0; border-top: 1px solid #1E293B; }
.qc-ws .qc-code { flex: none; width: 38px; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em;
  color: #60A5FA; padding-top: 2px; }
.qc-ws b { color: #E2E8F0; font-size: 0.88rem; font-weight: 600; }
.qc-ws span { display: block; color: #94A3B8; font-size: 0.78rem; margin-top: 2px; }
.qc-ws em { font-style: normal; color: #F59E0B; font-size: 0.7rem; font-weight: 600; margin-left: 6px; }
.qc-foot { margin-top: auto; padding-top: 18px; color: #64748B; font-size: 0.72rem; }
.qc-signin h3 { font-size: 1.35rem; font-weight: 700; margin: 0 0 4px 0; padding: 0; color: #0F172A; }
.qc-signin p { color: #64748B; font-size: 0.88rem; margin: 0 0 18px 0; }
</style>
"""


def login_page() -> None:
    """Enterprise sign-in: brand panel (left), sign-in form (right). After sign-in the user lands directly
    in their own workspace (app/Home.py makes it the default page)."""
    import html

    from app.components.personas import PERSONAS
    from app.components.ui import inject_css

    inject_css()
    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)
    users = configured_users()
    left, _, right = st.columns([1.15, 0.12, 0.9], vertical_alignment="center")
    with right:
        st.markdown("<div class='qc-signin'><h3>Sign in</h3>"
                    "<p>Use your workspace account. You will land directly in your workspace.</p></div>",
                    unsafe_allow_html=True)
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
            with st.expander("Demo accounts"):
                st.markdown("\n".join(f"- **{name}** — {ROLE_LABELS.get(u['role'], u['role'])}"
                                      for name, u in users.items()))
                st.caption("Passwords are set with `python -m scripts.create_demo_secrets`. "
                           "Demo-grade access control, not production security.")
    with left:
        rows = "".join(
            f"<div class='qc-ws'><div class='qc-code'>{p['code']}</div><div><b>{html.escape(p['title'])}</b>"
            f"{'<em>COMING SOON</em>' if p.get('on_hold') else ''}<span>{html.escape(' · '.join(p['use_cases']))}"
            "</span></div></div>" for p in PERSONAS.values())
        st.markdown(
            "<div class='qc-brandpanel'><div class='qc-mark'>QC</div>"
            "<h2>Quick-Commerce Intelligence</h2>"
            "<p class='qc-lede'>Decision support for a dark-store grocery network: demand, stock, revenue, "
            "operations and customers in one place.</p>"
            "<div class='qc-facts'><div class='qc-fact'><b>12</b><small>Dark stores</small></div>"
            "<div class='qc-fact'><b>3</b><small>Cities</small></div>"
            "<div class='qc-fact'><b>6 mo</b><small>Apr – Sep 2025</small></div></div>"
            f"{rows}"
            "<div class='qc-foot'>PySpark Medallion pipeline (Bronze → Silver → Gold → ML) over synthetic data · "
            "read-only snapshot</div></div>",
            unsafe_allow_html=True)


def logout() -> None:
    st.session_state.pop("user", None)
    st.rerun()
