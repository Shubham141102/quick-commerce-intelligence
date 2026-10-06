"""Shared page pieces: styling, page header, sidebar profile, workspace guard, notes."""

from __future__ import annotations

import html

import streamlit as st

from app.components.auth import current_user, logout
from app.components.personas import PERSONAS
from src.serving.permissions import ROLE_LABELS, WORKSPACES, AccessDenied, require
from src.serving.queries import snapshot_info

_CSS = """
<style>
/* KPI cards */
div[data-testid="stMetric"] {
  background: rgba(15, 118, 110, 0.05); border: 1px solid rgba(15, 118, 110, 0.18);
  border-radius: 0.6rem; padding: 0.7rem 0.9rem;
}
div[data-testid="stMetricLabel"] p { font-size: 0.85rem; opacity: 0.8; }
/* page header band */
.qc-hero { display: flex; gap: 1rem; align-items: center; padding: 1rem 1.25rem; margin-bottom: 0.5rem;
  border-radius: 0.8rem; background: linear-gradient(90deg, rgba(15,118,110,0.12), rgba(15,118,110,0.02));
  border: 1px solid rgba(15, 118, 110, 0.18); }
.qc-hero .qc-icon { font-size: 2.4rem; line-height: 1; }
.qc-hero h1 { font-size: 1.7rem; margin: 0; padding: 0; }
.qc-hero p { margin: 0.2rem 0 0 0; opacity: 0.8; }
.qc-chip { display: inline-block; font-size: 0.75rem; padding: 0.1rem 0.55rem; margin: 0.35rem 0.35rem 0 0;
  border-radius: 1rem; background: rgba(15, 118, 110, 0.12); }
.qc-question { font-size: 0.95rem; opacity: 0.85; margin: 0.2rem 0 0.6rem 0; }
/* sidebar profile */
.qc-avatar { width: 2.6rem; height: 2.6rem; border-radius: 50%; background: #0F766E; color: white;
  display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 1.05rem; }
.qc-profile { display: flex; gap: 0.7rem; align-items: center; margin-bottom: 0.4rem; }
.qc-profile small { opacity: 0.75; }
.block-container { padding-top: 2rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


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


def page_header(workspace: str) -> None:
    """Persona header band: icon, title, what the page is for, data chips."""
    inject_css()
    p, info = PERSONAS[workspace], snapshot_info()
    user = current_user() or {}
    published = info.get("published_at", "?").replace("T", " ").replace("Z", " UTC")
    chips = [f"👤 {ROLE_LABELS.get(user.get('role', ''), '')}", f"🗓️ Data Apr–Sep 2025 · {info.get('profile', '?')} run",
             f"📦 Snapshot published {published}"]
    st.markdown(
        f"<div class='qc-hero'><div class='qc-icon'>{p['icon']}</div><div><h1>{html.escape(p['title'])}</h1>"
        f"<p>{html.escape(p['tagline'])}</p>"
        + "".join(f"<span class='qc-chip'>{html.escape(c)}</span>" for c in chips) + "</div></div>",
        unsafe_allow_html=True)


def tab_intro(question: str) -> None:
    """One line at the top of a tab: the business question it answers."""
    st.markdown(f"<div class='qc-question'>❓ <b>{html.escape(question)}</b></div>", unsafe_allow_html=True)


def sidebar_profile() -> None:
    user = current_user()
    if not user:
        return
    initials = "".join(w[0] for w in user["name"].replace("(demo)", "").split()[:2]).upper()
    with st.sidebar:
        inject_css()
        st.markdown(f"<div class='qc-profile'><div class='qc-avatar'>{html.escape(initials)}</div><div>"
                    f"<b>{html.escape(user['name'])}</b><br><small>{html.escape(ROLE_LABELS.get(user['role'], ''))}"
                    "</small></div></div>", unsafe_allow_html=True)
        if st.button("Sign out", icon=":material/logout:", width="stretch"):
            logout()
        st.divider()
        with st.expander("About the data"):
            freshness_banner()


def freshness_banner() -> None:
    info = snapshot_info()
    st.caption(f"📦 **Precomputed snapshot** — pipeline run `{info.get('pipeline_run_id', '?')}`, published "
               f"{info.get('published_at', '?')} from generation run `{info.get('generation_run_id', '?')}`. "
               "The app reads published files; no pipeline is running live.")


def advisory_note() -> None:
    st.info("**Advisory only.** Risk tiers and suggested quantities support a manager's decision; "
            "nothing here places an order or changes stock.", icon="ℹ️")
