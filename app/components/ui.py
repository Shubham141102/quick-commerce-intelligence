"""Shared page pieces (enterprise design system): styling, page header, business-question callout,
sidebar profile, workspace guard, notes. Colours: slate neutrals, one accent blue (#1D4ED8), status colours
only for status. Theme values live in .streamlit/config.toml; this CSS only refines components."""

from __future__ import annotations

import html

import streamlit as st

from app.components.auth import current_user, logout
from app.components.personas import PERSONAS
from src.serving.permissions import ROLE_LABELS, WORKSPACES, AccessDenied, require
from src.serving.queries import snapshot_info

APP_NAME = "Quick-Commerce Intelligence"
DATA_PERIOD = "Apr – Sep 2025"

_CSS = """
<style>
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1440px; }
header[data-testid="stHeader"] { background: transparent; }

/* KPI cards */
div[data-testid="stMetric"] {
  background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 16px 12px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
div[data-testid="stMetricLabel"] p {
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: #64748B;
}
div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; color: #0F172A; }

/* page header */
.qc-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 1.5rem;
  padding: 0 0 14px 0; margin: 0 0 4px 0; border-bottom: 1px solid #E2E8F0; }
.qc-crumb { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
  color: #94A3B8; margin-bottom: 6px; }
.qc-crumb b { color: #1D4ED8; font-weight: 600; }
.qc-head h1 { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.01em; color: #0F172A; margin: 0; padding: 0; }
.qc-head p { margin: 4px 0 0 0; color: #475569; font-size: 0.9rem; }
.qc-meta { text-align: right; font-size: 0.75rem; color: #64748B; line-height: 1.6; white-space: nowrap; }
.qc-status { display: inline-flex; align-items: center; gap: 6px; font-weight: 600; color: #067647; }
.qc-dot { width: 7px; height: 7px; border-radius: 50%; background: #12B76A; display: inline-block; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 1.75rem; border-bottom: 1px solid #E2E8F0; }
.stTabs [data-baseweb="tab"] { padding: 12px 2px 10px 2px; }
.stTabs [data-baseweb="tab"] p { font-size: 0.9rem; font-weight: 500; color: #475569; }
.stTabs [aria-selected="true"] p { color: #0F172A; font-weight: 600; }

/* business question callout */
.qc-question { display: flex; gap: 12px; align-items: baseline; background: #FFFFFF; border: 1px solid #E2E8F0;
  border-left: 3px solid #1D4ED8; border-radius: 6px; padding: 10px 14px; margin: 8px 0 18px 0; }
.qc-question span { font-size: 0.66rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  color: #1D4ED8; white-space: nowrap; }
.qc-question div { font-size: 0.92rem; font-weight: 500; color: #1E293B; }

/* panels */
div[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; }
.qc-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  color: #64748B; margin: 2px 0 8px 0; }

/* sidebar */
.qc-profile { display: flex; gap: 10px; align-items: center; padding: 10px 12px; background: #1E293B;
  border-radius: 8px; margin: 4px 0 10px 0; }
.qc-avatar { width: 34px; height: 34px; border-radius: 50%; background: #334155; color: #E2E8F0;
  font-weight: 600; font-size: 0.8rem; display: flex; align-items: center; justify-content: center; flex: none; }
.qc-profile b { color: #F1F5F9; font-size: 0.85rem; font-weight: 600; }
.qc-profile small { color: #94A3B8; font-size: 0.72rem; }
.qc-side-label { font-size: 0.66rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  color: #64748B; margin: 14px 0 6px 0; }
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


def _published(info: dict) -> str:
    return info.get("published_at", "?").replace("T", " ").replace("Z", " UTC")


def page_header(workspace: str, section: str | None = None) -> None:
    """Breadcrumb (app › workspace), the section title and the workspace purpose on the left; data status on
    the right; a hairline underneath."""
    inject_css()
    p, info = PERSONAS[workspace], snapshot_info()
    st.markdown(
        "<div class='qc-head'><div>"
        f"<div class='qc-crumb'>{html.escape(APP_NAME)} &nbsp;›&nbsp; <b>{html.escape(p['title'])}</b></div>"
        f"<h1>{html.escape(section or p['title'])}</h1><p>{html.escape(p['tagline'])}</p></div>"
        "<div class='qc-meta'><span class='qc-status'><span class='qc-dot'></span>Snapshot current</span><br>"
        f"Data period {DATA_PERIOD} · {html.escape(info.get('profile', '?'))} profile<br>"
        f"Published {html.escape(_published(info))}</div></div>",
        unsafe_allow_html=True)


def tab_intro(question: str) -> None:
    """The business question a tab answers, as a quiet callout."""
    st.markdown(f"<div class='qc-question'><span>Business question</span><div>{html.escape(question)}</div></div>",
                unsafe_allow_html=True)


def label(text: str) -> None:
    """Small uppercase label above a panel's content (e.g. KEY TAKEAWAYS)."""
    st.markdown(f"<div class='qc-label'>{html.escape(text)}</div>", unsafe_allow_html=True)


def sidebar_profile() -> None:
    user = current_user()
    if not user:
        return
    initials = "".join(w[0] for w in user["name"].replace("(demo)", "").split()[:2]).upper()
    with st.sidebar:
        inject_css()
        st.markdown("<div class='qc-side-label'>Signed in</div>"
                    f"<div class='qc-profile'><div class='qc-avatar'>{html.escape(initials)}</div><div>"
                    f"<b>{html.escape(user['name'])}</b><br><small>{html.escape(ROLE_LABELS.get(user['role'], ''))}"
                    "</small></div></div>", unsafe_allow_html=True)
        if st.button("Sign out", icon=":material/logout:", width="stretch"):
            logout()
        st.markdown("<div class='qc-side-label'>Data</div>", unsafe_allow_html=True)
        with st.expander("About this snapshot"):
            freshness_banner()


def freshness_banner() -> None:
    info = snapshot_info()
    st.caption(f"**Precomputed snapshot** — pipeline run `{info.get('pipeline_run_id', '?')}`, published "
               f"{_published(info)} from generation run `{info.get('generation_run_id', '?')}`. "
               "The app reads published files; no pipeline is running live.")


def advisory_note() -> None:
    st.info("**Advisory only.** Risk tiers and suggested quantities support a manager's decision; "
            "nothing here places an order or changes stock.", icon=":material/info:")
