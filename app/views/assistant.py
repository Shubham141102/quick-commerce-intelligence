"""Business assistant page (Phase 6E): chat over the user's workspace data and our policies, with sources.

One page for every signed-in persona (in its sidebar group; the admin's spans all workspaces). Numbers come only
from the role-checked tools, guidance only from retrieved policy / method chunks (src/rag). The knowledge index and
lookups are built once per snapshot and cached.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components.auth import current_user  # noqa: E402
from app.components.insights import style_tiers  # noqa: E402
from app.components.personas import PERSONAS  # noqa: E402
from app.components.ui import header, notes  # noqa: E402
from src.rag.assistant import Assistant, log_record  # noqa: E402
from src.rag.composer import EXAMPLES  # noqa: E402
from src.rag.entities import Lookups  # noqa: E402
from src.rag.retrieval import Retriever  # noqa: E402
from src.serving.db import get_snapshot  # noqa: E402
from src.serving.permissions import allowed_workspaces  # noqa: E402
from src.serving.queries import snapshot_info  # noqa: E402

BADGE = {"data": "blue", "hybrid": "blue", "policy": "violet", "method": "gray", "insufficient": "orange",
         "clarify": "orange", "not_allowed": "red", "out_of_scope": "gray", "help": "gray"}
MODE_LABEL = "Template mode (no LLM)"


@st.cache_resource(show_spinner=False)
def _assistant(snapshot_run: str) -> Assistant:      # keyed by the snapshot, so a new publish rebuilds it
    return Assistant(Retriever(get_snapshot().query("SELECT * FROM rag_chunks")), Lookups.from_snapshot())


def _render(answer) -> None:
    st.markdown(f":{BADGE.get(answer.route, 'gray')}-badge[{answer.route_label}]")
    st.markdown(answer.markdown)
    if answer.table is not None:
        st.dataframe(style_tiers(answer.table, ["risk_tier", "severity"]), hide_index=True, width="stretch")
    if answer.sources or answer.understood:
        with st.expander("Evidence & sources"):
            points = []
            if answer.understood:
                points.append(f"**Understood as** — {answer.understood}")
            points += [f"**Source** — {s}" for s in answer.sources]
            points.append(f"**Answered in** — {answer.latency_ms:.0f} ms · {answer.mode}")
            notes(*points)


def assistant() -> None:
    user = current_user()
    if not user:
        st.warning("Please sign in first.")
        st.stop()
    role = user["role"]
    mine = allowed_workspaces(role)
    crumb = PERSONAS[mine[0]]["title"] if len(mine) == 1 else "All workspaces"
    header(crumb, "Assistant", "Ask about your data and our policies — every answer shows its sources.",
           status=MODE_LABEL)

    bot = _assistant(snapshot_info().get("pipeline_run_id", ""))
    chat = st.session_state.setdefault("assistant_chat", [])
    log = st.session_state.setdefault("assistant_log", [])

    examples = [q for ws in mine for q in EXAMPLES.get(ws, [])][: 3 if len(mine) == 1 else 4]
    picked = None
    if examples:
        cols = st.columns(len(examples))
        for i, (col, q) in enumerate(zip(cols, examples)):
            if col.button(q, key=f"example_{i}", width="stretch"):
                picked = q
    asked = st.chat_input("Ask a question about your workspace or our policies…")
    question = asked or picked
    if question:
        answer = bot.ask(question, role)
        chat.append(answer)
        log.append(log_record(answer, role))

    for answer in chat:
        with st.chat_message("user"):
            st.markdown(answer.question)
        with st.chat_message("assistant"):
            _render(answer)

    if chat:
        if st.button("Clear conversation", icon=":material/delete_sweep:"):
            chat.clear()
            st.rerun()
    else:
        notes("Numbers come only from your workspace's data · guidance only from our policies and method notes",
              "Questions about another workspace's data are refused")


if __name__ == "__main__":   # run as a script (tests)
    assistant()
