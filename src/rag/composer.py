"""Answer composer for the business assistant (Phase 6D): template answers, no LLM.

House style (same as the app): a one-line title with scope, headline numbers as `Label — value` points, a small
table, short quoted guidance (the 1–2 most relevant sentences of a retrieved chunk, never a whole section) with a
citation, an "understood as" line and the sources. Numbers come only from a ToolResult; quotes only from Hits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from src.rag.retrieval import Hit, expand

ROUTE_LABELS = {"data": "Data", "policy": "Policy", "method": "How it works", "hybrid": "Data + policy",
                "insufficient": "Data (no causal claim)", "help": "Help", "out_of_scope": "Out of scope",
                "not_allowed": "Not in your workspace", "clarify": "Need more detail"}

EXAMPLES = {
    "inventory": ["Which SKUs are at high risk at Dwarka?", "What is the demand forecast for Dairy & Eggs?",
                  "What should we do when stock falls below the reorder level?"],
    "business": ["What was net revenue last week?", "How is our on-time delivery rate?",
                 "What happens when a delivery is late?"],
    "marketing": ["What are our customer segments?", "What is bought with UrbanPantry Diapers?",
                  "What are the rules for discounts?"],
}


@dataclass
class Answer:
    question: str
    route: str
    markdown: str
    table: pd.DataFrame | None = None
    tool: str | None = None
    citations: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    understood: str = ""
    mode: str = "template"
    latency_ms: float = 0.0
    evidence_markdown: str = ""     # the exact template answer, kept when the optional LLM rephrased it (6G)

    @property
    def route_label(self) -> str:
        return ROUTE_LABELS.get(self.route, self.route)


def best_sentences(text: str, question: str, n: int = 2) -> str:
    """The n sentences of a chunk that share the most words with the question, in their original order."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if len(s.split()) >= 4]
    if not sentences:
        return text
    wanted = set(expand(question))
    scored = [(len(wanted & set(expand(s))), i) for i, s in enumerate(sentences)]
    keep = sorted(i for _, i in sorted(scored, key=lambda x: (-x[0], x[1]))[:n])
    return " ".join(sentences[i] for i in keep)


def _quotes(hits: list[Hit], question: str, limit: int = 2) -> tuple[str, list[str]]:
    lines, cites = [], []
    for h in hits[:limit]:
        if h.citation in cites:
            continue
        lines.append(f"> {best_sentences(h.text, question)}\n>\n> — *{h.citation}*")
        cites.append(h.citation)
    return "\n\n".join(lines), cites


def _understood(found: dict[str, str], tool_title: str | None) -> str:
    parts = [f"{k} {v}" for k, v in found.items()]
    if tool_title:
        parts.insert(0, tool_title)
    return " · ".join(parts)


def compose(question: str, route: str, *, result=None, hits: list[Hit] | None = None, found: dict | None = None,
            workspaces: list[str] | None = None, message: str = "") -> Answer:
    """Build the answer for one route from the evidence gathered by the assistant."""
    hits, found, workspaces = hits or [], found or {}, workspaces or []
    parts: list[str] = []
    table, cites, sources = None, [], []

    if route in ("data", "hybrid", "insufficient") and result is not None:
        if route == "insufficient":
            parts.append("**The data shows what happened, not why.** Here are the relevant numbers; possible "
                         "causes need investigation (see the anomaly and delivery pages).")
        parts.append(f"**{result.title}** — {result.scope}" + (f", as of {result.as_of}" if result.as_of else ""))
        parts.append("\n".join(f"- **{k}** — {v}" for k, v in result.headline.items()))
        if result.note:
            parts.append(f"*{result.note}*")
        if result.total_rows > len(result.table):       # a partial list must say so (6G finding)
            parts.append(f"*The table lists {len(result.table)} of {result.total_rows} rows; items not shown may exist.*")
        table = result.table if len(result.table) else None
        sources.append(f"{result.source} (as of {result.as_of})" if result.as_of else result.source)
    if route == "hybrid":
        quotes, cites = _quotes([h for h in hits if h.doc_type == "policy"], question)
        parts.append("**Policy guidance**\n\n" + quotes if quotes else
                     "*No policy section matched this question closely enough to quote.*")
    if route in ("policy", "method"):
        quotes, cites = _quotes(hits, question)
        head = "**What our policies say**" if route == "policy" else "**How it works**"
        parts.append(f"{head}\n\n{quotes}")
    if route == "not_allowed":
        parts.append(message)
        if hits:
            quotes, cites = _quotes([h for h in hits if h.doc_type == "policy"], question)
            if quotes:
                parts.append("**Policy guidance you can still use**\n\n" + quotes)
    if route == "clarify":
        parts.append(message)
    if route in ("help", "out_of_scope"):
        if route == "out_of_scope":
            parts.append("I can only answer questions about this business's data and policies: stock, sales, "
                         "delivery, customers, promotions, and how our metrics and models work.")
        else:
            parts.append("I answer questions about your workspace's data and our policies, with sources.")
        examples = [q for ws in workspaces for q in EXAMPLES.get(ws, [])][:6]
        if examples:
            parts.append("**Try asking**\n" + "\n".join(f"- {q}" for q in examples))

    sources += cites
    understood = _understood(found, result.title if result is not None else None)
    return Answer(question, route, "\n\n".join(p for p in parts if p), table, getattr(result, "tool", None),
                  cites, sources, understood)
