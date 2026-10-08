"""The business assistant (Phase 6D): question → entities → route → tool and / or retrieval → composed answer.

    assistant = Assistant(Retriever(chunks), Lookups.from_snapshot())
    answer = assistant.ask("Which SKUs are at high risk at Dwarka and what should we do?", role="inventory_manager")

Numbers come only from whitelisted, role-checked tools; guidance only from retrieved chunks. A question about
another workspace's data is refused (the policy part is still answered). Every answer carries a log record
(question, route, tool, citations, latency) for evaluation (Phase 6F).
"""

from __future__ import annotations

import time

from src.rag.composer import Answer, compose
from src.rag.entities import Entities, Lookups, extract
from src.rag.retrieval import Retriever
from src.rag.router import route_question
from src.rag.tools import TOOLS, ToolInputError, run_tool
from src.serving.permissions import WORKSPACES, AccessDenied, allowed_workspaces


def params_from(entities: Entities) -> dict:
    return {"store_ids": entities.store_ids or None, "tiers": entities.tiers or None,
            "category_id": entities.category_ids[0] if entities.category_ids else None,
            "product_id": entities.product_ids[0] if entities.product_ids else None,
            "detectors": entities.detectors or None, "start": entities.start, "end": entities.end,
            "limit": min(entities.limit, 20) if entities.limit else None}


class Assistant:
    def __init__(self, retriever: Retriever, lookups: Lookups):
        self.retriever = retriever
        self.lookups = lookups

    def ask(self, question: str, role: str) -> Answer:
        t0 = time.perf_counter()
        answer = self._answer(question.strip(), role)
        answer.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return answer

    def _answer(self, question: str, role: str) -> Answer:
        workspaces = allowed_workspaces(role)
        if not question:
            return compose(question, "help", workspaces=workspaces)
        entities = extract(question, self.lookups)
        route = route_question(question, entities)
        if route.kind == "help":
            return compose(question, "help", workspaces=workspaces)

        hits = self.retriever.search(question, doc_types=route.doc_types) if route.doc_types or route.kind == "policy" \
            else []
        if route.kind in ("policy", "method"):
            if not hits:
                return compose(question, "out_of_scope", workspaces=workspaces, found=entities.found)
            return compose(question, route.kind, hits=hits, found=entities.found)

        if route.tool is None:          # "why …?" without a data topic
            return compose(question, "clarify", found=entities.found, message=(
                "The data shows what happened, not why. Ask about a specific measure — for example "
                "*net revenue last week* or *on-time delivery at Dwarka* — and I will show the numbers."))
        if route.kind == "data" and route.generic:     # e.g. "How many minutes do we promise for delivery?"
            policy_hits = self.retriever.search(question, doc_types=("policy",))
            if policy_hits:
                return compose(question, "policy", hits=policy_hits, found=entities.found)
        tool = TOOLS[route.tool]
        if tool.workspace not in workspaces:
            policy_hits = self.retriever.search(question, doc_types=("policy",))
            return compose(question, "not_allowed", hits=policy_hits, found=entities.found, message=(
                f"**{tool.title}** is part of the **{WORKSPACES[tool.workspace]}** workspace, which your role "
                "cannot open. Ask a colleague with that role, or ask about your own workspace."))
        try:
            result = run_tool(tool.name, role, params_from(entities), self.lookups)
        except ToolInputError as exc:
            return compose(question, "clarify", found=entities.found, message=str(exc))
        except AccessDenied:            # defence in depth: the wrapped query checks the role again
            return compose(question, "not_allowed", found=entities.found,
                           message=f"Your role cannot open the {WORKSPACES[tool.workspace]} workspace.")
        return compose(question, route.kind, result=result, hits=hits, found=entities.found)


def log_record(answer: Answer, role: str) -> dict:
    """One row for meta_rag_runs."""
    return {"question": answer.question, "role": role, "route": answer.route, "tool": answer.tool or "",
            "citations": "|".join(answer.citations), "sources": "|".join(answer.sources), "mode": answer.mode,
            "latency_ms": answer.latency_ms}
