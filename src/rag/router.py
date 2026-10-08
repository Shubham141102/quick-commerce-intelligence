"""Question router for the business assistant (Phase 6D): rule-based, no LLM.

Routes
- data          a question about numbers → one whitelisted tool (src/rag/tools.py)
- policy        "what should we do / is it allowed" → policy chunks
- method        "how is X calculated / how does the model work" → metric definitions and model cards
- hybrid        a data question that also asks what to do → tool + policy chunks
- insufficient  "why did X happen" → the observed numbers, explicitly without a causal claim
- help          greetings / "what can you do" → example questions
- out_of_scope  nothing business-related (decided after retrieval finds nothing either)

Tool choice: every tool has trigger phrases; the tool whose matched phrases are longest wins (so "at risk customers"
beats "at risk"). Phrases match whole words only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.rag.entities import Entities
from src.rag.retrieval import stem

TOOL_PHRASES: dict[str, tuple[str, ...]] = {
    "inventory_overview": ("stock position", "inventory overview", "stock health", "stock status", "zero stock",
                           "inventory status", "how is our stock", "stock situation"),
    "at_risk_skus": ("at risk", "high risk", "medium risk", "stockout risk", "run out", "running out", "low stock",
                     "risk tier", "risky skus", "skus at risk"),
    "suggested_orders": ("suggested order", "suggested orders", "how much to order", "how much should we order",
                         "what to reorder", "what should we reorder", "order quantity", "purchase order",
                         "replenishment quantity", "how many units to order"),
    "category_forecast": ("forecast", "expected demand", "demand next", "next 7 days", "next week", "expected sales",
                          "predicted demand", "how many units will"),
    "forecast_reliability": ("forecast reliability", "trust the forecast", "forecast accuracy", "reliable forecast",
                             "how accurate is the forecast", "wape by category", "reliability"),
    "lost_sales": ("lost sales", "lost revenue", "cost of stockouts", "sales lost", "lost units", "stockouts cost"),
    "sales_overview": ("revenue", "net revenue", "sales performance", "gmv", "aov", "average order value",
                       "completed orders", "how are we selling", "how did we sell", "cancellation rate", "sales"),
    "top_products": ("top products", "best selling", "best-selling", "top selling", "best sellers", "top sellers",
                     "top skus by revenue", "highest revenue products"),
    "delivery_overview": ("delivery performance", "on-time", "on time", "delivery time", "late deliveries",
                          "failed deliveries", "deliveries", "delivery"),
    "cancellations": ("cancellations", "cancellation reasons", "why orders are cancelled", "cancelled orders",
                      "cancellation"),
    "anomalies": ("anomaly", "anomalies", "spike", "spikes", "unusual", "payment failure", "payment failures",
                  "outage", "outages"),
    "segments": ("segment", "segments", "customer groups", "customer segments", "who are our customers",
                 "personas", "clusters"),
    "bought_with": ("bought with", "bought together", "buy with", "goes with", "purchased with", "pairs with"),
    "recommendation_accuracy": ("recommendation", "recommendations", "precision@10", "precision at 10"),
    "retention": ("retention", "lapsed", "churn", "at risk customers", "inactive customers", "win back",
                  "active customers", "repeat rate", "cohort"),
    "promotions": ("promotion results", "promotions", "promotion", "uplift", "promo", "campaign results"),
}

POLICY_CUES = ("policy", "policies", "should we", "should i", "should a", "should the", "what should", "rule",
               "rules", "allowed", "procedure", "guideline", "what do we do", "what to do", "what happens when",
               "what happens if", "can a", "can we", "can i", "must", "eligible", "eligibility", "how long",
               "timeline", "escalate", "escalation", "approve", "approval", "limit", "when should", "recommend doing",
               "what does policy", "according to policy")
# "How is X calculated / chosen / estimated", definitions, formulas: always a method question, even when X is a
# data topic ("How is net revenue calculated?").
METHOD_STRONG = re.compile(
    r"\bhow (is|are|was|were|do|does|did) .*\b(calculated|computed|defined|chosen|selected|set|estimated|measured|"
    r"decided|built|trained|work|works|made|scored|named)\b|\bdefinition\b|\bformula\b|\bmethodology\b|"
    r"\bwhat is meant by\b|\bexplain how\b|\bhow does the (model|forecast|detector|recommend)|"
    # passive voice "how is / are X <verb>ed" asks how something is done (revised 2026-10-08, 6F)
    r"\bhow (is|are|was|were) (\w+ ){1,4}?\w+ed\b")
# Explicit report requests: a single generic topic word plus one of these is still a data question (6F revision).
REPORT_CUES = ("show", "list", "give me", "total", "count", "number of", "top", "compare", "trend", "overview",
               "report", "breakdown")
# Weaker signals: only a method question when no data topic matched ("What does lift mean?").
METHOD_WEAK = ("what does", "mean", "means", "how is", "how are", "how does", "what is a", "what is an")
DATA_CUES = ("which", "how many", "how much", "show", "list", "top", "give me", "current", "total", "compare",
             "number of", "count", "what was", "what were", "what is our", "what are our", "did we", "were there",
             "is there", "are there", "status", "overview")
WHY = re.compile(r"\bwhy\b|\bwhat caused\b|\breason for\b|\bcause of\b|\bexplain the (drop|fall|rise|increase)")
HELP = re.compile(r"^\s*(hi|hello|hey|help|what can you do|what can i ask|who are you)\b")


@dataclass(frozen=True)
class Route:
    kind: str                         # data / policy / method / hybrid / insufficient / help / out_of_scope
    tool: str | None = None
    doc_types: tuple[str, ...] | None = None
    reason: str = ""
    generic: bool = False             # data topic known only from one generic word (6F): prefer a matching policy


def _norm(text: str) -> str:
    """Words → stems, space-separated, padded: phrase matching ignores endings (lapsing ≈ lapsed) and hyphens."""
    return " " + " ".join(stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())) + " "


def _has(text: str, phrase: str) -> bool:
    return _norm(phrase) in _norm(text)


def _any(text: str, phrases) -> list[str]:
    return [p for p in phrases if _has(text, p)]


def _matched(question: str) -> dict[str, list[str]]:
    return {t: _any(question, phrases) for t, phrases in TOOL_PHRASES.items()}


def pick_tool(question: str) -> tuple[str | None, int]:
    """Best tool by total length of its matched trigger phrases (0 = none)."""
    scored = {t: sum(len(p) for p in ps) for t, ps in _matched(question).items()}
    best = max(scored, key=scored.get)
    return (best, scored[best]) if scored[best] else (None, 0)


def route_question(question: str, entities: Entities | None = None) -> Route:
    text = question.lower().strip()
    if HELP.search(text) and len(text.split()) <= 6:
        return Route("help", reason="greeting or help request")
    tool, _ = pick_tool(text)
    policy, data = _any(text, POLICY_CUES), _any(text, DATA_CUES)
    has_entities = bool(entities and (entities.store_ids or entities.tiers or entities.start or entities.limit
                                      or entities.product_ids or entities.category_ids))
    # 1. "why" questions: show what happened, never invent a cause
    if WHY.search(text) and (tool or data):
        return Route("insufficient", tool, ("policy", "method"), "asks for a cause the data cannot prove")
    # 2. how something is calculated / chosen / defined
    if METHOD_STRONG.search(text):
        return Route("method", None, ("metric", "method"), "method question")
    # 3. numbers + what to do
    if policy and tool and (data or has_entities):
        return Route("hybrid", tool, ("policy",), f"data + policy cue: {policy[0]}")
    # 4. what to do / what is allowed
    if policy:
        return Route("policy", None, ("policy",), f"policy cue: {policy[0]}")
    # 5. "what does X mean" with no data topic
    if not tool and _any(text, METHOD_WEAK):
        return Route("method", None, ("metric", "method"), "definition question")
    # 6. a data topic. When it rests on exactly ONE generic word ("delivery", "cancellation") with no report cue and
    #    no entity, the assistant prefers a clearly matching policy section (6F revision).
    if tool:
        stems = {_norm(p).strip() for p in _matched(text)[tool]}       # "delivery" / "deliveries" count once
        generic = (len(stems) == 1 and " " not in next(iter(stems))
                   and not _any(text, REPORT_CUES) and not has_entities)
        return Route("data", tool, None, "data topic" + (f" + cue: {data[0]}" if data else ""), generic=generic)
    # 7. anything else: search all knowledge; becomes out_of_scope when nothing relevant is found
    return Route("policy", None, None, "no data topic: search all knowledge")
