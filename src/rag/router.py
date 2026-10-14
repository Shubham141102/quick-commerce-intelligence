"""Question router for the business assistant (Phase 6D): rule-based, no LLM.

Routes
- data          a question about numbers → one whitelisted tool (src/rag/tools.py)
- policy        "what should we do / is it allowed" → policy chunks
- method        "how is X calculated / how does the model work" → metric definitions and model cards
- hybrid        a data question that also asks what to do → tool + policy chunks
- insufficient  "why did X happen" → the observed numbers, explicitly without a causal claim
- help          greetings / "what can you do" → example questions
- out_of_scope  schema / lineage questions (→ a pointer to the Lineage & tables page), or nothing business-related
                (decided after retrieval finds nothing either)

Tool choice: every tool has trigger phrases; the tool whose distinct matched phrases are longest wins (so "at risk
customers" beats "at risk"); phrases match whole words / stems. Counting questions ("how many rows…") only compete
between tools that report that measure (7E revision). A word belongs to one tool only: the longest phrase claims it
first (7F revision).
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
    # ranking by forecast (7F) — phrases from what the tool does: which items / categories lead future demand
    "top_forecast": ("most demanded", "highest demand", "in highest demand", "highest expected demand", "most in demand",
                     "sell the most", "will sell most", "top selling next week", "biggest demand", "demand ranking"),
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
    # Data Engineer (7E) — phrases from what each tool does (written before looking at the evaluation set)
    "pipeline_status": ("pipeline", "pipeline run", "pipeline runs", "last run", "run history", "failed runs",
                        "run failed", "runs failed", "success rate", "pipeline status", "pipeline health"),
    "pipeline_rows": ("rows processed", "rows written", "rows loaded", "records processed", "row count", "row counts",
                      "how many rows", "data volume", "rows rejected"),
    "data_quality": ("data quality", "clean rate", "quality checks", "dirty data", "data issues",
                     "duplicates removed", "validation results"),
    "quarantine": ("quarantine", "quarantined", "rejected rows", "rejected records", "quarantine reasons"),
    "model_health": ("model registry", "model version", "model versions", "live models", "model health",
                     "model performance", "models in production", "latest models"),
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


# What each tool COUNTS (7E revision): "how many <rows / records / runs / models>" goes to a tool that reports it.
TOOL_MEASURES = {"pipeline_rows": {"row", "record"}, "quarantine": {"row", "record"}, "pipeline_status": {"run"},
                 "model_health": {"model"}}
MEASURE_PHRASES = {"how many rows", "row count", "row counts", "how many records"}   # the ask itself, not a topic
COUNT_ASK = re.compile(r"\b(?:how many|number of|count of)\s+((?:\w+\s+){0,2}\w+)")
COUNT_STOP = frozenset("""can could are is was were be been did do does have has had will would should may might
the a an in of for at on to from by with that which""".split())
# Schema / catalog questions: answered by the Data Engineer catalog page, not by the knowledge base (7E revision).
SCHEMA_CUES = ("column", "columns", "schema", "grain", "lineage", "upstream", "downstream", "data type", "data types")
# Tools whose topic has a policy (policies/*.md: delivery, cancellation, promotion, inventory, refund → revenue). Only
# these may prefer a policy passage for a one-generic-word question; "What is the forecast?" has no policy (7F revision).
POLICY_TOPIC_TOOLS = frozenset({"delivery_overview", "cancellations", "promotions", "inventory_overview",
                                "at_risk_skus", "suggested_orders", "lost_sales", "sales_overview"})


def _distinct(phrases: list[str]) -> list[str]:
    """Matched phrases of one tool counted once: duplicates by stem and phrases inside a longer match are dropped
    ("pipeline" inside "pipeline run"; "pipeline run" = "pipeline runs") — 7E revision."""
    stems = sorted({_norm(p).strip() for p in phrases}, key=len, reverse=True)
    kept: list[str] = []
    for s in stems:
        if not any(f" {s} " in f" {k} " for k in kept):
            kept.append(s)
    return kept


def _scores(question: str, exclude: set[str] = frozenset()) -> dict[str, int]:
    """Each tool's score = total length of its distinct matched phrases. Words are claimed across tools, longest
    phrase first: a phrase overlapping a longer phrase of ANOTHER tool does not count ("highest expected demand"
    claims "expected demand", so the category forecast cannot also score it) — 7F revision. Overlaps within one
    tool are handled by _distinct, as before."""
    excluded = {_norm(p).strip() for p in exclude}
    words = _norm(question).split()
    hits = []
    for order, (tool, phrases) in enumerate(_matched(question).items()):
        for p in _distinct(phrases):
            if p in excluded:
                continue
            n = len(p.split())
            spans = [range(k, k + n) for k in range(len(words) - n + 1) if words[k:k + n] == p.split()]
            hits.append((len(p), order, tool, spans))
    claimed: dict[int, str] = {}
    scores = dict.fromkeys(TOOL_PHRASES, 0)
    for length, _, tool, spans in sorted(hits, key=lambda h: (-h[0], h[1])):
        free = [s for s in spans if all(claimed.get(k, tool) == tool for k in s)]
        if free:
            scores[tool] += length
            claimed.update({k: tool for s in free for k in s})
    return scores


def pick_tool(question: str) -> tuple[str | None, int]:
    """Best tool by the total length of its distinct matched trigger phrases (0 = none). For a counting question
    ("how many rows…"), only tools that report that measure compete, and the one whose topic is also named wins
    ("rows in quarantine" → quarantine; "rows in the last run" → rows processed)."""
    scored = _scores(question)
    ask = COUNT_ASK.search(question.lower())
    if ask:
        # the counted noun phrase stops at the first verb / auxiliary ("how many promotions CAN run" counts promotions)
        words = []
        for w in ask.group(1).split():
            if w in COUNT_STOP:
                break
            words.append(w)
        asked = {stem(w) for w in words}
        candidates = [t for t, measures in TOOL_MEASURES.items() if measures & asked]   # in ownership order
        if candidates:
            topic = _scores(question, MEASURE_PHRASES)
            # the candidate whose topic is named wins; otherwise the measure's primary owner (first in the order)
            best = max(candidates, key=lambda t: (topic[t], -candidates.index(t)))
            return best, max(scored[best], 1)
    best = max(scored, key=scored.get)
    return (best, scored[best]) if scored[best] else (None, 0)


def route_question(question: str, entities: Entities | None = None) -> Route:
    text = question.lower().strip()
    if HELP.search(text) and len(text.split()) <= 6:
        return Route("help", reason="greeting or help request")
    # schema / lineage questions are catalog questions: refuse with a pointer to the Lineage & tables page (7E)
    if _any(text, SCHEMA_CUES):
        return Route("out_of_scope", reason="schema / catalog question")
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
    #    no entity, the assistant prefers a clearly matching policy section (6F revision) — only for topics that have
    #    a policy (7F revision).
    if tool:
        stems = {_norm(p).strip() for p in _matched(text)[tool]}       # "delivery" / "deliveries" count once
        generic = (tool in POLICY_TOPIC_TOOLS and len(stems) == 1 and " " not in next(iter(stems))
                   and not _any(text, REPORT_CUES) and not has_entities)
        return Route("data", tool, None, "data topic" + (f" + cue: {data[0]}" if data else ""), generic=generic)
    # 7. anything else: search all knowledge; becomes out_of_scope when nothing relevant is found
    return Route("policy", None, None, "no data topic: search all knowledge")
