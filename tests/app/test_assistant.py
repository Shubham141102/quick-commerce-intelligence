"""The business assistant end to end (Phase 6D) on a real published snapshot: every route."""

import pytest

from src.rag.assistant import Assistant, log_record
from src.rag.entities import Lookups
from src.rag.retrieval import Retriever
from src.serving import queries as q
from src.serving.db import get_snapshot


@pytest.fixture(scope="module")
def bot(app_env):
    return Assistant(Retriever(get_snapshot().query("SELECT * FROM rag_chunks")), Lookups.from_snapshot())


def test_data_answer_numbers_equal_the_dashboard(bot):
    a = bot.ask("Which SKUs are at high risk?", "inventory_manager")
    k = q.inventory_kpis("inventory_manager")
    assert a.route == "data" and a.tool == "at_risk_skus"
    assert f"**High risk** — {int(k['high'])}" in a.markdown and a.sources[0].startswith("gld_stockout_risk")
    assert "tier High" in a.understood


def test_hybrid_answer_has_numbers_and_a_cited_policy(bot):
    a = bot.ask("Which SKUs are at high risk and what does the policy say we should do?", "inventory_manager")
    assert a.route == "hybrid" and a.tool == "at_risk_skus"
    assert any(c.startswith("POL-INV §") for c in a.citations) and "> " in a.markdown


def test_policy_and_method_answers_quote_and_cite(bot):
    a = bot.ask("What happens when a delivery takes longer than promised?", "business_analyst")
    assert a.route == "policy" and any(c.startswith("POL-DEL") for c in a.citations) and a.table is None
    m = bot.ask("How is net revenue calculated?", "marketing_manager")
    assert m.route == "method" and "Metric Definitions › Net revenue" in m.citations


def test_other_workspace_data_is_refused(bot):
    a = bot.ask("What was net revenue last week?", "inventory_manager")
    assert a.route == "not_allowed" and a.table is None and "Business & Revenue" in a.markdown
    assert "₹" not in a.markdown                                  # no revenue number leaks


def test_why_questions_get_numbers_without_a_cause(bot):
    a = bot.ask("Why did net revenue fall last week?", "business_analyst")
    assert a.route == "insufficient" and a.tool == "sales_overview" and "not why" in a.markdown


def test_out_of_scope_help_and_clarify(bot):
    out = bot.ask("Will it rain in Mumbai tomorrow?", "inventory_manager")
    assert out.route == "out_of_scope" and out.citations == [] and "Try asking" in out.markdown
    assert bot.ask("hi", "marketing_manager").route == "help"
    c = bot.ask("What is the demand forecast for next week?", "inventory_manager")
    assert c.route == "clarify" and "category" in c.markdown


def test_log_record(bot):
    a = bot.ask("Which promotions had the best uplift?", "marketing_manager")
    rec = log_record(a, "marketing_manager")
    assert rec["route"] == "data" and rec["tool"] == "promotions" and rec["latency_ms"] >= 0
