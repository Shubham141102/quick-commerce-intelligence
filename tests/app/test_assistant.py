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
    a = bot.ask("What happens when a delivery takes longer than promised?", "growth_manager")
    assert a.route == "policy" and any(c.startswith("POL-DEL") for c in a.citations) and a.table is None
    m = bot.ask("How is net revenue calculated?", "growth_manager")
    assert m.route == "method" and "Metric Definitions › Net revenue" in m.citations


def test_other_workspace_data_is_refused(bot):
    a = bot.ask("What was net revenue last week?", "inventory_manager")
    assert a.route == "not_allowed" and a.table is None and "Business & Growth" in a.markdown
    assert "₹" not in a.markdown                                  # no revenue number leaks


def test_why_questions_get_numbers_without_a_cause(bot):
    a = bot.ask("Why did net revenue fall last week?", "growth_manager")
    assert a.route == "insufficient" and a.tool == "sales_overview" and "not why" in a.markdown


def test_out_of_scope_help_and_clarify(bot):
    out = bot.ask("Will it rain in Mumbai tomorrow?", "inventory_manager")
    assert out.route == "out_of_scope" and out.citations == [] and "Try asking" in out.markdown
    assert bot.ask("hi", "growth_manager").route == "help"
    c = bot.ask("What is the demand forecast for next week?", "inventory_manager")
    assert c.route == "clarify" and "category" in c.markdown


def test_log_record(bot):
    a = bot.ask("Which promotions had the best uplift?", "growth_manager")
    rec = log_record(a, "growth_manager")
    assert rec["route"] == "data" and rec["tool"] == "promotions" and rec["latency_ms"] >= 0



def test_engineer_gets_pipeline_answers_and_other_roles_do_not(bot):
    a = bot.ask("Give me the row counts of the latest run", "data_engineer")
    assert a.route == "data" and a.tool == "pipeline_rows" and "**Rows written**" in a.markdown
    refused = bot.ask("Give me the row counts of the latest run", "growth_manager")
    assert refused.route == "not_allowed" and "Data Engineer" in refused.markdown


def test_unanswerable_question_points_to_the_right_page(bot):
    a = bot.ask("What is the grain of the reviews table?", "data_engineer")
    assert a.route == "out_of_scope" and "Lineage & tables" in a.markdown
    other = bot.ask("What is the grain of the reviews table?", "growth_manager")
    assert "Lineage & tables" not in other.markdown                     # only pages of your own workspace



def test_most_demanded_products_ranking(bot):
    a = bot.ask("What will be in highest demand over the coming days?", "inventory_manager")
    assert a.route == "data" and a.tool == "top_forecast" and "**Most demanded product**" in a.markdown
    assert "focus SKUs" in a.markdown                                    # coverage stated
    c = bot.ask("Forecast please", "inventory_manager")
    assert c.route == "clarify" and "most demanded" in c.markdown        # the clarification offers the ranking


def test_ranking_for_a_category_without_focus_skus_falls_back(bot):
    # 7F revision: Breakfast & Cereals has no focus SKUs — say so and give the category forecast, never an error
    a = bot.ask("Which breakfast cereals will be in highest demand?", "inventory_manager")
    assert a.route == "data" and a.tool == "top_forecast"
    assert "no focus SKUs" in a.markdown and "**Next 7 days**" in a.markdown
