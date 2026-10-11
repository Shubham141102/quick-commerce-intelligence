"""Router and quote selection (Phase 6D) on hand-written questions (not the 6F evaluation set)."""

import pytest

from src.rag.composer import best_sentences
from src.rag.entities import Entities
from src.rag.router import pick_tool, route_question


@pytest.mark.parametrize("question, kind, tool", [
    ("Show me the SKUs running out at Saket", "data", "at_risk_skus"),
    ("Top 5 best selling products in August", "data", "top_products"),
    ("Net revenue in Delhi last month", "data", "sales_overview"),
    ("How many failed deliveries did we have?", "data", "delivery_overview"),
    ("List the payment failures", "data", "anomalies"),
    ("How many at risk customers do we have?", "data", "retention"),          # longer phrase beats "at risk"
    ("Which promotions had the best uplift?", "data", "promotions"),
    ("What should a store do when a supplier delivery is three days late?", "policy", None),
    ("Is a 40% discount allowed?", "policy", None),
    ("How long do customers have to ask for a refund?", "policy", None),
    ("How is net revenue calculated?", "method", None),                        # method even though "revenue"
    ("How was the number of customer segments chosen?", "method", None),
    ("What does lift mean?", "method", None),
    ("Which SKUs are at high risk and what does the policy say?", "hybrid", "at_risk_skus"),
    ("Why did revenue drop in September?", "insufficient", "sales_overview"),
    ("hello", "help", None),
])
def test_routes(question, kind, tool):
    r = route_question(question, Entities())
    assert (r.kind, r.tool) == (kind, tool), r.reason


def test_entities_turn_a_policy_question_into_hybrid():
    e = Entities(store_ids=["S05"])
    assert route_question("What should we do about the stockout risk at Dwarka?", e).kind == "hybrid"


def test_unknown_topics_fall_back_to_knowledge_search():
    r = route_question("Tell me about the capital of France", Entities())
    assert r.kind == "policy" and r.doc_types is None and r.tool is None


def test_pick_tool_whole_words_only():
    assert pick_tool("the ontology of things")[0] is None          # "on t…" is not "on time"
    assert pick_tool("on time delivery")[0] == "delivery_overview"


def test_best_sentences_keeps_the_relevant_ones_in_order():
    text = ("Stores are reviewed weekly. A delivery is late when it takes more than 15 minutes from pickup. "
            "Partners are assigned in order of availability. Late deliveries are reported in the app.")
    out = best_sentences(text, "When is a delivery late?", n=2)
    assert out.startswith("A delivery is late") and "Late deliveries are reported" in out
    assert "Partners are assigned" not in out


# ---------------------------------------------------------------- 6F revision (general fixes)
def test_stem_matching_ignores_word_endings():
    assert pick_tool("customers who are lapsing")[0] == "retention"          # lapsing ≈ lapsed
    assert pick_tool("best-selling items")[0] == "top_products"              # hyphen ≈ space


def test_passive_how_questions_are_method_questions():
    assert route_question("How are refunds processed?", Entities()).kind == "method"
    assert route_question("How is our stock?", Entities()).kind == "data"    # no passive verb: still data


def test_generic_topic_flag():
    generic = route_question("When is a delivery considered complete?", Entities())
    assert generic.kind == "data" and generic.generic                       # one generic word, no report cue
    assert not route_question("Show delivery performance", Entities()).generic     # explicit report request
    assert not route_question("On time rate at Dwarka", Entities(store_ids=["S05"])).generic



# ---------------------------------------------------------------- Data Engineer tools (7E)
@pytest.mark.parametrize("question, tool", [
    ("Show me the run history of the pipeline", "pipeline_status"),
    ("What was the data volume written yesterday?", "pipeline_rows"),
    ("Any dirty data in payments?", "data_quality"),
    ("List the rejected records", "quarantine"),
    ("What is in the model registry?", "model_health"),
])
def test_engineering_tools_are_routed(question, tool):
    assert route_question(question, Entities()).tool == tool


def test_page_hints_only_for_own_workspace():
    from src.rag.composer import page_hints
    assert page_hints("schema of the reviews table", ["engineering"]) == ["Data Engineer → Lineage & tables"]
    assert page_hints("schema of the reviews table", ["business"]) == []



# ---------------------------------------------------------------- 7E revision: counting questions, schema questions
def test_counting_questions_go_to_the_tool_that_reports_the_measure():
    assert pick_tool("how many records were quarantined yesterday")[0] == "quarantine"
    assert pick_tool("number of rows the pipeline run wrote")[0] == "pipeline_rows"
    assert pick_tool("how many failed pipeline runs this month")[0] == "pipeline_status"
    assert pick_tool("how many payment failures happened")[0] == "anomalies"       # not a pipeline 'failure'


def test_overlapping_phrases_count_once():
    from src.rag.router import _distinct
    assert _distinct(["pipeline", "pipeline run", "pipeline runs"]) == ["pipeline run"]


def test_schema_questions_get_the_catalog_hint():
    assert route_question("Show the lineage of gld_inventory_daily", Entities()).kind == "out_of_scope"
    assert route_question("What data types does the payments table use?", Entities()).kind == "out_of_scope"


def test_counted_noun_stops_at_the_verb():
    # "run" here is a verb, not a pipeline run (found by the held-out set, 7E)
    assert pick_tool("how many offers can run together in a week")[0] != "pipeline_status"
