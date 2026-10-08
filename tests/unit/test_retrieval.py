"""BM25 retrieval (Phase 6B) on hand-made chunks and on the real knowledge base."""

import pandas as pd

from src.common.paths import PROJECT_ROOT
from src.rag.chunking import build_chunks
from src.rag.retrieval import Retriever, citation, expand, tokenize

CHUNKS = pd.DataFrame([
    {"chunk_id": "P-01", "doc_id": "POL-X", "doc_type": "policy", "doc_title": "Inventory Policy",
     "section": "3. Reorder rules", "text": "Reorder when stock is at or below the reorder level, every night."},
    {"chunk_id": "P-02", "doc_id": "POL-Y", "doc_type": "policy", "doc_title": "Delivery Policy",
     "section": "1. Delivery promise", "text": "Every order is delivered within the 15 minute SLA from pickup."},
    {"chunk_id": "M-01", "doc_id": "MET", "doc_type": "metric", "doc_title": "Metric Definitions",
     "section": "Average order value (AOV)", "text": "Metric: AOV; Formula: (GMV − Discount) ÷ completed orders."},
])


def test_tokenizer_drops_common_words_and_plurals():
    assert tokenize("What are the Stockouts in our stores?") == ["stockout", "store"]
    assert tokenize("categories") == ["category"] and tokenize("process") == ["process"]


def test_synonyms_expand_the_question():
    tokens = expand("Which items ran out of stock?")
    assert "stockout" in tokens and "item" in tokens
    assert "sla" in expand("Any late delivery today?")
    assert "deal" not in expand("ideal stock level")          # whole phrases only, not inside other words


def test_bm25_ranks_the_right_chunk_and_filters():
    r = Retriever(CHUNKS)
    assert r.search("When do we restock?", min_score=0)[0].chunk_id == "P-01"        # restock → reorder synonym
    assert r.search("Is delivery on time?", min_score=0)[0].chunk_id == "P-02"
    assert r.search("how is average order value computed", min_score=0)[0].chunk_id == "M-01"
    filtered = r.search("reorder", doc_types=("metric",), min_score=0)                  # filter by type
    assert filtered and all(h.doc_type == "metric" for h in filtered)
    assert r.search("capital of France") == []                                        # below threshold → nothing


def test_citations():
    assert citation("POL-INV", "policy", "Inventory Policy", "3. Reorder rules") == "POL-INV §3 Reorder rules"
    assert citation("MET-DEF", "metric", "Metric Definitions", "Net revenue") == "Metric Definitions › Net revenue"
    assert (citation("MC-FCST", "method", "5. Demand Forecasting (Phase 4B)", "Results (part 2)")
            == "Demand Forecasting › Results")


def test_real_knowledge_base_answers_core_questions():
    r = Retriever(build_chunks(PROJECT_ROOT))
    assert r.search("What is the reorder level rule?")[0].doc_id == "POL-INV"
    assert r.search("How is net revenue calculated?", doc_types=("metric",))[0].section == "Net revenue"
    assert r.search("What is the delivery SLA?")[0].doc_id in {"POL-DEL", "MET-DEF"}


def test_threshold_is_per_question_word():
    r = Retriever(CHUNKS)
    short, long_ = r.search("reorder level", min_score=0), r.search(
        "when should the store reorder because stock is at or below the reorder level every night", min_score=0)
    assert short and long_ and short[0].chunk_id == long_[0].chunk_id == "P-01"
    assert short[0].score > 0 and long_[0].score > 0                         # comparable scale for short and long
