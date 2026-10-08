"""Chunking rules (Phase 6A) on hand-made markdown and on the real knowledge base."""

from src.common.paths import PROJECT_ROOT
from src.rag.chunking import (
    KNOWLEDGE_SOURCES,
    MAX_WORDS,
    OVERLAP_WORDS,
    WINDOW_WORDS,
    build_chunks,
    chunk_document,
)

POLICY = """# Test Policy

Policy ID: POL-T · Owner: Ops · Applies to: all stores · a short preamble that is long enough to keep.

## 1. Scope

Covers **focus SKUs** only. See [the docs](docs/x.md) for details on how the scope was chosen for stores.

## 2. Rules

- Reorder when stock is at or below the reorder level, checked every night at 22:00.
- Never order a negative quantity.

## 3. Tiny

Too short.
"""


def test_policy_chunks_by_section_and_cleans_markup():
    chunks = chunk_document(POLICY, "POL-T", "policy", "policies/t.md")
    assert [c["section"] for c in chunks] == ["Overview", "1. Scope", "2. Rules"]      # tiny section dropped
    assert chunks[0]["doc_title"] == "Test Policy" and chunks[1]["chunk_id"] == "POL-T-02"
    scope = chunks[1]["text"]
    assert "**" not in scope and "focus SKUs" in scope and "the docs for details" in scope   # link text kept
    assert "Reorder when stock" in chunks[2]["text"] and "Never order" in chunks[2]["text"]


def test_long_section_split_with_overlap():
    body = " ".join(f"w{i}" for i in range(280))      # 0–159 and 130–279: two windows, 30 words shared
    chunks = chunk_document(f"# Doc\n\n## Long\n\n{body}\n", "D", "method")
    assert len(chunks) == 2 and chunks[0]["section"] == "Long (part 1)"
    first, second = chunks[0]["text"].split(), chunks[1]["text"].split()
    assert len(first) == WINDOW_WORDS and first[-OVERLAP_WORDS:] == second[:OVERLAP_WORDS]
    assert all(c["words"] <= max(MAX_WORDS, WINDOW_WORDS) for c in chunks)


def test_metric_table_one_chunk_per_metric_and_tables_flattened():
    md = ("# Metric Definitions\n\n| Metric | Definition | Formula |\n|---|---|---|\n"
          "| **Net revenue** | What the business keeps. | GMV − Discount − Refunds |\n"
          "| **AOV** | Average checkout value. | (GMV − Discount) ÷ completed orders |\n")
    chunks = chunk_document(md, "MET", "metric")
    assert [c["section"] for c in chunks] == ["Net revenue", "AOV"]
    assert "Formula: GMV − Discount − Refunds" in chunks[0]["text"]
    table_md = ("# M\n\n## Bars\n\n| Check | Bar |\n|---|---|\n| Routing accuracy for questions | ≥ 90% |\n"
                "| Retrieval hit at three for policy questions | ≥ 85% |\n")
    assert "Check: Routing accuracy for questions; Bar: ≥ 90%" in chunk_document(table_md, "M", "method")[0]["text"]


def test_real_knowledge_base():
    chunks = build_chunks(PROJECT_ROOT)
    assert set(chunks["doc_id"]) == {doc_id for _, doc_id, _ in KNOWLEDGE_SOURCES}     # every source contributes
    assert chunks["chunk_id"].is_unique and (chunks["words"] <= max(MAX_WORDS, WINDOW_WORDS)).all()
    assert len(chunks) < 200                                                              # small: BM25, no vector DB
    inv = chunks[chunks["doc_id"] == "POL-INV"]
    assert inv["section"].str.startswith("3. Reorder rules").any()
    assert "15 minutes" in " ".join(chunks.loc[chunks["doc_id"] == "POL-DEL", "text"])
    assert {"Net revenue", "On-time delivery rate"} <= set(chunks.loc[chunks["doc_id"] == "MET-DEF", "section"])
