"""Retrieval for the business assistant (Phase 6B): BM25 keyword search over the knowledge chunks.

No vector database and no extra package: the chunks come from data/demo/rag_chunks.csv and the index is built in
memory (≈100 chunks, milliseconds). BM25 Okapi, implemented here so the scoring is visible:

    score(q, d) = Σ_t∈q  idf(t) · f(t,d)·(k1+1) / ( f(t,d) + k1·(1 − b + b·|d|/avgdl) )
    idf(t)      = ln( (N − n(t) + 0.5) / (n(t) + 0.5) + 1 )

- tokenizer: lowercase, letters/digits, common words dropped, light plural stripping;
- synonyms: a fixed list expands the question (out of stock → stockout, late → SLA, …), because BM25 only
  matches shared words;
- the document side is "title + section + section + text" (the section twice: headings are strong signals);
- top-k = 3; hits below MIN_SCORE are dropped, so an off-topic question gets "not covered" instead of a weak match.
  MIN_SCORE is calibrated on the practice questions in tests/rag/dev_questions.yaml — never on the evaluation set.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd

K1, B = 1.5, 0.75
TOP_K = 3
MIN_SCORE = 3.24      # score ÷ √(question words), re-calibrated 2026-10-08 on tests/rag/dev_questions.yaml: midpoint
                      # of the lowest correct in-scope score (4.28) and the highest out-of-scope score (2.21). First
                      # version: raw score 7.8, which rejected short questions (docs/13_rag_assistant.md, 6F)

STOPWORDS = frozenset("""
a an and are as at be been by can could do does did for from has have how i if in is it its me my of on or our
should so than that the their them then there these they this to was we were what when where which who why will
with would you your about any all also into more most much not only per same such tell us please show give
""".split())

# Each group: phrases that mean the same thing. If a question contains any phrase of a group, the words of every
# phrase in the group are added to the query.
SYNONYMS = (
    ("stockout", "out of stock", "ran out", "run out", "empty shelf", "zero stock"),
    ("reorder", "replenish", "replenishment", "restock", "order more"),
    ("sla", "delivery promise", "on time", "on-time", "late delivery", "delivery time", "15 minutes"),
    ("aov", "average order value", "basket value"),
    ("refund", "return", "money back", "reimburse"),
    ("promotion", "discount", "offer", "deal", "coupon"),
    ("cancellation", "cancel", "cancelled", "canceled"),
    ("forecast", "prediction", "predict", "demand forecast"),
    ("segment", "segmentation", "customer group", "cluster"),
    ("anomaly", "unusual", "spike", "outlier"),
    ("wape", "forecast error", "accuracy"),
    ("lead time", "supplier delivery", "delivery from supplier"),
    ("write-off", "write off", "damaged", "damage", "expired"),
    ("net revenue", "revenue", "sales value"),
    ("recommendation", "recommend", "suggest products"),
    ("lift", "confidence", "bought together", "basket rule", "affinity"),
)


def stem(word: str) -> str:
    """Light suffix stripping, shared by retrieval and the router: lapsing / lapsed → laps, detected → detect,
    stockouts → stockout, categories → category (revised 2026-10-08, see docs/13_rag_assistant.md 6F)."""
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


_stem = stem


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in STOPWORDS]


def expand(question: str) -> list[str]:
    """Question tokens plus the words of every synonym group the question mentions."""
    q = f" {question.lower()} "
    extra = []
    for group in SYNONYMS:
        if any(re.search(rf"(?<![a-z]){re.escape(p)}(?![a-z])", q) for p in group):
            extra += [t for p in group for t in tokenize(p)]
    return tokenize(question) + extra


def citation(doc_id: str, doc_type: str, doc_title: str, section: str) -> str:
    """POL-INV §3 Reorder rules · Metric definitions › Net revenue · Demand Forecasting › Results."""
    section = re.sub(r"\s*\(part \d+\)$", "", section)
    if doc_type == "policy":
        m = re.match(r"(\d+)\.\s*(.*)", section)
        return f"{doc_id} §{m.group(1)} {m.group(2)}" if m else f"{doc_id} · {section}"
    title = re.sub(r"^\d+\.\s*", "", re.sub(r"\s*\(Phase[^)]*\)", "", doc_title))
    return f"{title} › {section}"


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    doc_type: str
    doc_title: str
    section: str
    text: str
    score: float

    @property
    def citation(self) -> str:
        return citation(self.doc_id, self.doc_type, self.doc_title, self.section)


class Retriever:
    """BM25 index over the knowledge chunks (a DataFrame with the rag_chunks columns)."""

    def __init__(self, chunks: pd.DataFrame):
        self.chunks = chunks.reset_index(drop=True)
        docs = [tokenize(f"{r.doc_title} {r.section} {r.section} {r.text}") for r in self.chunks.itertuples()]
        self.tf = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avgdl = sum(self.lengths) / max(1, len(docs))
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1) for t, c in df.items()}

    def scores(self, tokens: list[str]) -> list[float]:
        out = []
        for tf, length in zip(self.tf, self.lengths):
            s = 0.0
            for t in tokens:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (K1 + 1) / (f + K1 * (1 - B + B * length / self.avgdl))
            out.append(s)
        return out

    def search(self, question: str, k: int = TOP_K, doc_types: tuple[str, ...] | None = None,
               min_score: float = MIN_SCORE) -> list[Hit]:
        """Top-k chunks for a question, optionally only some doc types; [] when nothing reaches min_score.

        The threshold applies to the score ÷ √(number of question words). BM25 scores grow with the number of words:
        a raw threshold rejected short questions whose top hit was correct (6F finding), dividing by the word count
        diluted two-part questions, dividing by known words let one-word off-topic matches through; √words separated
        the practice set best (docs/13_rag_assistant.md, 6F)."""
        words = math.sqrt(max(1, len(set(tokenize(question)))))
        scores = self.scores(expand(question))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in ranked:
            row = self.chunks.iloc[i]
            if scores[i] / words < min_score or len(hits) == k:
                break
            if doc_types and row["doc_type"] not in doc_types:
                continue
            hits.append(Hit(row["chunk_id"], row["doc_id"], row["doc_type"], row["doc_title"], row["section"],
                            row["text"], round(scores[i] / words, 3)))
        return hits
