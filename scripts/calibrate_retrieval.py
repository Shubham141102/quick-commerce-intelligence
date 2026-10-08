"""Calibrate the retrieval threshold (Phase 6B) on the practice questions — never on the evaluation set.

    python -m scripts.calibrate_retrieval

For each practice question: the top hit, its score, and whether the expected section is in the top 3. The
suggested MIN_SCORE is the midpoint between the best out-of-scope score and the lowest score of a correct
in-scope hit (scores are ÷ √question words since the 6F revision); it is then written into src/rag/retrieval.py by hand and documented in docs/13_rag_assistant.md.
"""

from __future__ import annotations

import sys

import pandas as pd
import yaml

from src.common.paths import PROJECT_ROOT
from src.rag.retrieval import Retriever


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    chunks = pd.read_csv(PROJECT_ROOT / "data" / "demo" / "rag_chunks.csv", dtype=str)
    questions = yaml.safe_load((PROJECT_ROOT / "tests" / "rag" / "dev_questions.yaml").read_text(encoding="utf-8"))
    retriever = Retriever(chunks)
    in_scope, out_scope, found = [], [], 0
    print(f"{'question':58s} {'top hit':44s} {'score/√w':>10s}  expected in top 3")
    for item in questions:
        q, exp = item["question"], item.get("expect") or {}
        types = tuple(item["doc_types"]) if item.get("doc_types") else None
        hits = retriever.search(q, doc_types=types, min_score=0.0)
        top = hits[0] if hits else None
        ok = ""
        if exp:
            match = [h for h in hits if h.doc_id == exp["doc_id"] and h.section.startswith(exp["section"])]
            ok = "yes" if match else "NO"
            found += bool(match)
            if match:
                in_scope.append(match[0].score)
        else:
            out_scope.append(top.score if top else 0.0)
            ok = "(out of scope)"
        print(f"{q[:58]:58s} {(top.citation if top else '-')[:44]:44s} {top.score if top else 0:10.2f}  {ok}")
    n_in = sum(1 for i in questions if i.get("expect"))
    print(f"\nin-scope hit@3: {found}/{n_in}")
    if in_scope and out_scope:
        lo_in, hi_out = min(in_scope), max(out_scope)
        print(f"lowest correct in-scope score {lo_in:.2f} · highest out-of-scope score {hi_out:.2f}")
        if lo_in > hi_out:
            print(f"separable → suggested MIN_SCORE = {(lo_in + hi_out) / 2:.2f}")
        else:
            print("NOT separable by score alone → the router must also screen out-of-scope questions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
