"""Knowledge base for the business assistant (Phase 6A): split policies, metric definitions and model cards
into citable chunks.

Rules (Project_Plan_v2.md §9.3, §9.9):
- policies and model cards: one chunk per section (## / ### heading); a section longer than MAX_WORDS is split
  into windows of WINDOW_WORDS with OVERLAP_WORDS of overlap;
- metric definitions: one chunk per metric (one table row each), so a citation names the exact metric;
- tables are flattened to "column: value" text so their words are searchable; markdown markup is removed;
- every chunk keeps doc_id, doc_type, doc title, section and the source file's SHA-256, so the index can be
  rebuilt when a file changes and every answer can cite where it came from.

The publish stage writes the chunks to data/demo/rag_chunks.csv (CSV, like every other table).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MAX_WORDS, WINDOW_WORDS, OVERLAP_WORDS, MIN_WORDS = 200, 160, 30, 12

# (relative path, doc_id, doc_type). doc_type: policy (what to do), metric (how a number is defined),
# method (how a model / analysis works).
KNOWLEDGE_SOURCES = (
    ("policies/inventory.md", "POL-INV", "policy"),
    ("policies/delivery.md", "POL-DEL", "policy"),
    ("policies/cancellation.md", "POL-CAN", "policy"),
    ("policies/refund.md", "POL-REF", "policy"),
    ("policies/promotion.md", "POL-PRO", "policy"),
    ("docs/metric_definitions.md", "MET-DEF", "metric"),
    ("docs/05_demand_forecasting.md", "MC-FCST", "method"),
    ("docs/06_stockout_replenishment.md", "MC-STOCK", "method"),
    ("docs/08_inventory_explorer_lost_sales.md", "MC-LOST", "method"),
    ("docs/09_business_workspace.md", "MC-ANOM", "method"),
    ("docs/10_marketing_analytics.md", "MC-MKT", "method"),
)
CHUNK_COLUMNS = ["chunk_id", "doc_id", "doc_type", "doc_title", "section", "text", "words", "source", "doc_sha"]


@dataclass
class Section:
    heading: str
    lines: list[str]


def _clean(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)            # [label](link) -> label
    text = re.sub(r"[*_`]+", "", text)                                 # bold / italic / code marks
    text = re.sub(r"<[^>]+>", " ", text)                               # stray html
    return re.sub(r"\s+", " ", text).strip()


def _table_rows(lines: list[str]) -> list[dict[str, str]]:
    """Markdown table lines -> list of {header: cell}."""
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines if ln.strip().startswith("|")]
    rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]   # drop the |---| separator
    if len(rows) < 2:
        return []
    header = [_clean(h) for h in rows[0]]
    return [{h: _clean(c) for h, c in zip(header, r) if h or c} for r in rows[1:]]


def _section_text(lines: list[str]) -> str:
    """Section body as plain text: prose kept, list items kept, tables flattened row by row."""
    out, table = [], []
    for ln in lines + [""]:
        if ln.strip().startswith("|"):
            table.append(ln)
            continue
        if table:
            out += ["; ".join(f"{k}: {v}" for k, v in row.items() if v) + "." for row in _table_rows(table)]
            table = []
        if ln.strip() and not ln.strip().startswith("```"):
            out.append(ln.strip().lstrip("-> ").strip())
    return _clean(" ".join(out))


def _sections(markdown: str) -> tuple[str, list[Section]]:
    title, sections, parent = "", [Section("Overview", [])], ""
    for ln in markdown.splitlines():
        if ln.startswith("# ") and not title:
            title = _clean(ln[2:])
        elif ln.startswith("## "):
            parent = _clean(ln[3:])
            sections.append(Section(parent, []))
        elif ln.startswith("### "):
            sections.append(Section(f"{parent} › {_clean(ln[4:])}" if parent else _clean(ln[4:]), []))
        else:
            sections[-1].lines.append(ln)
    return title, sections


def _windows(words: list[str]) -> list[list[str]]:
    if len(words) <= MAX_WORDS:
        return [words]
    out, start = [], 0
    while True:
        out.append(words[start:start + WINDOW_WORDS])
        if start + WINDOW_WORDS >= len(words):
            return out
        start += WINDOW_WORDS - OVERLAP_WORDS


def chunk_document(markdown: str, doc_id: str, doc_type: str, source: str = "") -> list[dict]:
    """Chunks of one document (pure function, tested on hand-made markdown)."""
    sha = hashlib.sha256(markdown.encode("utf-8")).hexdigest()[:16]
    title, sections = _sections(markdown)
    pieces: list[tuple[str, str]] = []
    if doc_type == "metric":                                           # one chunk per metric row
        for s in sections:
            for row in _table_rows(s.lines):
                name = row.get("Metric", "")
                text = "; ".join(f"{k}: {v}" for k, v in row.items() if v)
                if name:
                    pieces.append((name, text + "."))
    else:
        for s in sections:
            words = _section_text(s.lines).split()
            if len(words) < MIN_WORDS:
                continue
            parts = _windows(words)
            for i, part in enumerate(parts):
                label = s.heading if len(parts) == 1 else f"{s.heading} (part {i + 1})"
                pieces.append((label, " ".join(part)))
    return [{"chunk_id": f"{doc_id}-{i + 1:02d}", "doc_id": doc_id, "doc_type": doc_type, "doc_title": title,
             "section": section, "text": text, "words": len(text.split()), "source": source, "doc_sha": sha}
            for i, (section, text) in enumerate(pieces)]


def build_chunks(root: Path) -> pd.DataFrame:
    """All knowledge-base chunks, in KNOWLEDGE_SOURCES order."""
    rows: list[dict] = []
    for rel, doc_id, doc_type in KNOWLEDGE_SOURCES:
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(f"knowledge source missing: {rel}")
        rows += chunk_document(path.read_text(encoding="utf-8"), doc_id, doc_type, rel)
    return pd.DataFrame(rows, columns=CHUNK_COLUMNS)
