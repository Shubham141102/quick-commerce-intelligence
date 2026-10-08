"""Entity extraction for the business assistant (Phase 6C): turn words in a question into tool parameters.

    "Which SKUs are at high risk in Dwarka?"        → stores [S..], tiers [High]
    "Revenue at Bandra last week"                    → stores [S..], period (24–30 Sep 2025)
    "What is bought with FreshFarm Baby Wipes?"      → product P..
    "top 5 products in September"                    → limit 5, period (1–30 Sep 2025)

Time phrases are resolved against the END OF THE DATA (the snapshot's last day, 30 Sep 2025), never today's date:
the data is a fixed historical snapshot. Lookups (store / category / product names) are read from the published
dimension tables; they hold names only, no customer data.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
TIERS = {"high": "High", "medium": "Medium", "low": "Low"}
DETECTORS = {"spike": "demand_spike", "payment": "payment_failure", "outage": "store_outage"}
SIZE = re.compile(r"\s+(\d+(\.\d+)?\s*(g|kg|ml|l)|pack of \d+)$", re.I)


@dataclass
class Lookups:
    """Names the extractor can recognise, plus the data period."""
    stores: pd.DataFrame        # store_id, store_name, city
    categories: pd.DataFrame    # category_id, category_name
    products: pd.DataFrame      # product_id, product_name
    first_day: date
    last_day: date

    @classmethod
    def from_snapshot(cls) -> "Lookups":
        from src.serving.db import get_snapshot
        s = get_snapshot()
        days = s.query("SELECT min(business_date) AS a, max(business_date) AS b FROM gld_daily_sales").iloc[0]
        return cls(s.query("SELECT store_id, store_name, city FROM dim_stores ORDER BY store_id"),
                   s.query("SELECT category_id, category_name FROM dim_categories"),
                   s.query("SELECT product_id, product_name FROM dim_products"),
                   pd.Timestamp(days["a"]).date(), pd.Timestamp(days["b"]).date())


@dataclass
class Entities:
    store_ids: list[str] = field(default_factory=list)
    category_ids: list[str] = field(default_factory=list)
    product_ids: list[str] = field(default_factory=list)
    tiers: list[str] = field(default_factory=list)
    detectors: list[str] = field(default_factory=list)
    start: date | None = None
    end: date | None = None
    period_label: str = ""
    limit: int | None = None
    found: dict[str, str] = field(default_factory=dict)     # what was recognised, for the "understood as" line


def _has(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase.lower())}(?![a-z0-9])", text) is not None


def _period(text: str, first: date, last: date) -> tuple[date, date, str] | None:
    m = re.search(r"last (\d+) days?", text)
    if m:
        n = max(1, int(m.group(1)))
        return max(first, last - timedelta(days=n - 1)), last, f"last {n} days"
    if _has(text, "last week") or _has(text, "this week") or _has(text, "past week"):
        return max(first, last - timedelta(days=6)), last, "last 7 days"
    if _has(text, "yesterday") or _has(text, "today") or _has(text, "latest day"):
        return last, last, f"{last:%d %b %Y}"
    if _has(text, "last month") or _has(text, "this month") or _has(text, "past month"):
        start = last.replace(day=1)
        return start, last, f"{start:%B %Y}"
    for name, num in MONTHS.items():
        if name == "may":   # also an ordinary word ("may I see…"): only "in may" / "may 2025" count
            mentioned = re.search(r"\b(in|during) may\b|\bmay \d{4}\b", text) is not None
        else:
            mentioned = _has(text, name) or (len(name) > 4 and _has(text, name[:3]))
        if mentioned:
            year = last.year if num <= last.month else last.year - 1
            start, end = date(year, num, 1), date(year, num, calendar.monthrange(year, num)[1])
            if end >= first and start <= last:
                return max(start, first), min(end, last), f"{start:%B %Y}"
    if _has(text, "all time") or _has(text, "whole period") or _has(text, "overall"):
        return first, last, f"{first:%d %b} – {last:%d %b %Y}"
    return None


def extract(question: str, lk: Lookups) -> Entities:
    text = question.lower()
    e = Entities()

    areas = []
    for r in lk.stores.itertuples():
        area = r.store_name.replace("QC Dark Store", "").strip()
        if _has(text, area) or _has(text, r.store_id):
            e.store_ids.append(r.store_id)
            areas.append(area)
    if areas:
        e.found["store"] = ", ".join(areas)
    if not e.store_ids:                                       # a city means all of its stores
        for city, grp in lk.stores.groupby("city"):
            if _has(text, city):
                e.store_ids += grp["store_id"].tolist()
                e.found["city"] = city

    for r in lk.categories.itertuples():
        parts = [r.category_name] + [p.strip() for p in re.split(r"[,&]", r.category_name) if len(p.strip()) >= 4]
        if any(_has(text, p) for p in parts):
            e.category_ids.append(r.category_id)
            e.found["category"] = ", ".join(filter(None, [e.found.get("category"), r.category_name]))

    best: tuple[int, str, str] | None = None                  # longest product name mentioned wins
    for r in lk.products.itertuples():
        base = SIZE.sub("", r.product_name).strip()
        if len(base) >= 6 and _has(text, base) and (best is None or len(base) > best[0]):
            best = (len(base), r.product_id, base)
    if best:
        e.product_ids = [best[1]]
        e.found["product"] = best[2]

    e.tiers = [TIERS[w] for w in TIERS if re.search(rf"\b{w}\b[\w\s-]{{0,12}}\b(risk|tier)", text)]
    if e.tiers:
        e.found["tier"] = ", ".join(e.tiers)
    e.detectors = [d for k, d in DETECTORS.items() if k in text]

    period = _period(text, lk.first_day, lk.last_day)
    if period:
        e.start, e.end, e.period_label = period
        e.found["period"] = period[2]

    m = re.search(r"\btop (\d+)\b", text)
    if m:
        e.limit = int(m.group(1))
        e.found["top"] = m.group(1)
    return e
