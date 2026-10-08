"""Phase 6 evaluation: the business assistant on the 40-question gold set + 5 role checks (tests/rag/questions.yaml).

    python -m scripts.check_phase6                  # the 40-question gold set (the bars apply here)
    python -m scripts.check_phase6 --set holdout    # 20 held-out questions written after the first run (6F)

Pass bars, fixed before measuring (2026-10-07, Project_Plan_v2.md §9.9):
  6F.1 routing accuracy ≥ 90%        right route (and right tool for data / hybrid questions)
  6F.2 retrieval hit@3 ≥ 85%         an acceptable policy / method section among the top 3 chunks
  6F.3 numbers = data 100%           every checked figure equals an INDEPENDENT SQL query on the Gold tables
  6F.4 citations correct 100%        every quoted sentence is in the chunk it cites
  6F.5 correct refusal ≥ 90%         out-of-scope / ambiguous questions refused or clarified, nothing invented
  6F.6 role check 100%               data outside the user's workspace refused, no figure shown
Every answer is logged to data/metadata/meta_rag_runs/<run>.csv. Reads the published snapshot (data/demo).
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone

import pandas as pd
import yaml

from src.common.config import load_config
from src.common.io import write_csv
from src.common.paths import PROJECT_ROOT, resolve
from src.rag.assistant import Assistant, params_from
from src.rag.entities import Lookups, extract
from src.rag.retrieval import Retriever, citation
from src.serving.db import get_snapshot

BARS = {"routing": 0.90, "hit3": 0.85, "numbers": 1.0, "citations": 1.0, "refusal": 0.90, "roles": 1.0}
DOC_TYPES = {"policy": ("policy",), "method": ("metric", "method"), "hybrid": ("policy",)}
LOG_COLUMNS = ["run_id", "question_id", "kind", "role", "question", "expected_routes", "route", "tool", "route_ok",
               "hit_at_3", "number_ok", "citations", "latency_ms", "mode"]


# ------------------------------------------------------------------------------------- independent figures (SQL)
def _in(col: str, ids) -> tuple[str, list]:
    return (f" AND {col} IN ({', '.join('?' * len(ids))})", list(ids)) if ids else ("", [])


def expected_text(check: str, p: dict, lk: Lookups) -> str:
    """The exact text the answer must contain, computed with SQL that does not use the assistant's code."""
    s = get_snapshot()
    start, end = p.get("start") or lk.first_day, p.get("end") or lk.last_day
    st, sp = _in("store_id", p.get("store_ids"))

    def one(sql, params):
        return s.query(sql, params).iloc[0, 0]

    if check == "high_risk_count":
        n = one(f"SELECT count(*) FROM gld_stockout_risk WHERE run_type = 'current' AND risk_tier = 'High'{st}", sp)
        return f"**High risk** — {int(n)}"
    if check == "medium_risk_count":
        n = one(f"SELECT count(*) FROM gld_stockout_risk WHERE run_type = 'current' AND risk_tier = 'Medium'{st}", sp)
        return f"**Medium risk** — {int(n)}"
    if check == "active_customers":
        n = one("SELECT count(*) FROM gld_customer_retention WHERE status = 'active'", [])
        return f"**Active (≤ 14 days)** — {int(n):,}"
    if check == "units_to_order":
        n = one(f"SELECT coalesce(sum(suggested_qty), 0) FROM gld_replenishment WHERE run_type = 'current' "
                f"AND suggested_qty > 0{st}", sp)
        return f"**Units** — {int(n)}"
    if check == "forecast_7d":
        x = one(f"SELECT sum(forecast_units) FROM gld_demand_predictions WHERE run_type = 'future' "
                f"AND category_id = ?{st}", [p["category_id"], *sp])
        return f"**Next 7 days** — {float(x):,.0f} units"
    if check == "lost_sales":
        x = one(f"SELECT coalesce(sum(lost_revenue), 0) FROM gld_lost_sales WHERE TRUE{st}", sp)
        return f"**Lost sales** — ₹{float(x):,.0f}"
    if check == "net_revenue":
        x = one(f"SELECT sum(net_revenue) FROM gld_daily_sales WHERE business_date BETWEEN ? AND ?{st}",
                [start, end, *sp])
        return f"**Net revenue** — ₹{float(x or 0):,.0f}"
    if check == "on_time_rate":
        x = one(f"SELECT sum(on_time_rate * delivered) / nullif(sum(delivered), 0) FROM gld_delivery_metrics "
                f"WHERE business_date BETWEEN ? AND ?{st}", [start, end, *sp])
        return f"**On-time rate** — {float(x):.1%}"
    if check == "top_product":
        return str(one("""SELECT p.product_name FROM gld_product_performance pp JOIN dim_products p USING (product_id)
                          WHERE pp.month BETWEEN date_trunc('month', ?::DATE) AND ?
                          GROUP BY ALL ORDER BY sum(pp.revenue) DESC LIMIT 1""", [start, end]))
    if check == "anomaly_count":
        dt, dp = _in("detector", p.get("detectors"))
        return f"**Anomalies** — {int(one(f'SELECT count(*) FROM gld_sales_anomalies WHERE TRUE{dt}{st}', dp + sp))}"
    if check == "segment_count":
        return f"**Segments** — {int(one('SELECT count(*) FROM gld_segment_profiles', []))}"
    if check == "cancellation_count":
        n = one(f"SELECT coalesce(sum(cancellations), 0) FROM gld_cancellation_metrics "
                f"WHERE week_start BETWEEN date_trunc('week', ?::DATE) AND ?{st}", [start, end, *sp])
        return f"**Cancellations** — {int(n):,}"
    if check == "promotion_count":
        return f"**Promotions** — {int(one('SELECT count(*) FROM gld_promotion_metrics', []))}"
    raise KeyError(check)


# ------------------------------------------------------------------------------------- citations
QUOTE = re.compile(r"> (.+?)\n>\n> — \*(.+?)\*")


def citations_ok(markdown: str, chunks: pd.DataFrame) -> tuple[int, int]:
    """(correct, total) quotes: every quoted sentence must appear in a chunk carrying that citation."""
    cites = chunks.assign(cite=[citation(r.doc_id, r.doc_type, r.doc_title, r.section) for r in chunks.itertuples()])
    good = total = 0
    for quote, cite in QUOTE.findall(markdown):
        total += 1
        texts = cites.loc[cites["cite"] == cite, "text"].tolist()
        sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", quote) if s]
        good += bool(texts) and all(any(s in t for t in texts) for s in sentences)
    return good, total


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    holdout = "--set" in sys.argv and sys.argv[sys.argv.index("--set") + 1] == "holdout"
    name = "holdout_questions.yaml" if holdout else "questions.yaml"
    spec = yaml.safe_load((PROJECT_ROOT / "tests" / "rag" / name).read_text(encoding="utf-8"))
    chunks = get_snapshot().query("SELECT * FROM rag_chunks")
    lk = Lookups.from_snapshot()
    bot = Assistant(Retriever(chunks), lk)
    run_id = ("rag_holdout_" if holdout else "rag_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    rows, misses = [], []
    route_ok = hit = hit_n = num_ok = num_n = cite_ok = cite_n = refuse_ok = refuse_n = 0
    for item in spec["questions"]:
        a = bot.ask(item["question"], item["role"])
        r_ok = a.route in item["routes"] and (not item.get("tool") or a.tool == item["tool"])
        route_ok += r_ok
        h = None
        if item.get("sections"):
            hits = bot.retriever.search(item["question"], doc_types=DOC_TYPES[item["kind"]])
            h = any(x.doc_id == d and x.section.startswith(sec) for x in hits for d, sec in item["sections"])
            hit += h
            hit_n += 1
        n = None
        if item.get("check") and a.tool == item.get("tool"):
            want = expected_text(item["check"], params_from(extract(item["question"], lk)), lk)
            n = want in a.markdown
            num_ok += n
            num_n += 1
            if not n:
                misses.append(f"{item['id']} number: expected «{want}»")
        g, t = citations_ok(a.markdown, chunks)
        cite_ok, cite_n = cite_ok + g, cite_n + t
        if item["kind"] in ("ambiguous", "out_of_scope"):
            ok = a.route in item["routes"] and (a.route != "out_of_scope" or (not a.citations and a.table is None))
            refuse_ok += ok
            refuse_n += 1
        if not r_ok:
            misses.append(f"{item['id']} route: expected {item['routes']}{' / ' + item['tool'] if item.get('tool') else ''}"
                          f", got {a.route}{' / ' + a.tool if a.tool else ''} — {item['question']}")
        if h is False:
            misses.append(f"{item['id']} hit@3: no {item['sections']} in top 3 — {item['question']}")
        rows.append({"run_id": run_id, "question_id": item["id"], "kind": item["kind"], "role": item["role"],
                     "question": item["question"], "expected_routes": "|".join(item["routes"]), "route": a.route,
                     "tool": a.tool or "", "route_ok": r_ok, "hit_at_3": "" if h is None else h,
                     "number_ok": "" if n is None else n, "citations": "|".join(a.citations),
                     "latency_ms": a.latency_ms, "mode": a.mode})

    role_ok = 0
    for item in spec["role_checks"]:
        a = bot.ask(item["question"], item["role"])
        ok = a.route == "not_allowed" and a.table is None and "₹" not in a.markdown
        role_ok += ok
        if not ok:
            misses.append(f"{item['id']} role: expected refusal, got {a.route} — {item['question']}")
        g, t = citations_ok(a.markdown, chunks)
        cite_ok, cite_n = cite_ok + g, cite_n + t
        rows.append({"run_id": run_id, "question_id": item["id"], "kind": "role_check", "role": item["role"],
                     "question": item["question"], "expected_routes": "not_allowed", "route": a.route,
                     "tool": a.tool or "", "route_ok": ok, "hit_at_3": "", "number_ok": "",
                     "citations": "|".join(a.citations), "latency_ms": a.latency_ms, "mode": a.mode})

    n_q, n_r = len(spec["questions"]), len(spec["role_checks"])
    rates = {"routing": route_ok / n_q, "hit3": hit / hit_n if hit_n else 0, "numbers": num_ok / num_n if num_n else 0,
             "citations": cite_ok / cite_n if cite_n else 1.0, "refusal": refuse_ok / refuse_n if refuse_n else 0,
             "roles": role_ok / n_r}
    lines = [("6F.1 Routing accuracy (route + tool)", "routing", f"{route_ok}/{n_q}"),
             ("6F.2 Retrieval hit@3", "hit3", f"{hit}/{hit_n}"),
             ("6F.3 Numbers equal independent SQL", "numbers", f"{num_ok}/{num_n} answers with the expected tool"),
             ("6F.4 Citations correct (quote is in the cited chunk)", "citations", f"{cite_ok}/{cite_n} quotes"),
             ("6F.5 Correct refusal / clarification", "refusal", f"{refuse_ok}/{refuse_n}"),
             ("6F.6 Role check (no data outside the workspace)", "roles", f"{role_ok}/{n_r}")]
    print("\nPhase 6 evaluation — business assistant (template mode)\n")
    for title, key, detail in lines:
        print(f"  [{status(rates[key] >= BARS[key])}] {title:52s} {rates[key]:6.1%}  (bar {BARS[key]:.0%})  {detail}")
    lat = pd.Series([r["latency_ms"] for r in rows])
    print(f"\n  latency: median {lat.median():.0f} ms, max {lat.max():.0f} ms over {len(rows)} answers")
    if misses:
        print("\n  Misses:")
        for m in misses:
            print(f"   - {m}")

    out = resolve(load_config("small").paths["metadata"]) / "meta_rag_runs" / f"{run_id}.csv"
    write_csv(pd.DataFrame(rows).astype(str), out, LOG_COLUMNS)
    print(f"\n  logged {len(rows)} answers → {out.relative_to(PROJECT_ROOT).as_posix()}")
    failed = sum(rates[k] < BARS[k] for k in BARS)
    print(f"\n{len(BARS) - failed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
