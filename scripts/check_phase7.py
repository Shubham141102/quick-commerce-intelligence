"""Phase 7 completion report: the Data Engineer workspace equals the recorded metadata and quarantine files.

    python -m scripts.check_phase7

Bars, fixed before measuring (2026-10-11) — all 100%:
  7.1 every ops table is published and typed in the snapshot schema
  7.2 pipeline runs: count, successes and failures equal the raw run log
  7.3 table runs: rows written / rejected / deduplicated equal the raw table log
  7.4 quarantine: rows per dataset equal the raw quarantine files (direct + cascade)
  7.5 quality results equal the raw quality log
  7.6 lineage: every Gold and ML table traces back to a source file (landing/…)
  7.7 catalog: every Bronze, Silver, Gold and ML table is in the table catalog
  7.8 role check: the 15 engineering functions work for the engineer and admin, refuse every other role
  7.9 privacy and size: no raw rejected rows, e-mail addresses or local paths published; snapshot < 50 MB
Raw records created after the snapshot was published (later runs, later evaluations) are excluded from the comparison.
"""

from __future__ import annotations

import re
import sys
from collections import deque

import pandas as pd

from src.common.config import load_config
from src.common.io import read_csv_strings
from src.common.paths import resolve
from src.common.schemas import SOURCE_SCHEMAS
from src.ml.tables import ML_TABLES
from src.orchestration.ops_snapshot import OPS_TABLES
from src.orchestration.tracking import read_meta
from src.serving import queries as q
from src.serving.db import get_snapshot
from src.serving.permissions import AccessDenied
from src.transformations.gold.run import GOLD_TABLES

NUM = ["rows_written", "rows_rejected", "rows_rejected_cascade", "rows_deduplicated"]


def _sum(df: pd.DataFrame, cols) -> dict:
    return {c: int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum()) for c in cols}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    paths = load_config("small").paths
    meta, qroot, demo = resolve(paths["metadata"]), resolve(paths["quarantine"]), resolve(paths["demo"])
    s = get_snapshot()
    info = dict(s.manifest)
    published = pd.Timestamp(info["published_at"].replace("Z", ""))
    snap_run = info["pipeline_run_id"]
    out: list[tuple[str, bool, str]] = []

    # 7.1 published + typed
    schema = pd.read_csv(demo / "snapshot_schema.csv", dtype=str)
    missing = [t for t in OPS_TABLES if not (demo / f"{t}.csv").exists() or t not in set(schema["table"])]
    out.append(("7.1 Every ops table published and typed", not missing,
                f"{len(OPS_TABLES) - len(missing)}/{len(OPS_TABLES)} ops tables" + (f"; missing {missing}" if missing else "")))

    # 7.2 pipeline runs (raw runs that existed at publish time)
    raw = read_meta(meta, "meta_pipeline_runs")
    raw = raw[(raw["run_id"] != snap_run) & (pd.to_datetime(raw["started_at"].str.replace("Z", "")) < published)]
    ops = s.query("SELECT * FROM ops_pipeline_runs")
    same = (len(ops) == len(raw) and set(ops["run_id"]) == set(raw["run_id"])
            and (ops["status"] == "success").sum() == (raw["status"] == "success").sum())
    out.append(("7.2 Pipeline runs equal the raw run log", bool(same),
                f"{len(ops)} runs published vs {len(raw)} recorded before publishing; "
                f"{int((ops['status'] == 'success').sum())} success / {int((ops['status'] == 'failed').sum())} failed"))

    # 7.3 table runs
    rt = read_meta(meta, "meta_table_runs")
    # the snapshot also holds the publishing run's own table runs recorded before publish (ingest … ml)
    rt = rt[rt["run_id"].isin(set(raw["run_id"])) | ((rt["run_id"] == snap_run) & (rt["stage"] != "publish"))]
    ot = s.query("SELECT * FROM ops_table_runs")
    a, b = _sum(rt, NUM), _sum(ot, NUM)
    out.append(("7.3 Table runs: rows written / rejected / deduplicated equal the raw log", a == b and len(rt) == len(ot),
                f"{len(ot)} table runs; rows written {b['rows_written']:,} (raw {a['rows_written']:,})"))

    # 7.4 quarantine
    files = [f for f in (qroot / "qtn_records").glob("*/*.csv") if not f.parent.name.startswith("_")]
    rq = pd.concat([read_csv_strings(f)[["source_dataset", "rejection_type", "silver_run_id"]] for f in files],
                   ignore_index=True) if files else pd.DataFrame(columns=["source_dataset", "rejection_type"])
    oq = s.query("SELECT dataset, rejection_type, sum(rows) AS n FROM ops_quarantine_summary GROUP BY ALL")
    raw_counts = rq.groupby(["source_dataset", "rejection_type"]).size().rename("n").reset_index() \
        .rename(columns={"source_dataset": "dataset"})
    merged = raw_counts.merge(oq, on=["dataset", "rejection_type"], how="outer", suffixes=("_raw", "_pub")).fillna(0)
    ok = bool((merged["n_raw"].astype(int) == merged["n_pub"].astype(int)).all())
    out.append(("7.4 Quarantine rows per dataset equal the raw quarantine files", ok,
                f"{int(oq['n'].sum()):,} rows in {oq['dataset'].nunique()} datasets "
                f"(direct {int(oq.loc[oq['rejection_type'] == 'direct', 'n'].sum()):,}, "
                f"cascade {int(oq.loc[oq['rejection_type'] == 'cascade', 'n'].sum()):,})"))

    # 7.5 quality results
    rqr = read_meta(meta, "meta_quality_results")
    oqr = s.query("SELECT * FROM ops_quality_results")
    ok = len(rqr) == len(oqr) and _sum(rqr, ["rows_affected"]) == _sum(oqr, ["rows_affected"])
    out.append(("7.5 Quality results equal the raw quality log", bool(ok),
                f"{len(oqr)} results, {_sum(oqr, ['rows_affected'])['rows_affected']:,} rows affected"))

    # 7.6 lineage: every Gold / ML table reaches a landing file upstream
    edges = s.query("SELECT source, target FROM ops_lineage")
    parents = edges.groupby("target")["source"].apply(set).to_dict()

    def reaches_source(table: str) -> bool:
        seen, todo = {table}, deque([table])
        while todo:
            for p in parents.get(todo.popleft(), ()):
                if p.startswith("landing/"):
                    return True
                if p not in seen:
                    seen.add(p)
                    todo.append(p)
        return False
    targets = [t.name for t in (*GOLD_TABLES, *ML_TABLES)]
    orphans = [t for t in targets if not reaches_source(t)]
    out.append(("7.6 Lineage: every Gold and ML table traces back to a source file", not orphans,
                f"{len(targets) - len(orphans)}/{len(targets)} tables; {len(edges)} edges"
                + (f"; no path for {orphans[:5]}" if orphans else "")))

    # 7.7 catalog coverage
    cat = set(s.query('SELECT "table" FROM ops_table_catalog')["table"])
    expected = {f"{p}_{d}" for d in SOURCE_SCHEMAS for p in ("brz", "slv")} | set(targets)
    gap = sorted(expected - cat)
    out.append(("7.7 Catalog lists every Bronze, Silver, Gold and ML table", not gap,
                f"{len(expected) - len(gap)}/{len(expected)} tables" + (f"; missing {gap[:5]}" if gap else "")))

    # 7.8 role check
    run = q.pipeline_runs("data_engineer")["run_id"].iloc[0]
    calls = [lambda r: q.ops_overview(r), lambda r: q.pipeline_runs(r), lambda r: q.stage_durations(r),
             lambda r: q.table_runs_for(r, run), lambda r: q.failed_runs(r), lambda r: q.quality_by_dataset(r),
             lambda r: q.quality_rules(r), lambda r: q.quarantine_summary(r), lambda r: q.quarantine_by_dataset(r),
             lambda r: q.file_loads_summary(r), lambda r: q.lineage_edges(r), lambda r: q.table_catalog(r),
             lambda r: q.model_registry(r), lambda r: q.key_metric_history(r), lambda r: q.assistant_eval_history(r)]
    allowed = all(c("data_engineer") is not None and c("admin") is not None for c in calls)
    refused = 0
    for c in calls:
        n = 0
        for role in ("inventory_manager", "growth_manager"):
            try:
                c(role)
            except AccessDenied:
                n += 1
        refused += n == 2
    out.append(("7.8 Role check: engineer and admin allowed, every other role refused", allowed and refused == len(calls),
                f"{len(calls)} functions; {refused}/{len(calls)} refuse every other role"))

    # 7.9 privacy + size
    qs = pd.read_csv(demo / "ops_quarantine_summary.csv", dtype=str)
    leaks = "raw_record" in qs.columns or qs.apply(lambda c: c.str.contains("@", na=False)).any().any()
    # no local paths / user names anywhere in the ops tables (they are committed to a public repository)
    home = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]|/home/[^/\s]+/|/Users/[^/\s]+/", re.I)
    path_hits = [t for t in OPS_TABLES if home.search((demo / f"{t}.csv").read_text(encoding="utf-8"))]
    leaks = leaks or bool(path_hits)
    size = sum(p.stat().st_size for p in demo.glob("*.csv")) / 1e6
    ops_size = sum((demo / f"{t}.csv").stat().st_size for t in OPS_TABLES) / 1e6
    out.append(("7.9 Privacy (no raw rows, e-mails or local paths) and snapshot < 50 MB", (not leaks) and size < 50,
                f"snapshot {size:.1f} MB (ops tables {ops_size:.2f} MB); leaks: "
                f"{', '.join(path_hits) if path_hits else ('YES' if leaks else 'none')}"))

    print("\nPhase 7 completion report — Data Engineer workspace\n")
    for name, ok, detail in out:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:72s} {detail}")
    failed = sum(not ok for _, ok, _ in out)
    print(f"\n{len(out) - failed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
