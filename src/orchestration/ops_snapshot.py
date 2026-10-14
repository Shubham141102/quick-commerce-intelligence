"""Operations snapshot for the Data Engineer workspace (Phase 7A), written by the publish stage.

The hosted app cannot read data/metadata/ or data/quarantine/, so publish copies small `ops_*` tables into
data/demo/: pipeline / stage / table runs, quality results, a quarantine SUMMARY, Bronze file loads, lineage,
a table catalog, the model registry and metrics, and the assistant's evaluation history.

Privacy: the quarantine summary holds counts, rules and record ids only — never the raw rejected rows (they can
contain customer names and e-mails and stay on the pipeline machine).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.common.io import read_csv_strings, to_source_strings
from src.common.paths import PROJECT_ROOT
from src.common.schemas import SOURCE_SCHEMAS, Column, TableSchema
from src.orchestration.tracking import META_TABLES, read_meta

ERROR_CHARS = 400

# name -> {column: dtype}; the order is the published column order
OPS_COLUMNS: dict[str, dict[str, str]] = {
    "ops_pipeline_runs": {"run_id": "string", "pipeline": "string", "profile": "string", "generation_run_id": "string",
                          "stages": "string", "status": "string", "started_at": "timestamp", "ended_at": "timestamp",
                          "duration_s": "double", "git_commit": "string", "error": "string"},
    "ops_stage_runs": {"run_id": "string", "stage": "string", "status": "string", "started_at": "timestamp",
                       "ended_at": "timestamp", "duration_s": "double", "error": "string"},
    "ops_table_runs": {"run_id": "string", "stage": "string", "job": "string", "source": "string", "target": "string",
                       "engine": "string", "rows_read": "int", "rows_written": "int", "rows_rejected": "int",
                       "rows_rejected_cascade": "int", "rows_deduplicated": "int", "rows_corrupt": "int",
                       "rows_flagged": "int", "status": "string", "duration_s": "double"},
    "ops_quality_results": {"run_id": "string", "dataset": "string", "check_type": "string", "rule": "string",
                            "column": "string", "rows_affected": "int", "rows_checked": "int", "outcome": "string"},
    "ops_quarantine_summary": {"dataset": "string", "primary_rule": "string", "rejection_type": "string",
                               "severity": "string", "rows": "int", "example_record_ids": "string",
                               "silver_run_id": "string"},
    "ops_file_loads": {"run_id": "string", "dataset": "string", "slice": "string", "status": "string", "files": "int",
                       "rows_read": "int", "rows_corrupt": "int", "size_mb": "double", "last_loaded_at": "timestamp"},
    "ops_lineage": {"source": "string", "target": "string", "transform": "string", "engine": "string",
                    "source_layer": "string", "target_layer": "string", "last_run_id": "string"},
    "ops_table_catalog": {"table": "string", "layer": "string", "rows": "int", "columns": "int", "grain": "string",
                          "description": "string"},
    "ops_model_runs": {"run_id": "string", "model_name": "string", "model_version": "string", "method": "string",
                       "trained_at": "timestamp", "train_start": "date", "train_end": "date", "test_start": "date",
                       "test_end": "date", "input_table": "string", "train_rows": "int", "metrics": "string",
                       "status": "string", "limitations": "string"},
    "ops_model_metrics": {"run_id": "string", "model_version": "string", "evaluation": "string", "model": "string",
                          "horizon": "int", "segment": "string", "metric": "string", "value": "double"},
    "ops_assistant_evals": {"run_id": "string", "question_set": "string", "mode": "string", "evaluated_at": "timestamp",
                            "answers": "int", "routing_rate": "double", "hit_at_3_rate": "double",
                            "numbers_rate": "double", "llm_verified": "int", "llm_fallbacks": "int",
                            "median_latency_ms": "double"},
}
OPS_TABLES = tuple(OPS_COLUMNS)
OPS_SOURCES = {   # lineage: which recorded data each ops table is built from
    "ops_pipeline_runs": "meta_pipeline_runs", "ops_stage_runs": "meta_stage_runs", "ops_table_runs": "meta_table_runs",
    "ops_quality_results": "meta_quality_results", "ops_quarantine_summary": "qtn_records",
    "ops_file_loads": "meta_file_loads", "ops_lineage": "meta_lineage_edges", "ops_table_catalog": "meta_table_runs",
    "ops_model_runs": "meta_model_runs", "ops_model_metrics": "meta_model_metrics",
    "ops_assistant_evals": "meta_rag_runs",
}

DIMENSION_DESCRIPTIONS = {
    "dim_products": "Product lookup for the app (id, name, category, brand, price).",
    "dim_stores": "Store lookup for the app (id, name, city, state).",
    "dim_categories": "Category lookup for the app.",
    "rag_chunks": "Business-assistant knowledge base: policy, metric-definition and model-card chunks.",
}


def _read(meta: Path, pending: dict | None, table: str) -> pd.DataFrame:
    """Recorded runs of one metadata table plus the CURRENT run's rows so far (the tracker writes them only when the
    run finishes, after publish — without them a first snapshot would miss this run's models and table runs)."""
    df = read_meta(meta, table)
    extra = (pending or {}).get(table) or []
    if not extra:
        return df
    cols = META_TABLES[table]
    new = pd.DataFrame(extra).reindex(columns=cols).astype(object)
    new = new.map(lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
    return pd.concat([df, new], ignore_index=True)


def layer_of(name: str, ml_tables: set[str]) -> str:
    if name.startswith("landing/"):
        return "source"
    if name.startswith(("policies/", "docs/")):
        return "knowledge"
    if name.startswith("demo/"):
        return "app"
    prefix = name.split("_", 1)[0]
    if prefix == "gld":
        return "ml" if name in ml_tables else "gold"
    return {"brz": "bronze", "slv": "silver", "qtn": "quarantine", "meta": "metadata"}.get(prefix, "other")


def _num(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")       # blanks → missing
    return df


# ------------------------------------------------------------------------------------------------- tables
_HOME = re.compile(r"[A-Za-z]:(?:\\\\|\\|/)+Users(?:\\\\|\\|/)+[^\\/\s'\"]+", re.I)
_UNIX_HOME = re.compile(r"/(?:home|Users)/[^/\s'\"]+")


def scrub(text: str) -> str:
    """Error text without local paths: the project folder → <project>, a user home → <home> (privacy, 7.9). The
    snapshot is committed to a public repository."""
    if not text:
        return ""
    root = str(PROJECT_ROOT)
    for variant in (root.replace("\\", "\\\\"), root, root.replace("\\", "/")):
        text = text.replace(variant, "<project>")
    text = _HOME.sub("<home>", text)
    return _UNIX_HOME.sub("<home>", text)


def pipeline_runs(meta: Path, pending: dict | None = None) -> pd.DataFrame:
    df = _read(meta, pending, "meta_pipeline_runs")
    df["error"] = df["error"].fillna("").map(scrub).str.slice(0, ERROR_CHARS)
    return _num(df, ["duration_s"]).sort_values("started_at")


def stage_runs(meta: Path, pending: dict | None = None) -> pd.DataFrame:
    df = _read(meta, pending, "meta_stage_runs")
    df["error"] = df["error"].fillna("").map(scrub).str.slice(0, ERROR_CHARS)
    return _num(df, ["duration_s"])


def table_runs(meta: Path, pending: dict | None = None) -> pd.DataFrame:
    return _num(_read(meta, pending, "meta_table_runs"),
                ["rows_read", "rows_written", "rows_rejected", "rows_rejected_cascade", "rows_deduplicated",
                 "rows_corrupt", "rows_flagged", "duration_s"])


def quality_results(meta: Path, pending: dict | None = None) -> pd.DataFrame:
    return _num(_read(meta, pending, "meta_quality_results"), ["rows_affected", "rows_checked"])


def quarantine_summary(quarantine_root: Path) -> pd.DataFrame:
    """Counts by dataset · primary rule · direct / cascade · severity, with up to 3 example record ids."""
    files = [f for f in (quarantine_root / "qtn_records").glob("*/*.csv") if not f.parent.name.startswith("_")]
    cols = ["source_dataset", "source_record_id", "primary_rule", "severity", "rejection_type", "silver_run_id"]
    if not files:
        return pd.DataFrame(columns=list(OPS_COLUMNS["ops_quarantine_summary"]))
    q = pd.concat([read_csv_strings(f)[cols] for f in files], ignore_index=True)    # raw_record is never read
    out = (q.groupby(["source_dataset", "primary_rule", "rejection_type", "severity"], dropna=False)
           .agg(rows=("source_record_id", "size"),
                example_record_ids=("source_record_id", lambda s: "|".join(s.astype(str).head(3))),
                silver_run_id=("silver_run_id", "max"))
           .reset_index().rename(columns={"source_dataset": "dataset"}))
    return out.sort_values(["rows", "dataset"], ascending=[False, True])


def file_loads(meta: Path, pending: dict | None = None) -> pd.DataFrame:
    df = _num(_read(meta, pending, "meta_file_loads"), ["size_bytes", "rows_read", "rows_corrupt"])
    out = (df.groupby(["run_id", "dataset", "slice", "status"], as_index=False)
           .agg(files=("file", "count"), rows_read=("rows_read", "sum"), rows_corrupt=("rows_corrupt", "sum"),
                size_mb=("size_bytes", lambda s: round(s.sum() / 1e6, 3)), last_loaded_at=("loaded_at", "max")))
    return out


def lineage(meta: Path, ml_tables: set[str], extra: list[dict] | None = None,
            pending: dict | None = None) -> pd.DataFrame:
    """Recorded edges, plus (a) Bronze → quality log edges derived from the quality log for runs recorded before the
    tracker logged them, and (b) the edges of the publish run in progress (recorded only when it finishes)."""
    e = _read(meta, pending, "meta_lineage_edges")
    quality = _read(meta, pending, "meta_quality_results")
    derived = [{"run_id": r, "source": f"brz_{d}", "target": "meta_quality_results",
                "transform": "silver quality checks (from the quality log)", "engine": "spark"}
               for r, d in quality[["run_id", "dataset"]].drop_duplicates().itertuples(index=False)]
    e = pd.concat([e, pd.DataFrame(derived), pd.DataFrame(extra or [])], ignore_index=True).sort_values("run_id")
    e = e.drop_duplicates(["source", "target"], keep="last").rename(columns={"run_id": "last_run_id"})
    e["source_layer"] = e["source"].map(lambda n: layer_of(n, ml_tables))
    e["target_layer"] = e["target"].map(lambda n: layer_of(n, ml_tables))
    return e


def table_catalog(meta: Path, gold_tables, ml_tables, published_counts: dict[str, int],
                  pending: dict | None = None) -> pd.DataFrame:
    """Every table a data engineer can ask about: Bronze and Silver per dataset, Gold, ML, app lookups."""
    runs = table_runs(meta, pending)
    rows = []
    for ds, schema in SOURCE_SCHEMAS.items():
        brz = runs[(runs["target"] == f"brz_{ds}") & (runs["status"] == "success")]
        slv = runs[(runs["target"] == f"slv_{ds}") & (runs["status"] == "success")].sort_values("run_id")
        rows.append({"table": f"brz_{ds}", "layer": "bronze", "rows": brz["rows_written"].sum() if len(brz) else None,
                     "columns": len(schema.columns), "grain": "raw row + ingestion metadata",
                     "description": f"Raw {ds.replace('_', ' ')} exactly as received (append-only, all strings)."})
        rows.append({"table": f"slv_{ds}", "layer": "silver",
                     "rows": slv["rows_written"].iloc[-1] if len(slv) else None, "columns": len(schema.columns),
                     "grain": ", ".join(schema.business_key),
                     "description": f"Cleaned, typed, de-duplicated {ds.replace('_', ' ')}."})
    for table, layer in [(t, "gold") for t in gold_tables] + [(t, "ml") for t in ml_tables]:
        rows.append({"table": table.name, "layer": layer, "rows": published_counts.get(table.name),
                     "columns": len(table.columns), "grain": ", ".join(table.grain), "description": table.description})
    for name, desc in DIMENSION_DESCRIPTIONS.items():
        rows.append({"table": name, "layer": "app", "rows": published_counts.get(name), "columns": None,
                     "grain": "", "description": desc})
    return pd.DataFrame(rows)


def model_runs(meta: Path, pending: dict | None = None) -> pd.DataFrame:
    return _num(_read(meta, pending, "meta_model_runs"), ["train_rows"])


def model_metrics(meta: Path, pending: dict | None = None) -> pd.DataFrame:
    return _num(_read(meta, pending, "meta_model_metrics"), ["horizon", "value"])


def assistant_evals(meta: Path, pending: dict | None = None) -> pd.DataFrame:
    """One row per assistant evaluation run (scripts/check_phase6.py)."""
    # meta_rag_runs is written by scripts/check_phase6.py, not by the pipeline tracker: absent until a first evaluation
    files = sorted((meta / "meta_rag_runs").glob("*.csv"))
    r = pd.concat([read_csv_strings(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    if r.empty:
        return pd.DataFrame(columns=list(OPS_COLUMNS["ops_assistant_evals"]))
    out = []
    for run_id, g in r.groupby("run_id"):
        stamp = re.search(r"(\d{8}T\d{6})", run_id)
        rate = lambda col: (g[col].astype(str) == "True").sum() / max(1, g[col].isin(["True", "False"]).sum())  # noqa: E731
        qset = ("forecast held-out" if "rag_forecast_holdout" in run_id else "forecast" if "rag_forecast" in run_id else
                "engineering held-out" if "engineering_holdout" in run_id else "engineering" if "engineering" in run_id
                else "held-out" if "holdout" in run_id else "gold")
        out.append({"run_id": run_id, "question_set": qset,
                    "mode": "llm" if "_llm_" in run_id else "template",
                    "evaluated_at": pd.to_datetime(stamp.group(1), format="%Y%m%dT%H%M%S") if stamp else None,
                    "answers": len(g), "routing_rate": round(rate("route_ok"), 4),
                    "hit_at_3_rate": round(rate("hit_at_3"), 4), "numbers_rate": round(rate("number_ok"), 4),
                    "llm_verified": int(g["mode"].str.startswith("llm").sum()),
                    "llm_fallbacks": int(g["mode"].str.contains("rejected|unavailable", regex=True).sum()),
                    "median_latency_ms": pd.to_numeric(g["latency_ms"], errors="coerce").median()})
    return pd.DataFrame(out).sort_values("evaluated_at")


def build_ops_tables(meta: Path, quarantine_root: Path, gold_tables, ml_tables,
                     published_counts: dict[str, int], extra_edges: list[dict] | None = None,
                     pending: dict[str, list[dict]] | None = None) -> dict[str, pd.DataFrame]:
    """All ops tables, formatted as CSV strings in OPS_COLUMNS order. `extra_edges`: lineage of the publish run in
    progress, so the published lineage includes the app snapshot it describes."""
    ml_names = {t.name for t in ml_tables}
    frames = {
        "ops_pipeline_runs": pipeline_runs(meta, pending), "ops_stage_runs": stage_runs(meta, pending),
        "ops_table_runs": table_runs(meta, pending), "ops_quality_results": quality_results(meta, pending),
        "ops_quarantine_summary": quarantine_summary(quarantine_root), "ops_file_loads": file_loads(meta, pending),
        "ops_lineage": lineage(meta, ml_names, extra_edges, pending),
        "ops_table_catalog": table_catalog(meta, gold_tables, ml_tables, published_counts, pending),
        "ops_model_runs": model_runs(meta, pending), "ops_model_metrics": model_metrics(meta, pending),
        "ops_assistant_evals": assistant_evals(meta, pending),
    }
    out = {}
    for name, frame in frames.items():
        cols = OPS_COLUMNS[name]
        frame = frame.reindex(columns=list(cols))
        schema = TableSchema(name, tuple(Column(c, d, True) for c, d in cols.items()), ())
        out[name] = to_source_strings(frame, schema)
    return out
