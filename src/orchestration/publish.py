"""The `publish` stage: copy the tables the app needs into `data/demo/` (Project_Plan_v2.md §11, §14.2).

The Streamlit app reads only this folder (through DuckDB) and never runs Spark or the pipeline code:
- one flat CSV per table (all Gold + ML tables, plus small product / store / category lookups)
- `snapshot_schema.csv`: column types for every table, so the app can read them without pipeline code
- `snapshot_manifest.csv`: which pipeline run and generation run produced the snapshot, and when

The folder is written to a staging folder, size-checked, then swapped in (a failed run never replaces
the current snapshot). It is committed to git so the hosted app can read it.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pandas as pd

from src.common.io import read_csv_strings, replace_with_retry, write_csv
from src.common.schemas import SOURCE_SCHEMAS
from src.orchestration.tracking import RunTracker, utc_now

MAX_SNAPSHOT_MB = 50
DIMENSIONS = {  # app lookup tables from Silver: name -> (silver dataset, columns)
    "dim_products": ("products", ["product_id", "product_name", "category_id", "brand", "price"]),
    "dim_stores": ("stores", ["store_id", "store_name", "city_id", "city", "state"]),
    "dim_categories": ("categories", ["category_id", "category_name"]),
}


def _tables():
    from src.ml.tables import ML_TABLES
    from src.transformations.gold.run import GOLD_TABLES
    return (*GOLD_TABLES, *ML_TABLES)


def run_publish(tracker: RunTracker, gold_root: Path, silver_root: Path, demo_root: Path,
                metadata_root: Path, generation_run_id: str, profile: str) -> dict:  # noqa: ARG001 (kept for callers)
    t0, started = time.perf_counter(), utc_now()
    staging = demo_root.parent / "_demo_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    schema_rows, counts = [], {}

    for table in _tables():
        parts = sorted((gold_root / table.name).glob("*.csv"))
        if not parts:
            raise FileNotFoundError(f"{table.name} missing; run the gold and ml stages first")
        frame = pd.concat([read_csv_strings(p) for p in parts], ignore_index=True)
        counts[table.name] = write_csv(frame, staging / f"{table.name}.csv", table.column_names)
        schema_rows += [{"table": table.name, "column": c.name, "dtype": c.dtype} for c in table.columns]

    for name, (dataset, columns) in DIMENSIONS.items():
        frame = pd.concat([read_csv_strings(p) for p in (silver_root / f"slv_{dataset}").glob("*.csv")])
        counts[name] = write_csv(frame.sort_values(columns[0]), staging / f"{name}.csv", columns)
        types = {c.name: c.dtype for c in SOURCE_SCHEMAS[dataset].columns}
        schema_rows += [{"table": name, "column": c, "dtype": types[c]} for c in columns]

    write_csv(pd.DataFrame(schema_rows), staging / "snapshot_schema.csv", ["table", "column", "dtype"])
    # The model version comes from the published predictions themselves: the run log (meta_model_runs) is only
    # written when a run ends, so within an `ml,publish` run it would still hold the previous model.
    preds = read_csv_strings(staging / "gld_demand_predictions.csv")
    model_version = preds["model_version"].iloc[0] if len(preds) else ""
    size_mb = sum(p.stat().st_size for p in staging.glob("*.csv")) / 1e6
    manifest = {"pipeline_run_id": tracker.run_id, "generation_run_id": generation_run_id, "profile": profile,
                "published_at": utc_now(), "tables": str(len(counts)), "rows": str(sum(counts.values())),
                "size_mb": f"{size_mb:.1f}", "forecast_model_version": model_version}
    write_csv(pd.DataFrame(manifest.items(), columns=["key", "value"]), staging / "snapshot_manifest.csv",
              ["key", "value"])
    if size_mb > MAX_SNAPSHOT_MB:
        raise RuntimeError(f"snapshot is {size_mb:.1f} MB, above the {MAX_SNAPSHOT_MB} MB limit")

    if demo_root.exists():
        previous = demo_root.parent / "_demo_previous"
        if previous.exists():
            shutil.rmtree(previous)
        replace_with_retry(demo_root, previous)
    replace_with_retry(staging, demo_root)
    tracker.table(stage="publish", job="publish_demo_snapshot", source="gold,ml,silver dims", target="data/demo",
                  engine="python", module="src.orchestration.publish", rows_written=sum(counts.values()),
                  status="success", started_at=started, ended_at=utc_now(), duration_s=round(time.perf_counter() - t0, 2))
    for name in counts:
        tracker.lineage(name if name.startswith("gld_") else f"slv_{DIMENSIONS[name][0]}", f"demo/{name}",
                        "publish", engine="python")
    return {"tables": len(counts), "rows": sum(counts.values()), "size_mb": round(size_mb, 1)}
