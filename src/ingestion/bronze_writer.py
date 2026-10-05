"""Shared Bronze step: add ingestion metadata to raw rows and write one batch (Project_Plan_v2.md §5).

Source columns are passed through untouched (as strings). Metadata columns start
with `_`. Two derived flags are best-effort (empty when the event time can't be parsed):
- `_is_late`: the event happened before the window its landing file covers
  (an earlier day for daily files, an earlier hour for stream drops);
- `_beyond_watermark` (stream only): the event is older than the stream's watermark
  (max event time seen in earlier micro-batches minus the allowed delay), i.e. a
  watermarked streaming aggregation would drop it. Bronze keeps the row either way.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from src.common.io import TIMESTAMP_FMT
from src.common.schemas import (
    CORRUPT_RECORD,
    EVENT_TIME_COLUMNS,
    SCHEMA_VERSION,
    SOURCE_SCHEMAS,
    SOURCE_SYSTEMS,
    bronze_columns,
)
from src.common.spark_io import relative_source_file, write_csv_atomic
from src.ingestion.file_tracker import LandingFile
from src.orchestration.tracking import RunTracker, utc_now

SOURCE_FILE_RAW = "_source_file_raw"
ISO_PATTERN = "yyyy-MM-dd'T'HH:mm:ssX"
BUSINESS_TZ = "Asia/Kolkata"
IST_OFFSET = timedelta(hours=5, minutes=30)


@dataclass
class BronzeContext:
    tracker: RunTracker
    bronze_root: Path
    generation_run_id: str
    files_by_rel: dict[str, LandingFile]


def event_time_utc(dataset: str) -> Column | None:
    col = EVENT_TIME_COLUMNS.get(dataset)
    return F.to_timestamp(F.col(col), ISO_PATTERN) if col else None


def _file_window_start(source_file: Column) -> Column:
    """IST wall-clock start of the window a landing file covers, parsed from its name."""
    key = F.regexp_extract(source_file, r"__([0-9T-]+)\.csv$", 1)
    return (F.when(F.length(key) == 7, F.to_timestamp(F.concat(key, F.lit("-01")), "yyyy-MM-dd"))
            .when(F.length(key) == 10, F.to_timestamp(key, "yyyy-MM-dd"))
            .when(F.length(key) == 16, F.to_timestamp(key, "yyyy-MM-dd'T'HH-mm")))


def _flag(condition: Column) -> Column:
    return F.when(condition.isNull(), F.lit(None)).when(condition, F.lit("true")).otherwise(F.lit("false"))


def add_bronze_metadata(df: DataFrame, dataset: str, batch_id: str, run_id: str, load_type: str,
                        ingestion_ts: datetime, beyond_watermark: Column | None = None) -> DataFrame:
    source_cols = SOURCE_SCHEMAS[dataset].names
    out = df.withColumn("_source_file", relative_source_file(F.col(SOURCE_FILE_RAW)))
    event = event_time_utc(dataset)
    is_late = (_flag(F.from_utc_timestamp(event, BUSINESS_TZ) < _file_window_start(F.col("_source_file")))
               if event is not None else F.lit(None))
    hash_input = F.concat_ws("\x1f", *[F.coalesce(F.col(c), F.lit("\x00")) for c in source_cols + [CORRUPT_RECORD]])
    ist_date = (ingestion_ts + IST_OFFSET).strftime("%Y-%m-%d")
    out = (out
           .withColumn("_batch_id", F.lit(batch_id))
           .withColumn("_pipeline_run_id", F.lit(run_id))
           .withColumn("_load_type", F.lit(load_type))
           .withColumn("_source_system", F.lit(SOURCE_SYSTEMS[dataset]))
           .withColumn("_schema_version", F.lit(SCHEMA_VERSION))
           .withColumn("_ingestion_ts", F.lit(ingestion_ts.strftime(TIMESTAMP_FMT)))
           .withColumn("_ingestion_date", F.lit(ist_date))
           .withColumn("_record_hash", F.sha2(hash_input, 256))
           .withColumn("_is_late", is_late)
           .withColumn("_beyond_watermark", _flag(beyond_watermark) if beyond_watermark is not None else F.lit(None)))
    return out.select(*[F.col(c).cast("string").alias(c) for c in bronze_columns(dataset)])


def write_bronze_batch(ctx: BronzeContext, raw: DataFrame, dataset: str, slice_name: str, batch_id: str,
                       load_type: str, beyond_watermark: Column | None = None) -> int:
    """Add metadata, write one Bronze batch folder, record table/file/lineage metadata. Returns rows written."""
    started, t0 = utc_now(), time.perf_counter()
    ingestion_ts = datetime.now(timezone.utc).replace(microsecond=0)
    out = add_bronze_metadata(raw, dataset, batch_id, ctx.tracker.run_id, load_type, ingestion_ts,
                              beyond_watermark).cache()
    try:
        per_file = (out.groupBy("_source_file")
                    .agg(F.count(F.lit(1)).alias("rows"),
                         F.sum(F.col(CORRUPT_RECORD).isNotNull().cast("int")).alias("corrupt"))
                    .collect())
        table_dir = ctx.bronze_root / f"brz_{dataset}"
        written = write_csv_atomic(out, table_dir / f"batch={batch_id}",
                                   ctx.bronze_root / "_staging" / ctx.tracker.run_id / f"brz_{dataset}" / batch_id)
        rows_read = sum(r["rows"] for r in per_file)
        corrupt = sum(r["corrupt"] or 0 for r in per_file)
        if written != rows_read:
            raise RuntimeError(f"brz_{dataset}/{batch_id}: read {rows_read} rows but wrote {written}")
        ctx.tracker.table(stage="ingest", job=f"bronze_{load_type}", source=f"landing/{slice_name}/{dataset}",
                          target=f"brz_{dataset}", engine="spark", module=f"src.ingestion.{_module(load_type)}",
                          batch_id=batch_id, rows_read=rows_read, rows_written=written, rows_corrupt=corrupt,
                          status="success", started_at=started, ended_at=utc_now(),
                          duration_s=round(time.perf_counter() - t0, 2))
        ctx.tracker.lineage(f"landing/{dataset}", f"brz_{dataset}", f"bronze_ingest_{dataset}")
        loaded_at = utc_now()
        for r in per_file:
            lf = ctx.files_by_rel.get(r["_source_file"])
            ctx.tracker.file_load(batch_id=batch_id, generation_run_id=ctx.generation_run_id, dataset=dataset,
                                  slice=slice_name, file=r["_source_file"],
                                  size_bytes=lf.size if lf else "", sha256=lf.sha256 if lf else "",
                                  rows_read=r["rows"], rows_corrupt=r["corrupt"] or 0, status="loaded",
                                  loaded_at=loaded_at)
        return written
    finally:
        out.unpersist()


def record_skipped(ctx: BronzeContext, files: list[LandingFile], reason: str) -> None:
    for lf in files:
        ctx.tracker.file_load(generation_run_id=ctx.generation_run_id, dataset=lf.dataset, slice=lf.slice,
                              file=lf.rel, size_bytes=lf.size, sha256=lf.sha256, status="skipped",
                              loaded_at=utc_now(), message=reason)


def _module(load_type: str) -> str:
    return "stream" if load_type == "stream" else "batch"
