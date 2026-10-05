"""Simulated-stream ingestion into Bronze with Spark Structured Streaming (pattern C).

For each stream dataset, a file-source stream watches `landing/stream/<dataset>/`,
processes up to `MAX_FILES_PER_TRIGGER` hourly drops per micro-batch with
`trigger(availableNow=True)`, and remembers processed files in a checkpoint.

Each micro-batch becomes one Bronze batch folder. Rows are never dropped; each row is
labelled `_is_late` (event before its file's hour) and `_beyond_watermark` (older than
max event time seen in earlier micro-batches minus WATERMARK_DELAY). The watermark state
is kept next to the checkpoint so it survives restarts.

`meta_file_loads` stays the source of truth: if a checkpoint is lost, files already
loaded are filtered out instead of being ingested twice.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.common.schemas import STREAM_DATASETS
from src.common.spark_io import relative_source_file, source_stream
from src.ingestion.bronze_writer import (
    ISO_PATTERN,
    SOURCE_FILE_RAW,
    BronzeContext,
    event_time_utc,
    record_skipped,
    write_bronze_batch,
)
from src.ingestion.file_tracker import LandingFile

MAX_FILES_PER_TRIGGER = 24          # one day of hourly drops per micro-batch
# Shorter than the generator's maximum lateness (6 h) on purpose: it shows the trade-off a
# streaming job makes, flagging the late events a 2-hour-watermarked aggregation would drop.
WATERMARK_DELAY = timedelta(hours=2)
UTC_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _load_state(path: Path) -> datetime | None:
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8")).get("max_event_ts")
        return datetime.fromisoformat(value) if value else None
    return None


def _save_state(path: Path, max_event: datetime | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"max_event_ts": max_event.isoformat() if max_event else None}), encoding="utf-8")


def ingest_stream(spark: SparkSession, ctx: BronzeContext, files: list[LandingFile],
                  already_loaded: set, force: bool, landing_dir: Path) -> int:
    total = 0
    for dataset in STREAM_DATASETS:
        source_dir = landing_dir / "stream" / dataset
        group = [f for f in files if f.dataset == dataset and f.slice == "stream"]
        if not group:
            continue
        if not force:
            record_skipped(ctx, [f for f in group if f.identity in already_loaded], "already loaded")
        checkpoint = ctx.bronze_root / "_checkpoints" / dataset
        state_file = ctx.bronze_root / "_checkpoints" / f"{dataset}_watermark.json"
        if force:
            shutil.rmtree(checkpoint, ignore_errors=True)
            state_file.unlink(missing_ok=True)
        loaded_names = {name for ds, name, _ in already_loaded if ds == dataset} if not force else set()
        written = {"rows": 0}

        def process(batch_df: DataFrame, epoch_id: int, dataset=dataset, state_file=state_file,
                    loaded_names=loaded_names, written=written) -> None:
            batch_df = batch_df.withColumn("_rel", relative_source_file(F.col(SOURCE_FILE_RAW)))
            if loaded_names:
                names = F.regexp_extract(F.col("_rel"), r"([^/]+)$", 1)
                batch_df = batch_df.filter(~names.isin(sorted(loaded_names)))
            batch_df = batch_df.drop("_rel").cache()
            try:
                if batch_df.isEmpty():
                    return
                # Timestamps cross the Python boundary as UTC strings: PySpark would otherwise
                # convert them through the machine's local timezone.
                event = event_time_utc(dataset)  # None for datasets without their own timestamp
                max_seen = _load_state(state_file) if event is not None else None
                watermark = (max_seen - WATERMARK_DELAY) if max_seen else None
                beyond = (event < F.to_timestamp(F.lit(watermark.strftime(UTC_FMT)), ISO_PATTERN)) if watermark else None
                batch_id = f"{ctx.tracker.run_id}__stream_{epoch_id:03d}"
                written["rows"] += write_bronze_batch(ctx, batch_df, dataset, "stream", batch_id,
                                                      load_type="stream", beyond_watermark=beyond)
                if event is None:
                    return
                batch_max = batch_df.select(F.date_format(F.max(event), "yyyy-MM-dd'T'HH:mm:ss'Z'")).first()[0]
                if batch_max is not None:
                    batch_max = datetime.strptime(batch_max, UTC_FMT).replace(tzinfo=timezone.utc)
                    _save_state(state_file, max(batch_max, max_seen) if max_seen else batch_max)
            finally:
                batch_df.unpersist()

        stream = source_stream(spark, source_dir, dataset, MAX_FILES_PER_TRIGGER).withColumn(
            SOURCE_FILE_RAW, F.input_file_name())
        query = (stream.writeStream.foreachBatch(process)
                 .option("checkpointLocation", checkpoint.as_posix())
                 .trigger(availableNow=True)
                 .queryName(f"bronze_stream_{dataset}")
                 .start())
        query.awaitTermination()
        total += written["rows"]
    return total
