"""Batch ingestion into Bronze: historical backfill (pattern B), daily batch (A) and logs (D).

Logs use the same reader: rows Spark cannot parse are kept with the raw line in
`_corrupt_record`, and per-file corrupt counts go to `meta_file_loads`.
"""

from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.common.schemas import SOURCE_SCHEMAS
from src.common.spark_io import read_source_csv
from src.ingestion.bronze_writer import (
    SOURCE_FILE_RAW,
    BronzeContext,
    record_skipped,
    write_bronze_batch,
)
from src.ingestion.file_tracker import LandingFile

BATCH_SLICES = ("historical", "batch")  # backfill first, then routine daily files


def ingest_batch(spark: SparkSession, ctx: BronzeContext, files: list[LandingFile],
                 already_loaded: set, force: bool) -> int:
    total = 0
    for slice_name in BATCH_SLICES:
        for dataset in SOURCE_SCHEMAS:
            group = [f for f in files if f.dataset == dataset and f.slice == slice_name]
            new = [f for f in group if force or f.identity not in already_loaded]
            record_skipped(ctx, [f for f in group if f not in new], "already loaded")
            if not new:
                continue
            raw = read_source_csv(spark, [f.path for f in new], dataset).withColumn(SOURCE_FILE_RAW, F.input_file_name())
            batch_id = f"{ctx.tracker.run_id}__{slice_name}"
            total += write_bronze_batch(ctx, raw, dataset, slice_name, batch_id, load_type=slice_name)
    return total
