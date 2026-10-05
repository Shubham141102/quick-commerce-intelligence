"""The `gold` stage: Silver -> Gold tables (full rebuild each run, staging -> rename, previous kept)."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from src.common.config import Config
from src.common.spark_io import read_typed_csv, write_csv_atomic
from src.orchestration.tracking import RunTracker, utc_now
from src.transformations.gold.base import GoldContext, GoldTable
from src.transformations.gold.customers import BASKET_PAIRS, CUSTOMER_360, CUSTOMER_CATEGORY
from src.transformations.gold.features import DEMAND_FEATURES
from src.transformations.gold.inventory import INVENTORY_DAILY
from src.transformations.gold.operations import (
    CANCELLATION_METRICS,
    DELIVERY_METRICS,
    PROMOTION_METRICS,
    QUALITY_SUMMARY,
)
from src.transformations.gold.sales import DAILY_CATEGORY_SALES, DAILY_SALES, PRODUCT_PERFORMANCE

# Build order: tables that other Gold tables read come first.
GOLD_TABLES: tuple[GoldTable, ...] = (
    DAILY_SALES, DAILY_CATEGORY_SALES, PRODUCT_PERFORMANCE, INVENTORY_DAILY, DEMAND_FEATURES,
    CUSTOMER_360, CUSTOMER_CATEGORY, BASKET_PAIRS,
    DELIVERY_METRICS, CANCELLATION_METRICS, PROMOTION_METRICS, QUALITY_SUMMARY,
)
GOLD = {t.name: t for t in GOLD_TABLES}


def read_gold(spark: SparkSession, gold_root: Path, name: str) -> DataFrame:
    return read_typed_csv(spark, gold_root / name, GOLD[name].schema())


def run_gold(spark: SparkSession, tracker: RunTracker, cfg: Config, silver_root: Path, gold_root: Path,
             metadata_root: Path) -> dict[str, int]:
    ctx = GoldContext(spark, cfg, silver_root, metadata_root)
    staging = gold_root / "_staging" / tracker.run_id
    rows = {}
    for table in GOLD_TABLES:
        started, t0 = utc_now(), time.perf_counter()
        df = table.build(ctx).select(*[ctx_col.name for ctx_col in table.columns])
        df = df.select(*[df[c.name].cast(table.schema()[c.name].dataType) for c in table.columns]).localCheckpoint()
        ctx.built[table.name] = df
        written = write_csv_atomic(df.repartition(1).sortWithinPartitions(*table.grain), gold_root / table.name,
                                   staging / table.name, replace=True)
        rows[table.name] = written
        tracker.table(stage="gold", job=table.name, source=",".join(table.sources), target=table.name,
                      engine="spark", module="src.transformations.gold", rows_written=written, status="success",
                      started_at=started, ended_at=utc_now(), duration_s=round(time.perf_counter() - t0, 2))
        for source in table.sources:
            tracker.lineage(source, table.name, table.name)
    shutil.rmtree(staging, ignore_errors=True)
    return rows
