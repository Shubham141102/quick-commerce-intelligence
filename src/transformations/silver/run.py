"""The `silver` stage: Bronze -> Silver tables + quarantine + quality metrics (full rebuild each run).

Datasets are built in dependency order (SILVER_ORDER) so foreign keys can be checked against
already-accepted parents. Orders get a final post-processing step once their child tables
exist (recomputed total, total-mismatch flag, bulk-order flag, completed flag). Everything is
written at the end with the staging -> rename pattern; the previous version is kept in `_previous/`.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.common.schemas import SOURCE_SCHEMAS
from src.common.spark_io import write_csv_atomic
from src.orchestration.tracking import RunTracker, utc_now
from src.quality.rules import ISSUE_FOR_CATEGORY
from src.serving.metrics import BULK_ORDER_QUANTITY  # a line of >= 8 units = bulk order
from src.transformations.silver.engine import (
    DatasetResult,
    SilverContext,
    build_dataset,
    output_columns,
)
from src.transformations.silver.specs import SILVER_ORDER, SPECS

# Silver tables read by lookups / post-processing (besides foreign keys), for lineage
LOOKUP_PARENTS = {
    "order_items": ["products"], "order_promotions": ["orders", "promotions"],
    "payments": ["orders", "order_items", "order_promotions", "products"], "deliveries": ["delivery_partners", "stores"],
    "cancellations": ["orders"], "reviews": ["order_items"], "returns_refunds": ["payments"],
    "orders": ["order_items", "order_promotions", "payments", "cancellations", "products"],
}


def finalize_orders(ctx: SilverContext) -> None:
    spec = SPECS["orders"]
    orders = ctx.accepted["orders"].drop("computed_total_amount", "dq_total_mismatch", "is_outlier", "is_completed")
    totals = ctx.order_totals().select("order_id", F.col("computed_total").alias("computed_total_amount"))
    max_qty = ctx.accepted["order_items"].groupBy("order_id").agg(F.max("quantity").alias("__max_qty"))
    paid = (ctx.accepted["payments"].where(F.col("status") == "success").select("order_id").distinct()
            .withColumn("__paid", F.lit(True)))
    cancelled = ctx.accepted["cancellations"].select("order_id").distinct().withColumn("__cancelled", F.lit(True))
    out = (orders.join(totals, "order_id", "left").join(max_qty, "order_id", "left")
           .join(paid, "order_id", "left").join(cancelled, "order_id", "left")
           # null = could not verify (an item or promotion of the order was rejected)
           .withColumn("dq_total_mismatch", F.abs(F.col("total_amount") - F.col("computed_total_amount")) > 0.005)
           .withColumn("is_outlier", F.coalesce(F.col("__max_qty") >= BULK_ORDER_QUANTITY, F.lit(False)))
           .withColumn("is_completed", (F.col("status") == "delivered") & F.col("__paid").isNotNull()
                       & F.col("__cancelled").isNull()))
    ctx.accepted["orders"] = out.select(*output_columns(spec)).localCheckpoint()


def run_silver(spark: SparkSession, tracker: RunTracker, bronze_root: Path, silver_root: Path,
               quarantine_root: Path, cities: tuple[str, ...]) -> dict[str, int]:
    ctx = SilverContext(spark, tracker.run_id, bronze_root, cities)
    results: dict[str, DatasetResult] = {}
    timings: dict[str, tuple[str, float]] = {}
    for ds in SILVER_ORDER:
        started, t0 = utc_now(), time.perf_counter()
        results[ds] = build_dataset(SPECS[ds], ctx)
        ctx.accepted[ds] = results[ds].valid
        timings[ds] = (started, t0)
    finalize_orders(ctx)

    staging = silver_root / "_staging" / tracker.run_id
    totals = {"rows_bronze": 0, "rows_silver": 0, "rows_quarantined": 0}
    for ds in SILVER_ORDER:
        spec, res = SPECS[ds], results[ds]
        keys = list(_keys(ds))
        written = write_csv_atomic(ctx.accepted[ds].repartition(1).sortWithinPartitions(*keys),
                                   silver_root / f"slv_{ds}", staging / f"slv_{ds}", replace=True)
        quarantined = write_csv_atomic(res.quarantine.repartition(1).sortWithinPartitions("source_record_id"),
                                       quarantine_root / "qtn_records" / ds, staging / f"qtn_{ds}", replace=True)
        m = res.metrics
        if written != m["rows_valid"] or quarantined != m["rows_rejected_direct"] + m["rows_rejected_cascade"]:
            raise RuntimeError(f"silver {ds}: row counts changed while writing")
        accounted = m["rows_corrupt"] + m["rows_duplicates"] + (quarantined - m["rows_corrupt"]) + written
        if accounted != m["rows_bronze"]:
            raise RuntimeError(f"silver {ds}: {m['rows_bronze']} Bronze rows but {accounted} accounted for")
        started, t0 = timings[ds]
        tracker.table(stage="silver", job=f"silver_{ds}", source=f"brz_{ds}", target=f"slv_{ds}", engine="spark",
                      module="src.transformations.silver", rows_read=m["rows_bronze"], rows_written=written,
                      rows_rejected=m["rows_rejected_direct"], rows_rejected_cascade=m["rows_rejected_cascade"],
                      rows_deduplicated=m["rows_duplicates"], rows_corrupt=m["rows_corrupt"],
                      rows_flagged=sum(m["rows_flagged"].values()), status="success", started_at=started,
                      ended_at=utc_now(), duration_s=round(time.perf_counter() - t0, 2))
        _record_quality(tracker, ds, res)
        tracker.lineage(f"brz_{ds}", f"slv_{ds}", f"silver_{ds}")
        tracker.lineage(f"brz_{ds}", "qtn_records", f"silver_{ds}_quarantine")
        for parent in sorted({fk.parent for fk in spec.fks} | set(LOOKUP_PARENTS.get(ds, []))):
            if parent != ds:
                tracker.lineage(f"slv_{parent}", f"slv_{ds}", f"silver_{ds}_reference_check")
        totals["rows_bronze"] += m["rows_bronze"]
        totals["rows_silver"] += written
        totals["rows_quarantined"] += quarantined
    shutil.rmtree(silver_root / "_staging" / tracker.run_id, ignore_errors=True)
    return totals


def _keys(ds: str) -> tuple[str, ...]:
    return SOURCE_SCHEMAS[ds].business_key


def _record_quality(tracker: RunTracker, ds: str, res: DatasetResult) -> None:
    m = res.metrics
    rows_checked = m["rows_bronze"] - m["rows_corrupt"] - m["rows_duplicates"]
    tracker.quality(dataset=ds, check_type="duplicates", rule="duplicate_key", column="", rows_affected=m["rows_duplicates"],
                    rows_checked=m["rows_bronze"] - m["rows_corrupt"], outcome="removed")
    if m["rows_corrupt"]:
        tracker.quality(dataset=ds, check_type="rule", rule="malformed:row", column="", rows_affected=m["rows_corrupt"],
                        rows_checked=m["rows_bronze"], outcome="quarantined")
    for column, count in res.standardized:
        if count:
            tracker.quality(dataset=ds, check_type="standardize", rule="format_fixed", column=column,
                            rows_affected=count, rows_checked=rows_checked, outcome="fixed")
    for rule, count in res.rule_counts:
        category, _, column = rule.partition(":")
        tracker.quality(dataset=ds, check_type="rule", rule=rule, column=column, rows_affected=count,
                        rows_checked=rows_checked,
                        outcome="cascade" if ISSUE_FOR_CATEGORY.get(category) == "cascade" else "quarantined")
    for flag, count in m["rows_flagged"].items():
        tracker.quality(dataset=ds, check_type="soft_rule", rule=flag, column="", rows_affected=count,
                        rows_checked=m["rows_valid"], outcome="flagged")
