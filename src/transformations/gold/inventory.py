"""gld_inventory_daily: daily stock per store × focus SKU, rebuilt from Silver.

Silver has weekly stock counts (snapshots), stock events (restock / damage / adjustment) and sales
(order lines). Stock at any instant t is

    stock(t) = last count before t + Σ movements between that count and t

(before the first count, the first count is used backwards). Each weekly count re-anchors the running
stock; `reconciliation_gap` = counted − calculated at each count. The gap is non-zero when Silver
quarantined some lines or events, which is what a data engineer monitors. Late events need no special
handling: Gold is rebuilt in full, so they sit at their event time and every affected day is recomputed.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.serving.metrics import ratio
from src.transformations.gold.base import Col, GoldContext, GoldTable, ist_day_start

PAIR = ["store_id", "product_id"]
# Ordering of rows at the same instant: close the old day, open the new day, then movements, then counts
P_DAY_END, P_DAY_START, P_MOVE, P_SNAPSHOT = 0, 1, 2, 3


def build_inventory_daily(ctx: GoldContext) -> DataFrame:
    snaps = ctx.silver("inventory_snapshots").select(*PAIR, F.col("snapshot_ts").alias("ts"),
                                                     F.col("stock_quantity").alias("snap_value"), "reorder_level")
    pairs = snaps.select(*PAIR).distinct()

    pre_dispatch = (ctx.silver("cancellations").where(F.col("stage") == "pre_dispatch")
                    .select("order_id").withColumn("__pre", F.lit(True)))
    sales = (ctx.order_lines().join(pre_dispatch, "order_id", "left").where(F.col("__pre").isNull())
             .join(pairs, PAIR)
             .select(*PAIR, F.col("order_ts").alias("ts"), (-F.col("quantity")).alias("delta"), F.lit("sale").alias("kind")))
    events = ctx.silver("inventory_events").join(pairs, PAIR).select(
        *PAIR, F.col("event_ts").alias("ts"), F.col("quantity").alias("delta"), F.col("event_type").alias("kind"))
    days = pairs.crossJoin(ctx.calendar().select("business_date"))
    starts = days.select(*PAIR, "business_date", ist_day_start("business_date").alias("ts"))
    ends = days.select(*PAIR, "business_date", F.expr("business_date + 1").alias("__next")) \
               .select(*PAIR, "business_date", ist_day_start("__next").alias("ts"))

    def rows(df: DataFrame, priority: int, kind: str | None = None) -> DataFrame:
        cols = {"delta": F.lit(0), "kind": F.lit(kind), "snap_value": F.lit(None).cast("int"),
                "reorder_level": F.lit(None).cast("int"), "business_date": F.lit(None).cast("date")}
        for c in cols:
            if c in df.columns:
                cols[c] = F.col(c)
        return df.select(*PAIR, "ts", *[v.alias(k) for k, v in cols.items()], F.lit(priority).alias("priority"))

    stream = (rows(sales, P_MOVE).unionByName(rows(events, P_MOVE)).unionByName(rows(snaps, P_SNAPSHOT, "snapshot"))
              .unionByName(rows(starts, P_DAY_START, "day_start")).unionByName(rows(ends, P_DAY_END, "day_end")))
    ordered = Window.partitionBy(*PAIR).orderBy("ts", "priority")
    upto = ordered.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    whole = Window.partitionBy(*PAIR)
    is_snap = F.col("kind") == "snapshot"
    stream = (stream.withColumn("cum", F.sum("delta").over(upto))
              .withColumn("anchor_value", F.last(F.col("snap_value"), ignorenulls=True).over(upto))
              .withColumn("anchor_cum", F.last(F.when(is_snap, F.col("cum")), ignorenulls=True).over(upto))
              .withColumn("first_value", F.first(F.col("snap_value"), ignorenulls=True).over(whole))
              .withColumn("first_cum", F.min(F.when(is_snap, F.col("cum"))).over(whole))
              .withColumn("reorder", F.coalesce(F.last(F.col("reorder_level"), ignorenulls=True).over(upto),
                                                F.first(F.col("reorder_level"), ignorenulls=True).over(whole)))
              .withColumn("stock", F.coalesce(F.col("anchor_value") + F.col("cum") - F.col("anchor_cum"),
                                              F.col("first_value") + F.col("cum") - F.col("first_cum")))
              .withColumn("day", F.coalesce(F.col("business_date"),
                                            F.to_date(F.from_utc_timestamp("ts", "Asia/Kolkata"))))
              .localCheckpoint())

    # reconciliation at each count: counted − (previous count + movements since it)
    snap_rows = stream.where(is_snap).select(*PAIR, "ts", "day", "snap_value", "cum")
    prev = Window.partitionBy(*PAIR).orderBy("ts")
    recon = (snap_rows.withColumn("__prev_value", F.lag("snap_value").over(prev))
             .withColumn("__prev_cum", F.lag("cum").over(prev))
             .select(*PAIR, F.col("day").alias("business_date"), F.col("snap_value").alias("snapshot_stock"),
                     (F.col("snap_value") - (F.col("__prev_value") + F.col("cum") - F.col("__prev_cum")))
                     .alias("reconciliation_gap")))

    moves = stream.where(F.col("priority") == P_MOVE)
    per_day = moves.groupBy(*PAIR, F.col("day").alias("business_date")).agg(
        F.sum(F.when(F.col("kind") == "sale", -F.col("delta"))).alias("units_sold"),
        F.sum(F.when(F.col("kind") == "restock", F.col("delta"))).alias("restocked"),
        F.sum(F.when(F.col("kind") == "damage", -F.col("delta"))).alias("damaged"),
        F.sum(F.when(F.col("kind") == "adjustment", F.col("delta"))).alias("adjusted"),
        F.min("stock").alias("__min_moving"))
    opening = stream.where(F.col("kind") == "day_start").select(*PAIR, "business_date", F.col("stock").alias("__open"))
    closing = stream.where(F.col("kind") == "day_end").select(*PAIR, "business_date", F.col("stock").alias("__close"),
                                                               F.col("reorder").alias("reorder_level"))
    out = (opening.join(closing, [*PAIR, "business_date"]).join(per_day, [*PAIR, "business_date"], "left")
           .join(recon, [*PAIR, "business_date"], "left")
           .fillna(0, ["units_sold", "restocked", "damaged", "adjusted"]))
    lowest = F.least(F.col("__open"), F.col("__close"), F.coalesce(F.col("__min_moving"), F.col("__close")))
    trailing = Window.partitionBy(*PAIR).orderBy("business_date").rowsBetween(-13, 0)
    return (out.withColumn("opening_stock", F.greatest(F.col("__open"), F.lit(0)).cast("int"))
            .withColumn("closing_stock", F.greatest(F.col("__close"), F.lit(0)).cast("int"))
            .withColumn("calculated_negative", lowest < 0)
            .withColumn("is_stockout", lowest <= 0)
            .withColumn("below_reorder_level", F.col("closing_stock") <= F.col("reorder_level"))
            .withColumn("avg_daily_units_14d", F.round(F.avg("units_sold").over(trailing), 3))
            .withColumn("days_of_inventory", ratio(F.col("closing_stock"), F.col("avg_daily_units_14d"), 2)))


INVENTORY_DAILY = GoldTable(
    "gld_inventory_daily", ("store_id", "product_id", "business_date"),
    "Daily stock per store × focus SKU, rebuilt from weekly counts + stock events − sales; each count re-anchors "
    "the running stock and the difference is reported as reconciliation_gap.",
    ("slv_inventory_snapshots", "slv_inventory_events", "slv_order_items", "slv_orders", "slv_cancellations",
     "slv_products"),
    (Col("store_id", "string", "store"), Col("product_id", "string", "focus SKU"),
     Col("business_date", "date", "business date (IST)"),
     Col("opening_stock", "int", "stock at 00:00 IST (calculated, floored at 0)"),
     Col("units_sold", "int", "units leaving stock (orders not cancelled before dispatch)"),
     Col("restocked", "int", "units received"), Col("damaged", "int", "units written off"),
     Col("adjusted", "int", "net cycle-count adjustment (±)"),
     Col("closing_stock", "int", "stock at 24:00 IST (calculated, floored at 0)"),
     Col("reorder_level", "int", "reorder level from the latest count"),
     Col("below_reorder_level", "boolean", "closing stock ≤ reorder level"),
     Col("is_stockout", "boolean", "stock reached 0 at some point during the day"),
     Col("calculated_negative", "boolean", "the calculation went below 0 (missing upstream rows), floored to 0"),
     Col("snapshot_stock", "int", "counted stock, on days with a weekly count"),
     Col("reconciliation_gap", "int", "counted − calculated at the count (empty for the first count)"),
     Col("avg_daily_units_14d", "double", "average units sold per day over the last 14 days"),
     Col("days_of_inventory", "double", "closing stock ÷ avg_daily_units_14d; empty if no recent sales")),
    build_inventory_daily)
