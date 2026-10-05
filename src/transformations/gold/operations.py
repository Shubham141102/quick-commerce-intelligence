"""Operations Gold tables (Tier 2, built now as agreed): delivery metrics, cancellation metrics,
promotion metrics, and the quality summary for the Data Engineer workspace."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.orchestration.tracking import read_meta
from src.serving.metrics import money, ratio
from src.transformations.gold.base import Col, GoldContext, GoldTable


def build_delivery_metrics(ctx: GoldContext) -> DataFrame:
    sla = ctx.cfg.demand.delivery_sla_minutes
    d = ctx.silver("deliveries").join(ctx.silver("orders").select("order_id", "store_id"), "order_id")
    delivered = F.col("status") == "delivered"
    agg = d.groupBy("store_id", "business_date").agg(
        F.count(F.lit(1)).alias("deliveries"),
        F.sum(delivered.cast("int")).alias("delivered"),
        F.sum((F.col("status") == "failed").cast("int")).alias("failed"),
        F.sum((F.col("status") == "cancelled").cast("int")).alias("cancelled_in_transit"),
        F.round(F.avg(F.when(delivered, F.col("delivery_minutes"))), 2).alias("avg_delivery_minutes"),
        F.percentile_approx(F.when(delivered, F.col("delivery_minutes")), 0.5).alias("p50_delivery_minutes"),
        F.percentile_approx(F.when(delivered, F.col("delivery_minutes")), 0.9).alias("p90_delivery_minutes"),
        F.sum((delivered & (F.col("delivery_minutes") <= sla)).cast("int")).alias("__on_time"),
        F.countDistinct("partner_id").alias("active_partners"))
    out = (ctx.store_days().select("store_id", "business_date").join(agg, ["store_id", "business_date"], "left")
           .fillna(0, ["deliveries", "delivered", "failed", "cancelled_in_transit", "__on_time", "active_partners"]))
    return out.withColumn("on_time_rate", ratio(F.col("__on_time"), F.col("delivered")))


def build_cancellation_metrics(ctx: GoldContext) -> DataFrame:
    week = F.to_date(F.date_trunc("week", F.col("business_date")))
    orders = ctx.silver("orders").select("order_id", "store_id", "business_date")
    placed = orders.groupBy("store_id", week.alias("week_start")).agg(F.count(F.lit(1)).alias("orders_placed"))
    c = ctx.silver("cancellations").drop("business_date").join(orders, "order_id")
    agg = c.groupBy("store_id", week.alias("week_start"), "stage", "reason").agg(F.count(F.lit(1)).alias("cancellations"))
    return (agg.join(placed, ["store_id", "week_start"], "left")
            .withColumn("share_of_orders", ratio(F.col("cancellations"), F.col("orders_placed"))))


def build_promotion_metrics(ctx: GoldContext) -> DataFrame:
    promos = ctx.silver("promotions")
    network = (ctx.built["gld_daily_category_sales"].groupBy("category_id", "business_date")
               .agg(F.sum("units").alias("units"), F.max("promo_active").alias("promo_active")))
    baseline = (network.where(~F.col("promo_active")).groupBy("category_id")
                .agg(F.avg("units").alias("baseline_daily_units")))
    during = (promos.select("promotion_id", "category_id", F.explode(F.sequence("start_date", "end_date")).alias("business_date"))
              .join(network, ["category_id", "business_date"]).groupBy("promotion_id")
              .agg(F.avg("units").alias("promo_daily_units")))
    facts = ctx.order_facts().where(F.col("is_completed")).select("order_id", "discount")
    used = (ctx.silver("order_promotions").join(facts.select("order_id"), "order_id").groupBy("promotion_id")
            .agg(F.count(F.lit(1)).alias("orders_with_promotion"), F.sum("discount_amount").alias("discount_cost")))
    return (promos.select("promotion_id", "name", "category_id", "discount_pct", "start_date", "end_date")
            .withColumn("days_active", F.datediff("end_date", "start_date") + 1)
            .join(used, "promotion_id", "left").join(during, "promotion_id", "left").join(baseline, "category_id", "left")
            .fillna(0, ["orders_with_promotion"])
            .withColumn("discount_cost", money(F.col("discount_cost")))
            .withColumn("promo_daily_units", F.round("promo_daily_units", 2))
            .withColumn("baseline_daily_units", F.round("baseline_daily_units", 2))
            .withColumn("uplift_pct", F.round(ratio(F.col("promo_daily_units") - F.col("baseline_daily_units"),
                                                    F.col("baseline_daily_units")) * 100, 1)))


def build_quality_summary(ctx: GoldContext) -> DataFrame:
    quality = read_meta(ctx.metadata_root, "meta_quality_results")
    tables = read_meta(ctx.metadata_root, "meta_table_runs")
    silver_runs = tables[tables["stage"] == "silver"].sort_values("started_at")
    run_id = silver_runs["run_id"].iloc[-1] if len(silver_runs) else ""
    quality = quality[quality["run_id"] == run_id]
    rows = [(run_id, r.dataset, r.check_type, r.rule, r.column or None, int(r.rows_affected or 0),
             int(r.rows_checked or 0), r.outcome) for r in quality.itertuples()]
    schema = ("silver_run_id string, dataset string, check_type string, rule string, column string, "
              "rows_affected int, rows_checked int, outcome string")
    df = ctx.spark.createDataFrame(rows, schema) if rows else ctx.spark.createDataFrame([], schema)
    return df.withColumn("affected_pct", F.round(ratio(F.col("rows_affected"), F.col("rows_checked"), 6) * 100, 3)) \
             .withColumn("column", F.coalesce("column", F.lit("")))


DELIVERY_METRICS = GoldTable(
    "gld_delivery_metrics", ("store_id", "business_date"),
    "Delivery performance per store and day (by pickup date); zero-filled.",
    ("slv_deliveries", "slv_orders", "slv_stores"),
    (Col("store_id", "string", "store"), Col("business_date", "date", "pickup date (IST)"),
     Col("deliveries", "int", "deliveries dispatched"), Col("delivered", "int", "delivered"),
     Col("failed", "int", "failed"), Col("cancelled_in_transit", "int", "cancelled after dispatch"),
     Col("avg_delivery_minutes", "double", "mean pickup → delivery minutes (delivered)"),
     Col("p50_delivery_minutes", "double", "median delivery minutes"),
     Col("p90_delivery_minutes", "double", "90th percentile delivery minutes"),
     Col("on_time_rate", "double", "delivered within the SLA ÷ delivered"),
     Col("active_partners", "int", "distinct riders that day")),
    build_delivery_metrics, tier=2)

CANCELLATION_METRICS = GoldTable(
    "gld_cancellation_metrics", ("store_id", "week_start", "stage", "reason"),
    "Cancellations per store, week, stage and reason, with the share of that week's orders.",
    ("slv_cancellations", "slv_orders"),
    (Col("store_id", "string", "store"), Col("week_start", "date", "Monday of the week (by order date)"),
     Col("stage", "string", "pre_dispatch / post_dispatch"), Col("reason", "string", "reason code"),
     Col("cancellations", "int", "cancelled orders"), Col("orders_placed", "int", "orders placed that store-week"),
     Col("share_of_orders", "double", "cancellations ÷ orders placed")),
    build_cancellation_metrics, tier=2)

PROMOTION_METRICS = GoldTable(
    "gld_promotion_metrics", ("promotion_id",),
    "Per promotion: usage, discount cost and uplift (average daily category units, all stores, during the "
    "promotion vs days when the category had no promotion).",
    ("slv_promotions", "slv_order_promotions", "slv_orders", "gld_daily_category_sales"),
    (Col("promotion_id", "string", "promotion"), Col("name", "string", "promotion name"),
     Col("category_id", "string", "discounted category"), Col("discount_pct", "decimal", "discount %"),
     Col("start_date", "date", "first day"), Col("end_date", "date", "last day"),
     Col("days_active", "int", "length in days"),
     Col("orders_with_promotion", "int", "completed orders that used it"),
     Col("discount_cost", "decimal", "Σ discount on those orders"),
     Col("promo_daily_units", "double", "avg daily category units during the promotion (all stores)"),
     Col("baseline_daily_units", "double", "avg daily category units on days without any promotion"),
     Col("uplift_pct", "double", "(promo − baseline) ÷ baseline × 100")),
    build_promotion_metrics, tier=2)

QUALITY_SUMMARY = GoldTable(
    "gld_quality_summary", ("silver_run_id", "dataset", "check_type", "rule", "column"),
    "Silver data-quality results of the latest Silver run, ready for the Data Engineer workspace.",
    ("meta_quality_results", "meta_table_runs"),
    (Col("silver_run_id", "string", "Silver run"), Col("dataset", "string", "dataset"),
     Col("check_type", "string", "duplicates / standardize / rule / soft_rule"),
     Col("rule", "string", "rule name"), Col("column", "string", "column checked"),
     Col("rows_affected", "int", "rows removed / fixed / quarantined / flagged"),
     Col("rows_checked", "int", "rows the check ran on"),
     Col("affected_pct", "double", "rows_affected ÷ rows_checked × 100"),
     Col("outcome", "string", "removed / fixed / quarantined / cascade / flagged")),
    build_quality_summary)
