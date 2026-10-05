"""Sales Gold tables: daily sales (store × day), daily category sales (store × category × day),
product performance (product × month)."""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.serving.metrics import BULK_ORDER_QUANTITY, money, ratio
from src.transformations.gold.base import Col, GoldContext, GoldTable


def build_daily_sales(ctx: GoldContext) -> DataFrame:
    facts = ctx.order_facts()
    done = F.col("is_completed")
    daily = facts.groupBy("store_id", "business_date").agg(
        F.count(F.lit(1)).alias("orders_placed"),
        F.sum((F.col("status") == "cancelled").cast("int")).alias("orders_cancelled"),
        F.sum(done.cast("int")).alias("orders_completed"),
        F.sum(F.when(done, F.col("gmv"))).alias("gmv"),
        F.sum(F.when(done, F.col("discount"))).alias("discount"),
        F.sum(F.when(done, F.col("units"))).alias("units"),
        F.countDistinct(F.when(done, F.col("customer_id"))).alias("unique_customers"),
        F.sum(F.when(done & (F.col("max_quantity") >= BULK_ORDER_QUANTITY), 1).otherwise(0)).alias("bulk_orders"),
    )
    refunds = ctx.completed_refunds().groupBy("store_id", "business_date").agg(F.sum("amount").alias("refunds"))
    out = (ctx.store_days().join(daily, ["store_id", "business_date"], "left")
           .join(refunds, ["store_id", "business_date"], "left")
           .fillna(0, ["orders_placed", "orders_cancelled", "orders_completed", "units", "unique_customers", "bulk_orders"]))
    for c in ("gmv", "discount", "refunds"):
        out = out.withColumn(c, money(F.col(c)))
    return (out.withColumn("net_revenue", (F.col("gmv") - F.col("discount") - F.col("refunds")).cast("decimal(12,2)"))
            .withColumn("avg_order_value", ratio(F.col("gmv") - F.col("discount"), F.col("orders_completed"), 2))
            .withColumn("cancellation_rate", ratio(F.col("orders_cancelled"), F.col("orders_placed"))))


def build_daily_category_sales(ctx: GoldContext) -> DataFrame:
    lines = ctx.order_lines().where(F.col("is_completed"))
    agg = lines.groupBy("store_id", "category_id", "business_date").agg(
        F.sum("quantity").alias("units"), F.sum("revenue").alias("revenue"),
        F.countDistinct("order_id").alias("orders"))
    promos = ctx.silver("promotions")
    promo_days = (promos.select("category_id", F.explode(F.sequence("start_date", "end_date")).alias("business_date"))
                  .distinct().withColumn("promo_active", F.lit(True)))
    spine = (ctx.store_days().select("store_id", "business_date")
             .crossJoin(ctx.silver("categories").select("category_id")))
    return (spine.join(agg, ["store_id", "category_id", "business_date"], "left")
            .join(promo_days, ["category_id", "business_date"], "left")
            .fillna(0, ["units", "orders"])
            .withColumn("revenue", money(F.col("revenue")))
            .withColumn("promo_active", F.coalesce("promo_active", F.lit(False)))
            .withColumn("avg_unit_price", ratio(F.col("revenue"), F.col("units"), 2)))


def build_product_performance(ctx: GoldContext) -> DataFrame:
    lines = ctx.order_lines().where(F.col("is_completed")).withColumn("month", F.trunc("business_date", "month"))
    agg = lines.groupBy("product_id", "month").agg(
        F.sum("quantity").alias("units"), F.sum("revenue").alias("revenue"), F.countDistinct("order_id").alias("orders"))
    months = ctx.calendar().select(F.trunc("business_date", "month").alias("month")).distinct()
    products = ctx.silver("products").select("product_id", "product_name", "brand", "category_id")
    out = (products.crossJoin(months).join(agg, ["product_id", "month"], "left")
           .fillna(0, ["units", "orders"]).withColumn("revenue", money(F.col("revenue"))))
    by_cat = Window.partitionBy("category_id", "month")
    return (out.withColumn("avg_selling_price", ratio(F.col("revenue"), F.col("units"), 2))
            .withColumn("rank_in_category", F.dense_rank().over(by_cat.orderBy(F.col("revenue").desc())))
            .withColumn("share_of_category_revenue", ratio(F.col("revenue"), F.sum("revenue").over(by_cat))))


DAILY_SALES = GoldTable(
    "gld_daily_sales", ("store_id", "business_date"),
    "Daily sales per store. Every store × day of the period has a row (days without orders are zeros).",
    ("slv_orders", "slv_order_items", "slv_products", "slv_order_promotions", "slv_returns_refunds", "slv_stores"),
    (Col("store_id", "string", "store"), Col("business_date", "date", "business date (IST)"),
     Col("city_id", "string", "city of the store"),
     Col("orders_placed", "int", "orders placed (any outcome)"),
     Col("orders_cancelled", "int", "orders with status cancelled"),
     Col("orders_completed", "int", "completed orders"),
     Col("gmv", "decimal", "GMV of completed orders"),
     Col("discount", "decimal", "discounts on completed orders"),
     Col("refunds", "decimal", "completed refunds dated this day"),
     Col("net_revenue", "decimal", "GMV − discount − refunds"),
     Col("avg_order_value", "double", "(GMV − discount) ÷ completed orders; empty if none"),
     Col("units", "int", "units in completed orders"),
     Col("unique_customers", "int", "distinct customers with a completed order"),
     Col("bulk_orders", "int", "completed bulk orders (a line of ≥ 8 units)"),
     Col("cancellation_rate", "double", "cancelled ÷ placed; empty if none placed")),
    build_daily_sales)

DAILY_CATEGORY_SALES = GoldTable(
    "gld_daily_category_sales", ("store_id", "category_id", "business_date"),
    "Daily sales per store and category (completed orders); zero-filled. The demand-forecasting series.",
    ("slv_orders", "slv_order_items", "slv_products", "slv_promotions", "slv_categories", "slv_stores"),
    (Col("store_id", "string", "store"), Col("category_id", "string", "category"),
     Col("business_date", "date", "business date (IST)"),
     Col("units", "int", "units sold"), Col("revenue", "decimal", "line revenue"),
     Col("orders", "int", "completed orders containing the category"),
     Col("promo_active", "boolean", "a promotion for the category was active that day"),
     Col("avg_unit_price", "double", "revenue ÷ units; empty if none")),
    build_daily_category_sales)

PRODUCT_PERFORMANCE = GoldTable(
    "gld_product_performance", ("product_id", "month"),
    "Monthly performance per product (completed orders), with its rank inside its category.",
    ("slv_orders", "slv_order_items", "slv_products"),
    (Col("product_id", "string", "product"), Col("month", "date", "first day of the month"),
     Col("product_name", "string", "product name"), Col("brand", "string", "brand"),
     Col("category_id", "string", "category"),
     Col("units", "int", "units sold"), Col("revenue", "decimal", "line revenue"),
     Col("orders", "int", "completed orders containing the product"),
     Col("avg_selling_price", "double", "revenue ÷ units"),
     Col("rank_in_category", "int", "1 = highest revenue in its category that month"),
     Col("share_of_category_revenue", "double", "share of its category's revenue that month")),
    build_product_performance)
