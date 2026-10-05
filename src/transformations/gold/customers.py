"""Customer and basket Gold tables: customer 360 (customer), customer × category, basket pairs."""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.serving.metrics import money, ratio
from src.transformations.gold.base import Col, GoldContext, GoldTable

NIGHT_HOURS_IST = (21, 22, 23, 0, 1)
MIN_PAIR_BASKETS = 8


def build_customer_360(ctx: GoldContext) -> DataFrame:
    facts = ctx.order_facts().withColumn("__hour", F.hour(F.from_utc_timestamp("order_ts", "Asia/Kolkata")))
    done = F.col("is_completed")
    per = facts.groupBy("customer_id").agg(
        F.count(F.lit(1)).alias("orders_placed"),
        F.sum(done.cast("int")).alias("orders_completed"),
        F.sum((F.col("status") == "cancelled").cast("int")).alias("orders_cancelled"),
        F.sum(F.when(done, F.col("gmv") - F.col("discount"))).alias("total_spend"),
        F.sum(F.when(done, F.col("units"))).alias("units"),
        F.avg(F.when(done, F.col("items"))).alias("avg_items_per_order"),
        F.min("business_date").alias("first_order_date"),
        F.max(F.when(done, F.col("business_date"))).alias("last_order_date"),
        F.sum(F.col("__hour").isin(*NIGHT_HOURS_IST).cast("int")).alias("__night"),
        F.sum(F.when(done & F.col("has_promotion"), 1).otherwise(0)).alias("__promo"),
        F.sum(F.dayofweek("business_date").isin(1, 7).cast("int")).alias("__weekend"))
    lines = ctx.order_lines().where(F.col("is_completed"))
    cats = lines.groupBy("customer_id").agg(F.countDistinct("category_id").alias("distinct_categories"))
    top = (lines.groupBy("customer_id", "category_id").agg(F.sum("revenue").alias("__rev"))
           .withColumn("__r", F.row_number().over(
               Window.partitionBy("customer_id").orderBy(F.col("__rev").desc(), F.col("category_id"))))
           .where("__r = 1").select("customer_id", F.col("category_id").alias("top_category")))
    reviews = ctx.silver("reviews").groupBy("customer_id").agg(
        F.count(F.lit(1)).alias("reviews"), F.round(F.avg("rating"), 2).alias("avg_rating"))
    as_of = F.lit(ctx.cfg.calendar.end_date.isoformat()).cast("date")
    out = (ctx.silver("customers").select("customer_id", "city_id", "signup_date")
           .join(per, "customer_id", "left").join(cats, "customer_id", "left").join(top, "customer_id", "left")
           .join(reviews, "customer_id", "left")
           .fillna(0, ["orders_placed", "orders_completed", "orders_cancelled", "units", "distinct_categories",
                       "reviews", "__night", "__promo", "__weekend"]))
    return (out.withColumn("total_spend", money(F.col("total_spend")))
            .withColumn("avg_order_value", ratio(F.col("total_spend"), F.col("orders_completed"), 2))
            .withColumn("avg_items_per_order", F.round("avg_items_per_order", 2))
            .withColumn("recency_days", F.datediff(as_of, F.col("last_order_date")))
            .withColumn("tenure_days", F.datediff(as_of, F.col("signup_date")))
            .withColumn("cancellation_rate", ratio(F.col("orders_cancelled"), F.col("orders_placed")))
            .withColumn("night_order_share", ratio(F.col("__night"), F.col("orders_placed")))
            .withColumn("weekend_order_share", ratio(F.col("__weekend"), F.col("orders_placed")))
            .withColumn("promo_order_share", ratio(F.col("__promo"), F.col("orders_completed"))))


def build_customer_category(ctx: GoldContext) -> DataFrame:
    lines = ctx.order_lines().where(F.col("is_completed"))
    per = lines.groupBy("customer_id", "category_id").agg(F.sum("quantity").alias("units"),
                                                          F.sum("revenue").alias("revenue"))
    total = per.groupBy("customer_id").agg(F.sum("revenue").alias("__total"))
    return (per.join(total, "customer_id").withColumn("revenue", money(F.col("revenue")))
            .withColumn("share_of_revenue", ratio(F.col("revenue"), F.col("__total"))))


def build_basket_pairs(ctx: GoldContext) -> DataFrame:
    baskets = ctx.order_lines().where(F.col("is_completed")).select("order_id", "product_id").distinct()
    n_baskets = baskets.select("order_id").distinct().count()
    counts = baskets.groupBy("product_id").agg(F.count(F.lit(1)).alias("baskets"))
    a, b = baskets.alias("a"), baskets.alias("b")
    pairs = (a.join(b, (F.col("a.order_id") == F.col("b.order_id")) & (F.col("a.product_id") < F.col("b.product_id")))
             .groupBy(F.col("a.product_id").alias("product_a"), F.col("b.product_id").alias("product_b"))
             .agg(F.count(F.lit(1)).alias("baskets_both"))
             .where(F.col("baskets_both") >= MIN_PAIR_BASKETS))
    ca = counts.select(F.col("product_id").alias("product_a"), F.col("baskets").alias("baskets_a"))
    cb = counts.select(F.col("product_id").alias("product_b"), F.col("baskets").alias("baskets_b"))
    names = ctx.silver("products").select("product_id", "product_name", "category_id")
    na = names.select(F.col("product_id").alias("product_a"), F.col("product_name").alias("product_a_name"),
                      F.col("category_id").alias("category_a"))
    nb = names.select(F.col("product_id").alias("product_b"), F.col("product_name").alias("product_b_name"),
                      F.col("category_id").alias("category_b"))
    total = F.lit(n_baskets)
    return (pairs.join(ca, "product_a").join(cb, "product_b").join(na, "product_a").join(nb, "product_b")
            .withColumn("total_baskets", total)
            .withColumn("support", F.round(F.col("baskets_both") / total, 6))
            .withColumn("confidence_a_to_b", F.round(F.col("baskets_both") / F.col("baskets_a"), 4))
            .withColumn("confidence_b_to_a", F.round(F.col("baskets_both") / F.col("baskets_b"), 4))
            .withColumn("lift", F.round(F.col("baskets_both") * total / (F.col("baskets_a") * F.col("baskets_b")), 3)))


CUSTOMER_360 = GoldTable(
    "gld_customer_360", ("customer_id",),
    "One row per customer with behaviour features (RFM, basket, timing, promotions, cancellations, reviews). "
    "Customers without orders are included with zeros. Input for segmentation and recommendations.",
    ("slv_customers", "slv_orders", "slv_order_items", "slv_products", "slv_order_promotions", "slv_reviews"),
    (Col("customer_id", "string", "customer"), Col("city_id", "string", "home city"),
     Col("signup_date", "date", "registration date"),
     Col("orders_placed", "int", "orders placed"), Col("orders_completed", "int", "completed orders (frequency)"),
     Col("orders_cancelled", "int", "cancelled orders"),
     Col("total_spend", "decimal", "Σ order value of completed orders (monetary)"),
     Col("avg_order_value", "double", "total_spend ÷ completed orders"),
     Col("units", "int", "units bought"), Col("avg_items_per_order", "double", "distinct products per completed order"),
     Col("distinct_categories", "int", "categories bought"), Col("top_category", "string", "category with most spend"),
     Col("first_order_date", "date", "first order (any outcome)"),
     Col("last_order_date", "date", "last completed order"),
     Col("recency_days", "int", "days from last completed order to the period end; empty if none"),
     Col("tenure_days", "int", "days from signup to the period end"),
     Col("cancellation_rate", "double", "cancelled ÷ placed"),
     Col("night_order_share", "double", "share of orders placed 21:00–01:59 IST"),
     Col("weekend_order_share", "double", "share of orders placed on Saturday/Sunday"),
     Col("promo_order_share", "double", "share of completed orders with a promotion"),
     Col("reviews", "int", "reviews written"), Col("avg_rating", "double", "average rating given")),
    build_customer_360)

CUSTOMER_CATEGORY = GoldTable(
    "gld_customer_category", ("customer_id", "category_id"),
    "Spend per customer and category (completed orders), for category-mix features and recommendations.",
    ("slv_orders", "slv_order_items", "slv_products"),
    (Col("customer_id", "string", "customer"), Col("category_id", "string", "category"),
     Col("units", "int", "units bought"), Col("revenue", "decimal", "line revenue"),
     Col("share_of_revenue", "double", "share of the customer's total spend")),
    build_customer_category)

BASKET_PAIRS = GoldTable(
    "gld_basket_pairs", ("product_a", "product_b"),
    f"Products bought together in completed orders (pairs seen in ≥ {MIN_PAIR_BASKETS} baskets), with support, "
    "confidence and lift. product_a < product_b.",
    ("slv_orders", "slv_order_items", "slv_products"),
    (Col("product_a", "string", "first product (lower id)"), Col("product_b", "string", "second product"),
     Col("product_a_name", "string", "name of product_a"), Col("product_b_name", "string", "name of product_b"),
     Col("category_a", "string", "category of product_a"), Col("category_b", "string", "category of product_b"),
     Col("baskets_both", "int", "completed orders containing both"),
     Col("baskets_a", "int", "completed orders containing product_a"),
     Col("baskets_b", "int", "completed orders containing product_b"),
     Col("total_baskets", "int", "completed orders"),
     Col("support", "double", "baskets_both ÷ total_baskets"),
     Col("confidence_a_to_b", "double", "baskets_both ÷ baskets_a"),
     Col("confidence_b_to_a", "double", "baskets_both ÷ baskets_b"),
     Col("lift", "double", "how much more often they co-occur than by chance (1 = independent)")),
    build_basket_pairs)
