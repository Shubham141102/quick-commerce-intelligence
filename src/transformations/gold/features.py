"""gld_demand_features: forecasting features per store × category × day (Project_Plan_v2.md §7.3).

Leakage rule: every feature for day t uses only data up to t − 1 (lags and rolling windows end at the
previous row of a zero-filled, gap-free daily series). Calendar, weather, holiday and promotion columns
describe day t itself; they are known in advance (weather as a forecast) and do not reveal the target.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.transformations.gold.base import Col, GoldContext, GoldTable

SERIES = ["store_id", "category_id"]


def build_demand_features(ctx: GoldContext) -> DataFrame:
    sales = ctx.built["gld_daily_category_sales"].select(*SERIES, "business_date", "units", "promo_active")
    w = Window.partitionBy(*SERIES).orderBy("business_date")
    past7, past14 = w.rowsBetween(-7, -1), w.rowsBetween(-14, -1)
    feats = (sales
             .withColumn("lag_1", F.lag("units", 1).over(w))
             .withColumn("lag_7", F.lag("units", 7).over(w))
             .withColumn("lag_14", F.lag("units", 14).over(w))
             .withColumn("rolling_mean_7", F.round(F.avg("units").over(past7), 3))
             .withColumn("rolling_mean_14", F.round(F.avg("units").over(past14), 3))
             .withColumn("rolling_std_7", F.round(F.stddev_samp("units").over(past7), 3))
             .withColumn("history_days", F.count("units").over(w.rowsBetween(Window.unboundedPreceding, -1))))

    weather = (ctx.silver("weather").groupBy("city_id", "business_date")
               .agg(F.round(F.sum("rainfall_mm"), 1).alias("rainfall_mm"),
                    F.round(F.avg("temperature_c"), 1).alias("temperature_c")))
    stores = ctx.silver("stores").select("store_id", "city_id")
    focus = ctx.built["gld_inventory_daily"].join(ctx.silver("products").select("product_id", "category_id"), "product_id")
    stockouts = (focus.groupBy("store_id", "category_id", "business_date")
                 .agg(F.avg(F.col("is_stockout").cast("double")).alias("__stockout_share")))
    prev_day = Window.partitionBy(*SERIES).orderBy("business_date")
    train_end = ctx.cfg.ml_split.train_end.isoformat()
    return (feats.join(stores, "store_id")
            .join(weather, ["city_id", "business_date"], "left")
            .join(ctx.calendar(), "business_date")
            .join(stockouts, [*SERIES, "business_date"], "left")
            .withColumn("stockout_share", F.coalesce("__stockout_share", F.lit(0.0)))
            .withColumn("stockout_share_lag_1", F.lag("stockout_share", 1).over(prev_day))
            .withColumn("month", F.month("business_date"))
            .withColumn("day_of_month", F.dayofmonth("business_date"))
            .withColumn("target_units", F.col("units"))
            .withColumn("split", F.when(F.col("business_date") <= F.lit(train_end).cast("date"), "train").otherwise("test")))


DEMAND_FEATURES = GoldTable(
    "gld_demand_features", ("store_id", "category_id", "business_date"),
    "Forecasting features per store × category × day. Lags and rolling statistics use only earlier days "
    "(no leakage); target_units is the value to predict. Train = up to the configured train_end, test = after.",
    ("gld_daily_category_sales", "gld_inventory_daily", "slv_weather", "slv_stores", "slv_products"),
    (Col("store_id", "string", "store"), Col("category_id", "string", "category"),
     Col("business_date", "date", "business date (IST)"), Col("city_id", "string", "city of the store"),
     Col("target_units", "int", "units sold that day (the value to predict)"),
     Col("lag_1", "int", "units 1 day earlier"), Col("lag_7", "int", "units 7 days earlier"),
     Col("lag_14", "int", "units 14 days earlier"),
     Col("rolling_mean_7", "double", "mean units over the previous 7 days"),
     Col("rolling_mean_14", "double", "mean units over the previous 14 days"),
     Col("rolling_std_7", "double", "standard deviation of units over the previous 7 days"),
     Col("history_days", "int", "earlier days available (features are partial while < 14)"),
     Col("day_of_week", "int", "1 = Monday … 7 = Sunday"), Col("is_weekend", "boolean", "Saturday or Sunday"),
     Col("is_holiday", "boolean", "date is in the configured holiday list"),
     Col("holiday_name", "string", "holiday name"),
     Col("month", "int", "month number"), Col("day_of_month", "int", "day of month"),
     Col("promo_active", "boolean", "a promotion for the category is active that day"),
     Col("rainfall_mm", "double", "total rainfall that day in the store's city"),
     Col("temperature_c", "double", "mean temperature that day in the store's city"),
     Col("stockout_share", "double", "share of the category's focus SKUs in the store that stocked out that day "
         "(sales understate demand when > 0). Same-day information: use it to weight or filter training rows, "
         "NOT as a model input"),
     Col("stockout_share_lag_1", "double", "stockout_share 1 day earlier (safe to use as a model input)"),
     Col("split", "string", "train / test")),
    build_demand_features)
