"""Gold framework: table declarations and the shared build context (Project_Plan_v2.md §7).

Each Gold table declares its grain, columns (with types and descriptions), source tables and a build
function. The declaration is used for the written schema, the typed reader, the completion checks and
the generated catalog (docs/gold_catalog.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Callable

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.common.config import Config
from src.quality.standardize import BUSINESS_TZ, SPARK_TYPES
from src.serving.metrics import line_revenue
from src.transformations.silver.engine import read_silver


@dataclass(frozen=True)
class Col:
    name: str
    dtype: str
    description: str


@dataclass(frozen=True)
class GoldTable:
    name: str                       # e.g. "gld_daily_sales"
    grain: tuple[str, ...]          # columns that identify one row
    description: str
    sources: tuple[str, ...]        # Silver / Gold tables read
    columns: tuple[Col, ...]
    build: Callable[["GoldContext"], DataFrame] | None   # None for tables written by the Python ML stage
    tier: int = 1
    engine: str = "spark"                                  # "spark" (gold stage) or "python" (ml stage)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def typed_columns(self) -> list[tuple[str, str]]:
        return [(c.name, c.dtype) for c in self.columns]

    def schema(self) -> T.StructType:
        return T.StructType([T.StructField(c.name, SPARK_TYPES[c.dtype], True) for c in self.columns])


@dataclass
class GoldContext:
    spark: SparkSession
    cfg: Config
    silver_root: Path
    metadata_root: Path
    built: dict[str, DataFrame] = field(default_factory=dict)
    _silver: dict[str, DataFrame] = field(default_factory=dict)
    _cache: dict[str, DataFrame] = field(default_factory=dict)

    # ---------------------------------------------------------------- sources
    def silver(self, dataset: str) -> DataFrame:
        if dataset not in self._silver:
            self._silver[dataset] = read_silver(self.spark, self.silver_root, dataset).cache()
        return self._silver[dataset]

    def _memo(self, key: str, make: Callable[[], DataFrame]) -> DataFrame:
        if key not in self._cache:
            self._cache[key] = make().localCheckpoint()
        return self._cache[key]

    def calendar(self) -> DataFrame:
        """Every business date of the period with calendar attributes."""
        def make() -> DataFrame:
            start, n = self.cfg.calendar.start_date, self.cfg.calendar.n_days
            dates = [(start + timedelta(days=i),) for i in range(n)]
            holidays = {h.date: h.name for h in self.cfg.holidays}
            df = self.spark.createDataFrame(dates, "business_date date")
            names = self.spark.createDataFrame([(d, n) for d, n in holidays.items()] or [(start, None)],
                                               "business_date date, holiday_name string")
            return (df.join(names.where(F.col("holiday_name").isNotNull()), "business_date", "left")
                    .withColumn("day_of_week", ((F.dayofweek("business_date") + 5) % 7) + 1)  # Mon=1 … Sun=7
                    .withColumn("is_weekend", F.col("day_of_week") >= 6)
                    .withColumn("is_holiday", F.col("holiday_name").isNotNull()))
        return self._memo("calendar", make)

    def order_lines(self) -> DataFrame:
        """Every Silver order line with its order and product attributes and trusted line revenue."""
        def make() -> DataFrame:
            orders = self.silver("orders").select("order_id", "store_id", "customer_id", "business_date", "order_ts",
                                                  "status", "is_completed")
            products = self.silver("products").select("product_id", "category_id", F.col("price").alias("catalog_price"))
            return (self.silver("order_items").select("order_id", "product_id", "quantity", "unit_price", "line_amount",
                                                      "dq_unit_price_mismatch")
                    .join(orders, "order_id").join(products, "product_id")
                    .withColumn("revenue", line_revenue()))
        return self._memo("order_lines", make)

    def order_facts(self) -> DataFrame:
        """One row per Silver order: store, date, customer, outcome, GMV, discount, units, items."""
        def make() -> DataFrame:
            lines = (self.order_lines().groupBy("order_id")
                     .agg(F.sum("revenue").alias("gmv"), F.sum("quantity").alias("units"),
                          F.countDistinct("product_id").alias("items"), F.max("quantity").alias("max_quantity")))
            disc = self.silver("order_promotions").groupBy("order_id").agg(F.sum("discount_amount").alias("discount"))
            return (self.silver("orders")
                    .select("order_id", "store_id", "customer_id", "business_date", "order_ts", "status", "is_completed")
                    .join(lines, "order_id", "left").join(disc, "order_id", "left")
                    .fillna(0, ["units", "items", "max_quantity"])
                    .withColumn("gmv", F.coalesce("gmv", F.lit(0)).cast("decimal(12,2)"))
                    .withColumn("discount", F.coalesce("discount", F.lit(0)).cast("decimal(12,2)"))
                    .withColumn("has_promotion", F.col("discount") > 0))
        return self._memo("order_facts", make)

    def completed_refunds(self) -> DataFrame:
        """Completed refunds with the store of their order, by refund business date."""
        def make() -> DataFrame:
            stores = self.silver("orders").select("order_id", "store_id")
            return (self.silver("returns_refunds").where(F.col("status") == "completed")
                    .join(stores, "order_id").select("store_id", "business_date", "amount"))
        return self._memo("completed_refunds", make)

    def store_days(self) -> DataFrame:
        return self._memo("store_days", lambda: self.silver("stores").select("store_id", "city_id")
                          .crossJoin(self.calendar().select("business_date")))


def ist_day_start(date_col: str) -> F.Column:
    """UTC instant of 00:00 IST on a business date."""
    return F.to_utc_timestamp(F.col(date_col).cast("timestamp"), BUSINESS_TZ)
