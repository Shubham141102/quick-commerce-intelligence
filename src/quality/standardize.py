"""Standardisation and type parsing for Silver (Project_Plan_v2.md §6.2).

All functions return Spark Columns/DataFrames; nothing is collected to the driver.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

ISO_PATTERN = "yyyy-MM-dd'T'HH:mm:ssX"
# Every timestamp format seen in the sources; all are UTC. Tried in order.
TIMESTAMP_FORMATS = (ISO_PATTERN, "yyyy-MM-dd'T'HH:mm:ssXXX", "dd/MM/yyyy HH:mm:ss",
                     "yyyy-MM-dd HH:mm:ss", "yyyy/MM/dd HH:mm:ss")
ISO_REGEX = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
EMAIL_REGEX = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
BUSINESS_TZ = "Asia/Kolkata"

SPARK_TYPES = {
    "string": T.StringType(), "int": T.IntegerType(), "decimal": T.DecimalType(12, 2),
    "double": T.DoubleType(), "timestamp": T.TimestampType(), "date": T.DateType(),
    "boolean": T.BooleanType(),
}


def clean_string(c: Column) -> Column:
    """Trim; an empty or blank value becomes null."""
    trimmed = F.trim(c)
    return F.when(trimmed == "", F.lit(None)).otherwise(trimmed)


def parse_timestamp(c: Column) -> Column:
    return F.coalesce(*[F.to_timestamp(c, fmt) for fmt in TIMESTAMP_FORMATS])


def cast_value(c: Column, dtype: str) -> Column:
    """Cast a standardised string; values that don't parse become null (detected as type_invalid)."""
    if dtype == "timestamp":
        return parse_timestamp(c)
    if dtype == "date":
        return F.to_date(c, "yyyy-MM-dd")
    if dtype == "int":
        # reject "2.5"-style values instead of silently truncating them
        return F.when(c.rlike(r"^[+-]?\d+$"), c.cast("int"))
    return c.cast(SPARK_TYPES[dtype])


def is_non_iso_timestamp(c: Column) -> Column:
    return c.isNotNull() & ~c.rlike(ISO_REGEX)


def business_date(ts_column: str) -> Column:
    return F.to_date(F.from_utc_timestamp(F.col(ts_column), BUSINESS_TZ))


def majority_spelling(df: DataFrame, column: str) -> DataFrame:
    """Replace each value by the most frequent spelling among values equal ignoring case
    (ties: alphabetically first). Used where the right casing can't be derived by rule, e.g. brands."""
    key = f"__maj_key_{column}"
    counts = (df.where(F.col(column).isNotNull())
              .groupBy(F.lower(F.col(column)).alias(key), F.col(column).alias("__spelling"))
              .count())
    w = Window.partitionBy(key).orderBy(F.col("count").desc(), F.col("__spelling").asc())
    canonical = (counts.withColumn("__rank", F.row_number().over(w)).where("__rank = 1")
                 .select(key, F.col("__spelling").alias(f"__canon_{column}")))
    out = df.withColumn(key, F.lower(F.col(column))).join(F.broadcast(canonical), key, "left")
    return out.withColumn(column, F.coalesce(F.col(f"__canon_{column}"), F.col(column))).drop(key, f"__canon_{column}")
