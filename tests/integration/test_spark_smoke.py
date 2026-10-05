"""Spark environment check: session starts, Python UDF workers run, CSV write/read round-trips."""

import pytest

pyspark = pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql import types as T  # noqa: E402


def test_csv_round_trip_with_explicit_schema(spark, tmp_path):
    schema = T.StructType([
        T.StructField("id", T.StringType()),
        T.StructField("amount", T.DecimalType(12, 2)),
        T.StructField("ts", T.TimestampType()),
        T.StructField("note", T.StringType()),
    ])
    df = spark.createDataFrame(
        [("a", 149.0, "2025-09-01T10:00:00Z", 'Fresh, "well" packed'), ("b", 20.5, "2025-09-02T11:30:00Z", None)],
        "id string, amount double, ts string, note string",
    ).select("id", F.col("amount").cast("decimal(12,2)").alias("amount"),
             F.to_timestamp("ts", "yyyy-MM-dd'T'HH:mm:ssX").alias("ts"), "note")
    out = str(tmp_path / "t")
    df.coalesce(1).write.mode("overwrite").option("header", True).option("escape", '"') \
        .option("timestampFormat", "yyyy-MM-dd'T'HH:mm:ss'Z'").csv(out)
    back = spark.read.schema(schema).option("header", True).option("escape", '"') \
        .option("timestampFormat", "yyyy-MM-dd'T'HH:mm:ssX").csv(out)
    assert sorted(map(tuple, back.collect())) == sorted(map(tuple, df.collect()))


def test_python_workers_run(spark):
    plus_one = F.udf(lambda x: x + 1, T.IntegerType())
    assert spark.range(3).select(plus_one(F.col("id").cast("int")).alias("y")).agg(F.sum("y")).first()[0] == 6
