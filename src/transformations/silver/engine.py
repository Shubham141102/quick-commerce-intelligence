"""Generic Silver engine: Bronze rows of one dataset -> valid typed rows + quarantined rows + metrics.

Steps (Project_Plan_v2.md §6, agreed Silver plan):
 1. separate rows Spark could not parse (`_corrupt_record`)
 2. standardise strings (trim, casing, majority spelling, email)
 3. parse & cast to logical types (value present but unparseable -> type_invalid)
 4. remove duplicates per business key (latest ingestion, then latest event time, then record hash)
 5. validate: required, type, allowed values, ranges, sequences, business rules, foreign keys
 6. split: hard failures -> quarantine (direct, or cascade when only the parent was rejected);
    soft failures -> `dq_*` flags on the kept row
 7. derive extra columns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from operator import and_
from pathlib import Path

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.common.schemas import CORRUPT_RECORD, EVENT_TIME_COLUMNS, SOURCE_SCHEMAS
from src.common.spark_io import read_bronze, read_typed_csv
from src.quality.rules import primary_rule
from src.quality.standardize import (
    EMAIL_REGEX,
    SPARK_TYPES,
    cast_value,
    clean_string,
    is_non_iso_timestamp,
    majority_spelling,
)
from src.transformations.silver.specs import CITY, FK, SPECS, SilverSpec

LINEAGE = ("_source_file", "_bronze_batch_id", "_record_hash", "_silver_run_id")


@dataclass
class SilverContext:
    spark: SparkSession
    run_id: str
    bronze_root: Path
    cities: tuple[str, ...]
    accepted: dict[str, DataFrame] = field(default_factory=dict)      # valid Silver rows (typed)
    rejected: dict[str, DataFrame] = field(default_factory=dict)      # quarantined rows (standardised strings)
    deduped: dict[str, DataFrame] = field(default_factory=dict)       # unique rows before validation (strings)
    _order_totals: DataFrame | None = None

    def order_totals(self) -> DataFrame:
        """Per accepted order: total recomputed from catalog prices - discount, when every child row is valid.

        `computed_total` is null ("cannot verify") if any of the order's items or promotions was rejected.
        `trusted_total` = computed_total when available, else the order's own total_amount.
        """
        if self._order_totals is None:
            prices = self.accepted["products"].select("product_id", "price")
            gross = (self.accepted["order_items"].join(F.broadcast(prices), "product_id")
                     .groupBy("order_id").agg(F.sum(F.round(F.col("quantity") * F.col("price"), 2)).alias("__gross")))
            disc = self.accepted["order_promotions"].groupBy("order_id").agg(F.sum("discount_amount").alias("__disc"))
            bad = (self.rejected["order_items"].select("order_id")
                   .union(self.rejected["order_promotions"].select("order_id"))
                   .where(F.col("order_id").isNotNull()).distinct().withColumn("__unverifiable", F.lit(True)))
            computed = F.when(F.col("__unverifiable").isNull() & F.col("__gross").isNotNull(),
                              F.round(F.col("__gross") - F.coalesce(F.col("__disc"), F.lit(0)), 2))
            self._order_totals = (self.accepted["orders"].select("order_id", "total_amount")
                                  .join(gross, "order_id", "left").join(disc, "order_id", "left")
                                  .join(F.broadcast(bad), "order_id", "left")
                                  .select("order_id", computed.cast("decimal(12,2)").alias("computed_total"),
                                          F.coalesce(computed, F.col("total_amount")).cast("decimal(12,2)").alias("trusted_total"))
                                  .localCheckpoint())
        return self._order_totals


@dataclass
class DatasetResult:
    valid: DataFrame
    quarantine: DataFrame
    metrics: dict
    rule_counts: list[tuple[str, int]]
    standardized: list[tuple[str, int]]


def output_columns(spec: SilverSpec) -> list[str]:
    cols = SOURCE_SCHEMAS[spec.dataset].names + [c for c, _ in spec.derived]
    return cols + [r.flag_column for r in spec.soft_rules()] + list(LINEAGE)


def silver_schema(dataset: str) -> T.StructType:
    """Typed schema of slv_<dataset>, in written column order."""
    spec = SPECS[dataset]
    types = {c.name: c.dtype for c in SOURCE_SCHEMAS[dataset].columns}
    types.update(dict(spec.derived))
    types.update({r.flag_column: "boolean" for r in spec.soft_rules()})
    return T.StructType([T.StructField(c, SPARK_TYPES[types.get(c, "string")], True) for c in output_columns(spec)])


def read_silver(spark: SparkSession, silver_root: Path, dataset: str) -> DataFrame:
    return read_typed_csv(spark, silver_root / f"slv_{dataset}", silver_schema(dataset))


def _standardize(df: DataFrame, spec: SilverSpec, cols: list[str]) -> DataFrame:
    for c in cols:
        df = df.withColumn(c, clean_string(F.col(c)))
    for c in spec.lower:
        df = df.withColumn(c, F.lower(c))
    for c in spec.upper:
        df = df.withColumn(c, F.upper(c))
    for c in spec.initcap:
        df = df.withColumn(c, F.initcap(c))
    for c in spec.majority:
        df = majority_spelling(df, c)
    for c in spec.email:
        lowered = F.lower(F.col(c))
        invalid = lowered.isNotNull() & ~lowered.rlike(EMAIL_REGEX)
        df = df.withColumn("dq_email_invalid", invalid).withColumn(c, F.when(invalid, F.lit(None)).otherwise(lowered))
    return df


def _add_fk(df: DataFrame, fk: FK, ctx: SilverContext) -> DataFrame:
    accepted = ctx.accepted[fk.parent].select(F.col(fk.parent_column).alias("__pk")).distinct()
    rejected = (ctx.rejected[fk.parent].select(F.col(fk.parent_column).alias("__pk"))
                .where(F.col("__pk").isNotNull()).distinct().join(accepted, "__pk", "left_anti"))
    ok, rej = f"__fk_ok_{fk.column}", f"__fk_rej_{fk.column}"
    df = df.join(F.broadcast(accepted.withColumn(ok, F.lit(True))), df[fk.column] == F.col("__pk"), "left").drop("__pk")
    return df.join(F.broadcast(rejected.withColumn(rej, F.lit(True))), df[fk.column] == F.col("__pk"), "left").drop("__pk")


def build_dataset(spec: SilverSpec, ctx: SilverContext) -> DatasetResult:
    ds = spec.dataset
    schema = SOURCE_SCHEMAS[ds]
    cols, keys = schema.names, list(schema.business_key)
    bronze = read_bronze(ctx.spark, ctx.bronze_root, ds).cache()
    n_bronze = bronze.count()

    # 1. rows Spark could not parse
    corrupt = bronze.where(F.col(CORRUPT_RECORD).isNotNull())
    df = bronze.where(F.col(CORRUPT_RECORD).isNull())
    df = df.select(*cols, *[F.col(c).alias(f"__raw_{c}") for c in cols],
                   "_source_file", F.col("_batch_id").alias("_bronze_batch_id"), "_record_hash", "_ingestion_ts")

    # 2-3. standardise, then cast
    df = _standardize(df, spec, cols)
    for c in cols:
        df = df.withColumn(f"__t_{c}", cast_value(F.col(c), schema.column(c).dtype))

    # 4. duplicates: one row per business key (rows without a full key go on to validation)
    key_ok = reduce(and_, [F.col(f"__t_{k}").isNotNull() for k in keys])
    order: list[Column] = [F.col("_ingestion_ts").desc()]
    if ds in EVENT_TIME_COLUMNS:
        order.append(F.col(f"__t_{EVENT_TIME_COLUMNS[ds]}").desc_nulls_last())
    order.append(F.col("_record_hash").desc())
    window = Window.partitionBy(*[F.col(f"__t_{k}") for k in keys]).orderBy(*order)
    df = df.withColumn("__rn", F.when(key_ok, F.row_number().over(window))).localCheckpoint()
    n_duplicates = df.where(F.col("__rn") > 1).count()
    df = df.where(F.col("__rn").isNull() | (F.col("__rn") == 1)).drop("__rn")

    standardized = []
    for c in cols:
        changed = F.coalesce(F.col(f"__raw_{c}"), F.lit("")) != F.coalesce(F.col(c), F.lit(""))
        if schema.column(c).dtype == "timestamp":
            changed = changed | (is_non_iso_timestamp(F.col(c)) & F.col(f"__t_{c}").isNotNull())
        standardized.append((c, changed))
    std_counts = df.select(*[F.sum(ch.cast("int")).alias(c) for c, ch in standardized]).first().asDict()

    # localCheckpoint() materialises and cuts the query plan: later tables join these, and without
    # truncation every child's plan would carry its parents' whole history (planning time / memory blow up)
    ctx.deduped[ds] = df.select(*cols).localCheckpoint()

    # typed columns take the source names; standardised strings stay as __s_<col>
    for c in cols:
        df = df.withColumnRenamed(c, f"__s_{c}").withColumnRenamed(f"__t_{c}", c)
    if spec.lookups:
        df = spec.lookups(df, ctx)
    for fk in spec.fks:
        df = _add_fk(df, fk, ctx)

    # 5. validation
    checks: list[Column] = []
    for col in schema.columns:
        if not col.nullable:
            checks.append(F.when(F.col(f"__s_{col.name}").isNull(), F.lit(f"required_missing:{col.name}")))
        if col.dtype != "string":
            checks.append(F.when(F.col(f"__s_{col.name}").isNotNull() & F.col(col.name).isNull(),
                                 F.lit(f"type_invalid:{col.name}")))
    for c, allowed in spec.enums.items():
        values = list(ctx.cities) if allowed == (CITY,) else list(allowed)
        checks.append(F.when(F.col(c).isNotNull() & ~F.col(c).isin(values), F.lit(f"enum_invalid:{c}")))
    for rule in spec.rules:
        if rule.hard:
            checks.append(F.when(rule.fails(), F.lit(rule.name)))
    for fk in spec.fks:
        ok, rej = F.col(f"__fk_ok_{fk.column}").isNotNull(), F.col(f"__fk_rej_{fk.column}").isNotNull()
        present = F.col(fk.column).isNotNull()
        checks.append(F.when(present & ~ok & ~rej, F.lit(f"fk_missing:{fk.column}")))
        checks.append(F.when(present & ~ok & rej, F.lit(f"parent_rejected:{fk.parent}")))
    df = df.withColumn("__failures", F.array_compact(F.array(*checks)))
    for rule in spec.soft_rules():
        flag = F.coalesce(rule.fails(), F.lit(False)) if rule.unknown_is_pass else rule.fails()
        df = df.withColumn(rule.flag_column, flag)
    df = df.withColumn("_silver_run_id", F.lit(ctx.run_id)).localCheckpoint()

    # 6. split
    rejected = df.where(F.size("__failures") > 0)
    cascade_only = F.forall("__failures", lambda x: x.startswith("parent_rejected"))
    rejected = rejected.withColumn("rejection_type", F.when(cascade_only, "cascade").otherwise("direct"))
    ctx.rejected[ds] = rejected.select(*[F.col(f"__s_{c}").alias(c) for c in cols], "rejection_type").localCheckpoint()

    valid = df.where(F.size("__failures") == 0)
    # 7. derived columns (columns filled by later post-processing start as typed nulls)
    if spec.derive:
        valid = spec.derive(valid, ctx)
    for name, dtype in spec.derived:
        if name not in valid.columns:
            valid = valid.withColumn(name, F.lit(None).cast(SPARK_TYPES[dtype]))
    valid = valid.select(*output_columns(spec)).localCheckpoint()

    quarantine = _quarantine_rows(ds, keys, cols, rejected, corrupt, ctx.run_id).localCheckpoint()
    rule_counts = (rejected.select(F.explode("__failures").alias("rule")).groupBy("rule").count()
                   .orderBy("rule").collect())
    n_valid = valid.count()
    split = {r["rejection_type"]: r["count"] for r in rejected.groupBy("rejection_type").count().collect()}
    flags = {r.flag_column: valid.where(F.col(r.flag_column) == F.lit(True)).count() for r in spec.soft_rules()}
    if spec.email:
        flags["dq_email_invalid"] = valid.where(F.col("dq_email_invalid")).count()
    n_corrupt = corrupt.count()
    bronze.unpersist()
    metrics = {
        "rows_bronze": n_bronze, "rows_corrupt": n_corrupt, "rows_duplicates": n_duplicates,
        "rows_rejected_direct": split.get("direct", 0) + n_corrupt, "rows_rejected_cascade": split.get("cascade", 0),
        "rows_valid": n_valid, "rows_flagged": flags,
    }
    return DatasetResult(valid, quarantine, metrics, [(r["rule"], r["count"]) for r in rule_counts],
                         [(c, int(std_counts[c] or 0)) for c in cols])


def _quarantine_rows(ds: str, keys: list[str], cols: list[str], rejected: DataFrame, corrupt: DataFrame,
                     run_id: str) -> DataFrame:
    def record_id(prefix: str) -> Column:
        return F.concat_ws(";", *[F.concat(F.lit(f"{k}="), F.coalesce(F.col(f"{prefix}{k}"), F.lit(""))) for k in keys])

    common = ["source_dataset", "source_record_id", "failed_rules", "primary_rule", "severity", "rejection_type",
              "raw_record", "bronze_batch_id", "source_file", "silver_run_id", "rejected_at"]
    now = F.date_format(F.current_timestamp(), "yyyy-MM-dd'T'HH:mm:ss'Z'")
    direct = rejected.select(
        F.lit(ds).alias("source_dataset"), record_id("__s_").alias("source_record_id"),
        F.concat_ws("|", "__failures").alias("failed_rules"), primary_rule(F.col("__failures")).alias("primary_rule"),
        F.when(F.col("rejection_type") == "cascade", "warning").otherwise("error").alias("severity"),
        "rejection_type",
        F.to_json(F.struct(*[F.col(f"__raw_{c}").alias(c) for c in cols])).alias("raw_record"),
        F.col("_bronze_batch_id").alias("bronze_batch_id"), F.col("_source_file").alias("source_file"),
        F.lit(run_id).alias("silver_run_id"), now.alias("rejected_at"))
    broken = corrupt.select(
        F.lit(ds).alias("source_dataset"), record_id("").alias("source_record_id"),
        F.lit("malformed:row").alias("failed_rules"), F.lit("malformed:row").alias("primary_rule"),
        F.lit("error").alias("severity"), F.lit("direct").alias("rejection_type"),
        F.to_json(F.struct(*cols, CORRUPT_RECORD)).alias("raw_record"),
        F.col("_batch_id").alias("bronze_batch_id"), F.col("_source_file").alias("source_file"),
        F.lit(run_id).alias("silver_run_id"), now.alias("rejected_at"))
    return direct.select(*common).unionByName(broken.select(*common))
