"""Spark side of table I/O (Project_Plan_v2.md §2.1). The pandas side is in `io.py`.

Rules enforced here:
- every read uses an explicit schema (never inferSchema);
- CSV writes keep leading/trailing whitespace (Spark trims it by default, which would
  silently "fix" dirty values);
- writes go to a staging folder, are counted back, then moved into place with a rename.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.common.schemas import CORRUPT_RECORD, SOURCE_SCHEMAS, bronze_columns

CSV_READ_OPTIONS = {
    "header": "true", "quote": '"', "escape": '"', "multiLine": "false", "encoding": "UTF-8",
    "mode": "PERMISSIVE", "ignoreLeadingWhiteSpace": "false", "ignoreTrailingWhiteSpace": "false",
}
CSV_WRITE_OPTIONS = {
    "header": "true", "quote": '"', "escape": '"', "encoding": "UTF-8", "lineSep": "\n", "nullValue": "",
    "ignoreLeadingWhiteSpace": "false", "ignoreTrailingWhiteSpace": "false",
    "timestampFormat": "yyyy-MM-dd'T'HH:mm:ss'Z'", "dateFormat": "yyyy-MM-dd",   # typed layers (session tz = UTC)
}
TYPED_READ_OPTIONS = {**CSV_READ_OPTIONS, "timestampFormat": "yyyy-MM-dd'T'HH:mm:ssX", "dateFormat": "yyyy-MM-dd"}
# When reading our own tables, `_corrupt_record` is an ordinary stored column; point Spark's
# corrupt-row capture at a name that isn't in the schema so it never overwrites it.
_UNUSED_CORRUPT_COLUMN = "__spark_corrupt_unused"


def string_schema(columns: list[str]) -> T.StructType:
    return T.StructType([T.StructField(c, T.StringType(), True) for c in columns])


def read_source_csv(spark: SparkSession, paths: list[Path], dataset: str) -> DataFrame:
    """Read landing CSVs: every source column as string, unparseable rows kept in `_corrupt_record`."""
    schema = string_schema(SOURCE_SCHEMAS[dataset].names + [CORRUPT_RECORD])
    return (spark.read.schema(schema).options(**CSV_READ_OPTIONS)
            .option("columnNameOfCorruptRecord", CORRUPT_RECORD)
            .csv([p.as_posix() for p in paths]))


def source_stream(spark: SparkSession, directory: Path, dataset: str, max_files_per_trigger: int) -> DataFrame:
    schema = string_schema(SOURCE_SCHEMAS[dataset].names + [CORRUPT_RECORD])
    return (spark.readStream.schema(schema).options(**CSV_READ_OPTIONS)
            .option("columnNameOfCorruptRecord", CORRUPT_RECORD)
            .option("maxFilesPerTrigger", max_files_per_trigger)
            .csv(directory.as_posix()))


def read_table_csv(spark: SparkSession, path: Path, columns: list[str]) -> DataFrame:
    """Read one of our own CSV tables (all files under `path`, recursively) with an all-string schema."""
    return (spark.read.schema(string_schema(columns)).options(**CSV_READ_OPTIONS)
            .option("columnNameOfCorruptRecord", _UNUSED_CORRUPT_COLUMN)
            .option("recursiveFileLookup", "true").option("pathGlobFilter", "*.csv")
            .csv(path.as_posix()))


def read_typed_csv(spark: SparkSession, path: Path, schema: T.StructType) -> DataFrame:
    """Read one of our typed CSV tables (Silver, Gold) with an explicit typed schema."""
    return (spark.read.schema(schema).options(**TYPED_READ_OPTIONS)
            .option("columnNameOfCorruptRecord", _UNUSED_CORRUPT_COLUMN)
            .option("recursiveFileLookup", "true").option("pathGlobFilter", "*.csv")
            .csv(path.as_posix()))


def read_bronze(spark: SparkSession, bronze_root: Path, dataset: str) -> DataFrame:
    return read_table_csv(spark, bronze_root / f"brz_{dataset}", bronze_columns(dataset))


def write_csv_atomic(df: DataFrame, final_dir: Path, staging_dir: Path, single_file: bool = True,
                     replace: bool = False) -> int:
    """Write `df` to staging, verify the row count by reading it back, then rename into `final_dir`.

    With `replace=False` (append-only, as Bronze does) an existing `final_dir` is an error.
    With `replace=True` the old folder is kept as a sibling `_previous/<name>` (one generation back).
    Returns the number of rows written.
    """
    if final_dir.exists() and not replace:
        raise FileExistsError(f"{final_dir} already exists (append-only target)")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.parent.mkdir(parents=True, exist_ok=True)
    out = df.coalesce(1) if single_file else df
    out.write.mode("overwrite").options(**CSV_WRITE_OPTIONS).csv(staging_dir.as_posix())
    for crc in staging_dir.glob(".*.crc"):
        crc.unlink()
    written = read_table_csv(df.sparkSession, staging_dir, df.columns).count()

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        previous = final_dir.parent / "_previous" / final_dir.name
        if previous.exists():
            shutil.rmtree(previous)
        previous.parent.mkdir(parents=True, exist_ok=True)
        os.replace(final_dir, previous)
    os.replace(staging_dir, final_dir)
    return written


def relative_source_file(raw: Column) -> Column:
    """`file:///C:/.../gen_x/landing/batch/orders/f.csv` -> `gen_x/landing/batch/orders/f.csv`."""
    rel = F.regexp_extract(raw, r"(gen_[^/]+/landing/.+)$", 1)
    return F.when(rel != "", rel).otherwise(raw)
