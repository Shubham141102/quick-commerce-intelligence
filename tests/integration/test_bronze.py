"""Bronze ingestion on the small profile: completeness, idempotency, raw-value preservation, metadata."""

import csv
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from src.common.schemas import SOURCE_SCHEMAS  # noqa: E402
from src.common.spark_io import read_bronze  # noqa: E402
from src.generation.generate_all import generate  # noqa: E402
from src.ingestion.bronze import BronzeSourceMismatch  # noqa: E402
from src.orchestration.pipeline import run_pipeline  # noqa: E402
from src.orchestration.tracking import read_meta  # noqa: E402


@pytest.fixture(scope="module")
def env(pipeline_env):
    return pipeline_env


@pytest.fixture(scope="module")
def manifest(small_run):
    return pd.DataFrame(small_run.datasets).set_index("dataset")


def test_every_bronze_table_matches_the_manifest(env, spark, manifest):
    for ds in SOURCE_SCHEMAS:
        assert read_bronze(spark, env["bronze"], ds).count() == manifest.loc[ds, "final_rows"], ds


def test_every_landing_file_loaded_once(env, small_run):
    loads = read_meta(env["metadata"], "meta_file_loads")
    loaded = loads[loads["status"] == "loaded"]
    assert loaded["file"].nunique() == len(small_run.files)
    assert not loaded["file"].duplicated().any()


def test_rerun_is_idempotent(env, spark, small_run, manifest):
    again = run_pipeline(["ingest"], small_run.run_id, env["gen_root"], overrides=env["overrides"], spark=spark)
    assert again.results["ingest"]["rows_batch"] == 0
    assert again.results["ingest"]["rows_stream"] == 0
    loads = read_meta(env["metadata"], "meta_file_loads")
    assert (loads[loads["run_id"] == again.run_id]["status"] == "skipped").sum() == len(small_run.files)
    assert read_bronze(spark, env["bronze"], "orders").count() == manifest.loc["orders", "final_rows"]


def test_malformed_log_rows_are_kept_as_corrupt(env, spark, small_run):
    malf = sum(1 for r in small_run.injected["application_logs"].issue_rows if r["issue_code"] == "MALF")
    logs = read_bronze(spark, env["bronze"], "application_logs")
    assert logs.filter(F.col("_corrupt_record").isNotNull()).count() == malf > 0


@pytest.mark.parametrize("dataset", ["orders", "customers", "reviews"])
def test_source_values_are_preserved_exactly(env, spark, small_run, dataset):
    """Whitespace, casing, quotes and commas must survive Bronze unchanged."""
    cols = SOURCE_SCHEMAS[dataset].names
    expected = Counter()
    for f in small_run.files:
        if f["dataset"] == dataset:
            with open(Path(small_run.out_dir) / f["file"], encoding="utf-8", newline="") as fh:
                rows = list(csv.reader(fh))[1:]
            expected.update(tuple(r) for r in rows)
    bronze = read_bronze(spark, env["bronze"], dataset).select(*cols).toPandas().fillna("")
    actual = Counter(map(tuple, bronze.itertuples(index=False, name=None)))
    assert actual == expected
    padded = sum(1 for row in expected for v in row if v != v.strip())
    if dataset in ("orders", "customers"):
        assert padded > 0  # the fixture really contains whitespace-dirty values


def test_metadata_columns_are_filled(env, spark):
    orders = read_bronze(spark, env["bronze"], "orders")
    for col in ["_batch_id", "_pipeline_run_id", "_source_file", "_load_type", "_ingestion_ts", "_record_hash"]:
        assert orders.filter(F.col(col).isNull()).count() == 0, col
    load_types = {r[0] for r in orders.select("_load_type").distinct().collect()}
    assert load_types == {"historical", "batch", "stream"}


def test_stream_runs_in_micro_batches_and_flags_late_rows(env, spark):
    orders = read_bronze(spark, env["bronze"], "orders").filter(F.col("_load_type") == "stream")
    assert orders.select("_batch_id").distinct().count() >= 2
    assert orders.filter(F.col("_is_late") == "true").count() > 0
    assert orders.filter(F.col("_beyond_watermark").isNotNull()).count() > 0


def test_run_metadata_and_lineage_recorded(env):
    runs = read_meta(env["metadata"], "meta_pipeline_runs")
    assert (runs[runs["run_id"] == env["first"].run_id]["status"] == "success").all()
    edges = read_meta(env["metadata"], "meta_lineage_edges")
    edges = edges[edges["run_id"] == env["first"].run_id]
    assert set(edges["target"]) == {f"brz_{ds}" for ds in SOURCE_SCHEMAS}


def test_other_generation_run_is_refused(env, spark, tmp_path_factory):
    other_root = tmp_path_factory.mktemp("other_generation")  # separate root: leaves the shared LATEST alone
    other = generate("small", seed=7, output_root=other_root, verbose=False)
    with pytest.raises(BronzeSourceMismatch):
        run_pipeline(["ingest"], other.run_id, other_root, overrides=env["overrides"], spark=spark)
