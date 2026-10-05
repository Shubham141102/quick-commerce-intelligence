"""Silver on the small profile: every injected issue handled as planned, typed output, idempotent rebuilds."""

import pandas as pd
import pytest

pytest.importorskip("pyspark")

from pyspark.sql import types as T  # noqa: E402

from scripts.check_silver import evaluate  # noqa: E402
from src.common.io import read_csv_strings  # noqa: E402
from src.common.schemas import SOURCE_SCHEMAS  # noqa: E402
from src.orchestration.pipeline import run_pipeline  # noqa: E402
from src.orchestration.tracking import read_meta  # noqa: E402
from src.transformations.silver.engine import read_silver  # noqa: E402

CITIES = ("C01", "C02", "C03")


@pytest.fixture(scope="module")
def report(silver_env):
    return evaluate(silver_env["paths"], CITIES)


def test_completion_report_passes(report):
    _, _, checks = report
    failed = [(name, detail) for name, ok, detail in checks if not ok]
    assert not failed, failed


def test_every_bronze_row_accounted_for(report):
    accounting, _, _ = report
    assert (accounting["balanced"] == "yes").all(), accounting


def test_injected_issues_handled_at_full_rate(report):
    _, detection, _ = report
    measured = detection[detection["rate"] != "n/a"]
    assert (measured["rate"] == "100.0%").all(), measured[measured["rate"] != "100.0%"]


def test_children_of_rejected_orders_are_cascaded(silver_env):
    qtn = pd.concat([read_csv_strings(p) for p in (silver_env["quarantine"] / "qtn_records" / "order_items").glob("*.csv")])
    cascade = qtn[qtn["rejection_type"] == "cascade"]
    assert len(cascade) > 0
    assert cascade["failed_rules"].str.fullmatch(r"parent_rejected:orders").all()


def test_every_quarantined_row_has_a_primary_rule(silver_env):
    qtn = pd.concat([read_csv_strings(p) for p in (silver_env["quarantine"] / "qtn_records").glob("*/*.csv")])
    assert (qtn["primary_rule"] != "").all()
    first = qtn["failed_rules"].str.split("|")
    assert all(p in rules for p, rules in zip(qtn["primary_rule"], first))


def test_silver_reads_back_with_types(silver_env, spark):
    orders = read_silver(spark, silver_env["silver"], "orders")
    types = dict((f.name, f.dataType) for f in orders.schema.fields)
    assert isinstance(types["total_amount"], T.DecimalType)
    assert isinstance(types["order_ts"], T.TimestampType)
    assert isinstance(types["is_completed"], T.BooleanType)
    assert orders.where("order_ts is null or total_amount is null").count() == 0
    assert orders.where("is_completed").count() > 0


def test_rebuild_is_idempotent(silver_env, small_run, spark):
    def snapshot(ds):
        frame = pd.concat([read_csv_strings(p) for p in (silver_env["silver"] / f"slv_{ds}").glob("*.csv")])
        return frame.drop(columns="_silver_run_id").reset_index(drop=True)

    before = {ds: snapshot(ds) for ds in ["orders", "order_items", "payments", "reviews"]}
    run_pipeline(["silver"], small_run.run_id, silver_env["gen_root"], overrides=silver_env["overrides"], spark=spark)
    for ds, frame in before.items():
        pd.testing.assert_frame_equal(snapshot(ds), frame, obj=ds)


def test_quality_metrics_recorded(silver_env):
    run_id = silver_env["result"].run_id
    tables = read_meta(silver_env["metadata"], "meta_table_runs")
    assert set(tables.loc[tables["run_id"] == run_id, "target"]) == {f"slv_{d}" for d in SOURCE_SCHEMAS}
    quality = read_meta(silver_env["metadata"], "meta_quality_results")
    quality = quality[quality["run_id"] == run_id]
    assert {"duplicates", "rule", "standardize", "soft_rule"} <= set(quality["check_type"])
