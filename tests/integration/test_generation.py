"""Small-profile generation: counts, 90% clean, consistency rules C1-C11, landing files, determinism."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.schemas import SOURCE_SCHEMAS
from src.generation.generate_all import generate


@pytest.fixture(scope="module")
def clean(small_run):
    return small_run.clean


@pytest.fixture(scope="module")
def manifest(small_run):
    return pd.DataFrame(small_run.datasets).set_index("dataset")


# ---------------------------------------------------------------------------------- counts

def test_generated_rows_hit_targets(manifest):
    off = manifest[manifest["generated_rows"] != manifest["target_rows"]]
    # order_promotions / reviews may fall short only if too few eligible orders exist
    assert off.index.difference(["order_promotions", "reviews", "returns_refunds"]).empty, off


def test_every_dataset_is_about_ninety_percent_clean(manifest):
    big = manifest[manifest["final_rows"] >= 100]
    assert big["clean_pct"].between(89.5, 90.5).all(), big["clean_pct"]
    overall = manifest["clean_rows"].sum() / manifest["final_rows"].sum()
    assert overall == pytest.approx(0.90, abs=0.002)


def test_injected_issue_counts_match_plan(small_run):
    issues = pd.DataFrame(small_run.issues)
    assert (issues["planned"] == issues["injected"]).all(), issues[issues["planned"] != issues["injected"]]


# ---------------------------------------------------------------------- consistency (clean data)

def test_c1_orders_after_signup_and_mostly_same_city(clean):
    o = clean["orders"].merge(clean["customers"][["customer_id", "signup_date", "city_id"]], on="customer_id")
    o = o.merge(clean["stores"][["store_id", "city_id"]], on="store_id", suffixes=("_cust", "_store"))
    assert (pd.to_datetime(o["order_ts"]).dt.normalize() >= pd.to_datetime(o["signup_date"]) - pd.Timedelta(days=1)).all()
    assert (o["city_id_cust"] == o["city_id_store"]).mean() > 0.94


def test_c2_order_total_equals_lines_minus_discount(clean):
    gross = clean["order_items"].groupby("order_id")["line_amount"].sum()
    disc = clean["order_promotions"].set_index("order_id")["discount_amount"]
    o = clean["orders"].set_index("order_id")
    expected = (gross - disc.reindex(gross.index).fillna(0)).round(2)
    assert np.allclose(o.loc[expected.index, "total_amount"], expected, atol=0.01)


def test_c3_at_most_one_success_and_failures_first(clean):
    p = clean["payments"]
    assert p[p["status"] == "success"].groupby("order_id").size().max() == 1
    first = p.sort_values(["order_id", "attempt_no"]).groupby("order_id")
    assert (first["payment_ts"].apply(lambda s: s.is_monotonic_increasing)).all()


def test_c4_c5_cancellations_vs_deliveries(clean):
    stage = clean["cancellations"].set_index("order_id")["stage"]
    d = clean["deliveries"].set_index("order_id")
    assert not d.index.isin(stage[stage == "pre_dispatch"].index).any()
    post = d.loc[d.index.isin(stage[stage == "post_dispatch"].index), "status"]
    assert set(post) <= {"cancelled", "failed"}


def test_c6_delivery_time_order(clean):
    d = clean["deliveries"].merge(clean["orders"][["order_id", "order_ts"]], on="order_id")
    done = d[d["status"] == "delivered"]
    assert (done["order_ts"] < done["pickup_ts"]).all() and (done["pickup_ts"] < done["delivered_ts"]).all()
    assert d.loc[d["status"] != "delivered", "delivered_ts"].isna().all()


def test_c7_refunds_within_payment(clean):
    r = clean["returns_refunds"].merge(clean["payments"][["payment_id", "amount", "payment_ts"]],
                                       on="payment_id", suffixes=("", "_paid"))
    assert (r["amount"] <= r["amount_paid"] + 0.005).all()
    assert (r["refund_ts"] > r["payment_ts"]).all()


def test_c8_promotions_active_and_eligible(clean):
    op = clean["order_promotions"].merge(clean["orders"][["order_id", "order_ts"]], on="order_id")
    op = op.merge(clean["promotions"][["promotion_id", "start_date", "end_date", "category_id"]], on="promotion_id")
    day = (pd.to_datetime(op["order_ts"]) + pd.Timedelta(hours=5, minutes=30)).dt.normalize()
    assert ((day >= op["start_date"]) & (day <= op["end_date"])).all()
    items = clean["order_items"].merge(clean["products"][["product_id", "category_id"]], on="product_id")
    have = set(zip(items["order_id"], items["category_id"]))
    assert all((o, c) in have for o, c in zip(op["order_id"], op["category_id"]))


def test_c9_reviews_for_bought_products_after_delivery(clean):
    bought = set(zip(clean["order_items"]["order_id"], clean["order_items"]["product_id"]))
    rv = clean["reviews"]
    assert all((o, p) in bought for o, p in zip(rv["order_id"], rv["product_id"]))
    rv = rv.merge(clean["deliveries"][["order_id", "delivered_ts"]], on="order_id")
    assert (rv["review_ts"] > rv["delivered_ts"]).all()


def test_c10_unit_price_is_catalog_price_and_items_distinct(clean):
    it = clean["order_items"].merge(clean["products"][["product_id", "price"]], on="product_id")
    assert np.allclose(it["unit_price"], it["price"])
    assert not clean["order_items"].duplicated(["order_id", "product_id"]).any()


def test_c11_snapshot_stock_non_negative(clean):
    assert (clean["inventory_snapshots"]["stock_quantity"] >= 0).all()


# ------------------------------------------------------------------------------- landing files

def test_landing_files_contain_every_row(small_run):
    out = Path(small_run.out_dir)
    for name, res in small_run.injected.items():
        files = [f for f in small_run.files if f["dataset"] == name]
        total = 0
        for f in files:
            with open(out / f["file"], encoding="utf-8", newline="") as fh:
                rows = list(csv.reader(fh))
            assert rows[0] == SOURCE_SCHEMAS[name].names
            total += len(rows) - 1
        assert total == res.final_rows, name


def test_stream_files_only_for_stream_datasets(small_run):
    streamed = {f["dataset"] for f in small_run.files if f["slice"] == "stream"}
    assert streamed == {"orders", "order_items", "payments", "inventory_events", "application_events"}


def test_manifest_and_ground_truth_written(small_run):
    out = Path(small_run.out_dir)
    for name in ["manifest_run", "manifest_datasets", "manifest_issues", "manifest_files"]:
        assert (out / "manifest" / f"{name}.csv").exists()
    for name in ["gt_personas", "gt_affinity_pairs", "gt_anomalies", "gt_stockouts", "gt_injected_issues"]:
        assert (out / "ground_truth" / f"{name}.csv").exists()
    assert (out.parent / "LATEST").read_text().strip() == small_run.run_id


# ---------------------------------------------------------------------- determinism and signals

def test_same_seed_gives_identical_data(small_run):
    again = generate("small", write=False, verbose=False)
    for name, res in small_run.injected.items():
        pd.testing.assert_frame_equal(res.rows, again.injected[name].rows, obj=name)


def test_planted_affinity_pairs_are_detectable(small_run):
    items = small_run.clean["order_items"]
    baskets = items.groupby("order_id")["product_id"].apply(set)
    n = len(baskets)
    lifts = []
    for pair in small_run.ground_truth["gt_affinity_pairs"]:
        a, b = pair["antecedent_product_id"], pair["consequent_product_id"]
        has_a = baskets.apply(lambda s: a in s)
        has_b = baskets.apply(lambda s: b in s)
        if has_a.sum() and has_b.sum():
            lifts.append(((has_a & has_b).sum() / has_a.sum()) / (has_b.sum() / n))
    assert lifts and np.median(lifts) > 3
