"""Marketing analytics on hand-made data: segment names, basket rules, recommendation metrics, retention status."""

import pandas as pd
import pytest

pytest.importorskip("sklearn")

from src.ml import recommendations, segmentation  # noqa: E402
from src.ml.marketing import basket_rules, retention  # noqa: E402


def test_segment_label_uses_distinctive_traits_and_top_category():
    z = pd.Series({"night_order_share": 1.4, "promo_order_share": 0.7, "log_spend": 0.2, "recency_days": -0.3})
    assert segmentation._label(z, "Snacks") == "Night-time · Deal-seeking — Snacks"
    assert segmentation._label(pd.Series({"log_spend": 0.1, "recency_days": 0.0}), "Dairy") == "Everyday — Dairy"
    assert segmentation._label(pd.Series({"recency_days": 1.2, "log_orders": 0.9}), "Dairy").startswith("Lapsing")


def test_basket_rules_keep_both_directions_above_thresholds():
    pairs = pd.DataFrame({
        "product_a": ["P1", "P3"], "product_b": ["P2", "P4"], "product_a_name": ["Chips", "Soap"],
        "product_b_name": ["Dip", "Rice"], "category_a": ["C1", "C2"], "category_b": ["C1", "C3"],
        "baskets_both": [50, 5], "support": [0.05, 0.005], "confidence_a_to_b": [0.5, 0.02],
        "confidence_b_to_a": [0.25, 0.03], "lift": [5.0, 1.1]})
    rules = basket_rules(pairs)
    assert set(zip(rules["antecedent_id"], rules["consequent_id"])) == {("P1", "P2"), ("P2", "P1")}
    assert rules.set_index("antecedent_id").loc["P2", "confidence"] == 0.25


def test_recommendation_metrics_by_hand():
    recs = pd.DataFrame({"customer_id": ["A"] * 10 + ["B"] * 10, "product_id": [f"P{i}" for i in range(10)] * 2,
                         "rank": list(range(1, 11)) * 2})
    actual = pd.DataFrame({"customer_id": ["A", "A", "B", "B"], "product_id": ["P0", "P1", "X1", "X2"]})
    m = recommendations.evaluate(recs, actual, catalog_size=20)
    assert m["precision_at_10"] == 0.1          # (2/10 + 0/10) / 2
    assert m["recall_at_10"] == 0.5             # (2/2 + 0/2) / 2
    assert m["hit_rate"] == 0.5 and m["coverage"] == 0.5 and m["customers"] == 2


def test_hybrid_puts_repeat_purchases_first():
    lines = pd.DataFrame({"order_id": ["o1", "o1", "o2", "o3", "o3"], "customer_id": ["A", "A", "A", "B", "B"],
                          "product_id": ["milk", "bread", "milk", "bread", "jam"]})
    recs = recommendations.recommend(recommendations.fit(lines), ["A", "Z"], "hybrid")
    a = recs[recs["customer_id"] == "A"].sort_values("rank")
    assert a.iloc[0]["product_id"] == "milk" and a.iloc[0]["reason"] == "you buy this often"
    assert "jam" in set(a["product_id"])                       # bought with bread by B
    assert set(recs[recs["customer_id"] == "Z"]["reason"]) == {"popular"}   # no history -> popular list


def test_retention_status_and_cohorts():
    orders = pd.DataFrame({
        "order_id": ["1", "2", "3", "4", "5"], "customer_id": ["A", "A", "B", "C", "C"],
        "business_date": ["2025-04-05", "2025-09-25", "2025-04-10", "2025-05-01", "2025-07-15"],
        "total_amount": ["100", "200", "50", "80", "90"], "is_completed": ["true", "true", "true", "true", "false"]})
    customers = pd.DataFrame({"customer_id": ["A", "B", "C", "D"]})
    cohorts, status = retention(orders, customers, pd.Timestamp("2025-09-30"))
    s = status.set_index("customer_id")
    assert s.loc["A", "status"] == "active" and s.loc["B", "status"] == "lapsed" and s.loc["D", "status"] == "never_ordered"
    assert s.loc["C", "orders_completed"] == 1            # the cancelled order does not count
    april = cohorts[cohorts["cohort_month"] == pd.Timestamp("2025-04-01")].set_index("month_offset")
    assert april.loc[0, "cohort_size"] == 2 and april.loc[5, "active_customers"] == 1 and april.loc[5, "retention_rate"] == 0.5
