"""Lost-sales estimate on a hand-made SKU history (no Spark, no files)."""

import pandas as pd
import pytest

pytest.importorskip("sklearn")

from src.ml.lost_sales import estimate_lost_sales  # noqa: E402


def _history(units: list[int], stockout_days: set[int]) -> pd.DataFrame:
    days = pd.date_range("2025-09-01", periods=len(units))
    return pd.DataFrame({"store_id": "S01", "product_id": "P1", "business_date": days, "units_sold": units,
                         "is_stockout": [i in stockout_days for i in range(len(units))]})


PRODUCTS = pd.DataFrame({"product_id": ["P1"], "category_id": ["CAT01"], "price": ["50.00"]})


def test_lost_units_are_expected_minus_sold_on_stockout_days():
    lost = estimate_lost_sales(_history([4] * 20 + [1], {20}), PRODUCTS)
    assert len(lost) == 1
    row = lost.iloc[0]
    assert row["expected_units"] == 4 and row["lost_units"] == 3 and row["lost_revenue"] == 150


def test_earlier_stockout_days_do_not_drag_the_expectation_down():
    units = [4] * 12 + [0] + [4] * 10 + [0]           # two stockout days with zero sales
    lost = estimate_lost_sales(_history(units, {12, len(units) - 1}), PRODUCTS)
    assert list(lost["expected_units"]) == [4.0, 4.0]  # day 12's zero is excluded from day 23's average


def test_no_estimate_without_enough_in_stock_history():
    assert estimate_lost_sales(_history([4, 4, 0], {2}), PRODUCTS).empty


def test_never_negative_when_a_stockout_day_still_sold_a_lot():
    lost = estimate_lost_sales(_history([2] * 10 + [9], {10}), PRODUCTS)
    assert lost.iloc[0]["lost_units"] == 0
