"""Stockout-risk tiers and replenishment on a hand-made example (no Spark, no files)."""

import math

import pandas as pd
import pytest

pytest.importorskip("sklearn")

from src.ml.stockout.run import LEAD_TIME_DAYS, Z, build_risk  # noqa: E402

DAY = pd.Timestamp("2025-09-10")


def _case(stock: int, reorder: int, daily_forecast: float, doi: float | None):
    days = pd.date_range(DAY - pd.Timedelta(days=27), DAY)
    inventory = pd.DataFrame({"store_id": "S01", "product_id": "P1", "business_date": days,
                              "units_sold": [2, 4] * 14, "closing_stock": stock, "reorder_level": reorder,
                              "days_of_inventory": doi})
    sku = pd.DataFrame({"store_id": "S01", "product_id": "P1", "category_id": "CAT01", "origin_date": DAY,
                        "forecast_date": [DAY + pd.Timedelta(days=h) for h in range(1, 8)], "horizon": range(1, 8),
                        "forecast_units": daily_forecast, "upper_units": daily_forecast * 2,
                        "run_type": "backtest", "model_version": "v"})
    risk, repl = build_risk(inventory, sku)
    return risk.iloc[0], repl.iloc[0]


def test_zero_stock_is_high():
    risk, _ = _case(stock=0, reorder=3, daily_forecast=1.0, doi=0.0)
    assert risk["risk_tier"] == "High" and "ZERO_STOCK" in risk["reason_codes"]


def test_demand_over_lead_time_exceeding_stock_is_high():
    risk, _ = _case(stock=5, reorder=2, daily_forecast=2.0, doi=4.0)   # 3 days × 2 = 6 ≥ 5
    assert risk["risk_tier"] == "High" and risk["reason_codes"] == "DEMAND_EXCEEDS_STOCK"


def test_below_reorder_level_is_medium():
    risk, _ = _case(stock=4, reorder=5, daily_forecast=0.5, doi=8.0)
    assert risk["risk_tier"] == "Medium" and risk["reason_codes"] == "BELOW_REORDER"


def test_plenty_of_stock_is_low_and_orders_nothing():
    risk, repl = _case(stock=50, reorder=5, daily_forecast=1.0, doi=25.0)
    assert risk["risk_tier"] == "Low" and risk["reason_codes"] == ""
    assert repl["suggested_qty"] == 0


def test_suggested_quantity_formula():
    _, repl = _case(stock=3, reorder=5, daily_forecast=2.0, doi=1.5)
    std = pd.Series([2, 4] * 14).std()
    expected = math.ceil(2.0 * 3 + Z * std * math.sqrt(LEAD_TIME_DAYS) - 3)
    assert repl["suggested_qty"] == expected
