"""Chart summaries and tier colours on hand-made data."""

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from app.components.insights import forecast_summary, stock_summary, style_tiers  # noqa: E402


def _days(start, n):
    return pd.date_range(start, periods=n, freq="D")


def test_forecast_summary_totals_and_table():
    hist = pd.DataFrame({"day": _days("2025-09-24", 7), "actual": [4] * 7, "backtest_forecast": [5.0] * 7,
                         "future_forecast": None, "lower": None, "upper": None})
    fut = pd.DataFrame({"day": _days("2025-10-01", 7), "actual": None, "backtest_forecast": None,
                        "future_forecast": [5, 5, 5, 5, 5, 8, 2], "lower": [1] * 7, "upper": [9] * 7})
    lines, table = forecast_summary(pd.concat([hist, fut], ignore_index=True))
    text = " ".join(lines)
    assert "35 units" in text and "25% more" in text            # 35 forecast vs 28 sold
    assert "Mon 06 Oct" in text and "off by **1.0 units a day**" in text
    assert len(table) == 7 and list(table.columns)[0] == "date"


def test_stock_summary_predicts_run_out_day():
    hist = pd.DataFrame({"day": _days("2025-09-01", 30), "closing_stock": [10] * 29 + [4], "units_sold": [2] * 30,
                         "reorder_level": [5] * 30, "forecast": None})
    fut = pd.DataFrame({"day": _days("2025-10-01", 7), "closing_stock": None, "units_sold": None,
                        "reorder_level": None, "forecast": [1.5] * 7})
    risk = pd.Series({"risk_tier": "Medium", "reason_codes": "BELOW_REORDER", "suggested_qty": 6})
    text = " ".join(stock_summary(pd.concat([hist, fut], ignore_index=True), risk))
    assert "**4 units**" in text and "at or below it" in text
    assert "run out around Fri 03 Oct" in text                  # 1.5 + 1.5 + 1.5 >= 4 on the 3rd day
    assert "Risk tier: Medium" in text and "**6 units**" in text


def test_tier_colours():
    df = pd.DataFrame({"risk_tier": ["High", "Medium", "Low"], "high": [2, 0, 1]})
    html = style_tiers(df, ["risk_tier"], {"high": "High"}).to_html()
    assert "#FDE2E1" in html and "#FEF3C7" in html and "#DCFCE7" in html


def test_reliability_bands_and_headline():
    from app.components.insights import accuracy_headline, reliability
    assert [reliability(w) for w in (0.469, 0.55, 0.70, 0.701, 0.927)] == ["High", "Medium", "Medium", "Low", "Low"]
    pooled = pd.Series({"model": 0.619, "baseline_moving_avg": 0.689, "baseline_seasonal_naive": 0.866})
    by_h = pd.DataFrame({"model": [0.621, 0.617, 0.623]})
    text = accuracy_headline(pooled, by_h)
    assert "reliable for ordering" in text and "10% less error" in text and "up to 7 days ahead" in text
