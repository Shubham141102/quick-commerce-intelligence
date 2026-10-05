"""Phase 4B forecasting on the small profile (30 days): structure, no leakage, intervals, registry.
Accuracy targets are only meaningful on the medium profile, so they are not asserted here."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from scripts.check_ml import evaluate  # noqa: E402
from src.ml.forecasting.model import recursive_forecast, train  # noqa: E402
from src.ml.tables import DEMAND_PREDICTIONS, SKU_FORECAST, read_table  # noqa: E402
from src.orchestration.pipeline import run_pipeline  # noqa: E402
from src.transformations.gold.features import DEMAND_FEATURES  # noqa: E402


@pytest.fixture(scope="module")
def ml_env(gold_env, small_run):
    result = run_pipeline(["ml"], small_run.run_id, gold_env["gen_root"], overrides=gold_env["overrides"])
    return {**gold_env, "ml_result": result}


def test_structural_checks_pass(ml_env):
    checks = evaluate(ml_env["paths"], strict_accuracy=False, include_app=False)  # app checks: tests/app
    failed = [(name, detail) for name, ok, detail in checks if not ok]
    assert not failed, failed


def test_future_forecast_follows_the_data(ml_env):
    preds = read_table(ml_env["gold"], DEMAND_PREDICTIONS)
    future = preds[preds["run_type"] == "future"]
    last = preds.loc[preds["run_type"] == "backtest", "forecast_date"].max()
    assert sorted(future["horizon"].unique()) == list(range(1, 8))
    assert (future["origin_date"] == last).all()
    assert (future["forecast_units"] >= 0).all()


def test_sku_forecasts_split_the_category_forecast(ml_env):
    preds = read_table(ml_env["gold"], DEMAND_PREDICTIONS)
    sku = read_table(ml_env["gold"], SKU_FORECAST)
    merged = sku.merge(preds, on=["store_id", "category_id", "forecast_date", "origin_date"], suffixes=("", "_cat"))
    assert np.allclose(merged["forecast_units"], merged["forecast_units_cat"] * merged["sku_share"], atol=1e-3)
    assert (sku["sku_share"] >= 0).all()


def test_recursive_forecast_never_sees_the_future(ml_env):
    """Changing actual sales after the origin must not change the forecast."""
    features = read_table(ml_env["gold"], DEMAND_FEATURES)
    model = train(features, seed=1)
    units = features.pivot_table(index=["store_id", "category_id"], columns="business_date", values="target_units")
    exog = features[["store_id", "category_id", "business_date", "day_of_week", "is_weekend", "is_holiday", "month",
                     "day_of_month", "promo_active", "rainfall_mm", "temperature_c", "stockout_share_lag_1"]]
    first, last = units.columns[-7], units.columns[-1]
    base = recursive_forecast(model, units, exog, [first], 7, last)
    tampered = units.copy()
    tampered.loc[:, tampered.columns >= first] = 999.0
    again = recursive_forecast(model, tampered, exog, [first], 7, last)
    pd.testing.assert_series_equal(base["forecast_units"], again["forecast_units"])
