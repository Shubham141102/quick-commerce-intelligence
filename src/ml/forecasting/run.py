"""Demand forecasting stage (Phase 4B): train, backtest on September, refit, forecast the next 7 days,
split to focus SKUs, record metrics / importance / model registry.

Backtest: a model trained on Apr–Aug forecasts 1–7 days ahead from every day of September (recursive),
next to two baselines. Production: the same recipe refit on all data forecasts the 7 days after the data
ends. Both models are saved under data/ml/models/demand_forecast/<version>/.
"""

from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.common.config import Config
from src.common.io import read_csv_strings
from src.common.paths import PROJECT_ROOT
from src.ml.evaluation import forecast_metrics
from src.ml.forecasting.model import FEATURES, SERIES, recursive_forecast, train
from src.ml.tables import (
    DEMAND_PREDICTIONS,
    FEATURE_IMPORTANCE,
    FORECAST_METRICS,
    SKU_FORECAST,
    read_table,
    write_table,
)
from src.orchestration.tracking import RunTracker, utc_now
from src.transformations.gold.features import DEMAND_FEATURES
from src.transformations.gold.inventory import INVENTORY_DAILY
from src.transformations.gold.sales import DAILY_CATEGORY_SALES

HORIZON = 7
SKU_SHARE_DAYS = 28
EXOG = ["day_of_week", "is_weekend", "is_holiday", "month", "day_of_month", "promo_active", "rainfall_mm",
        "temperature_c", "stockout_share_lag_1"]
LIMITATIONS = ("Synthetic data; ~4 units per store-category-day so daily error is high; weather for future days "
               "is assumed equal to the last 7-day average; no promotions assumed after the data ends; "
               "holiday dates are approximate and the generator planted no holiday effect; backtest model is "
               "not refit during September.")


def _project_path(path: Path) -> str:
    """Project-relative path when inside the project, absolute otherwise (e.g. test folders)."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _future_exog(features: pd.DataFrame, cfg: Config, days: list[pd.Timestamp]) -> pd.DataFrame:
    last = features["business_date"].max()
    recent = features[features["business_date"] > last - pd.Timedelta(days=7)]
    weather = recent.groupby(SERIES)[["rainfall_mm", "temperature_c"]].mean()
    stock = features[features["business_date"] == last].set_index(SERIES)["stockout_share"]
    holidays = {pd.Timestamp(h.date) for h in cfg.holidays}
    rows = []
    for (store, cat), w in weather.iterrows():
        for d in days:
            rows.append({"store_id": store, "category_id": cat, "business_date": d,
                         "day_of_week": d.dayofweek + 1, "is_weekend": d.dayofweek >= 5, "is_holiday": d in holidays,
                         "month": d.month, "day_of_month": d.day, "promo_active": False,
                         "rainfall_mm": w["rainfall_mm"], "temperature_c": w["temperature_c"],
                         "stockout_share_lag_1": stock.get((store, cat), 0.0)})
    return pd.DataFrame(rows)


def _metrics_rows(frame: pd.DataFrame, evaluation: str, models: dict[str, str], segment_col: str | None,
                  version: str) -> list[dict]:
    rows = []
    groups = [("all", frame)] + (list(frame.groupby(segment_col)) if segment_col else [])
    for segment, part in groups:
        for horizon in [0, *range(1, HORIZON + 1)]:
            sub = part if horizon == 0 else part[part["horizon"] == horizon]
            if sub.empty:
                continue
            for name, col in models.items():
                bands = (sub["lower_units"], sub["upper_units"]) if name == "model" else (None, None)
                m = forecast_metrics(sub["actual_units"], sub[col], *bands)
                rows.append({"evaluation": evaluation, "model": name, "horizon": horizon, "segment": segment,
                             **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
                             "model_version": version})
    return rows


def _sku_split(preds: pd.DataFrame, gold_root: Path, silver_root: Path) -> pd.DataFrame:
    inv = read_table(gold_root, INVENTORY_DAILY)[["store_id", "product_id", "business_date", "units_sold"]]
    cat_sales = read_table(gold_root, DAILY_CATEGORY_SALES)[["store_id", "category_id", "business_date", "units"]]
    products = pd.concat([read_csv_strings(p) for p in (silver_root / "slv_products").glob("*.csv")])
    inv = inv.merge(products[["product_id", "category_id"]], on="product_id")
    inv = inv.sort_values(["store_id", "product_id", "business_date"])
    g = inv.groupby(["store_id", "product_id"])["units_sold"]
    inv["sku_28d"] = g.transform(lambda s: s.rolling(SKU_SHARE_DAYS, min_periods=1).sum())
    inv["sku_ma7"] = g.transform(lambda s: s.rolling(7, min_periods=1).mean())
    cat_sales = cat_sales.sort_values(["store_id", "category_id", "business_date"])
    cat_sales["cat_28d"] = cat_sales.groupby(["store_id", "category_id"])["units"].transform(
        lambda s: s.rolling(SKU_SHARE_DAYS, min_periods=1).sum())
    at_origin = inv.merge(cat_sales[["store_id", "category_id", "business_date", "cat_28d"]],
                          on=["store_id", "category_id", "business_date"])
    at_origin["sku_share"] = np.where(at_origin["cat_28d"] > 0, at_origin["sku_28d"] / at_origin["cat_28d"], 0.0)
    at_origin = at_origin.rename(columns={"business_date": "origin_date", "sku_ma7": "baseline_moving_avg"})
    category_level = preds.drop(columns=["baseline_seasonal_naive", "baseline_moving_avg"])
    sku = category_level.merge(at_origin[["store_id", "product_id", "category_id", "origin_date", "sku_share",
                                 "baseline_moving_avg"]], on=["store_id", "category_id", "origin_date"])
    for col in ("forecast_units", "lower_units", "upper_units"):
        sku[col] = sku[col] * sku["sku_share"]
    actual = inv[["store_id", "product_id", "business_date", "units_sold"]].rename(
        columns={"business_date": "forecast_date", "units_sold": "actual_units"})
    sku = sku.drop(columns=["actual_units"]).merge(actual, on=["store_id", "product_id", "forecast_date"], how="left")
    return sku


def run_forecasting(cfg: Config, tracker: RunTracker, gold_root: Path, silver_root: Path, ml_root: Path) -> dict:
    t0, started = time.perf_counter(), utc_now()
    version = f"demand_forecast_{tracker.run_id}"
    features = read_table(gold_root, DEMAND_FEATURES)
    train_end = pd.Timestamp(cfg.ml_split.train_end)
    last_day = features["business_date"].max()

    # backtest: model trained on Apr–Aug, recursive forecasts from every September day
    backtest_model = train(features[features["business_date"] <= train_end], cfg.seed)
    units = features.pivot_table(index=SERIES, columns="business_date", values="target_units").sort_index()
    future_days = [last_day + timedelta(days=i) for i in range(1, HORIZON + 1)]
    for d in future_days:
        units[d] = np.nan
    exog = pd.concat([features[[*SERIES, "business_date", *EXOG]], _future_exog(features, cfg, future_days)],
                     ignore_index=True)
    test_days = sorted(features.loc[features["business_date"] > train_end, "business_date"].unique())
    backtest = recursive_forecast(backtest_model, units, exog, [pd.Timestamp(d) for d in test_days], HORIZON, last_day)
    backtest["run_type"] = "backtest"

    # production: refit on all data, forecast the days after the data ends
    production_model = train(features, cfg.seed)
    future = recursive_forecast(production_model, units, exog, [future_days[0]], HORIZON, future_days[-1])
    future["run_type"] = "future"
    preds = pd.concat([backtest, future], ignore_index=True)
    preds["model_version"] = version
    n_preds = write_table(preds, gold_root, DEMAND_PREDICTIONS)

    sku = _sku_split(preds, gold_root, silver_root)
    sku["model_version"] = version
    n_sku = write_table(sku, gold_root, SKU_FORECAST)

    # accuracy (backtest only)
    bt, sku_bt = preds[preds["run_type"] == "backtest"], sku[sku["run_type"] == "backtest"].dropna(subset=["actual_units"])
    metric_rows = _metrics_rows(bt, "category_daily", {"model": "forecast_units",
                                                       "baseline_seasonal_naive": "baseline_seasonal_naive",
                                                       "baseline_moving_avg": "baseline_moving_avg"}, "category_id", version)
    metric_rows += _metrics_rows(sku_bt, "sku_daily", {"model": "forecast_units",
                                                      "baseline_moving_avg": "baseline_moving_avg"}, None, version)
    metrics = pd.DataFrame(metric_rows)
    write_table(metrics, gold_root, FORECAST_METRICS)

    # permutation importance on September, 1 day ahead (features built from actual history)
    test_rows = features[(features["business_date"] > train_end) & (features["history_days"] >= 14)]
    imp = permutation_importance(backtest_model.point, backtest_model.matrix(test_rows),
                                 test_rows["target_units"].to_numpy(dtype=float), scoring="neg_mean_absolute_error",
                                 n_repeats=5, random_state=cfg.seed)
    importance = pd.DataFrame({"feature": FEATURES, "importance_mae": imp.importances_mean.round(4),
                               "importance_std": imp.importances_std.round(4)})
    importance["rank"] = importance["importance_mae"].rank(ascending=False, method="first").astype(int)
    importance["model_version"] = version
    write_table(importance, gold_root, FEATURE_IMPORTANCE)

    # registry + artifacts
    model_dir = ml_root / "models" / "demand_forecast" / version
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in (("backtest", backtest_model), ("production", production_model)):
        joblib.dump({"model": model, "features": FEATURES, "version": version}, model_dir / f"{name}.joblib")
    pooled = metrics[(metrics["horizon"].isin([0, 1])) & (metrics["segment"] == "all")]
    summary = {f"{r.evaluation}/{r.model}/h{r.horizon}": {"wape": r.wape, "mae": r.mae} for r in pooled.itertuples()}
    test_start, test_end = pd.Timestamp(test_days[0]).date().isoformat(), last_day.date().isoformat()
    for status, model, t_end, rows in (("evaluated", backtest_model, train_end, backtest_model.params["train_rows"]),
                                       ("current", production_model, last_day, production_model.params["train_rows"])):
        tracker.model_run(model_name="demand_forecast", model_version=f"{version}/{'backtest' if status == 'evaluated' else 'production'}",
                          method="HistGradientBoostingRegressor (Poisson) + quantile 0.1/0.9", trained_at=utc_now(),
                          train_start=features["business_date"].min().date().isoformat(),
                          train_end=t_end.date().isoformat(), test_start=test_start if status == "evaluated" else "",
                          test_end=test_end if status == "evaluated" else "", input_table="gld_demand_features",
                          train_rows=rows, features="|".join(FEATURES), params=json.dumps(model.params, default=str),
                          metrics=json.dumps(summary) if status == "evaluated" else "",
                          artifact_path=_project_path(model_dir / f"{'backtest' if status == 'evaluated' else 'production'}.joblib"),
                          status=status, limitations=LIMITATIONS)
    for r in metrics[metrics["segment"] == "all"].itertuples():
        for metric in ("mae", "rmse", "wape", "bias", "interval_coverage"):
            value = getattr(r, metric)
            if pd.notna(value):
                tracker.model_metric(model_version=version, evaluation=r.evaluation, model=r.model,
                                     horizon=r.horizon, segment=r.segment, metric=metric, value=value)
    for table, n in ((DEMAND_PREDICTIONS, n_preds), (SKU_FORECAST, n_sku), (FORECAST_METRICS, len(metrics)),
                     (FEATURE_IMPORTANCE, len(importance))):
        tracker.table(stage="ml", job=table.name, source=",".join(table.sources), target=table.name, engine="python",
                      module="src.ml.forecasting", rows_written=n, status="success", started_at=started,
                      ended_at=utc_now(), duration_s=round(time.perf_counter() - t0, 2))
        for source in table.sources:
            tracker.lineage(source, table.name, table.name, engine="python")
    h1 = metrics[(metrics["evaluation"] == "category_daily") & (metrics["horizon"] == 1) & (metrics["segment"] == "all")]
    return {"model_version": version, "predictions": n_preds, "sku_predictions": n_sku,
            "wape_h1": dict(zip(h1["model"], h1["wape"]))}
