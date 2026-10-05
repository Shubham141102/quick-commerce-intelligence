"""Demand forecasting model: one global gradient-boosting model over all store × category series.

- point forecast: HistGradientBoostingRegressor with Poisson loss (daily unit counts are non-negative)
- uncertainty band: two quantile models (10th and 90th percentile)
- training rows are down-weighted on stockout days, where sales understate demand
- multi-day forecasts are recursive: each predicted day feeds the next day's lag / rolling features
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

NUMERIC_FEATURES = [
    "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
    "day_of_week", "is_weekend", "is_holiday", "month", "day_of_month", "promo_active",
    "rainfall_mm", "temperature_c", "stockout_share_lag_1",
]
CATEGORICAL_FEATURES = ["store_id", "category_id"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
SERIES = ["store_id", "category_id"]
MIN_HISTORY_DAYS = 14
STOCKOUT_WEIGHT = 0.5   # weight = 1 − 0.5 × stockout_share


@dataclass
class ForecastModel:
    point: HistGradientBoostingRegressor
    lower: HistGradientBoostingRegressor
    upper: HistGradientBoostingRegressor
    stores: list[str]
    categories: list[str]
    params: dict

    def matrix(self, rows: pd.DataFrame) -> pd.DataFrame:
        X = rows[FEATURES].copy()
        for c in ("is_weekend", "is_holiday", "promo_active"):
            X[c] = X[c].astype(float)
        X["store_id"] = pd.Categorical(X["store_id"], categories=self.stores)
        X["category_id"] = pd.Categorical(X["category_id"], categories=self.categories)
        return X

    def predict(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = self.matrix(rows)
        point = np.clip(self.point.predict(X), 0, None)
        lo = np.clip(self.lower.predict(X), 0, None)
        hi = np.clip(self.upper.predict(X), 0, None)
        return point, np.minimum(lo, point), np.maximum(hi, point)


def train(rows: pd.DataFrame, seed: int) -> ForecastModel:
    rows = rows[rows["history_days"] >= MIN_HISTORY_DAYS]
    params = {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 31, "min_samples_leaf": 40,
              "l2_regularization": 0.1, "random_state": seed, "categorical_features": "from_dtype",
              "early_stopping": False}
    stores, categories = sorted(rows["store_id"].unique()), sorted(rows["category_id"].unique())
    shell = ForecastModel(None, None, None, stores, categories, params)  # type: ignore[arg-type]
    X, y = shell.matrix(rows), rows["target_units"].to_numpy(dtype=float)
    weight = 1 - STOCKOUT_WEIGHT * rows["stockout_share"].fillna(0).to_numpy()
    point = HistGradientBoostingRegressor(loss="poisson", **params).fit(X, y, sample_weight=weight)
    lower = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, **params).fit(X, y, sample_weight=weight)
    upper = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **params).fit(X, y, sample_weight=weight)
    return ForecastModel(point, lower, upper, stores, categories,
                         {**params, "loss": "poisson", "quantiles": [0.1, 0.9], "train_rows": int(len(rows)),
                          "stockout_weight": STOCKOUT_WEIGHT})


def recursive_forecast(model: ForecastModel, units: pd.DataFrame, exog: pd.DataFrame,
                       first_days: list[pd.Timestamp], horizon: int, last_day: pd.Timestamp) -> pd.DataFrame:
    """Forecast up to `horizon` days from each first forecast day, feeding predictions back in.

    units: wide actual units, index = (store_id, category_id), columns = consecutive dates (may extend
           into the future with NaN). exog: rows per (store_id, category_id, business_date) with the
           calendar / promotion / weather columns and stockout_share_lag_1 (known for the first day).
    """
    dates = list(units.columns)
    pos = {d: i for i, d in enumerate(dates)}
    actual = units.to_numpy(dtype=float)
    keys = units.index.to_frame(index=False)
    exog = exog.set_index(["store_id", "category_id", "business_date"])
    out = []
    for first in first_days:
        o = pos[first]
        hist = actual.copy()
        hist[:, o:] = np.nan
        stockout_lag = exog.loc[[(s, c, first) for s, c in units.index], "stockout_share_lag_1"].to_numpy()
        for h in range(1, horizon + 1):
            d = o + h - 1
            if d >= len(dates) or dates[d] > last_day:
                break
            window7, window14 = hist[:, d - 7:d], hist[:, d - 14:d]
            feats = keys.copy()
            feats["lag_1"], feats["lag_7"], feats["lag_14"] = hist[:, d - 1], hist[:, d - 7], hist[:, d - 14]
            feats["rolling_mean_7"] = window7.mean(axis=1)
            feats["rolling_mean_14"] = window14.mean(axis=1)
            feats["rolling_std_7"] = window7.std(axis=1, ddof=1)
            day = exog.loc[[(s, c, dates[d]) for s, c in units.index]].reset_index(drop=True)
            for col in ("day_of_week", "is_weekend", "is_holiday", "month", "day_of_month", "promo_active",
                        "rainfall_mm", "temperature_c"):
                feats[col] = day[col].to_numpy()
            feats["stockout_share_lag_1"] = stockout_lag
            point, lo, hi = model.predict(feats)
            hist[:, d] = point
            block = keys.copy()
            block["forecast_date"] = dates[d]
            block["origin_date"] = dates[o - 1]
            block["horizon"] = h
            block["forecast_units"], block["lower_units"], block["upper_units"] = point, lo, hi
            block["baseline_seasonal_naive"] = actual[:, d - 7]
            block["baseline_moving_avg"] = actual[:, o - 7:o].mean(axis=1)
            block["actual_units"] = actual[:, d]
            out.append(block)
    return pd.concat(out, ignore_index=True)
