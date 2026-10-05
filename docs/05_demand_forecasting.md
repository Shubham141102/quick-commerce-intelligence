# 5. Demand Forecasting (Phase 4B)

**Status:** complete · **Code:** `src/ml/forecasting/`, `src/ml/evaluation.py`, `src/ml/tables.py` · **Run:** `python -m scripts.run_pipeline --stages ml`
**Outputs (columns in [gold_catalog.md](gold_catalog.md)):** `gld_demand_predictions`, `gld_sku_demand_forecast`, `gld_forecast_metrics`, `gld_forecast_feature_importance`

## Question answered

*How many units will each store sell in each category over the next 7 days?* Also per focus SKU, which the stockout-risk step (4C) needs.

## Data

`gld_demand_features`: 240 series (12 stores × 20 categories) × 183 days, zero-filled. Lags and rolling statistics use only earlier days (no leakage; verified in Gold and again here).

| Split | Dates | Rows |
|---|---|---:|
| Train | 1 Apr – 31 Aug 2025 (first 14 days dropped: incomplete lags) | 33,360 |
| Test (backtest) | 1 – 30 Sep 2025, never seen in training | 7,200 |

ML runs in **pandas + scikit-learn**, not Spark: the table is small, and the plan uses Python where data is bounded.

## Models

| Model | How it forecasts |
|---|---|
| Baseline: seasonal naive | same weekday one week earlier |
| Baseline: moving average | mean of the last 7 days (flat for all 7 days ahead) |
| **Main model** | one **HistGradientBoostingRegressor** across all 240 series, **Poisson loss** (daily unit counts) |
| Uncertainty band | two more gradient-boosting models with **quantile loss** (10th and 90th percentile) |

**Features (17):** lag 1/7/14, rolling mean 7/14, rolling std 7, day of week, weekend, holiday, month, day of month, promotion active, rainfall, temperature, yesterday's stockout share, store, category.

**Stockout days are down-weighted** in training (weight = 1 − 0.5 × stockout share), because sales on those days understate real demand.

**Recursive multi-day forecasting:** for day 2–7 the model's own forecasts fill the lag / rolling features, which is exactly how the forecast is used live.

## Evaluation

The backtest model (trained on Apr–Aug) forecasts **1–7 days ahead from every day of September**: 45,360 forecasts next to the actuals and both baselines. Metrics: MAE, RMSE, **WAPE** (Σ|error| ÷ Σ actual, the headline), bias, and band coverage.

**Rule from the plan:** the model counts as useful only if it beats both baselines on WAPE.

## Results

| WAPE (lower is better) | Model | Moving average | Seasonal naive |
|---|---:|---:|---:|
| Store × category, 1 day ahead | **0.621** | 0.686 | 0.873 |
| Store × category, all horizons 1–7 | **0.619** | 0.689 | 0.866 |
| Focus SKU, all horizons | **0.878** | 0.949 | — |

**Is 0.62 good? Context matters.** An average store-category-day sells only **3.3 units** (17% of days sell nothing), so a large part of the error is pure randomness no model can predict. We measured how good a *perfect* model could be on this data:

| Reference point | WAPE |
|---|---:|
| Seasonal naive | 0.87 |
| Moving average (best baseline) | 0.69 |
| **Our model** | **0.62** |
| Perfect model, if each sale were 1 unit (plain Poisson noise) | 0.44 |
| **Perfect model, realistic** (each order line sells 1–5 units, as in the data) | **≈ 0.58** |

The model closes roughly **two-thirds of the gap** between the best baseline and the best achievable result. The remaining error is mostly irreducible noise, not a model weakness.

| Other result | Value | Meaning |
|---|---|---|
| Error growth from 1 → 7 days ahead | 0.621 → 0.623 | almost none: demand here depends on calendar/level, not on recent days, so recursion adds little error |
| 10–90% band coverage | 86.6% (target 80%) | the band is honest, slightly cautious |
| Bias | −7% | slight under-forecast, likely from down-weighting stockout days |
| MAE (1 day) | 2.06 units | |

**What the model learned** (permutation importance, September):

| Rank | Feature | Interpretation |
|---|---|---|
| 1–2 | category, store | demand level differs by category and store size |
| 3 | day of week | the planted weekend effect (weekends +30%) |
| 4 | promotion active | the planted promotion boost |
| 7 | rainfall | the planted monsoon effect (rainy days +14%) |
| low | lag features | recent days carry little extra signal in this data |

The three planted demand drivers are recognised, which is a check that the model learned real structure rather than noise.

## SKU-level forecast (for stockout risk)

Each focus SKU's daily demand = category forecast × the SKU's share of the category over the 28 days before the forecast was made (top-down split). Simple and explainable; it beats the SKU moving average (0.878 vs 0.949). SKU demand is sparse (~0.9 units/day), so SKU-level error is naturally higher.

## Outputs

| Table | Rows | Content |
|---|---:|---|
| `gld_demand_predictions` | 47,040 | September backtest (forecast, band, both baselines, actual) + **live forecast for 1–7 Oct 2025** |
| `gld_sku_demand_forecast` | 58,800 | the same per store × focus SKU |
| `gld_forecast_metrics` | 520 | MAE / RMSE / WAPE / bias / coverage by model, horizon and category |
| `gld_forecast_feature_importance` | 17 | importance per feature |
| `data/ml/models/demand_forecast/<version>/` | 2 files | backtest model (trained to 31 Aug) and production model (refit to 30 Sep), joblib |
| `meta_model_runs`, `meta_model_metrics` | | model registry: method, training window, features, parameters, metrics, artifact, limitations |

## Model card

| | |
|---|---|
| Purpose | 1–7 day demand forecast per store × category (and focus SKU) for inventory planning |
| Method | HistGradientBoostingRegressor, Poisson loss; quantile models for the 10–90% band |
| Training data | `gld_demand_features`, Apr–Aug (backtest) / Apr–Sep (production), synthetic |
| Evaluation | September backtest from every day, 1–7 days ahead, vs two baselines |
| Headline | WAPE 0.62 vs 0.69 (best baseline); realistic floor ≈ 0.58 |
| Limitations | synthetic data; ~3 units per cell so daily error is high; future weather assumed = last 7-day average; no promotions assumed after the data ends; holiday dates approximate and no holiday effect planted; backtest model not refit during September; slight under-forecast bias |
| Intended use | advisory input to stockout risk and replenishment; not for automated ordering |

## How it is verified

`python -m scripts.check_ml`: **6/6 checks.**

1. Predictions complete: actuals for every backtest row, 7 future days for every series, no duplicate keys.
2. Model beats both baselines (category and SKU).
3. Band honest: lower ≤ forecast ≤ upper everywhere; coverage 70–95%.
4. No train/test overlap: backtest model trained up to 31 Aug, tested from 1 Sep.
5. Registry filled and both model files load.
6. Planted drivers recognised (weekday top 5, promotion and rain top 10).

`tests/integration/test_ml.py` (small profile): structural checks, future forecast starts right after the data, SKU forecast = category forecast × share, and **recursive forecasting never sees the future** (changing actual sales after the origin does not change the forecast).

## Problems found and fixed

- A column-name clash in the SKU split (category and SKU baselines both called `baseline_moving_avg`) crashed the first run; fixed by dropping the category baseline before the join.
- Model file paths were stored relative to the project, which broke when tests wrote models to a temporary folder; now relative inside the project, absolute outside.
- The pipeline started Spark even for the ML stage, which doesn't need it; Spark now starts only for ingest / silver / gold.
