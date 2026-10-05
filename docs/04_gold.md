# 4. Gold Layer

**Status:** complete · **Code:** `src/transformations/gold/`, `src/serving/metrics.py` · **Run:** `python -m scripts.run_pipeline --stages gold`
**References (generated from the code):** [gold_catalog.md](gold_catalog.md) (every table and column) · [metric_definitions.md](metric_definitions.md) (every metric formula)

## Purpose

Silver holds clean records. Gold turns them into **business answers and ML inputs**: tables with a fixed grain (what one row means) and **one shared definition for every metric**, so every dashboard, model and assistant answer uses the same revenue, order counts and stock levels.

## Input and output

| | Location |
|---|---|
| Input | `data/silver/slv_<dataset>/` (19 typed tables, 272,971 rows) |
| Output | `data/gold/<table>/`: 12 typed CSV tables, rebuilt in full each run (previous version kept in `_previous/`) |
| Metadata | `meta_table_runs` (rows, timings), `meta_lineage_edges` (Silver → Gold) |

## Shared metric definitions

All metrics are defined once in `src/serving/metrics.py` (full list: [metric_definitions.md](metric_definitions.md)). The key ones:

| Metric | Definition |
|---|---|
| Completed order | delivered + successful payment + not cancelled (Silver `is_completed`) |
| Line revenue | quantity × price, using the **catalog price** when the line's unit price was flagged in Silver |
| GMV / Discount / Refunds | Σ line revenue / Σ promotion discount of completed orders / Σ completed refunds (on the refund date) |
| Net revenue | GMV − discount − refunds |
| AOV | (GMV − discount) ÷ completed orders |
| Days of inventory | closing stock ÷ average daily units (last 14 days); empty when there were no recent sales, never infinite |

### Decisions (agreed before building)

| # | Question | Decision |
|---|---|---|
| 1 | Revenue of lines with a flagged unit price | use the catalog price (the trusted value) |
| 2 | Which orders count toward revenue | completed orders only; cancelled and unpaid orders are reported separately |
| 3 | Refunds | subtract only `completed` refunds, on the refund's business date |
| 4 | Tier 2 tables (delivery, cancellation, promotion metrics) | built now |
| 5 | Holidays | short list of Indian holidays in `configs/default.yaml` (lunar dates approximate) |

One clarification made while building: **AOV excludes refunds**, because refunds are dated by refund date, not order date, so mixing them into a daily average would misstate it. Net revenue includes them.

## The 12 Gold tables

| Table | Grain (one row =) | Rows | Used for |
|---|---|---:|---|
| `gld_daily_sales` | store × day | 2,196 | executive overview, anomaly detection |
| `gld_daily_category_sales` | store × category × day | 43,920 | the demand-forecasting series |
| `gld_product_performance` | product × month | 3,600 | sales analytics, category ranking |
| `gld_inventory_daily` | store × focus SKU × day | 54,900 | stockout risk, replenishment |
| `gld_demand_features` | store × category × day | 43,920 | the forecasting model (train Apr–Aug, test Sep) |
| `gld_customer_360` | customer | 6,000 | segmentation, recommendations |
| `gld_customer_category` | customer × category | 44,923 | category-mix features, recommendations |
| `gld_basket_pairs` | product pair | 3,506 | basket analysis (support, confidence, lift) |
| `gld_delivery_metrics` | store × day | 2,196 | delivery performance |
| `gld_cancellation_metrics` | store × week × stage × reason | 1,312 | cancellation analysis |
| `gld_promotion_metrics` | promotion | 57 | promotion usage, cost and uplift |
| `gld_quality_summary` | Silver run × dataset × rule | 114 | Data Engineer workspace |

Grids are **zero-filled**: every store × day (× category) has a row even when nothing sold, so a day without sales is 0, not missing. Forecasting depends on this.

## How the two hard tables work

### Daily inventory (`gld_inventory_daily`)

Silver has *weekly* stock counts, stock events (restocks, damage, adjustments) and sales. Gold rebuilds stock at every instant:

```
stock(t) = last count before t + Σ movements since that count       (before the first count: first count, backwards)
```

- Each Monday count **re-anchors** the running stock. `reconciliation_gap` = counted − calculated at each count. It is non-zero when Silver quarantined some order lines or events, and it is exactly what a data engineer monitors.
- Units sold = every focus-SKU order line except orders cancelled *before* dispatch.
- A day is a **stockout day** if stock reached 0 at any moment. Calculated stock below 0 (possible when upstream rows are missing) is floored to 0 and flagged `calculated_negative`.
- **Late events** need no special logic: Gold is rebuilt in full, so they sit at their event time and every affected day is recomputed.

### Forecasting features without leakage (`gld_demand_features`)

- Lags (1/7/14 days) and rolling means/std (7/14 days) use **only earlier days** of a gap-free daily series.
- Calendar, holiday, promotion and weather columns describe the day itself and are known in advance.
- `stockout_share` (same day) shows when sales understate demand. It is for weighting training rows, **not** a model input; `stockout_share_lag_1` is safe to use.
- `split` = train (≤ 31 Aug) / test (September).

## Results (medium run)

| Measure | Value |
|---|---|
| Orders placed / completed / cancelled | 34,249 / 30,321 / 2,386 (cancellation rate 7.0%) |
| GMV − discount − refunds = **net revenue** | ₹2,52,87,969 − ₹2,15,742 − ₹5,51,373.45 = **₹2,45,20,853.55** |
| Average order value | ₹826.89 |
| On-time delivery (≤ 15 min) | 50.2% (median delivery 14.5 min, right at the SLA) |
| Promotion uplift (median over 57 promotions) | +40% category units |
| Stockout days (store × SKU) | 1,917 |
| Customers with at least one order | 5,403 of 6,000 |
| Feature rows train / test | 36,720 / 7,200 |
| Runtime | ~100 s |

## How it is verified

`python -m scripts.check_gold`: **10/10 checks.** It recomputes everything independently from the Silver files with pandas.

| Check | Result |
|---|---|
| 1. All 12 tables written | ✅ |
| 2. Grain unique | ✅ 0 duplicates |
| 3. Complete grids | ✅ 2,196 store-days, 54,900 SKU-days, … |
| 4. **Revenue, discount, refunds, orders reconcile with Silver to the paisa** | ✅ (daily, category and product tables all equal the Silver total) |
| 5. Inventory continuity (closing = next day's opening; every count anchored) | ✅ 0 breaks; gap 0 at 53% of counts, mean \|gap\| 1.5 units |
| 6. No leakage (features recomputed using past days only) | ✅ |
| 7. Weekend uplift visible | ✅ weekend ÷ weekday units = 1.30 |
| 8. Rain uplift visible | ✅ rainy ÷ dry store-days = 1.14 |
| 9. Planted product pairs at the top by lift | ✅ 23 of 25 in the top 50 |
| 10. Real stockout days detected | ✅ recall 67%, precision 62% (bar of 60% set before running) |

Checks 7–10 compare against the generator's ground truth (read only by the report). Stockout detection is not 100% because Silver quarantined some lines and events, so the rebuilt stock drifts between counts. That drift is what `reconciliation_gap` measures.

`tests/integration/test_gold.py` (small profile): core checks, one store-day recomputed by hand, typed read-back of every table, days of inventory never infinite, and identical output on rebuild.

## Problems found and decisions made while building

- **`stockout_share` would leak** (same-day information) if used as a model input. It is documented as a weighting column, and a lagged version was added for model input.
- **AOV and refunds:** refunds are excluded from AOV (see above).
- **The holiday feature is expected to show little effect**: the generator didn't plant holiday demand. That is an honest result to report, not a bug.
- **Silver late-data test added** before Gold (`tests/integration/test_late_data.py`): a record arriving in a later file is ingested, flagged late in Bronze, and dated by its event time in Silver. This closed the last Silver checklist item.

## Presentation copy

`python -m scripts.export_flat_tables --layer gold` writes `data_sep_gold/`: one CSV per Gold table (12 files).
Suggested demo: open `gld_daily_sales.csv` for one store and show that `net_revenue = gmv − discount − refunds`, then run `check_gold` to show the totals match Silver to the paisa.
