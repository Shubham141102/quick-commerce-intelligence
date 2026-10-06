# 9. Business & Revenue Workspace (Phases 5B, 5C)

**Status:** complete, with one documented known limitation (5C.2) · **Code:** `app/views/business.py`, `src/serving/queries.py` (business functions), `src/ml/anomaly.py` · **Run:** `python -m scripts.run_pipeline --stages ml,publish`, then `streamlit run app/Home.py` · **Check:** `python -m scripts.check_phase5`
**Outputs (columns in [gold_catalog.md](gold_catalog.md)):** `gld_sales_anomalies`, `gld_anomaly_series` · **Login:** the Business Analyst demo account (and admin)

## Business questions

- *How are we selling?* Revenue, orders, basket size and cancellations, and how they compare with the previous period.
- *Are we delivering on our promise?* On-time rate, delivery time, failed deliveries, and the reasons for cancellations.
- *Did something unusual happen?* Products that suddenly took off, payment-gateway failures, and quiet hours at a store.

## The page

Two filters at the top (date range, stores) apply to every tab.

| Tab | What it shows | Source |
|---|---|---|
| **Sales & revenue performance** | Net revenue, completed orders, AOV, units, cancellation rate, each vs the previous period of equal length; daily / weekly revenue trend; revenue by store; category contribution; top products | `gld_daily_sales`, `gld_monthly_product_sales` |
| **Delivery & operations** | On-time rate (≤ 15 min after pickup), average and typical P90 delivery time, failed deliveries, in-transit cancellations; trend; worst stores first; cancellations by reason | `gld_delivery_metrics`, `gld_cancellation_metrics` |
| **Anomaly detection** | Count per detector, filterable list sorted by how unusual each event is, and a chart of observed vs expected around a selected anomaly | `gld_sales_anomalies`, `gld_anomaly_series` |

All metric definitions are in [metric_definitions.md](metric_definitions.md). Every number comes from a role-checked function in `src/serving/queries.py`. `stores()` and `categories()` are shared filters open to the Inventory, Business and Marketing roles.

## Results (medium run)

| Measure | Value |
|---|---|
| Net revenue (Apr–Sep) | ₹24,520,853.55 |
| Completed orders | 30,321 |
| AOV | ₹826.89 |
| Delivered orders | 30,575 |
| On-time rate | 50.2% (avg 16.3 min, against a 15-minute promise) |

The page totals match Gold exactly (check 5B.1/5B.2).

## Anomaly detection (5C)

Each detector compares an observed count with what is normal over the **previous 28 days** and scores how unlikely the count is (p-value, shown as score = −log10 p). An anomaly is a **signal to investigate**, not proof of a problem.

| Detector | Observed | Expected | Flag when |
|---|---|---|---|
| **Demand spike** | orders containing a product at a store on a day | product's share of the store's orders over the last 28 days × that day's store orders | p < 0.0001 and ≥ 4 orders; consecutive days merged |
| **Payment failure** | network-wide failed payments and payment-service ERROR logs per hour | same hour, previous 28 days | p < 0.000001 and ≥ 4 events; consecutive hours merged |
| **Store outage (experimental)** | orders in a 2–8 hour window (06:00–23:59 IST) | same store, hours and day type (weekday/weekend), previous 28 days | p < 0.001 and ≥ 6 orders expected |

### Method revision (approved 2026-10-05, after the first measurement)

The first version used **Poisson** p-values, which assume the variance equals the mean. The data varies much more than that: store-day orders have a mean of 15.6 and a variance of 27.7, driven by weekdays, rain and promotions. Ordinary quiet or busy spells therefore looked "impossible", and the first measurement failed precision (7 of 30, 23%).

Changes, all decided **before** re-measuring once:
1. Spikes and outages use a **negative binomial** model (var = μ + α·μ²), with α estimated from the data by method of moments.
2. The spike baseline scales with **that day's store orders**, so a busy day is not a spike.
3. **Store outages are labelled experimental** and left out of the precision figure. A store gets less than 1 order an hour, so a 3–6 hour outage cannot be told apart from chance using orders alone. Real systems use store heartbeats, which this data does not have.
4. Thresholds, minimum counts and the pass bars were **not** changed. Payment detection was unchanged (it had passed).

### Verification against the planted anomalies

Pass bars were **fixed before measuring**. A detection matches a planted anomaly with the same type, store and product within 1 hour (1 day for spikes).

| Check | First run | After revision |
|---|---|---|
| 5C.2 Demand spikes found (≥ 80%) | 4 of 6 | ⚠️ **LIMIT: 4 of 6**, known limitation |
| 5C.3 Payment failures found (≥ 2/3) | 3 of 3 | ✅ PASS: 3 of 3 |
| 5C.4 Store outages (no bar, reported) | 0 of 6; 11 false alarms | 0 of 6; 5 false alarms |
| 5C.5 Precision of spike + payment detections (≥ 50%) | 23% (all detectors) | ✅ PASS: 7 of 11 (64%) |

**Why 5C.2 is not met.** Both missed spikes are undetectable from this data:
- S02 / P0062 on 4 April is day 4 of the data, so there is no 28-day history to compare with.
- S11 / P0282 on 23 July was planted as "~25% of orders" but produced only **3** orders at that store, below the 4-order minimum.

Lowering the minimum would flood the list with 3-order false alarms. The bar stays and the result is reported as LIMIT.

## Limitations

- Anomaly p-values are per-test; with thousands of store × product × day tests a few false alarms are expected (4 of 11 here).
- Store outages from order volume alone are not reliable at ~1 order per store-hour.
- Weekly delivery and cancellation tables are matched to the filter by week, not exact days.
