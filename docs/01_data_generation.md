# 1. Synthetic Data Generation

**Status:** complete · **Code:** `src/generation/` · **Run:** `python -m src.generation.generate_all --profile medium`

## Purpose

The platform needs realistic quick-commerce data with known problems in it. A generator gives three things real data can't:

1. **Controlled dirt.** Exactly 90% of each dataset's rows are clean, and every problem is injected on purpose, so later layers can be checked against a known answer.
2. **Planted signal.** Patterns that ML models should find (customer types, products bought together, anomalies, stockouts) are put in deliberately and recorded separately as *ground truth*.
3. **Reproducibility.** The same seed always gives byte-identical data.

## Output

One run writes everything under `data/generation/<run_id>/` (all CSV):

| Folder | Contents |
|---|---|
| `landing/historical/<dataset>/` | Apr–Jun 2025, one file per month; the 5 reference tables as one file each |
| `landing/batch/<dataset>/` | Jul–23 Sep 2025, one file per business day (IST) |
| `landing/stream/<dataset>/` | 24–30 Sep 2025, hourly drops, for orders, order_items, payments, inventory_events, application_events |
| `manifest/` | `manifest_run.csv` (seed, profile, dates), `manifest_datasets.csv` (targets and actual counts), `manifest_issues.csv` (planned vs injected issues), `manifest_files.csv` (every landing file) |
| `ground_truth/` | personas, affinity pairs, anomalies, stockouts, and **every injected issue** (`gt_injected_issues.csv`). Used only for evaluation; the pipeline never reads it |

`data/generation/LATEST` holds the newest run id.

## How it works

1. **Reference data:** 20 categories, 600 products (fictional brands, Zipf popularity), 12 dark stores in 3 cities, 120 delivery riders, 60 category promotions.
2. **Customers:** 6,000, each with a hidden *persona* (daily essentials, weekend stock-up, snacks at night, premium, deal seeker) that drives how often they order, basket size, category mix and night ordering.
3. **Weather:** 3-hourly readings per city with a monsoon pattern (rain raises order volume and slows deliveries).
4. **Orders:** exactly 36,000, spread over store-days by store size × day of week × rain × trend. Customers come from the store's city 97% of the time.
5. **Baskets:** 122,400 lines (avg 3.4). Category from the persona's mix (boosted by active promotions), product by popularity. 25 product pairs are planted to be bought together.
6. **Inventory simulation:** for the 25 most popular products in every store, stock is tracked minute by minute: orders deplete it, nightly checks reorder, restocks arrive 1–2 days later, damage and adjustments happen, and 40 planted *stockout episodes* block restocks. When an item is out of stock the customer gets a substitute.
7. **Downstream records,** consistent with the orders (rules C1–C11 in the plan): payments (with failed retries), deliveries, cancellations, refunds, reviews, app events, service logs.
8. **Planted anomalies (15):** store outages, product demand spikes, payment-gateway failures.
9. **Dirty-data injection** (next section).
10. **Landing files:** rows are split by event date into historical, batch and stream files; ~1% of batch rows and ~3% of stream rows arrive late, and ~2% of stream rows are shuffled out of order.

## Dirty-data design

- **Budget:** every dataset is exactly **90% clean, 10% dirty**. Each dirty row has exactly one issue.
- **Issue types:** duplicates (DUP 2.5%), missing required value (MISS 2%), inconsistent format (FMT 1.5%), invalid number (NUM 1%), business-rule violation (BIZ 1%), invalid type (TYPE 0.75%), broken reference (FK 0.75%), bad time order (SEQ 0.5%); malformed payloads (MALF) for logs and app events.
- **Per dataset:** issue types that don't apply hand their share to those that do. Reference tables (stores, products, …) only get fixable issues (DUP, FMT); otherwise one bad store would invalidate thousands of orders.
- **Self-contained:** every issue is detectable from its own row (plus lookups). Duplicate "conflicting" copies differ only in formatting, so cleaning makes them identical.
- **Not counted as dirty:** late arrivals, out-of-order rows, optional empty values (e.g. 30% of review comments), and valid outliers.

Full rules: `configs/dirty_data.yaml`; per-column details: [data_dictionary.md](data_dictionary.md).

## Results (medium profile, seed 42)

| Measure | Value |
|---|---|
| Generated rows (targets met exactly) | 298,124 |
| Duplicate copies added | 9,660 |
| **Source rows in landing files** | **307,784** |
| Clean rows | **90.00%** (every dataset 89.8–90.5%) |
| Injected issues (ground truth) | 30,778 |
| Landing files | 1,864 (22 MB) |
| Runtime | ~40 s |

Planted signal check: affinity pairs show support 0.03–0.50%, lift 8–74 (median 29), against ~1 for normal pairs.

## How it is verified

`python -m pytest tests/unit tests/integration/test_generation.py`:
- targets met, every dataset ~90% clean, planned issue counts = injected counts
- all 11 cross-dataset consistency rules hold on the clean data (e.g. order total = lines − discount, refund ≤ payment, review only for bought products)
- every row lands in exactly one landing file; the same seed gives identical output
- every configured issue type actually applies to real rows; planted pairs are detectable

## Decisions and problems found

- **Volumes rebalanced** from the original plan: with 30 stores × 800 products there were only ~3 sales per store-product pair for the whole period, too sparse for forecasting. Now 12 stores × 600 products, avg 3.4 items per basket, and forecasting at store × category level.
- **CSV only** for all data (project decision).
- **Inventory event count** first came out 1 short of target; fixed by tuning the background-event count and topping up with end-of-period stock adjustments.
- **Basket-analysis thresholds** in the plan were too high for the measured support of planted pairs; lowered to "pair seen in ≥ 8 baskets".
