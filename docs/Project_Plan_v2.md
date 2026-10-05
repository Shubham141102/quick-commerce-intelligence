# Quick-Commerce Intelligence Platform — Project Plan v2

**Title:** Quick-Commerce Intelligence Platform: An End-to-End PySpark Medallion Pipeline with ML, RAG and Multi-Persona Analytics
**Type:** End-to-end data engineering, analytics and AI application
**Domain:** Quick-commerce (dark-store model, inspired by Blinkit)
**Status:** Revised plan (v2). Supersedes the v1 master plan in `Readme.md`.

| Area | Decision |
|---|---|
| Language | Python 3.11 |
| Processing engine | PySpark 3.5.x on Java 17 |
| Storage format | **CSV only**, for every layer: source, Bronze, Silver, quarantine, Gold, ML outputs, metadata and the published demo snapshot. Conventions in §2.1. |
| Serving query engine | DuckDB querying the published CSV files (no Spark in the hosted app) |
| ML | scikit-learn, mlxtend (bounded), statsmodels optional |
| RAG | BM25 retrieval by default; optional sentence-transformer embeddings; optional LLM behind a provider-agnostic interface |
| Frontend | Streamlit + Plotly |
| Hosting | Streamlit Community Cloud |
| Dev environment | WSL2 (Ubuntu) or a Docker dev container. Native Windows is not a supported dev target for Spark. |
| Orchestration | Custom Python CLI with a declared stage DAG (no Airflow) |
| Config | YAML profiles validated with Pydantic |
| Testing | pytest, session-scoped SparkSession fixture, `chispa` for DataFrame assertions |

---

## 0. What changed from v1

| # | Change | Why |
|---|---|---|
| 1 | **Dataset volumes rebalanced** (fewer stores, fewer products, more items per order, a fixed 6-month window) | In v1, 30 stores × 800 products gave ~3 sales per store-product pair for the *whole period*, ~2 items per basket and fewer inventory snapshots than store-product pairs. Forecasting, basket analysis and stockout risk could not produce meaningful results. |
| 2 | **Date range fixed:** 2025-04-01 → 2025-09-30 (183 days), timezone rules defined | v1 never specified a date range, although forecasting, weather density and snapshot frequency all depend on it. |
| 3 | **Realistic demand model in the generator:** Zipf product popularity, day-of-week and rain effects, planted co-purchase pairs, latent customer personas, planted anomalies, planted stockouts | Synthetic data only yields useful ML if the signal is deliberately put there. Planted ground truth also lets us *measure* whether models recover it. |
| 4 | **Forecast grain changed** to store × category × day (primary), plus focus-SKU × store × day (secondary, for stockout) | Daily SKU-level series are too sparse at this volume. |
| 5 | **CSV-only storage with a full-rebuild strategy** (§2.1) | Project decision: all data stays in CSV for now. CSV has no upserts or transactions, so Silver and Gold are rebuilt in full each run (seconds at this scale) and swapped in atomically. All storage access goes through one I/O module, so a later move to Parquet or Delta is a config change. |
| 6 | **Cross-dataset consistency rules** added to the generator | e.g. a cancelled-before-dispatch order must not have a delivery; refunds must not exceed the payment. |
| 7 | **Weather keyed by `city_id`**, and stores carry `city_id` | v1's `location_id` had no mapping to stores. |
| 8 | **Application logs added to the volume table** | v1 listed the dataset but gave no row target. |
| 9 | **Scope split into Tier 1 (MVP) and Tier 2 (stretch)** | v1 was team-sized: 19 datasets, ~20 Gold tables, 6 ML use cases, RAG and ~27 pages, all required. |
| 10 | **Open technical choices resolved:** retrieval method, LLM handling, auth, artifact storage, streaming mechanism, quarantine shape, dedup tie-break | v1 left these to the implementer. |
| 11 | **Formulas corrected** (v1 rendering inverted numerator and denominator) and tables restored | The v1 document was a broken chat export. |
| 12 | **Feasibility assessment added** (§21) | Requested. |

---

## 1. Vision and objectives

### 1.1 Vision

Build a realistic quick-commerce intelligence platform that simulates how a dark-store business processes data on customers, products, stores, orders, payments, inventory, deliveries, promotions, app activity, weather and application logs. Raw data should be traceable all the way to a business decision.

The platform must be **reproducible, modular, observable, explainable, testable, and honest** about what is simulated, precomputed or actually executed.

### 1.2 Objectives

| Objective | Expected outcome |
|---|---|
| End-to-end lifecycle | A raw record can be traced Source → Bronze → Silver → Gold → ML/RAG → UI |
| PySpark Medallion | PySpark is the default engine for ingestion and transformation; every layer is stored as CSV |
| Selective Python | pandas/NumPy/scikit-learn only for bounded, specialised work |
| Controlled dirty data | Issues are intentional, configurable, measured and visible |
| Four ingestion patterns | Batch, historical backfill, simulated stream (Structured Streaming), logs |
| Business-ready Gold | Curated tables with documented grain feed dashboards and ML |
| Analytics/ML | Six use cases (three in Tier 1), each evaluated against a baseline and against planted ground truth where available |
| RAG assistant | Policy retrieval, controlled metric functions, hybrid answers with citations |
| Personas | Three business workspaces plus one Data Engineer workspace |
| Pipeline visibility | Quality, transformations, lineage, run history and failures are inspectable |
| Affordable deployment | The hosted app reads a published CSV snapshot; no running Spark cluster |
| Demonstrability | An interviewer can follow one dirty record to a business outcome |

### 1.3 Processing rule

PySpark is the default for transformations. Python complements Spark; it does not replace the Medallion layers. Never `collect()` or `toPandas()` an unbounded DataFrame. Aggregate or filter in Spark first.

**Why Spark at this scale?** The Medium profile (~298K rows) is small enough that pandas would also cope. Spark is chosen to show distributed-processing patterns (windowing, deterministic deduplication, Structured Streaming, partitioned writes) that carry over to production. The **Large profile (~3M rows)** exists to back this up with *measured* runtime comparisons (§17).

---

## 2. Architecture

```
Source files (batch, historical, stream drops, logs)          (CSV)
        │
        ▼
Bronze  — raw values preserved + ingestion metadata           (CSV)
        │
        ▼
Silver  — typed, standardised, deduplicated, validated         (CSV)
        ├──► Quarantine (rejected rows + failed rules)          (CSV)
        ▼
Gold    — business aggregates, features, ML outputs             (CSV)
        │
        ▼
Publication — bounded snapshot + metadata                       (CSV, data/demo/)
        │
        ▼
Streamlit app — DuckDB queries over the CSV snapshot; RAG assistant; persona workspaces
```

**Cross-cutting:** config and secrets, run tracking and metadata, data quality, lineage, structured logging, error handling, role-based access, tests.

**Boundary rules**

- Business logic lives in `src/`, never in Streamlit page files.
- Metric definitions live in exactly one place (`src/serving/metrics.py` plus `docs/metric_definitions.md`).
- The hosted app never imports PySpark.
- **All reads and writes go through `src/common/io.py`** (`read_table`, `write_table`, `append_table`). No module calls `spark.read.csv` or `pd.read_csv` on project tables directly.

### 2.1 CSV storage conventions

**Scope of "CSV only".** Every *data table* is CSV: source datasets, Bronze, Silver, quarantine, Gold, ML predictions and evaluation results, metadata/run tables, the generation manifest and ground truth, and the published snapshot. A few things are not tabular data and keep their natural formats: config (`.yaml`), policy documents (`.md`), trained model binaries (`.joblib`) and Spark Structured Streaming checkpoint folders (internal to Spark). Pipeline logs are CSV too (§12.3).

| Convention | Rule |
|---|---|
| Encoding / dialect | UTF-8, comma delimiter, header row, RFC 4180 quoting (`"` quote, `"` escape), `\n` line endings |
| Schemas | **Never `inferSchema`.** Each table has an explicit schema in `src/common/schemas.py` (column, type, nullable). It is applied on every read and documented in the data dictionary. |
| Nulls | Written as an empty field. An empty string and null are **treated as the same thing** (documented limitation of CSV). |
| Timestamps | ISO-8601 UTC, `yyyy-MM-dd'T'HH:mm:ss'Z'`; dates as `yyyy-MM-dd` |
| Decimals | Money written as plain text with 2 decimals (e.g. `149.00`) and read as `decimal(12,2)` |
| Booleans | `true` / `false` |
| Lists | Pipe-separated in one column (e.g. `failed_rules = "qty_positive\|fk_product"`) |
| Nested data | A JSON *string* inside a quoted CSV column (e.g. `application_events.metadata`, `quarantine.raw_record`). The file is still CSV; the column is parsed with `from_json` in Silver. |
| Table layout | Each table is a folder: `data/<layer>/<table>/part-*.csv` (Spark's output layout). Small tables (< 200K rows) are written with `coalesce(1)` to give one part file. |
| Partitioning | Bronze is partitioned by folder `_batch_id=<id>/`. Silver and Gold are not partitioned (they are rebuilt in full; see below). |
| Published snapshot | One flat file per table, `data/demo/<table>.csv`, so DuckDB and pandas read it directly |
| Atomic writes | Spark CSV overwrite is not atomic. `write_table` writes to `data/<layer>/_staging/<run_id>/<table>/`, validates row count and schema, then swaps the folder in with a rename. A failed write leaves the previous version in place. |
| Versioning | The previous version of each Silver/Gold table is kept as `data/<layer>/_previous/<table>/` (one generation back), so the before/after comparison is possible |
| Metadata appends | Each run writes its own file, e.g. `data/metadata/meta_stage_runs/run_<run_id>.csv`. Readers glob the folder. This avoids concurrent appends to one file. |

**Idempotency strategy (no upserts).** At ~300K rows, Spark rebuilds every Silver and Gold table from Bronze in seconds, so the plan uses **full rebuilds rather than upserts**:

1. Bronze is append-only, one folder per batch; reloading a file that was already loaded is blocked by `meta_file_loads`.
2. Silver reads **all** Bronze batches and applies the deterministic dedup tie-break (§6.3), so backfills, repeated batches and late events all settle on the same result.
3. Gold is rebuilt in full from Silver.
4. Running the pipeline twice on the same Bronze produces byte-identical Silver and Gold (sorted output, tested).

**Known trade-offs of CSV (documented for interviews):** no type information in the file (handled by explicit schemas), larger files and slower reads than columnar formats, no column pruning, no transactions (handled by staging + rename), and empty string = null. The I/O module isolates these, so switching to Parquet or Delta is a later, contained change (Tier 2 option).

---

## 3. Data generation and source design

### 3.1 Time and calendar

| Item | Value |
|---|---|
| Business period | 2025-04-01 → 2025-09-30 (183 days) |
| Storage timezone | All timestamps stored in UTC, ISO-8601 with `Z` |
| Business date | Derived in `Asia/Kolkata`. Daily aggregates use the IST date. |
| Historical backfill slice | 2025-04-01 → 2025-06-30, delivered as monthly archive files |
| Routine batch slice | 2025-07-01 → 2025-09-23, delivered as daily files |
| Simulated stream slice | 2025-09-24 → 2025-09-30, delivered as micro-batch drops with ~3% late events (up to 6 h late) |
| ML split | Train 2025-04-01 → 2025-08-31; test 2025-09-01 → 2025-09-30 |

The window covers the monsoon months (Jun–Sep), which gives the rainfall feature a real effect to learn.

### 3.2 Geography

- **3 cities** (`city_id` C01–C03; e.g. Bengaluru, Mumbai, Delhi), **4 dark stores per city**, 12 stores in total.
- Weather is generated per city at a 3-hourly cadence and joined to stores via `city_id`.

### 3.3 Dataset inventory and relationships

| Dataset | Key fields | Business key | Relationships | Tier |
|---|---|---|---|---|
| categories | category_id, category_name | category_id | ← products | 1 |
| products | product_id, product_name, category_id, brand, price, popularity_rank* | product_id | → categories | 1 |
| stores | store_id, store_name, city_id, city, state, latitude, longitude | store_id | ← orders, inventory | 1 |
| customers | customer_id, name, email, city_id, signup_date | customer_id | ← orders, events | 1 |
| orders | order_id, customer_id, store_id, order_ts, status, total_amount | order_id | → customers, stores | 1 |
| order_items | order_item_id, order_id, product_id, quantity, unit_price | (order_id, product_id) | → orders, products | 1 |
| payments | payment_id, order_id, method, amount, status, attempt_no, payment_ts | payment_id | → orders | 1 |
| inventory_snapshots | snapshot_id, store_id, product_id, stock_quantity, reorder_level, snapshot_ts | (store_id, product_id, snapshot_ts) | → stores, products | 1 |
| inventory_events | event_id, store_id, product_id, event_type, quantity, event_ts | event_id | → stores, products | 1 |
| delivery_partners | partner_id, partner_name, city_id, availability_status | partner_id | ← deliveries | 1 |
| deliveries | delivery_id, order_id, partner_id, pickup_ts, delivered_ts, status | delivery_id | → orders, partners | 1 |
| cancellations | cancellation_id, order_id, reason, stage, cancelled_ts | cancellation_id | → orders | 1 |
| application_logs | log_id, service, log_level, event_ts, message | log_id | independent | 1 |
| promotions | promotion_id, name, discount_pct, category_id, start_date, end_date | promotion_id | ← order_promotions | 2 |
| order_promotions | order_id, promotion_id, discount_amount | (order_id, promotion_id) | → orders, promotions | 2 |
| returns_refunds | refund_id, payment_id, amount, status, refund_ts | refund_id | → payments | 2 |
| reviews | review_id, customer_id, product_id, rating, comment, review_ts | review_id | → customers, products | 2 |
| application_events | event_id, customer_id, event_name, event_ts, metadata (JSON string in a CSV column) | event_id | → customers (nullable) | 2 |
| weather | city_id, observation_ts, temperature_c, rainfall_mm, humidity_pct | (city_id, observation_ts) | → stores via city_id | 2 |

\* `popularity_rank` is used internally by the generator and is **not** written to source files.

**Tier 2 datasets are still generated from day one** (generation is cheap). Only their Silver/Gold processing is deferred.

Every dataset is documented in `docs/data_dictionary.md` with: column, type, nullability, business key, FKs, valid ranges, timestamp semantics, generation rules, injected issue types, and Bronze/Silver table names.

### 3.4 Volume targets (Medium profile)

| Dataset | Rows | Derivation |
|---|---:|---|
| categories | 20 | |
| products | 600 | ~30 per category |
| stores | 12 | 3 cities × 4 |
| customers | 6,000 | |
| delivery_partners | 120 | 10 per store |
| promotions | 60 | |
| orders | 36,000 | ≈ 16.4 orders / store / day |
| order_items | 122,400 | avg 3.4 lines per order |
| payments | 37,800 | 36,000 + ~5% retry/failed attempts |
| deliveries | 34,000 | orders minus pre-dispatch cancellations |
| cancellations | 2,520 | 7% of orders (2,000 pre-dispatch, 520 post-dispatch) |
| returns_refunds | 1,800 | ~5% of orders |
| order_promotions | 7,200 | ~20% of orders |
| reviews | 5,400 | ~15% of orders |
| inventory_snapshots | 7,800 | 300 focus pairs × 26 weekly snapshots |
| inventory_events | 12,000 | ~1.5 restock/adjust/damage events per focus pair per week |
| application_events | 15,000 | sampled app activity |
| weather | 4,392 | 3 cities × 183 days × 8 readings |
| application_logs | 5,000 | |
| **Total** | **298,124** | within the 200K–300K target |

Dirty-data injection then adds duplicate copies (§4.2), so the landing files hold **307,784 source rows** (298,124 + 9,660 duplicates), measured on the seed-42 medium run.

These are **generation targets**. Actual counts in each layer differ (dedup, quarantine, aggregation) and are reported only from real runs.

**Profiles**

| Profile | Multiplier | Use |
|---|---|---|
| small | ~0.05× (~15K rows; 3 stores, 30 days) | unit/integration tests, CI; must run end-to-end in < 3 min |
| medium | 1× (~298K rows) | the demo dataset |
| large | ~10× (~3M rows; 30 stores, 2,000 products) | performance comparisons only |

### 3.5 Density check (why these numbers work)

| Use case | Grain | Expected density |
|---|---|---|
| Demand forecasting (primary) | store × category × day: 12 × 20 × 183 = 43,920 cells | ~122K lines × ~1.4 qty ≈ 171K units → **~3.9 units per cell**, varying by Zipf weight |
| Stockout / SKU forecast (secondary) | 12 stores × 25 focus SKUs = 300 pairs × 183 days | focus SKUs carry ~30% of units → **~0.9 units per pair-day**. This is intermittent demand, handled with intermittent-demand baselines (§8.1). |
| Basket analysis | 36,000 baskets, avg 3.4 items | 25 planted affinity pairs. Measured on the medium run: support 0.03–0.50% (≈ 11–180 baskets), confidence 0.12–0.32, lift 8–74 (median 29). Organic pairs sit near lift 1. |
| Segmentation / recommendations | 6,000 customers, avg 6 orders | heavy-tailed: ~20% of customers with ≥ 12 orders |

### 3.6 Generator demand model

Generation runs in Python (NumPy, seeded) and writes source files. It does not use Spark. Steps:

1. **Catalog.** Assign Zipf(s≈1.1) popularity across products. Within each store, the top 25 products by popularity are the *focus SKUs* (tracked inventory).
2. **Customers.** Assign each customer one of **5 latent personas** (e.g. daily-essentials, weekend-stock-up, snacks-at-night, premium, deal-seeker). Each persona has its own order frequency, basket size, category mix, time-of-day profile and promo sensitivity. Persona labels go to the **ground-truth file only**.
3. **Daily demand.** For each store and day: `orders = base_store × dow_factor × (1 + rain_uplift·rain) × trend × promo_factor + noise`.
4. **Baskets.** Draw categories from the customer's persona mix and products by Zipf within category. With probability *p*, add the planted partner of an affinity pair.
5. **Inventory (focus pairs).** Simulate daily stock: sales deplete stock, restocks arrive 1–2 days after stock falls below `reorder_level`, and there is occasional damage or adjustment. **Sales are capped at available stock** (lost demand is recorded in ground truth). Planted stockout episodes: ~40.
6. **Anomalies.** Plant ~15 events: store outage (sales drop to ~0 for 2–6 h), viral spike for a product, a payment-gateway failure window (raised failed-payment rate plus an ERROR log burst).
7. **Downstream records.** Generate payments, deliveries, cancellations, refunds, reviews, events and logs consistent with the orders (§3.7).
8. **Dirty data.** Inject issues (§4).
9. **Manifest.** Write the manifest (§3.8) and the ground-truth file.

**Ground truth** (`data/generation/<run_id>/ground_truth/`) holds persona labels, planted affinity pairs, planted anomalies, planted stockouts and true lost demand. It is read **only by evaluation code and tests, never by the pipeline.**

### 3.7 Cross-dataset consistency rules (clean records)

| Rule | Detail |
|---|---|
| C1 | `order_ts` ≥ the customer's `signup_date`; customer `city_id` = store `city_id` for 97% of orders |
| C2 | `orders.total_amount` = Σ(quantity × unit_price) − Σ(order_promotions.discount_amount), to 2 decimals |
| C3 | Each order has ≥ 1 payment attempt; at most one attempt has `status = success`; failed attempts precede the success |
| C4 | Pre-dispatch cancellation → no delivery record, payment refunded or voided |
| C5 | Post-dispatch cancellation → delivery `status ∈ {cancelled, failed}` |
| C6 | Delivered orders: `order_ts < pickup_ts < delivered_ts`, with delivery duration ~ lognormal (median ~14 min) |
| C7 | `refund.amount` ≤ the successful payment amount; `refund_ts` > `payment_ts` |
| C8 | `order_promotions` only reference promotions active on the order date and eligible for the order's categories |
| C9 | Reviews only for products the customer actually bought, with `review_ts` > delivery time |
| C10 | `unit_price` = product price at order time (prices may change at most twice in the period) |
| C11 | Inventory: snapshot stock = simulated stock at the snapshot time |

### 3.8 Generation manifest

Each run writes the manifest as three CSV files in `data/generation/<run_id>/manifest/`:

- `manifest_run.csv` (one row): run_id, timestamp, generator version, git commit, seed, profile, date range start/end, config file hash.
- `manifest_datasets.csv` (one row per dataset): dataset, configured target, **actual count**, output file, schema version.
- `manifest_issues.csv` (one row per dataset × issue type): configured rate, **actual injected count**.

Ground truth is also CSV: `gt_personas.csv`, `gt_affinity_pairs.csv`, `gt_anomalies.csv`, `gt_stockouts.csv`.

### 3.9 Ingestion patterns

| Pattern | Input | Mechanism | Demonstrates |
|---|---|---|---|
| A. Batch | daily CSV drops (Jul–Sep) | `spark.read.csv` with explicit all-string schema → appended to a new Bronze batch folder, keyed by `_source_file` | load tracking, idempotent re-ingest, header/column-count mismatch capture |
| B. Historical backfill | monthly archive CSVs (Apr–Jun) | same reader, `_load_type = 'historical'`, separate batch ID | backfill isolation, preserved event dates |
| C. Simulated stream | micro-batch CSV drops into `data/landing/stream/` | **Spark Structured Streaming** CSV file source (`readStream.schema(...).csv(...)`), up to 24 hourly drops per micro-batch, `trigger(availableNow=True)`, checkpointed, written with `foreachBatch` to Bronze CSV. Bronze never drops rows: each row gets `_is_late` (event before its file's hour) and `_beyond_watermark` (older than a **2-hour** watermark, deliberately shorter than the 6-hour maximum lateness to show the trade-off) | event time vs ingestion time, late data, watermarking as real Spark behaviour |
| D. Logs | application-log CSVs with malformed rows (wrong column count, broken quoting, unparseable timestamps) | `mode=PERMISSIVE` + `columnNameOfCorruptRecord='_corrupt_record'` | malformed-record capture, level normalisation |

**Idempotency:** the load-tracking table `meta_file_loads` (CSV) records each file's path, size and SHA-256. A file that has already loaded is skipped unless `--force-reload` is passed.

---

## 4. Controlled dirty data

### 4.1 Issue catalog

| Issue | Example | Silver treatment |
|---|---|---|
| Exact duplicate | same order row twice | dedupe by business key |
| Conflicting duplicate | same order_id, different status | dedupe by tie-break rule (§6.3) |
| Missing optional value | null review comment | keep null |
| Missing required value | null order_id | quarantine |
| Invalid format | malformed email or timestamp | email: set to null and flag; timestamp on a required field: quarantine |
| Inconsistent casing/whitespace | `" Delivered"`, `DELIVERED` | normalise to the canonical value |
| Invalid numeric | negative quantity or price | quarantine |
| Broken FK | order item → unknown product | quarantine |
| Invalid temporal order | delivered_ts < pickup_ts | quarantine the delivery record |
| Out of range | rating = 8 | quarantine |
| Malformed JSON column | broken `metadata` JSON string | Bronze keeps the raw string; Silver quarantines |
| Malformed CSV row | wrong column count or broken quoting | Bronze keeps the line in `_corrupt_record`; Silver quarantines |
| Late event | stream event arrives 4 h late | processed by event time, flagged `_is_late` |
| Duplicate snapshot | same store/product/timestamp twice | snapshot tie-break |
| Unusual but valid | order of ₹25,000 | keep and flag `is_outlier` |

### 4.2 Dirty-data budget (90% clean)

**Unit:** every rate is a **percentage of the rows in one source dataset**. Rates are not mixed with "% of fields" or "% of entities".

**Assignment:** each dirty row gets **exactly one** primary issue (mutually exclusive), so every dataset is exactly **90% clean rows, 10% dirty rows**. The manifest records the exact counts.

| Code | Issue | Budget | Example | Silver treatment |
|---|---|---:|---|---|
| DUP | Duplicate records (half exact, half conflicting) | 2.50% | same order row twice; same order_id with a different status | dedupe by business key + tie-break (§6.3) |
| MISS | Missing **required** value | 2.00% | null order_id, null delivered_ts on a delivered order | quarantine (fill only where a documented rule justifies it) |
| FMT | Inconsistent format | 1.50% | `DELIVERED` / `" delivered"`; `05/04/2025 14:03` instead of ISO | normalise |
| NUM | Invalid numeric / out of range | 1.00% | negative price, quantity 0, rating 8, humidity 140 | quarantine |
| BIZ | Business-rule violation | 1.00% | order total ≠ sum of lines − discount; refund > payment; partner from another city | apply the documented rule (flag as ineligible or quarantine) |
| TYPE | Invalid data type | 0.75% | `"two"` in quantity; `"N/A"` in price | the cast fails, so quarantine |
| FK | Referential-integrity error | 0.75% | order item → nonexistent product | quarantine |
| SEQ | Invalid temporal sequence | 0.50% | delivered_ts < pickup_ts; promotion end < start | quarantine |
| | **Total dirty** | **10.00%** | | |

**Dataset-specific substitute:**

| Code | Issue | Applies to | Example | Treatment |
|---|---|---|---|---|
| MALF | Malformed payload | application_logs (broken CSV row), application_events (invalid JSON in `metadata`) | extra fields or a truncated row; `{"screen": "cart"` | Bronze keeps raw content in `_corrupt_record` / the raw string; Silver routes it to quarantine |

Every injected issue is contained in a single row, so it can be detected from that row (plus lookups) without changing other datasets. **Conflicting duplicates** are re-sent copies of a record that differ only in formatting (casing, whitespace, `149.0` vs `149.00`, `+00:00` vs `Z`). After Silver standardisation they become exact duplicates, so deduplication gives one unambiguous result.

**Outside the budget.** These are valid data or timing behaviour, not dirty rows, so they do **not** reduce the 90%:

| Behaviour | Rate | Treatment |
|---|---|---|
| Late-arriving records | ~3% of simulated-stream rows; ~1% of batch rows arrive in the next day's file | event-time processing, watermark, `_is_late` flag |
| Out-of-order arrival | ~2% of stream rows arrive after a newer event | ordering by event time |
| Missing **optional** values | per field, e.g. review comment 30%, customer email 3% | keep null |
| Unusual but valid values | ~0.2% (very large orders) | keep and flag `is_outlier` |

#### Applicability per dataset

Not every issue fits every dataset (for example, SEQ needs two timestamps). Each dataset keeps the 10% total by **redistributing the budget of issue types that don't apply, proportionally**, across the ones that do.

**Dimension tables only get correctable issues** (DUP, FMT). If 10% of stores or products were quarantined, every order pointing to them would fail its FK check. Rejections would cascade far past the planned budget, and broken relationships would no longer come only from deliberately injected FK errors.

| Dataset | DUP | MISS | FMT | NUM | BIZ | TYPE | FK | SEQ | MALF |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| categories, stores, products, customers, delivery_partners | ✓ | | ✓ | | | | | | |
| orders | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| order_items | ✓ | ✓ | | ✓ | ✓ | ✓ | ✓ | | |
| payments | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| deliveries | ✓ | ✓ | ✓ | | ✓ | | ✓ | ✓ | |
| cancellations | ✓ | ✓ | ✓ | | | | ✓ | ✓ | |
| inventory_snapshots | ✓ | ✓ | | ✓ | | ✓ | ✓ | | |
| inventory_events | ✓ | ✓ | ✓ | ✓ | | | ✓ | | |
| promotions | ✓ | ✓ | ✓ | ✓ | | | | ✓ | |
| order_promotions | ✓ | | | ✓ | ✓ | | ✓ | | |
| returns_refunds | ✓ | ✓ | | | ✓ | | ✓ | ✓ | |
| reviews | ✓ | | ✓ | ✓ | ✓ | | ✓ | | |
| application_events | ✓ | ✓ | ✓ | | | | ✓ | | ✓ |
| weather | ✓ | ✓ | ✓ | ✓ | | ✓ | | | |
| application_logs | ✓ | ✓ | ✓ | | | | | | ✓ |

#### Direct vs cascaded rejections

Child records are FK-checked against the **accepted Silver parents**. If an order is quarantined (e.g. for MISS), its order items, payments and delivery fail their FK checks too, even though they were generated clean. Silver therefore reports two numbers per dataset:

- `rejected_direct`: rows that broke a rule themselves (≈ the injected budget)
- `rejected_cascade`: rows rejected only because a parent was rejected

As a result, the **actual** Silver rejection rate for child tables will be somewhat above the injected rate. That is expected and is explained in the Data Quality page.

All rates are configurable in `configs/dirty_data.yaml`.

### 4.3 Rules for the generator

- Clean data is generated first, then mutated deterministically from the seed.
- Each issue type can be turned on or off.
- Injected-issue metadata goes to the manifest only. Source rows never carry a "dirty" marker.
- The pipeline detects issues with real validation rules. Tests then compare detected against injected counts.

### 4.4 Before/after inspection

For any Silver transformation, the Data Engineer workspace shows the Bronze rows, the rule applied, affected rows, the Silver result, quarantined rows with their failed rules, before/after counts, and run-level quality metrics.

---

## 5. Bronze layer

**Objectives:** preserve source values, keep dirty data, add ingestion metadata, track files and batches.

**Tables:** `brz_<dataset>` for all 19 datasets, stored as CSV folders partitioned by `_batch_id=<id>/`.

**Storage choice:** Bronze reads with an explicit schema in which **every source column is a string**. Original values (including malformed numbers and timestamps) are kept exactly. Typing happens in Silver.

| Metadata column | Meaning |
|---|---|
| `_ingestion_ts` | when the row entered Bronze (UTC) |
| `_ingestion_date` | IST date of ingestion |
| `_batch_id` | ingestion batch |
| `_pipeline_run_id` | overall run |
| `_source_file` | input path |
| `_source_system` | logical source (e.g. `oms`, `wms`, `app`) |
| `_load_type` | `batch` / `historical` / `stream` / `log` |
| `_schema_version` | source schema version |
| `_record_hash` | SHA-256 of the source columns |
| `_corrupt_record` | raw line when the CSV row could not be parsed |

**Rules:** no normalisation or repair; file-level errors (missing file, header mismatch) are logged to `meta_file_loads` separately from row-level quality issues.

**Done when:** every dataset has an ingestion path, metadata is present, re-ingesting the same file creates no duplicates (tested), and counts per file and batch appear in the Data Engineer UI.

---

## 6. Silver layer

### 6.1 Objectives

Cast types, standardise, deduplicate, validate (required, range, format, FK, temporal), quarantine with reasons, record quality metrics, handle incremental and late data.

### 6.2 Transformation catalog

| Transformation | Example |
|---|---|
| Schema enforcement / casting | `quantity` → int, `amount` → decimal(12,2), timestamps → timestamp (UTC) |
| Column standardisation | rename inconsistent headers via a mapping config |
| Trim and case normalisation | statuses mapped to canonical lowercase enums |
| Null handling | required → quarantine; optional → keep |
| Deduplication | latest record per business key (§6.3) |
| Range / format / FK / temporal validation | rules declared in `src/quality/rules.py` |
| Derived columns | `line_amount`, `delivery_minutes`, `business_date_ist` |
| Quarantine | invalid rows to `qtn_records` |
| Late-data handling | full rebuild over all Bronze batches; event-time ordering |

### 6.3 Determinism rules

- **Deduplication tie-break:** within a business key, keep the row with the latest `_ingestion_ts`. If tied, keep the row with the greatest source timestamp (e.g. `order_ts`, `event_ts`). If still tied, keep the lexicographically greatest `_record_hash`.
- **Snapshot tie-break:** within (store, product, snapshot_ts), keep the latest `_ingestion_ts`, then the max `_record_hash`.
- **Writes:** each Silver table is **rebuilt in full** from all Bronze batches and swapped in atomically (§2.1). Output is sorted by business key, so reprocessing the same Bronze produces identical CSVs (idempotency test).
- **History:** entities (customers, products, stores) are SCD Type 1 (latest version wins) in Tier 1; SCD Type 2 for product price is a Tier 2 option. Facts (orders, events, snapshots) are all kept; a later fact never replaces an earlier one with a different key.

### 6.4 Dataset-specific rules (summary)

| Dataset | Key rules |
|---|---|
| customers | trim; email regex (invalid → null + flag); signup_date parse; required customer_id |
| products / categories | price ≥ 0 and ≤ configured max; category FK; normalise names |
| stores | lat/long ranges; city_id FK to the city config |
| orders | status enum; customer/store FK; total_amount ≥ 0; `is_completed` flag from the shared eligibility rule |
| order_items | quantity ≥ 1; unit_price ≥ 0; order/product FK; `line_amount` computed only after validation |
| payments | method/status enums; order FK; multiple attempts per order allowed; `is_success` |
| inventory_snapshots | stock ≥ 0 (negative → quarantine); FK; snapshot tie-break; history kept |
| inventory_events | event_type enum (`restock`, `adjustment`, `damage`); quantity sign consistent with type; late events by event_ts |
| deliveries | status enum; temporal ordering; `delivery_minutes` only when valid; > 120 min flagged |
| cancellations / refunds | order/payment FK; refund ≤ payment amount; stage enum |
| reviews | rating 1–5; FK; a missing comment is valid |
| application_events | parse the JSON-string `metadata` column with `from_json` (malformed → quarantine); customer FK when not null |
| weather | numeric ranges; city_id FK; one reading per (city, ts) |
| application_logs | level normalisation (`warn` → `WARN`, etc.); `_corrupt_record` → quarantine |

**Shared eligibility rule (one definition, used everywhere):**
`is_completed = status = 'delivered' AND has a successful payment AND NOT cancelled`.

### 6.5 Quarantine design

One standard CSV table, `qtn_records`, with **one row per rejected source record**:

| Column | Description |
|---|---|
| source_dataset | e.g. `orders` |
| source_record_id | business key if parseable |
| batch_id, pipeline_run_id | traceability |
| rejected_at | UTC timestamp |
| failed_rules | pipe-separated list of every rule that failed (e.g. `qty_positive\|fk_product`) |
| primary_rule | the highest-severity failed rule |
| severity | `critical` / `error` |
| raw_record | original Bronze row as a JSON string (quoted CSV column) |

Rule-level counts are computed with `explode(split(failed_rules, '\\|'))`. This keeps record counts and rule counts both correct.

### 6.6 Quality metrics (per dataset, per run)

Rows in, rows valid, rows quarantined (split into `rejected_direct` and `rejected_cascade`, §4.2), rows deduplicated, values standardised, null counts on key columns, failures per rule, FK failures, and pass rate (`valid / (rows_in − duplicates_removed)`). Written to `meta_quality_results`.

The **actual** rejection rate is shown next to the **configured** injection rate. These differ on purpose (corrected issues, valid nulls, overlapping issues), and the UI explains why.

### 6.7 Incremental and late data

| Question | Answer |
|---|---|
| How are new records detected? | new files via `meta_file_loads` become new Bronze batch folders; Silver always reads all batches |
| Repeated files? | skipped by hash unless forced |
| Update vs version? | dedup tie-break keeps the latest version per business key (SCD1) |
| Late inventory events? | placed at their `event_ts` during the full Gold rebuild, so daily inventory state is correct for every affected date |
| Backfills? | historical batches are just more Bronze folders; the rebuild plus dedup makes overlapping keys converge |

**Done when:** every Tier 1 dataset has a documented schema, business key and rules; dedup is deterministic; quarantine has reasons; metrics are saved; reprocessing is idempotent (tested); and late-event handling is tested.

---

## 7. Gold layer

### 7.1 Tables

| Table | Grain | Purpose | Tier |
|---|---|---|---|
| gld_daily_sales | business_date × store | revenue, orders, AOV, units, customers | 1 |
| gld_daily_category_sales | business_date × store × category | **primary forecast series**; zero-filled | 1 |
| gld_product_performance | product × month | units, revenue, order count, ASP | 1 |
| gld_inventory_daily | business_date × store × focus SKU | end-of-day stock, sales, stockout flag, restocks | 1 |
| gld_demand_features | business_date × store × category (and focus-SKU variant) | lag/rolling/calendar/weather features | 1 |
| gld_customer_360 | customer | RFM, category mix, cancellations, promo use | 1 |
| gld_basket_pairs | product pair | co-occurrence counts and support | 1 |
| gld_quality_summary | run × dataset × rule | quality overview for the engineering UI | 1 |
| gld_demand_predictions | forecast_date × store × category | ML output | 1 |
| gld_stockout_risk | as_of_date × store × focus SKU | risk tier and reason codes | 1 |
| gld_replenishment | as_of_date × store × focus SKU | suggested quantity and assumptions | 1 |
| gld_customer_segments | customer × segment_run | cluster assignment | 1 |
| gld_basket_rules | antecedent → consequent | support, confidence, lift | 1 |
| gld_store_performance | store × week | store comparisons | 2 |
| gld_delivery_metrics | store × day | delivery-time percentiles, failure rate | 2 |
| gld_cancellation_metrics | store × reason × week | cancellation trends | 2 |
| gld_promotion_metrics | promotion | uplift vs baseline, discount cost | 2 |
| gld_recommendations | customer × rank | ranked products | 2 |
| gld_sales_anomalies | store × hour/day | anomaly signals | 2 |

Gold tables are fully rebuilt from Silver on each run and swapped in atomically (§2.1), so reruns are idempotent.

### 7.2 Metric definitions (canonical)

| Metric | Definition |
|---|---|
| Gross Merchandise Value (GMV) | Σ `line_amount` over completed orders (before discounts) |
| Net Revenue | GMV − order discounts − completed refunds |
| Completed Orders | count of orders where `is_completed` |
| Average Order Value | **AOV = Net Revenue from completed orders ÷ Number of completed orders** |
| Cancellation Rate | cancelled orders ÷ all orders placed |
| Days of Inventory | **DOI = Current usable stock ÷ Average daily demand (trailing 14 days)**. If average demand is 0, DOI is shown as "No recent demand", never ∞. |
| Stockout day | a focus pair with end-of-day stock = 0, or sales capped by stock |
| On-time delivery | `delivery_minutes` ≤ promised SLA (default 15 min) |

All pages read these definitions from `src/serving/metrics.py`. No page computes its own version.

### 7.3 Demand features (leakage-safe)

Features for date *t* use only data up to *t − 1*. They are lag 1/7/14, rolling mean/std over 7 and 14 days (`rowsBetween(-14, -1)`), day of week, weekend flag, month, Indian public-holiday flag, active promotion and discount % (Tier 2), daily rainfall and mean temperature by city, store and category encodings, and a stockout-on-previous-day flag.

Missing dates are **zero-filled** using a calendar × store × category spine. Days where sales were capped by stock are flagged (censored demand) so they can be excluded or down-weighted in training.

### 7.4 Basket preparation

A basket is one completed order. Distinct products are taken per basket. Pairs are generated in Spark with a self-join on `order_id` where `product_a < product_b`. Baskets with more than 15 distinct products are capped (documented). Pairs with count ≥ 8 are kept (the weakest planted pair appears in ~11 baskets). Only this bounded pair table (a few thousand rows) is collected for mlxtend/rule generation.

**Done when:** each table has a documented grain, metrics match hand-computed fixtures on the small profile, features pass the leakage test, and Gold output counts and lineage are recorded.

---

## 8. Intelligence layer

Each use case documents: persona, question, inputs, method, baseline, evaluation (including **recovery of planted ground truth** where applicable), outputs, and UI.

### 8.1 Demand forecasting — Tier 1
*Persona: Inventory & Supply Chain. Question: what demand should we expect per store and category over the next 7 days?*

- **Series:** `gld_daily_category_sales` (primary); focus-SKU × store (secondary).
- **Baselines:** seasonal naïve (same weekday last week) and trailing 7-day mean. For intermittent SKU series, also a Croston/SBA-style estimate.
- **Models:** one global `HistGradientBoostingRegressor` across all series, trained on the §7.3 features. Recursive 7-day horizon.
- **Evaluation:** chronological hold-out (September), with MAE, RMSE and **WAPE** reported per category and overall, and compared to both baselines. *The model is only presented as useful if it beats the baselines on WAPE.*
- **Outputs:** forecast, horizon day, model version, generated_at, eval context. Prediction intervals come from quantile loss (`loss='quantile'`, α = 0.1/0.9) in Tier 2.
- **UI:** history vs forecast chart, store/category filters, horizon selector (1–7), metric table vs baseline, feature importance (permutation), limitations note.

### 8.2 Stockout risk and replenishment — Tier 1 (rules)
*Question: which focus SKUs may run out, and what should be reordered first?*

- **Transparent rules first.** Signals: stock ≤ reorder_level; stock = 0; DOI < lead time + 1; 7-day forecast demand > usable stock. Risk tiers: **High / Medium / Low**, each with reason codes.
- **Replenishment:** `max(0, forecast_demand(horizon = lead_time + review_period) + safety_stock − usable_stock)`, where `safety_stock = z × σ_daily_demand × √(lead_time)`, z = 1.65, lead time = 1–2 days (a **stated assumption**, matching the generator).
- **Evaluation:** backtest over September. Measure precision/recall of "High risk" against actual stockouts in the next 3 days and against **planted stockout episodes**.
- **Tier 2:** a classifier (logistic regression / gradient boosting) on the same signals, used only if it beats the rules in the backtest.
- **UI:** high-risk list, stock vs demand chart, suggested quantities with reasons, filters, and an "advisory only" disclaimer.

### 8.3 Customer segmentation — Tier 1
*Persona: Growth & Marketing. Question: which customers behave alike?*

- **Features:** recency, frequency, monetary (log-transformed), AOV, category-share vector (top 8 categories), night-order share, promo-use share, cancellation rate.
- **Method:** RobustScaler → K-means for k = 3–8. Pick k by silhouette plus interpretability. Name segments only after profiling.
- **Evaluation:** silhouette, cluster sizes (no cluster < 3%), stability across 5 seeds (adjusted Rand index), and **ARI against the 5 planted personas** (a check unique to synthetic data that segmentation recovers real structure).
- **UI:** segment distribution, profile comparison (original-scale metrics), customer drill-down (role-restricted), campaign ideas with stated assumptions.

### 8.4 Basket analysis — Tier 1
*Question: which products are bought together?*

- **Definitions** (corrected):
  - **Support(A→B) = Baskets containing both A and B ÷ Total eligible baskets**
  - **Confidence(A→B) = Baskets containing both A and B ÷ Baskets containing A**
  - **Lift(A→B) = Confidence(A→B) ÷ Support(B)**
- **Thresholds:** pair count ≥ 8 (support ≈ 0.02%), confidence ≥ 10%, lift ≥ 2 (configurable). The low support floor is deliberate: with 600 products and ~3.4 items per basket, even strong real associations have small support.
- **Evaluation:** recall of the 25 planted pairs in the top-50 rules by lift; spot-check example baskets.
- **UI:** top rules, product search, category-level rules, example baskets, plain-language interpretation.

### 8.5 Product recommendations — Tier 2
- **Methods:** global popularity (baseline), segment popularity, item-to-item co-purchase (from `gld_basket_pairs`), and a hybrid weighted rank excluding recently bought items.
- **Evaluation:** chronological split (last 30 days as hold-out) with Precision@10, Recall@10 and catalog coverage, each compared to popularity.
- **Outputs:** customer, product, rank, method, supporting signal.

### 8.6 Sales anomaly detection — Tier 2
- **Baseline:** robust z-score of hourly/daily store sales against the same-weekday median ± MAD over the trailing 4 weeks. **Optional:** Isolation Forest on residuals.
- **Evaluation:** precision/recall against the **15 planted anomalies**.
- **Outputs:** store, period, actual, expected, deviation, score, context (promo, rain, holiday), priority.
- An anomaly is a **signal to investigate**, never evidence of fraud.

### 8.7 Model registry

`meta_model_runs` (CSV) records method, version, run ID, input table and the pipeline run that produced it, training window, split, parameters, metrics, baseline metrics, ground-truth recovery metrics, output table, created_at, status and known limitations. Model artifacts are saved with `joblib` under `data/ml/models/<model>/<version>/`.

---

## 9. RAG business assistant

### 9.1 Principles

- Exact numbers come **only** from controlled structured functions. Retrieval never estimates numbers.
- The assistant works with **no LLM** (default). An LLM is an optional enhancement for answer phrasing.

### 9.2 Architecture

```
Question
  └─► Router (Tier 1: keyword/intent rules; Tier 2: optional LLM classifier)
        ├─ structured  → whitelisted query function(s) → DuckDB over published Gold
        ├─ policy      → BM25 retrieval over policy chunks (threshold + top-k)
        └─ hybrid      → both
  └─► Answer composer
        ├─ no-LLM mode: template answer (metrics table + quoted policy passages + sources)
        └─ LLM mode:    LLM writes from the evidence only; must cite; must say when evidence is insufficient
```

### 9.3 Documents

Five policies in `policies/`: inventory, cancellation, delivery, promotion, refund. They are written as Markdown with numbered sections and concrete thresholds that **match the generator's assumptions** (e.g. reorder at DOI < 2 days, 15-minute delivery SLA).

Chunking is by section heading, then about 120–200 words with a 30-word overlap. Each chunk keeps `doc_id`, title, section number and chunk_id. The index is rebuilt when a policy file's hash changes.

### 9.4 Retrieval

- **Default:** BM25 (`rank_bm25`). Fewer than 200 chunks, so no vector database is needed.
- **Optional (Tier 2):** `sentence-transformers/all-MiniLM-L6-v2` embeddings with NumPy cosine similarity, compared to BM25 on the evaluation set.
- Minimum score threshold, top-k = 3, source metadata always attached. The UI shows which retrieval method was used.

### 9.5 Structured functions (whitelist)

`get_daily_revenue(store?, start, end)`, `get_top_products(store?, category?, start, end, k ≤ 20)`, `get_inventory_risks(store?, tier?)`, `get_replenishment(store?)`, `get_segment_summary()`, `get_basket_rules(product?, k)`, plus Tier 2: `get_delivery_performance`, `get_sales_anomalies`.

Every function validates its parameters (Pydantic), enforces date bounds within the data range, checks the caller's role, returns a typed result with `source_table` and `as_of`, and never returns customer PII.

### 9.6 Routing examples

| Type | Example | Behaviour |
|---|---|---|
| Structured | "What was revenue at store S03 last week?" | `get_daily_revenue` |
| Policy | "What does the inventory policy say about low stock?" | BM25 → inventory policy §x |
| Hybrid | "Which products are at high stockout risk and what does policy recommend?" | `get_inventory_risks` + inventory policy |
| Out of scope | "What's the weather tomorrow?" | refuse politely; explain scope |
| Insufficient | "Why did sales drop?" | report the observed change and available context; no causal claim |

### 9.7 Evaluation

A 40-question gold set (`tests/rag/questions.yaml`): 10 policy, 10 structured, 10 hybrid, 5 ambiguous, 5 out-of-scope. Metrics are routing accuracy, retrieval hit@3, **exact numerical match** for structured questions, citation correctness, and correct refusal on insufficient or out-of-scope questions. Each Q&A is logged to `meta_rag_runs` with latency.

### 9.8 LLM handling (optional)

A provider-agnostic `LLMClient` interface. The API key comes from Streamlit secrets or an env var. If the key is missing or the call fails, the assistant falls back silently to no-LLM mode and the UI shows the active mode.

---

## 10. Streamlit application

### 10.1 Roles

| Role | Workspace |
|---|---|
| Inventory & Supply Chain Manager | inventory, forecasting, stockout, replenishment |
| Growth & Marketing Manager | segments, customer insights, recommendations, affinity |
| Business & Revenue Analyst | executive overview, sales, anomalies, basket analysis |
| Data Engineer | pipeline, tables, quality, lineage, runs, ML/RAG monitoring |
| Demo admin | all (only if enabled in config) |

**Auth (demo-grade):** usernames, bcrypt password hashes and roles live in `st.secrets`. Authorization is checked in `src/serving/permissions.py` **inside every data-access function**, not just in navigation. Documented as not production security. Streamlit's built-in OIDC `st.login` is an optional upgrade.

### 10.2 Pages by tier

| Workspace | Tier 1 pages | Tier 2 pages |
|---|---|---|
| Inventory | Overview, Demand Forecasting, Stockout & Replenishment | Inventory Explorer |
| Marketing | Customer Segments | Customer Insights, Recommendations, Product Affinity |
| Business | Executive Overview, Basket Analysis | Sales Analytics, Anomaly Detection |
| Data Engineer | Pipeline Overview, Table Explorer (with lineage), Data Quality (with quarantine drill-down), Execution History | Transformations catalog, ML & RAG Monitoring |
| All | Business Assistant (shared component) | |

Tier 1 has **11 pages**, against v1's ~27 required pages.

### 10.3 Shared UI rules

Consistent page header with data freshness ("Snapshot from run `<run_id>`, generated `<ts>`, **precomputed**"). KPI cards with tooltips linking to metric definitions. Plotly charts only where they answer the page's question. Tables paginated or limited (≤ 500 rows). Helpful empty and error states. All queries go through DuckDB with filters pushed down, never loading full tables into memory.

### 10.4 Data Engineer workspace

- **Pipeline Overview:** stage status (green only if that stage succeeded in the latest run), last successful run, row counts per layer, rejected and deduplicated counts.
- **Table Explorer:** choose layer and table → schema, actual row count, sample (≤ 100 rows), null/distinct profile for chosen columns, upstream/downstream tables (from `meta_lineage_edges`), and the run that last wrote it.
- **Data Quality:** pass rate per dataset (definition shown), failures by rule, configured vs actual issue rates, trend across runs, quarantine drill-down with `failed_rules` and `raw_record`.
- **Execution History:** runs → stages → tables with rows read/written/rejected/deduplicated, duration, errors, environment.

Every panel is labelled **Actual** (from run metadata), **Declared** (from the catalog) or **Illustrative**.

### 10.5 Lineage capture

Lineage edges are **emitted by code**, not drawn by hand: each transformation is wrapped in `@tracked_transform(inputs=[...], output=...)`, which writes an edge to `meta_lineage_edges` and a row to `meta_table_runs` at runtime. The lineage view therefore reflects only what actually ran.

---

## 11. Orchestration

### 11.1 Stages

| # | Stage | Output |
|---|---|---|
| 0 | config | validated run config |
| 1 | generate | source files, manifest, ground truth |
| 2 | ingest | Bronze tables, `meta_file_loads` |
| 3 | silver | Silver + quarantine + quality results |
| 4 | gold | Gold tables |
| 5 | ml | predictions, segments, rules, risk, model runs |
| 6 | rag_index | chunk index |
| 7 | validate | data/ML/RAG checks |
| 8 | publish | `data/demo/` CSV snapshot + metadata |

### 11.2 CLI

```bash
python -m scripts.run_pipeline --profile medium                 # full run
python -m scripts.run_pipeline --profile small --stages silver,gold
python -m scripts.run_pipeline --backfill 2025-04-01:2025-06-30
python -m scripts.run_pipeline --validate-only
python -m scripts.run_pipeline --publish-only                   # demo refresh
```

### 11.3 Failure handling

Each stage runs in a try/except that records status (`success` / `failed` / `skipped`), error message and traceback reference. Downstream stages are `skipped` when an upstream stage fails. **Publish is atomic:** write to `data/demo/_staging/`, validate, then swap. A failed run never replaces the current snapshot.

---

## 12. Metadata and observability

### 12.1 Metadata tables (CSV, `data/metadata/`, one file per run per table)

`meta_pipeline_runs`, `meta_stage_runs`, `meta_table_runs`, `meta_file_loads`, `meta_layer_progress`, `meta_lineage_edges`, `meta_quality_results`, `meta_model_runs`, `meta_rag_runs`, and `meta_transform_catalog` (declared).

### 12.2 Execution record fields

run_id, stage, job, source_table(s), target_table, engine (`spark` / `python` / `duckdb`), module, started_at, ended_at, duration_s, status, rows_read, rows_written, rows_rejected, rows_deduplicated, error_message, environment (host, Spark/Python versions), profile, git commit.

Unavailable values stay **null**, never invented.

### 12.3 Logging

Structured pipeline logs are written as CSV (`timestamp, level, run_id, stage, module, message, exc_info`) to `logs/<run_id>.csv`, so the Data Engineer workspace can query them like any other table. No secrets or PII.

---

## 13. Repository structure

```
quick-commerce-intelligence/
├── README.md                    # short: what, how to run, screenshots, limitations
├── pyproject.toml               # deps split: [pipeline] (pyspark, delta) vs [app] (streamlit, duckdb)
├── requirements.txt             # app-only deps for Streamlit Cloud (no pyspark)
├── .env.example
├── .devcontainer/               # Java 17 + Python 3.11 + Spark
├── configs/  default.yaml small.yaml medium.yaml large.yaml dirty_data.yaml cities.yaml
├── policies/ inventory.md cancellation.md delivery.md promotion.md refund.md
├── docs/     architecture.md data_dictionary.md transformation_catalog.md
│             metric_definitions.md ml_design.md rag_design.md deployment.md demo_walkthrough.md
├── src/
│   ├── common/          config.py spark.py logging.py paths.py schemas.py
│   ├── generation/      generate_all.py catalog.py customers.py orders.py inventory.py
│   │                    downstream.py weather.py logs.py dirty_data.py manifest.py
│   ├── ingestion/       batch.py historical.py stream.py logs.py file_tracker.py
│   ├── quality/         rules.py validate.py quarantine.py metrics.py
│   ├── transformations/ silver/<dataset>.py   gold/<table>.py
│   ├── ml/              forecasting/ stockout/ segmentation/ basket/ recommendations/ anomaly/
│   │                    evaluation.py registry.py
│   ├── rag/             loader.py chunking.py retrieval.py router.py functions.py composer.py llm.py
│   ├── orchestration/   pipeline.py stages.py tracking.py lineage.py publish.py
│   └── serving/         db.py (DuckDB) metrics.py permissions.py queries.py
├── app/
│   ├── Home.py
│   ├── pages/           1_Inventory.py 2_Marketing.py 3_Business.py 4_Data_Engineer.py
│   └── components/      kpi.py charts.py assistant.py auth.py freshness.py
├── scripts/             run_pipeline.py validate_outputs.py
├── data/                # git-ignored except data/demo/
│   ├── landing/ bronze/ silver/ quarantine/ gold/ ml/ rag/ metadata/ generation/
│   └── demo/            # published CSV snapshot, one file per table, committed, < 50 MB
└── tests/               unit/ integration/ quality/ ml/ rag/ app/ e2e/
```

**Dependency split:** the hosted app installs only `streamlit, duckdb, pandas, plotly, rank_bm25, bcrypt, pydantic`. PySpark is pipeline-only, which keeps the cloud build small and fast.

---

## 14. Environments and deployment

### 14.1 Local

- **Decision (2026-10-04): native Windows.** Python 3.11 venv + PySpark 3.5.9 + Java 17, with `winutils.exe`/`hadoop.dll` (Hadoop 3.3.6 community build) in a git-ignored `tools/hadoop/bin`. `src/common/spark.py` handles `HADOOP_HOME`, the worker Python, and 8.3 short paths, because Spark's `.cmd` launchers break on paths with spaces. A Spark smoke test guards the setup. WSL2 or a dev container is still the fallback if Windows-specific problems appear.
- `make test` (small profile), `make demo` (medium profile + publish), `make app`.

### 14.2 Streamlit Community Cloud

- Entry point `app/Home.py`; `requirements.txt` holds app-only deps.
- Reads `data/demo/*.csv` committed to the repo. Target < 50 MB in total and **< 100 MB per file** (GitHub's hard limit). Gold aggregates are small; Silver samples are capped at 5,000 rows per table. DuckDB queries the CSVs with explicit column types (`read_csv(..., columns={...})`), never auto-detection.
- `@st.cache_resource` for the DuckDB connection and BM25 index; `@st.cache_data` for query results.
- Secrets: user and role table, optional LLM key.
- Every page shows "Precomputed snapshot from run X at time Y". It never claims live processing.
- The free tier has limited memory and sleeps when idle. Keep the snapshot small and verify current platform limits before deployment.

### 14.3 CI (GitHub Actions)

On every push: set up Java 17 and Python, install the pipeline deps, and run `pytest` on the small profile (target < 5 min), plus `ruff` lint.

---

## 15. Testing

| Level | Examples |
|---|---|
| Unit | dirty-data injectors; each validation rule; dedup tie-break; metric functions on fixtures; feature lags (no leakage); BM25 ranking; role checks |
| Integration | source → Bronze → Silver → Gold on the small profile; quarantine counts = rejected rows; Gold → DuckDB serving queries |
| Data quality | Silver required keys non-null; FKs resolve; ranges hold; **detected issues ≈ injected issues** (within tolerance for overlaps) |
| Idempotency | running ingest + silver + gold twice gives identical row counts and table hashes |
| Late data | inject late inventory events → affected daily stock rows recomputed correctly |
| ML | chronological split; leakage test (shuffle future target → metrics unchanged); model beats or is reported against baseline; ground-truth recovery reported |
| RAG | 40-question set: routing, hit@3, numeric exact match, refusal behaviour |
| App | auth/role gating; data functions reject unauthorised roles; missing-artifact empty states; pages render with the small snapshot (`streamlit.testing.AppTest`) |
| E2E | small profile: generate → publish → AppTest loads each Tier 1 page |

Tests use one session-scoped SparkSession (`local[2]`, shuffle partitions = 4, UI disabled) to keep the suite fast.

---

## 16. Security and privacy

- No secrets in git; `.env.example` holds placeholders only; secrets via env vars or `st.secrets`.
- All customer data is synthetic (Faker with a fixed seed, Indian locale).
- Customer-level drill-down is limited to the Marketing and Data Engineer roles; emails are masked in the UI (`a***@domain`).
- Structured functions never return names or emails.
- The demo-auth limitation is documented in the README.

---

## 17. Performance

- Spark for joins, windows, aggregation, deduplication and streaming. Collect only bounded results (pair tables, model feature matrices of ~44K rows, UI samples).
- Partitioning: Bronze by `_batch_id` folder; Silver and Gold unpartitioned (full rebuild). Avoid tiny-file explosion by using `coalesce(1)` on small tables.
- CSV is read with explicit schemas (no inference pass). Within a run, Silver tables reused by several Gold jobs are `cache()`d to avoid re-parsing CSV.
- Large profile: CSV parsing will dominate runtime. Measure it, and note it in the benchmark as the main argument for a later move to a columnar format.
- **Benchmark plan:** run small, medium and large profiles on a documented machine. Record wall time per stage, Spark/Python versions and cold vs warm runs. Optionally compare a pandas implementation of two Silver jobs at the medium and large sizes to show *where* Spark starts paying off. Report only measured numbers.

---

## 18. Documentation and interview readiness

**Required docs:** README, architecture, data dictionary, transformation catalog, metric definitions, ML design, RAG design, deployment, demo walkthrough. They live in `docs/`; `README.md` stays short.

**Be ready to explain:** why Medallion; why Bronze keeps dirty values as strings; the dedup tie-break; why CSV for now, and how idempotency works without upserts (full rebuild + deterministic dedup + atomic swap), plus what Parquet/Delta would add; why Gold grains were chosen; **why Spark at this scale** (§1.3); how late data and watermarks work; how lineage is captured from code; how leakage is prevented; why exact numbers bypass RAG; **how planted ground truth validates the models**; what the free host can and cannot do.

---

## 19. Demonstration scenarios

| # | Scenario | Flow | Expected result |
|---|---|---|---|
| 1 | Dirty record → Silver | DE workspace → orders → a conflicting duplicate and a negative-quantity item → Silver result + quarantine row with `failed_rules` | user can explain what changed, why, and where rejects went |
| 2 | Forecast → replenishment | Inventory → store S03, category Dairy → forecast vs baseline → high-risk SKU → suggested quantity with formula and assumptions | the recommendation traces to stock and demand signals; marked advisory |
| 3 | Segments | Marketing → segment profiles → ARI vs planted personas shown in a methodology note | segment names reflect computed profiles |
| 4 | Basket rules | Business → top rules → planted-pair recall → example baskets | support/confidence/lift explained on real baskets |
| 5 | Lineage & runs | DE → `gld_daily_category_sales` → upstream to Bronze → run history with counts | actual execution evidence, not a static diagram |
| 6 | Hybrid RAG | "Which products are at high stockout risk at S03, and what does the inventory policy recommend?" | numbers from `get_inventory_risks`, guidance from inventory policy §3, both cited |

---

## 20. Acceptance checklist

### Tier 1 — MVP (definition of done for the core project)

**Foundation and data**
- [ ] Dev container / WSL2 setup documented and working
- [ ] Config profiles validated by Pydantic
- [ ] Generator produces all 19 datasets reproducibly from a seed
- [ ] Demand model: Zipf popularity, personas, planted pairs, anomalies, stockouts
- [ ] Cross-dataset consistency rules C1–C11 enforced and tested
- [ ] Dirty-data injection configurable; manifest records actual injected counts
- [ ] Ground-truth files written and isolated from the pipeline

**Bronze / Silver**
- [ ] Batch, historical, Structured Streaming (watermark) and log ingestion work
- [ ] Bronze keeps raw string values plus metadata; file reloads are idempotent
- [ ] Silver for the 13 Tier 1 datasets: typing, standardisation, dedup, validation
- [ ] `qtn_records` with `failed_rules`; quality metrics per run
- [ ] Rebuild idempotency (identical CSV output on rerun) and late-event tests pass
- [ ] All tables read and written through `src/common/io.py` with explicit schemas

**Gold and ML**
- [ ] Tier 1 Gold tables with documented grains; metrics match fixtures
- [ ] Leakage-safe demand features
- [ ] Forecasting evaluated vs two baselines (WAPE, MAE, RMSE)
- [ ] Rule-based stockout risk and replenishment, backtested
- [ ] Segmentation with silhouette, stability and persona ARI
- [ ] Basket rules with support/confidence/lift and planted-pair recall
- [ ] Model registry populated

**RAG**
- [ ] Five policies indexed; BM25 retrieval with citations
- [ ] Whitelisted structured functions with role checks
- [ ] Rule-based router; hybrid answers; no-LLM mode works end to end
- [ ] 40-question evaluation recorded

**App and ops**
- [ ] Demo auth; role checks inside data functions
- [ ] 11 Tier 1 pages working on the published snapshot via DuckDB
- [ ] DE workspace shows actual runs, quality, quarantine and code-emitted lineage
- [ ] Atomic publish; failed runs never replace the snapshot
- [ ] Unit, integration, quality, ML, RAG, app and e2e tests in CI on the small profile
- [ ] Deployed on Streamlit Community Cloud with a precomputed-data banner
- [ ] README + docs + demo walkthrough (scenarios 1, 2, 3, 4, 5, 6)

### Tier 2 — Stretch
- [ ] Silver/Gold for promotions, refunds, reviews, app events, weather features
- [ ] Recommendations with Precision@K/Recall@K/coverage vs popularity
- [ ] Anomaly detection with recall on planted anomalies
- [ ] Forecast prediction intervals (quantile GBM)
- [ ] Stockout classifier vs rules
- [ ] Embedding retrieval compared to BM25; optional LLM answer composer
- [ ] Remaining 8 pages (Inventory Explorer, Customer Insights, Recommendations, Product Affinity, Sales Analytics, Anomaly Detection, Transformations, ML & RAG Monitoring)
- [ ] SCD2 product prices
- [ ] Large-profile benchmark report (Spark vs pandas)
- [ ] Optional storage upgrade: switch `src/common/io.py` to Parquet or Delta and compare runtime and size against CSV

---

## 21. Feasibility assessment

### 21.1 Verdict

| Scope | Feasibility | Notes |
|---|---|---|
| **Tier 1 (MVP)** | **High** | All components use mature, well-documented libraries. The main risk is execution discipline, not unknown technology. |
| **Tier 1 + Tier 2 (full plan)** | **Medium** | Achievable solo given time. Tier 2 items are independent add-ons, so partial completion still leaves a coherent product. |
| v1 plan as originally written | Medium-Low | Data volumes could not support the ML claims, and with everything mandatory there was no shippable intermediate state. |

**Effect of the CSV-only decision on feasibility:** neutral to slightly positive at this scale. There is one less dependency (no Delta), setup is simpler, and every file can be opened and inspected directly, which helps demos and debugging. The cost is extra care around types and atomic writes, covered by explicit schemas, the single I/O module and the staging-swap pattern. At the Large profile (~3M rows), CSV parsing will be noticeably slower than a columnar format, which is acceptable because Large is only used for benchmarking.

### 21.2 Effort estimate (solo developer, ~15–20 h/week, intermediate Python/Spark)

These are planning estimates, not commitments. Adjust after Phase 2 using actual velocity.

| Phase | Scope | Estimate |
|---|---|---|
| 1. Foundation | dev env, config, generator, demand model, consistency rules, dirty data, manifest | 2–3 weeks |
| 2. Bronze/Silver | 4 ingestion patterns, 13 Silver datasets, quarantine, quality, idempotency tests | 3–4 weeks |
| 3. Gold | Tier 1 Gold tables, metrics, features, lineage capture | 1.5–2 weeks |
| 4. Inventory slice | forecasting + baselines, stockout rules, replenishment, Inventory pages, publish + DuckDB serving | 2–3 weeks |
| 5. Segments + basket | segmentation, basket rules, Marketing/Business Tier 1 pages | 1.5–2 weeks |
| 6. RAG | policies, BM25, functions, router, composer, eval set | 1.5–2 weeks |
| 7. DE workspace | 4 Tier 1 pages over metadata | 1.5–2 weeks |
| 8. Hardening + deploy | test gaps, CI, Streamlit Cloud, docs, demo rehearsal | 1.5–2 weeks |
| **Tier 1 total** | | **~15–20 weeks** |
| Tier 2 (all items) | | +8–12 weeks |

### 21.3 Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Spark setup problems on Windows | High (on native Windows) | High | Dev container or WSL2 from day one; pin versions; CI proves the setup |
| CSV type drift (dates, decimals, nulls misread) | Medium | Medium | Explicit schemas on every read, a schema round-trip test per table, one I/O module |
| Partial CSV writes after a crash | Low–Medium | Medium | Staging folder + validate + rename swap; previous version kept |
| Generator becomes a project of its own | Medium | High | Timebox Phase 1; build the demand model incrementally (Zipf + DOW first, personas and plants next) |
| Forecast fails to beat the baseline | Medium | Medium | Expected and acceptable: report it honestly. The generator's DOW/rain signal makes improvement likely at category grain. |
| Scope creep into Tier 2 before Tier 1 ships | High | High | Tier 1 checklist is the gate; no Tier 2 work until Tier 1 is deployed |
| Snapshot too large or slow for Streamlit Cloud | Low–Medium | Medium | Publish Gold only plus capped samples; DuckDB pushdown; size check in the publish stage |
| Slow Spark tests | Medium | Medium | Small profile, session fixture, `local[2]`, few shuffle partitions |
| Metadata/lineage plumbing adds friction | Medium | Medium | One decorator (`@tracked_transform`) built in Phase 2 and reused everywhere |
| RAG answers look weak without an LLM | Medium | Low | Template answers with clear tables and quotes are a deliberate, explainable design. The LLM is optional polish. |
| Synthetic data called "unrealistic" in interviews | Medium | Low | The documented demand model plus planted-truth evaluation turns this into a strength |

### 21.4 Critical success factors

1. **Ship the Inventory vertical slice end to end (Phases 1–4) before anything else.** It is the project's proof of concept.
2. **Keep planted ground truth separate from the pipeline.** It is what makes the ML claims verifiable.
3. **Capture metadata from code from the start.** Retrofitting lineage and run tracking is expensive.
4. **Deploy early** (end of Phase 4) with a basic snapshot, so hosting problems surface while there is time to fix them.

---

## 22. Non-goals

- Not a production retail platform; streaming is simulated with Structured Streaming on files.
- No continuously running Spark in hosting; the app reads precomputed snapshots and says so.
- Demo auth is not production security.
- Stockout and replenishment outputs are advisory.
- RAG never computes exact metrics.
- No infrastructure added only to look impressive (no Kafka, Airflow or vector DB at this scale).
- No performance claims without measured, documented runs.
- No claim that source row counts survive into every layer.

---

## 23. Definition of done

The project is done (Tier 1) when a reviewer can open the hosted app and do the following:

1. Pick a dirty source record and follow it through Bronze and Silver (or into quarantine, with its failed rules).
2. See the Gold aggregation it feeds, with code-emitted lineage and actual run counts.
3. See a forecast, stockout risk and replenishment suggestion built on that data, evaluated against baselines.
4. See segments and basket rules evaluated against planted ground truth.
5. Ask the assistant a hybrid question and get cited numbers plus cited policy.
6. Confirm everything shown is labelled as actual, declared or illustrative, and that the app clearly states it is a precomputed snapshot.

All of this must be reproducible from a seed with one command, covered by tests in CI.
