# 3. Silver Transformations

**Status:** complete · **Code:** `src/transformations/silver/`, `src/quality/` · **Run:** `python -m scripts.run_pipeline --stages silver`
**Rule-by-rule reference:** [transformation_catalog.md](transformation_catalog.md) (generated from the code)

## Purpose

Silver turns Bronze (raw text, ~10% dirty) into **trusted tables**: typed values, one standard format, one row per business key, and valid references. Nothing disappears silently. Every rejected row goes to a quarantine table with the rules it failed and its original values, and suspicious-but-real rows stay in Silver with a `dq_*` flag.

## Input and output

| | Location |
|---|---|
| Input | `data/bronze/brz_<dataset>/` (19 tables, 307,784 rows) |
| Clean tables | `data/silver/slv_<dataset>/` (typed CSV; previous version kept in `_previous/`) |
| Rejected rows | `data/quarantine/qtn_records/<dataset>/` |
| Quality metrics | `data/metadata/meta_quality_results/<run_id>.csv`, plus counts in `meta_table_runs` |

Silver is **rebuilt in full on every run** from all Bronze batches (seconds at this size). That is what makes late data, backfills and re-runs safe with CSV storage, which has no in-place updates.

## The seven steps (every dataset)

1. **Separate unparseable rows** (`_corrupt_record`) → quarantine as `malformed:row`.
2. **Standardise:** trim all text, blank → empty, lowercase code values (status, method, …), uppercase log levels, Title Case names, *most frequent spelling* for brands (their correct casing can't be derived by rule), lowercase and check emails.
3. **Parse and cast** text to int / decimal / timestamp / date. Timestamps arrive in 5 formats and all are accepted. A value that is present but can't be parsed is a `type_invalid` failure; an empty required value is `required_missing`.
4. **Remove duplicates:** one row per business key, keeping the latest ingestion, then the latest event time, then the highest record hash. Standardising first means formatting-only duplicates (`DELIVERED` vs `delivered`) become identical and collapse.
5. **Validate:** required values, types, allowed values, ranges, time order, business rules, and foreign keys against the **already-accepted** parent tables.
6. **Split:** any hard rule failed → quarantine. It is `direct` if the row broke a rule itself and `cascade` if its only problem is a rejected parent. Soft (business) rules → a `dq_*` flag on the kept row.
7. **Derive** helper columns: business date (IST), line amount, delivery minutes, completed-order flag, bulk-order flag, recomputed order total.

Tables are built in dependency order (categories, stores, customers → products, promotions, riders → orders → order items, payments, deliveries, … → refunds), then an **orders post-processing** step fills the order columns that need child tables.

## Decisions (agreed before building)

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | Children of a rejected parent | **Strict cascade:** quarantined as `cascade` | Silver keeps perfect referential integrity; the knock-on effect is visible and measurable |
| 2 | Business-rule violations | **Flag** (`dq_*`) and keep; quarantine only **refund > payment** | The row is real but one value is suspect; Gold can use the trusted value. A refund larger than its payment is financially impossible |
| 3 | Invalid email | **Blank it and flag** `dq_email_invalid`, keep the customer | Email is optional; a bad email shouldn't remove a customer |

## How each injected issue is handled

| Issue | Example in the data | Rule | Outcome |
|---|---|---|---|
| DUP | `CUST00220` twice | business key + tie-break | removed |
| FMT | `DELIVERED`, `" delivered"`, `05/04/2025 14:03:00` | standardise / multi-format parse | fixed |
| MISS | `ORD0000004` with empty `order_ts` | `required_missing:order_ts` | quarantined |
| TYPE | `total_amount = "unknown"` | `type_invalid:total_amount` | quarantined |
| NUM | `ORD0000002` with total −1291.00 | `range_invalid:total_amount` | quarantined |
| FK | order item → product `P9xxx` (doesn't exist) | `fk_missing:product_id` | quarantined |
| SEQ | delivered before pickup | `sequence_invalid:delivered_before_pickup` | quarantined |
| MALF | broken log row, cut-off JSON | `malformed:row`, `malformed:metadata` | quarantined |
| BIZ | order total ≠ its lines; rider from another city | `business:*` | flagged (refund > payment: quarantined) |

## Results (medium run)

```
307,784 Bronze rows = 272,971 in Silver + 25,153 quarantined + 9,660 duplicates removed
```

| Dataset | Bronze | Duplicates | Quarantined (direct) | Quarantined (cascade) | Silver |
|---|---:|---:|---:|---:|---:|
| categories | 21 | 1 | 0 | 0 | 20 |
| products | 640 | 40 | 0 | 0 | 600 |
| stores | 13 | 1 | 0 | 0 | 12 |
| customers | 6,400 | 400 | 0 | 0 | 6,000 |
| delivery_partners | 128 | 8 | 0 | 0 | 120 |
| orders | 36,973 | 973 | 1,751 | 0 | 34,249 |
| order_items | 126,348 | 3,948 | 7,717 | 5,001 | 109,682 |
| payments | 38,822 | 1,022 | 2,020 | 1,538 | 34,242 |
| deliveries | 35,062 | 1,062 | 1,547 | 1,403 | 31,050 |
| cancellations | 2,610 | 90 | 130 | 114 | 2,276 |
| inventory_snapshots | 8,089 | 289 | 520 | 0 | 7,280 |
| inventory_events | 12,400 | 400 | 600 | 0 | 11,400 |
| application_logs | 5,172 | 172 | 241 | 0 | 4,759 |
| promotions | 62 | 2 | 3 | 0 | 57 |
| order_promotions | 7,560 | 360 | 284 | 584 | 6,332 |
| returns_refunds | 1,869 | 69 | 122 | 162 | 1,516 |
| reviews | 5,608 | 208 | 169 | 230 | 5,001 |
| application_events | 15,469 | 469 | 797 | 0 | 14,203 |
| weather | 4,538 | 146 | 220 | 0 | 4,172 |

Orders came out at **exactly 34,249**, the number predicted from the injected issues before Silver was built.

**Detected vs injected** (all 30,778 injected issues, matched row by row against the generator's ground truth):

| Expected outcome | Handled |
|---|---|
| Duplicates removed | 9,314 of 9,314 |
| Format issues fixed | 2,932 of 2,932 |
| Invalid rows quarantined | 15,088 of 15,088 |
| Business-rule issues flagged | 2,824 of 2,824 |
| Unexplained flags on untouched rows | **0** (1,355 flags on untouched rows are each caused by a damaged related row) |

548 injected issues sit on rows that were rejected for another reason first (usually a rejected or unidentifiable parent), and 72 order/review flags are "could not verify" (some of the order's items were rejected). Both are reported separately, not counted as misses.

Runtime ~80 s.

## How it is verified

- `python -m scripts.check_silver`: **10/10 checks.** Rows accounted for; keys unique; no dirty values left (no padded text, allowed values only, required values present, ISO timestamps); detected vs injected per dataset and issue type; no unexplained flags; tables and lineage written.
- `tests/integration/test_silver.py`: the same report on the small profile at 100%, plus typed read-back, cascade correctness, `primary_rule` always set, quality metrics recorded, and **idempotency** (a second run gives identical tables).

## Problems found and fixed

1. **Out of memory on payments.** Each child table's Spark plan carried the full history of its parents (orders → items → promotions …), and the planner analysed it all every time. Fixed by *checkpointing* each finished table (save the result, cut the history): from a crash to ~80 s.
2. **Flags on the wrong rows.** When an item's `order_id` was damaged, its order lost a line, so the order's total no longer matched. The order flag is *correct* (the order really is missing a line), but payments and reviews were flagged too. Fixed:
   - a payment is flagged only if it matches neither the stored nor the recomputed order total;
   - a review is "could not verify" (empty flag) when its order has rejected items.
3. **Empty `primary_rule` in quarantine.** A PySpark quirk: a lambda with a default argument is treated as the two-argument `(element, index)` form. Fixed, and a test now guards it.
4. **Report matching limits** (not pipeline errors): when the injected bad value *is* part of the key (a fake `product_id` in an order item), the original key can't be found, so those issues are matched by rule counts. Order promotions are matched by `order_id` (one promotion per order).

## Presentation copy

`python -m scripts.export_flat_tables --layer silver` writes `data_sep_silver/`: one clean CSV per table plus `quarantine.csv` with every rejected row and its reason.

Suggested demo: `ORD0000002` has total −1291.00 in `data_sep/orders.csv`, is missing from `data_sep_silver/orders.csv`, and appears in `quarantine.csv` with `range_invalid:total_amount`.
