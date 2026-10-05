# 8. Inventory Explorer and Lost-Sales Estimate (Phase 5A)

**Status:** complete, with one documented known limitation · **Code:** `src/ml/lost_sales.py`, `src/serving/queries.py` (explorer functions), `app/views/inventory.py` (Inventory explorer tab) · **Run:** `python -m scripts.run_pipeline --stages ml,publish`
**Output (columns in [gold_catalog.md](gold_catalog.md)):** `gld_lost_sales` · **Metric:** "Lost sales (estimate)" in [metric_definitions.md](metric_definitions.md)

## Business question

*What did empty shelves cost us, and which SKUs did it hit hardest?* An inventory manager is judged on availability. The lost-sales figure turns stockouts into money and tells the manager where better replenishment would pay off. The explorer lets them drill into one store × SKU to see *why* it ran out.

## Lost-sales estimate

For every **stockout day** (stock reached 0 at some point, from `gld_inventory_daily`):

```
expected_units = average daily units sold on the SKU's IN-STOCK days in the previous 28 days
lost_units     = max(0, expected_units − units actually sold)
lost_revenue   = lost_units × catalog price
```

- **Stockout days are excluded from the expectation**, because their sales were cut short by empty shelves and would drag the expectation down.
- At least 7 in-stock days of history are required; otherwise no estimate is made.
- **Business caveat:** in the simulation a shopper facing an empty shelf often bought a **substitute**, so this is revenue lost *for that SKU*. The store as a whole lost less.

## Inventory explorer (4th tab of the Inventory workspace)

- **Network view:** estimated lost sales across all stores, lost units, stockout days; the 10 SKUs that lost the most; lost sales by store.
- **One SKU in detail:** pick a store and SKU (it defaults to the worst one) → stockout days, units sold, restocked / damaged, estimated lost sales, and the average gap between weekly counts and calculated stock.
- **Chart:** calculated closing stock, weekly counted stock (hover shows the gap), restocks, write-offs, the reorder level, and shaded stockout days.
- **Daily rows table** for that SKU.

All data comes from four new role-checked functions in `src/serving/queries.py`; only the Inventory role (and admin) can call them.

## Results (medium run)

| Measure | Value |
|---|---|
| Stockout days with an estimate | 1,898 (store × SKU × day) |
| Estimated lost units | 811 |
| Estimated lost revenue | ₹1,57,101 (before substitutes) |
| Simulator's **true** lost units | 809 |

## Verification against ground truth

Pass bars were **fixed before measuring**: total within ±50% of the simulator's true lost units, and SKU-level ranking agreement (Spearman) ≥ 0.4.

| Check | Result |
|---|---|
| 5A.1 Rows only on stockout days; never negative; units × price = revenue | ✅ PASS |
| 5A.2 Total close to the truth (±50%) | ✅ PASS: 811 vs 809 (ratio 1.00) |
| **5A.3 Ranks the right SKUs (Spearman ≥ 0.4)** | ⚠️ **LIMIT: bar not met (0.35)**, documented known limitation |
| 5A.4 Explorer functions role-checked | ✅ PASS |

### Why 5A.3 is not met (diagnostic, 2026-10-05)

| Finding | Value |
|---|---|
| True lost units per SKU over six months | median **1**; 65% of SKUs lost ≤ 2 units |
| True lost units on days the rebuilt inventory also marks as stockout | **65%** (35% missed) |
| SKUs with no true loss but an estimate | 80 (false stockout days) |
| Ranking agreement, all 291 SKUs | Spearman 0.35 (Pearson 0.48) |
| Ranking agreement, SKUs that lost ≥ 10 units | **Spearman 0.47, Pearson 0.68** |

**Interpretation:**
- The weak point is **not the lost-sales formula** but **which days count as stockouts**. Stock is rebuilt in Gold from weekly counts, events and sales, and drifts between counts because Silver quarantined some lines and events. The same root cause gives the 67% stockout recall measured in Gold.
- The near-exact total is **partly luck**: missed stockout days and false ones roughly cancel out.
- In practice the estimate is **reliable for the total and for the biggest losers**, which is what a manager acts on, and unreliable for ranking SKUs that lost only a unit or two. The app shows this caveat.

**Decision (2026-10-05):** keep the pre-set bar unchanged, record the result as a **known limitation** (shown as `LIMIT` in `check_phase5`, never as `PASS`), and treat better stockout detection as a Tier 2 improvement.

## How it is verified

- `python -m scripts.check_phase5`: section 5A (above).
- `tests/unit/test_lost_sales.py`: hand-made cases (expected − sold × price; earlier stockout days excluded from the expectation; no estimate without enough history; never negative).
- `tests/app/test_app.py`: the explorer functions reject other roles; the Inventory workspace renders the explorer KPIs.

## Problems found and fixed

- **Publish recorded the previous model version** when `ml` and `publish` ran in the same command, because the run log is only written when a run ends. The snapshot now takes the model version from the published predictions; Phase 4 check 11 caught it.

## Possible improvement (Tier 2)

Better stockout-day detection in Gold, e.g. weighting the weekly counts more strongly or requiring a run of low readings, then re-measuring 5A.3 against the **same** 0.4 bar.
