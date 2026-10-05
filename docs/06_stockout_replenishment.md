# 6. Stockout Risk and Replenishment (Phase 4C)

**Status:** complete · **Code:** `src/ml/stockout/run.py` · **Run:** `python -m scripts.run_pipeline --stages ml` (runs after forecasting)
**Outputs (columns in [gold_catalog.md](gold_catalog.md)):** `gld_stockout_risk`, `gld_replenishment`, `gld_stockout_backtest`

## Question answered

*Which focus SKUs may run out of stock, and how much should be reordered?* Every result is **advisory**; nothing places an order.

## Inputs

| Input | From |
|---|---|
| End-of-day stock, reorder level, days of inventory, daily units sold | `gld_inventory_daily` (Gold, rebuilt from weekly counts + events − sales) |
| Forecast demand 1–7 days ahead per store × focus SKU, with 10–90% band | `gld_sku_demand_forecast` (Phase 4B) |

Decisions are made **every day of the September backtest** (using the forecast made that day by the model trained on Apr–Aug) and for the **last day of data** (30 Sep, "current").

## Assumptions (stored on every replenishment row)

| Assumption | Value | Why |
|---|---|---|
| Supplier lead time | 2 days | the simulator uses 1–2 days; the worst case is the safe choice |
| Review period | 1 day | decisions are taken daily |
| Service level | 95% (z = 1.65) | common retail default |
| "Low cover" threshold | 3 days of inventory | |

## Risk tiers (transparent rules)

| Tier | Condition | Reason codes |
|---|---|---|
| **High** | stock is 0, **or** forecast demand over the next 3 days (lead time + 1) ≥ stock | `ZERO_STOCK`, `DEMAND_EXCEEDS_STOCK` |
| **Medium** | stock ≤ reorder level, **or** days of inventory < 3 | `BELOW_REORDER`, `LOW_DAYS_COVER` |
| **Low** | none of the above | — |

All applicable reason codes are listed, so every flag explains itself. Also stored: forecast demand over 3 days (and its 90th-percentile version) and **days of cover** = stock ÷ forecast daily demand.

## Replenishment quantity

```
suggested_qty = ceil( max(0, forecast demand over (lead time + review) + safety stock − current stock) )
safety stock  = 1.65 × standard deviation of daily units sold (last 28 days) × √(lead time)
```

## Results (medium run)

**Current position (30 Sep 2025):** 10 SKUs High, 30 Medium, 260 Low; **49 SKUs to reorder, 133 units in total.**

**Backtest:** for each decision taken while the SKU was **still in stock** (8,200 store × SKU × day decisions in September), did the SKU actually stock out within the next 3 days? Decisions on SKUs already at zero are excluded, because predicting those is trivial and would flatter the result.

| Rule | Flags | Precision | Recall |
|---|---:|---:|---:|
| **High tier (forecast-based)** | 4.2% of decisions | **20.5%** | 19.4% |
| High or Medium tier | 14.4% | 11.5% | 37.5% |
| Naive: stock ≤ reorder level | 14.2% | 11.6% | 37.5% |
| *No rule (base rate)* | — | *4.4%* | — |

**How to read this:**
- The **forecast-based High tier is about twice as precise as the naive reorder rule** and 4.7× better than chance. It is the right short list for a manager's attention.
- The naive rule catches more stockouts (recall 37.5%) but with a lot more noise. That is a genuine precision/recall trade-off, shown on screen rather than hidden.
- Recall is modest by nature here: many stockouts come from **planted supplier outages** (no stock or demand signal can predict a supplier failing), and stock between weekly counts drifts when Silver quarantined some lines or events (see the inventory reconciliation gap in Gold).
- **Planted supplier outages:** 2 of 2 outages that start inside the backtest window were flagged High or Medium in the 3 days before they started. The pass bar (50%) was fixed before measuring, but **2 is too small a sample to prove much**.

## How it is verified

`python -m scripts.check_ml`: checks 7–10 (checks 1–6 cover forecasting):

7. A tier for every focus SKU on every decision day, and every High/Medium row has a reason.
8. Every suggested quantity follows the formula and is never negative.
9. The forecast-based High tier is more precise than the naive reorder rule.
10. Planted supplier outages are warned (ground truth read only by the report).

`tests/unit/test_stockout_rules.py`: hand-made cases for each tier (zero stock → High; demand over lead time > stock → High; below reorder level → Medium; plenty of stock → Low with order 0) and the exact replenishment formula.

## Problems found and decisions made

- **Backtest restricted to in-stock decisions**, so the reported precision measures *warning before* a stockout, not recognising one that has already happened.
- **Windows file locking:** writing a new table occasionally failed with "Access is denied" while renaming a just-written folder (antivirus/indexing briefly locks new files). All writers now retry the rename for a few seconds; the same fix protects every layer.
- A rule name containing "≤" could not be printed by the Windows console; rule names use plain "<=" and the scripts print UTF-8.

## Possible next improvements (Tier 2)

- A learned stockout classifier, adopted only if it beats these rules in the same backtest.
- Lead time per supplier instead of one assumption.
