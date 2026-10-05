# 7. Inventory Workspace — the first app screen (Phase 4D)

**Status:** complete · **Code:** `app/`, `src/serving/`, `src/orchestration/publish.py` · **Run:** `streamlit run app/Home.py`

## Purpose

This is the first screen a business user sees. It closes the plan's first **vertical slice**: raw data → Bronze → Silver → Gold → forecast → stockout risk → a decision on screen. An inventory manager can see which SKUs are at risk, why, how much to reorder, and how far to trust the forecast.

## How to run it

```powershell
python -m scripts.run_pipeline --stages publish     # (re)publish the snapshot after a pipeline run
python -m scripts.create_demo_secrets               # once: creates .streamlit/secrets.toml with demo logins
streamlit run app/Home.py                           # opens http://localhost:8501
```

| Demo login | Password | Role | Can open |
|---|---|---|---|
| `inventory` | `inventory-demo` | Inventory & Supply Chain Manager | Inventory workspace |
| `marketing` | `marketing-demo` | Growth & Marketing Manager | (Marketing workspace — Phase 5) |
| `business` | `business-demo` | Business & Revenue Analyst | (Business workspace — Phase 5) |
| `engineer` | `engineer-demo` | Data Engineer | (Data Engineer workspace — Phase 7) |
| `admin` | `admin-demo` | Demo administrator | everything |

These are the default **demo** passwords printed by `create_demo_secrets` (stored only as bcrypt hashes in the git-ignored secrets file). `--password X` sets a different one. This is demo-grade login, not production security.

## Architecture

```
pipeline (Spark + Python)                       app (no Spark, no pipeline code)
Gold + ML tables ──publish──► data/demo/*.csv ──► DuckDB (in memory) ──► src/serving/queries.py ──► Streamlit pages
                              + snapshot_schema.csv     typed tables       role check in every         app/views/
                              + snapshot_manifest.csv                      function
```

- **Publish stage** (`--stages publish`): copies all 19 Gold/ML tables plus product, store and category lookups into `data/demo/` as one CSV each (22 tables, 331,056 rows, **27.3 MB**, limit 50 MB). It writes a schema file and a manifest (pipeline run, generation run, forecast model version, time), and swaps the folder in atomically. The snapshot is **committed to git** so the hosted app can read it.
- **DuckDB** loads each CSV once into memory, typed from the schema file, so the app never needs PySpark or the pipeline code. That is what lets it run on Streamlit Community Cloud. Filters run in DuckDB with bound parameters.
- **Access control is checked twice:** the menu only shows the workspaces a role may open, **and** every data function in `src/serving/queries.py` checks the role itself (`AccessDenied`), so hiding a page is never the only protection.
- **Honesty banner** on every page: "Precomputed snapshot — pipeline run X, published at Y … no pipeline is running live."

## The Inventory workspace

| Tab | What it shows | Data |
|---|---|---|
| **Overview** | KPIs: High-risk SKUs (10), Medium (30), zero stock (4), SKUs to reorder (49), units forecast for the next 7 days (5,144). Stock health per store: tiers, zero stock, median days of cover | `gld_stockout_risk`, `gld_replenishment`, `gld_demand_predictions` |
| **Demand forecasting** | Pick a store and category → chart of actual units, the September 1-day-ahead backtest forecast and the 7-day future forecast with its 10–90% band. Model vs both baselines (WAPE), WAPE by days ahead and by category, feature importance, limitations | `gld_daily_category_sales`, `gld_demand_predictions`, `gld_forecast_metrics`, `gld_forecast_feature_importance` |
| **Stockout risk & replenishment** | Filter by store and tier → SKU list with stock, reorder level, 3-day forecast, days of cover, suggested order and reason codes; **CSV download of suggested orders**. One-SKU chart: closing stock, units sold, reorder level and forecast demand. The replenishment formula with its assumptions, and the backtest table | `gld_stockout_risk`, `gld_replenishment`, `gld_inventory_daily`, `gld_sku_demand_forecast`, `gld_stockout_backtest` |

An "advisory only" note appears with every risk and order figure: nothing places an order or changes stock.

## How it is verified

`python -m scripts.check_ml`: **12/12** (checks 11–12 are the app):

11. Snapshot published, under 50 MB, built from the current forecast model.
12. All 10 Inventory data functions work for the inventory manager and are blocked for another role.

`tests/app/test_app.py` (small profile, Streamlit's own test runner):
- snapshot published with 22 tables and the right run id;
- **every data function rejects** marketing, business, engineer and anonymous callers, and works for inventory and admin;
- login: a wrong password is refused; the demo password signs in with the right role;
- the Inventory workspace renders with its KPIs and tables and no errors;
- a marketing manager opening the Inventory page gets "cannot open" and sees no data.

A real launch was also checked: `streamlit run` starts and the health endpoint answers within about a second.

## Problems found and fixed

- An ambiguous `risk_tier` column (present in both joined tables) broke the risk list query; now qualified.
- Streamlit's percentage format is the preset `"percent"`, not a printf pattern.
- `use_container_width` is being retired by Streamlit; replaced by `width="stretch"`.
- App tests must use absolute paths, because Streamlit's test runner resolves paths relative to the test file.

## Next

- Deploy this app to **Streamlit Community Cloud** (Phase 8, or earlier if you want a public link): the repo already has `requirements.txt` with app-only dependencies and the committed snapshot; the demo logins go into the Cloud app's Secrets settings.
- The Marketing, Business and Data Engineer workspaces reuse the same login, snapshot and serving layer.
