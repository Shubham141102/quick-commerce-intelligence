# Documentation

Documentation for the Quick-Commerce Intelligence Platform. Each completed process gets its own document as soon as it is finished; reference documents are generated from the code so they cannot drift from what the pipeline actually does.

## Process documents (narrative: what, why, how, results)

| # | Process | Document | Status |
|---|---|---|---|
| 1 | Synthetic data generation (19 datasets, 90% clean) | [01_data_generation.md](01_data_generation.md) | Done |
| 2 | Bronze ingestion (landing CSV → raw tables + metadata) | [02_bronze_ingestion.md](02_bronze_ingestion.md) | Done |
| 3 | Silver transformations (clean, validate, quarantine) | [03_silver_transformations.md](03_silver_transformations.md) | Done |
| 4 | Gold (business tables, metrics, ML features) | [04_gold.md](04_gold.md) | Done |
| 5 | Demand forecasting (Phase 4B) | [05_demand_forecasting.md](05_demand_forecasting.md) | Done |
| 6 | Stockout risk and replenishment (Phase 4C) | [06_stockout_replenishment.md](06_stockout_replenishment.md) | Done |
| 7 | Inventory workspace — Streamlit app (Phase 4D) | [07_inventory_workspace.md](07_inventory_workspace.md) | Done |
| 8 | Inventory explorer + lost-sales estimate (Phase 5A) | [08_inventory_explorer_lost_sales.md](08_inventory_explorer_lost_sales.md) | Done (1 known limitation) |
| 9 | Business workspace: sales, delivery & operations, anomaly detection (Phases 5B, 5C) | [09_business_workspace.md](09_business_workspace.md) | Done (1 known limitation) |
| 10 | Marketing analytics — model cards: segmentation, basket rules, recommendations, retention (Phase 5D) | [10_marketing_analytics.md](10_marketing_analytics.md) | Done |
| 11 | Customer Growth & Marketing workspace (Phase 5E) | [11_marketing_workspace.md](11_marketing_workspace.md) | Done |
| 12 | Frontend: login page, persona landing, page design | [12_frontend.md](12_frontend.md) | Done (Data Engineer workspace on hold) |
| 13 | Business assistant — RAG (Phase 6) | [13_rag_assistant.md](13_rag_assistant.md) | In progress (6A–6G done; all evaluation bars pass) |

## Reference documents (generated — do not edit by hand)

| Document | Contents | Regenerate |
|---|---|---|
| [data_dictionary.md](data_dictionary.md) | Every source dataset and column: type, required, valid values, references, injected issues; Bronze metadata columns | `python -m scripts.build_docs` |
| [transformation_catalog.md](transformation_catalog.md) | Every Bronze → Silver rule per dataset: standardisation, casts, validation rules, outcomes, derived columns, lineage | `python -m scripts.build_docs` |
| [gold_catalog.md](gold_catalog.md) | Every Gold table: grain, sources, every column with type and meaning, lineage | `python -m scripts.build_docs` |
| [metric_definitions.md](metric_definitions.md) | Every business metric: definition and formula (the single source of truth) | `python -m scripts.build_docs` |
| [file_registry.md](file_registry.md) | Every project file: purpose, inputs, outputs, who uses it, how to run it | `python -m scripts.build_docs` |

A unit test fails if a generated document is out of date.

## Pipeline at a glance

```
Generator (Python)            Bronze (PySpark)                 Silver (PySpark)
19 datasets, 90% clean   →    raw rows kept exactly       →    typed, standardised, de-duplicated
landing CSVs:                 + ingestion metadata             rejected rows → quarantine (with reasons)
historical / batch / stream   file-level load tracking         business-rule flags (dq_*)
data/generation/<run>/        data/bronze/brz_<dataset>/       data/silver/slv_<dataset>/
                                                               data/quarantine/qtn_records/<dataset>/
        ↓                              ↓                                ↓
   manifest + ground truth      run metadata (CSV) in data/metadata/: runs, stages, tables, files, quality, lineage
```

Then **Gold** (PySpark): 12 business tables in `data/gold/<table>/` with shared metric definitions.

Then the **ML stage** (Python / scikit-learn): demand forecasts, stockout risk and replenishment, written as Gold tables.
The **publish stage** copies the app's tables to `data/demo/`, and the **Streamlit app** (`streamlit run app/Home.py`) reads them through DuckDB.

Completion checks: `python -m scripts.check_bronze` (9 checks), `python -m scripts.check_silver` (10 checks), `python -m scripts.check_gold` (10 checks), `python -m scripts.check_ml` (12 checks, including the app).
Overall design: [Project_Plan_v2.md](Project_Plan_v2.md).
