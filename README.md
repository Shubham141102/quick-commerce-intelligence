# Quick-Commerce Intelligence Platform

End-to-end PySpark Medallion pipeline (Source → Bronze → Silver → Gold, all stored as CSV) with ML, RAG and multi-persona Streamlit analytics, built on synthetic quick-commerce data.

The full design is in [`docs/Project_Plan_v2.md`](docs/Project_Plan_v2.md); all documentation is indexed in [`docs/README.md`](docs/README.md).

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Foundation: config, schemas, data generator, dirty data, manifest, data dictionary | Done |
| 2 | Bronze / Silver | Done (Bronze 9/9, Silver 10/10 completion checks) |
| 3 | Gold | Done (10/10 completion checks) |
| 4 | Inventory vertical slice | Done (forecasting, stockout risk, Inventory workspace app; check_ml 12/12) |
| 5 | Segmentation + basket analysis | Not started |
| 6 | RAG assistant | Not started |
| 7 | Data Engineer workspace | Not started |
| 8 | Hardening + deployment | Not started |

## Setup

Requires Python 3.11 and Java 17 (Java is needed only from Phase 2, for PySpark).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on WSL/Linux)
pip install -e ".[generation,pipeline,app,dev]"
```

### Spark on Windows (native)

Spark runs directly in the Windows venv (PySpark 3.5 + Java 17). Spark's file writes on Windows need Hadoop's
native helpers, which are kept out of git in `tools/hadoop/bin/`:

```powershell
New-Item -ItemType Directory -Force tools\hadoop\bin
$base = "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin"   # community build, not an Apache release
foreach ($f in "winutils.exe","hadoop.dll") { Invoke-WebRequest "$base/$f" -OutFile "tools\hadoop\bin\$f" -UseBasicParsing }
```

Hashes of the copies in use: `winutils.exe` SHA-256 `496A591E…EFD8553`, `hadoop.dll` SHA-256 `D7AB36A6…200405BE3`.

Always create sessions with `src.common.spark.get_spark()`. It sets `HADOOP_HOME`, adds the helpers to `PATH`, and
pins Spark's Python workers to the venv interpreter (otherwise Anaconda's `python` on PATH would be used). It also
passes space-free 8.3 paths to Spark's `.cmd` launchers, which break on the spaces in this folder's path.
`tests/integration/test_spark_smoke.py` checks the setup.

## Generate data

```bash
python -m src.generation.generate_all --profile small     # ~14K rows, ~5 s  (tests / CI)
python -m src.generation.generate_all --profile medium    # ~308K rows, ~40 s (demo)
python -m scripts.build_docs                              # regenerate docs/data_dictionary.md + transformation_catalog.md
python -m pytest                                          # 65 tests (Spark tests included)
```

## Run the pipeline

```bash
python -m scripts.run_pipeline --stages ingest                 # latest generation run -> Bronze (~2 min, medium)
python -m scripts.run_pipeline --stages ingest --force-reload  # re-ingest files already loaded
python -m scripts.run_pipeline --stages ingest --reset-bronze  # rebuild Bronze from scratch
python -m scripts.run_pipeline --stages silver                 # Bronze -> Silver + quarantine (~1.5 min, full rebuild)
python -m scripts.run_pipeline --stages gold                   # Silver -> 12 Gold tables (~1.5 min)
python -m scripts.run_pipeline --stages ingest,silver,gold     # everything after generation
python -m scripts.check_gold                                   # Gold completion report (reconciles with Silver)
python -m scripts.run_pipeline --stages ml                     # forecasting + stockout risk + replenishment (~40 s, no Spark)
python -m scripts.check_ml                                     # Phase 4 report (vs baselines, backtests, app)
python -m scripts.run_pipeline --stages publish                # copy app tables to data/demo/ (committed snapshot)
```

## Run the app

```bash
python -m scripts.create_demo_secrets      # once: .streamlit/secrets.toml with demo logins (git-ignored)
streamlit run app/Home.py                  # http://localhost:8501  — e.g. login: inventory / inventory-demo
```

The app reads only `data/demo/` through DuckDB (no Spark needed). Details and all demo logins:
[docs/07_inventory_workspace.md](docs/07_inventory_workspace.md).

```bash
python -m scripts.export_flat_tables --layer gold              # data_sep_gold/ one CSV per Gold table

python -m scripts.check_bronze                                 # Bronze completion report
python -m scripts.check_silver                                 # Silver completion report (detected vs injected)
python -m scripts.export_flat_tables                           # data_sep/        one raw CSV per table
python -m scripts.export_flat_tables --layer silver            # data_sep_silver/ one clean CSV per table + quarantine.csv
```

Silver: `data/silver/slv_<dataset>/` (typed, standardised, de-duplicated). Rejected rows:
`data/quarantine/qtn_records/<dataset>/` with `failed_rules`, `primary_rule`, `rejection_type` (direct / cascade)
and the original row. Rules live in `src/transformations/silver/specs.py`: strict cascade; business rules are
`dq_*` flags except refund > payment; invalid emails are blanked and flagged.

Bronze tables: `data/bronze/brz_<dataset>/batch=<batch_id>/*.csv` (raw values + `_` metadata columns).
Run metadata: `data/metadata/meta_{pipeline_runs,stage_runs,table_runs,file_loads,lineage_edges}/<run_id>.csv`.
Re-running is safe: files already loaded (same name + SHA-256) are skipped. Bronze is tied to one generation
run; switching runs needs `--reset-bronze`.
```

Each run writes to `data/generation/<run_id>/`: `landing/` (source CSVs split into historical / batch / stream),
`manifest/` (targets, actual counts, injected issues, file list) and `ground_truth/` (planted patterns, used only for
evaluation). `data/generation/LATEST` holds the newest run id.

## Layout

```
configs/   profiles (small/medium/large), dirty-data budget, cities
src/       generation, ingestion, quality, transformations, ml, rag, orchestration, serving
app/       Streamlit app
policies/  business policy documents for RAG
docs/      architecture, data dictionary, metric definitions, ...
scripts/   CLI entry points
tests/     unit, integration, quality, ml, rag, app, e2e
data/      generated data (git-ignored except data/demo/)
```
