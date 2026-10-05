# 2. Bronze Ingestion

**Status:** complete · **Code:** `src/ingestion/`, `src/common/spark_io.py`, `src/orchestration/` · **Run:** `python -m scripts.run_pipeline --stages ingest`

## Purpose

Bronze is the raw layer. It loads every landing CSV into project tables **exactly as received** (dirty values included), adds metadata saying where and when each row came from, and keeps a log of what it loaded. It fixes nothing; cleaning is Silver's job. Keeping the raw values means any later result can be traced back to the original record.

## Input and output

| | Location |
|---|---|
| Input | `data/generation/<run_id>/landing/{historical,batch,stream}/<dataset>/*.csv` (1,864 files) |
| Output | `data/bronze/brz_<dataset>/batch=<batch_id>/part-*.csv`, 19 tables |
| Load log | `data/metadata/meta_file_loads/<run_id>.csv` |
| Run log | `data/metadata/meta_{pipeline_runs,stage_runs,table_runs,lineage_edges}/<run_id>.csv` |

## Steps

1. **Find new files.** List the landing files and compute each file's SHA-256. A file whose (dataset, name, SHA-256) is already in the load log is skipped, so re-running never duplicates rows.
2. **Read every column as text** with an explicit schema. A value like `"unknown"` in an amount column arrives as written instead of failing or turning into null.
3. **Keep unparseable rows.** Rows with the wrong number of fields are kept, with the raw line in `_corrupt_record`.
4. **Add metadata columns:** `_batch_id`, `_pipeline_run_id`, `_source_file`, `_load_type`, `_source_system`, `_schema_version`, `_ingestion_ts`, `_ingestion_date`, `_record_hash` (SHA-256 of the row), `_is_late`, `_beyond_watermark`. Definitions: [data_dictionary.md](data_dictionary.md#bronze-columns-added-to-every-table).
5. **Write safely.** Write to a staging folder, count the rows back, then rename into place. Bronze only adds new batch folders and never overwrites.
6. **Record** per-file row and corrupt counts, per-table timings, and lineage (landing → `brz_<dataset>`).

## The four ingestion patterns

| Pattern | Files | Mechanism |
|---|---|---|
| A. Batch | daily files | Spark batch read, one Bronze batch per run |
| B. Historical backfill | monthly files + reference tables | same reader, loaded first, `_load_type = historical` |
| C. Simulated stream | hourly drops (24–30 Sep) | **Spark Structured Streaming**: file source, up to 24 files per micro-batch, `trigger(availableNow)`, checkpoint. Each micro-batch is one Bronze batch |
| D. Logs | application logs | same reader; malformed rows kept in `_corrupt_record` |

**Late data in the stream.** Bronze never drops rows. It labels them:
- `_is_late`: the event happened before the hour its file covers.
- `_beyond_watermark`: the event is older than a **2-hour watermark** (latest event time seen in earlier micro-batches minus 2 hours), i.e. a watermarked streaming aggregation would drop it.

The watermark is deliberately shorter than the generator's maximum lateness (6 hours) to show the trade-off.

## Design decisions

- **Text-only Bronze:** typing happens in Silver, so no value is ever lost or altered at load time.
- **Append-only, tied to one generation run:** loading a different generation run is refused unless `--reset-bronze` is given, because the two runs' IDs would collide.
- **Metadata log is the source of truth for idempotency,** also for the stream: if a checkpoint is lost, already-loaded files are filtered out instead of loaded twice.
- **Windows specifics** (handled in `src/common/spark.py`): Hadoop helpers (`winutils.exe`, `hadoop.dll`) in `tools/hadoop/bin`; Spark's Python workers pinned to the project venv; space-free 8.3 paths for Spark's `.cmd` launchers.

## Results (medium run)

| Measure | Value |
|---|---|
| Files loaded | 1,864 of 1,864, none twice |
| Bronze rows | **307,784** = landing rows exactly, all 19 tables |
| Malformed log rows kept | 103 (= injected) |
| Stream micro-batches | 4–7 per stream dataset; 54 late orders; a few rows beyond the 2-hour watermark |
| Runtime | ~100 s (clean rebuild) |
| Re-run | 0 new rows, every file skipped |

## How it is verified

- `python -m scripts.check_bronze`: **9/9 checks**. Every dataset has a table, row counts match the manifest, every file loaded once, values identical to the landing files (dirty values preserved), malformed rows kept, metadata filled, all patterns used, latest run succeeded, lineage recorded.
- `tests/integration/test_bronze.py`: the same properties on the small profile, plus re-run idempotency and refusal of a different generation run.

## Problems found and fixed

- **Spark trims whitespace when writing CSV** by default, which would silently "fix" dirty values like `" delivered"`. Disabled; a test compares Bronze with the landing files value by value.
- **Crash on the first run:** `order_items` has no timestamp, so the watermark code failed. After the fix, the re-run *without* reset skipped everything already loaded and still matched the manifest exactly (a real recovery test).
- **PySpark converts timestamps to Python using the PC's timezone (IST).** The stream code passes timestamps as UTC text to avoid a 5.5-hour shift.
- **Spark's `.cmd` launchers break on spaces** in the project path ("Complete new Project"); solved with Windows short paths.

## Presentation copy

`python -m scripts.export_flat_tables` merges each table's batches into one file per table in `data_sep/` (19 files, raw values, sorted by key so duplicates sit together).
