# 14. Data Engineer Workspace (Phase 7)

**Status:** ✅ complete — 9/9 completion checks · **Code:** `src/orchestration/ops_snapshot.py` (7A), `src/serving/queries.py` (15 engineering functions, 7B), `app/views/engineering.py` (7C), `scripts/check_phase7.py` (7D) · **Login:** `engineer` (and `admin`)

## Business question

*Is the platform healthy?* A data engineer needs to know whether last night's run worked, what each step wrote or
rejected, how clean the data is, where every table comes from, and whether the models and the assistant still pass
their checks — without opening files on the pipeline machine.

## 7A Ops snapshot

The hosted app cannot read `data/metadata/` or `data/quarantine/`, so the **publish** stage writes 11 small
`ops_*` tables into `data/demo/` (≈ 0.3 MB in total), each with lineage back to what it is built from:

| Table | Built from | Content |
|---|---|---|
| `ops_pipeline_runs` | `meta_pipeline_runs` | every run: stages, status, duration, error (≤ 400 chars) |
| `ops_stage_runs` | `meta_stage_runs` | time and status per stage |
| `ops_table_runs` | `meta_table_runs` | rows read / written / rejected / cascade / deduplicated / corrupt / flagged per table |
| `ops_quality_results` | `meta_quality_results` | every Silver check: rows affected, outcome |
| `ops_quarantine_summary` | `qtn_records` | rows by dataset · primary rule · direct / cascade · severity, with 3 example record ids |
| `ops_file_loads` | `meta_file_loads` | Bronze files and rows per dataset and slice |
| `ops_lineage` | `meta_lineage_edges` | table-to-table edges with their layers |
| `ops_table_catalog` | declarations + `meta_table_runs` | every Bronze, Silver, Gold, ML and app table: layer, rows, columns, grain, description |
| `ops_model_runs`, `ops_model_metrics` | model registry | versions, train / test windows, metrics |
| `ops_assistant_evals` | `meta_rag_runs` | each assistant evaluation: question set, mode, routing / hit@3 / numbers rates, LLM fallbacks, latency |

**Privacy:** the quarantine summary holds counts, rules and record ids only; run error messages are scrubbed of local paths. The raw rejected rows (20 MB, may
contain customer names and e-mails) are never read by publish and stay on the pipeline machine (check 7.9).

## 7C The four pages

Sidebar group **Data Engineer**: Pipeline runs (landing page) · Data quality & quarantine · Lineage & tables · Model
monitoring · Assistant.

| Page | Shows |
|---|---|
| **Pipeline runs** | Last run, runs recorded, success rate, median run time, last success; time per stage for recent runs (the failed stage outlined in red); run history with status colours; for any run: tables written, rows written, rejected, deduplicated and the per-table table; failed runs with their errors |
| **Data quality & quarantine** | Datasets and rows checked, rows quarantined (direct / cascade), lowest clean rate; clean rate per dataset; quarantined rows by dataset; quarantine detail by rule with example ids; rules that fired; Bronze file loads |
| **Lineage & tables** | A **pipeline map**: one row of boxes left to right (Source files → Bronze → Silver → Gold → ML → App snapshot), each with its number of tables and what it holds, Quarantine below Silver ("rejected"), run metadata and policies & docs behind a toggle; a searchable table catalog; **one table in context**: three columns *Reads from → this table → Used by*, boxes coloured by layer (≤ 10 per side, "+N more"); the transformation catalog. (First version was a Sankey; replaced on user feedback 2026-10-11 as hard to read.) |
| **Model monitoring** | Models in the registry, latest forecast WAPE, assistant evaluations, latest routing rate; the registry; one headline metric per model across runs (forecast WAPE, stockout High-tier precision, recommendation Precision@10); the assistant's evaluation history |

## Results (medium snapshot)

| Measure | Value |
|---|---|
| Pipeline runs recorded | 25 (18 success, 7 failed); median successful run 55 s |
| Rows written across all table runs | 6.3 million |
| Silver checks | 342 results, 128,086 rows affected; mean clean rate 90.0% (lowest: returns_refunds 76.4%) |
| Quarantine | 25,153 rows in 14 datasets — 16,121 direct, 9,032 cascade (largest: order items whose order was rejected, 5,001) |
| Lineage | 248 edges; every Gold and ML table traces back to a source file |
| Table catalog | 72 tables |
| Model monitoring | 4 models; forecast WAPE 0.619 (8 runs), stockout precision 0.205, recommendations Precision@10 0.148; 11 assistant evaluations |

## 7D Verification — `python -m scripts.check_phase7`

Bars fixed before measuring, all 100%. The comparison uses only raw records that existed when the snapshot was
published.

| Check | Result |
|---|---|
| 7.1 Every ops table published and typed | ✅ 11/11 |
| 7.2 Pipeline runs equal the raw run log | ✅ 25 = 25 (18 / 7) |
| 7.3 Table runs equal the raw log (rows written / rejected / deduplicated) | ✅ 286 runs, 6,310,944 rows |
| 7.4 Quarantine rows per dataset equal the raw quarantine files | ✅ 25,153 rows, 14 datasets |
| 7.5 Quality results equal the raw log | ✅ 342 results |
| 7.6 Every Gold and ML table traces back to a source file | ✅ 30/30 (after the fix below) |
| 7.7 The catalog lists every Bronze, Silver, Gold and ML table | ✅ 68/68 |
| 7.8 Role check: engineer and admin allowed, every other role refused | ✅ 15/15 functions |
| 7.9 Privacy (no raw rejected rows, e-mails or local paths) and snapshot < 50 MB | ✅ 33.1 MB (ops 0.32 MB) |

**Tests** (`tests/app/test_app.py`): all 15 functions role-checked; consistency (run counts, quarantine totals,
direct + cascade, clean rates in 0–1, lineage layers, catalog, no `raw_record`); the workspace renders; other roles
blocked; the engineer lands on Pipeline runs; every headline metric present.

## Problems found and fixed

| Finding | Fix |
|---|---|
| **7.6 failed (29/30):** `gld_quality_summary` had no path to a source file — the pipeline never recorded that the quality log is computed from the Bronze datasets during Silver | the tracker now records `brz_<dataset> → meta_quality_results` with every quality result; for runs recorded before, the ops snapshot derives the same edges from the quality log itself (labelled as derived) |
| The published lineage lagged one publish (publish records its own edges only when it finishes) | publish now works out its edges first and includes them in the lineage it publishes; one list is used for both |
| The forecast's metric history was empty (the registry stores `<run>/backtest`, the metrics `<run>`) — the page would have shown "nan" | the join maps the metrics to the backtest registry entry; a test guards all three headline metrics |
| **Before the first push:** run error messages contained full local paths (`C:\Users\<name>\…`), and the snapshot is committed to a public repository | errors are scrubbed before publishing (project folder → `<project>`, a user home → `<home>`); check 7.9 now fails on any local path in the ops tables (verified: it failed on the old snapshot and passes after re-publishing); the whole snapshot was scanned clean |
| On a fresh pipeline the first snapshot had no models: the tracker writes a run's records only when the run finishes, after publish | the ops snapshot also includes the current run's records so far (models, metrics, table runs, quality, lineage), held in memory by the tracker; check 7.3 counts the publishing run's pre-publish table runs |

## 7E Assistant tools for the Data Engineer

**Why:** asked "how many rows are being processed", the engineer's Assistant answered "Out of scope" — it had no
tools for pipeline data and the refusal did not say where the answer was.

**What was added:**

| Part | Detail |
|---|---|
| 5 tools (`src/rag/tools.py`) | `pipeline_status` (last run, runs, failed, success rate) · `pipeline_rows` (rows written / rejected / deduplicated in the last run, by stage) · `data_quality` (clean rate per dataset) · `quarantine` (rows by dataset and rule, direct / cascade) · `model_health` (registry + latest headline metrics) — each wraps the role-checked engineering queries |
| Datasets as entities | "order items" / "order_items" → the dataset (longest name wins, so not "orders"); shown in "understood as" only for engineering answers |
| Role rule | only the Data Engineer and admin get these answers; other roles get "part of the Data Engineer workspace" |
| Better refusal | an unanswerable question now names the dashboard page **of the user's own workspace** that shows it (e.g. "Data Engineer → Lineage & tables") |

**Evaluation — bars unchanged** (`python -m scripts.check_phase6 --set engineering`; questions written before the
tools; figures checked with SQL on the ops tables, which check 7.x verifies against the raw logs):

| Bar | First measurement | After the approved general fix (binding) | Engineering held-out (new) |
|---|---|---|---|
| Routing ≥ 90% | **62.5%** (5/8) — FAIL | **100%** (8/8) | 87.5% (7/8) |
| Numbers = independent SQL | 5/5 | **7/7** | 5/5 |
| Refusal ≥ 90% | **0/1** — FAIL | **1/1** | 2/2 |
| Role check | 2/2 | **2/2** | 2/2 |
| Hit@3, citations | n/a (no retrieval questions) | n/a | citations 2/2 |

**Root causes of the first measurement and the general fix (user decision 2026-10-11, same process as 6F: a new
held-out set written before the fix, one re-measure):**
1. *Tool choice summed overlapping phrases* ("pipeline" + "pipeline run" + "pipeline runs" = 33 beat "how many rows"
   = 13) → matched phrases of one tool now count once (`_distinct`).
2. *Counting questions ignored what is counted* ("how many rows are in quarantine" → rows processed) → each tool
   declares the measure it reports (rows, runs, models); "how many <X>" competes only between tools that report X,
   the one whose topic is named wins, otherwise the measure's primary owner.
3. *Schema questions fell through to a weak knowledge match* ("What columns does the orders table have?" quoted an
   irrelevant forecasting passage) → column / schema / grain / lineage / upstream / downstream questions go straight
   to the refusal with the Lineage & tables pointer.

**Found by the held-out sets and fixed:** the counting rule first read "how many promotions can **run**" as pipeline
runs (a business question refused) → the counted noun now stops at the first verb / auxiliary.

**Known characteristics (documented, not chased):**
- *"What was the rejected **row** count in the latest run?"* (engineering held-out) goes to quarantine — "rejected
  rows" is a quarantine phrase.
- The 6F held-out set is now **17/20** routing (was 18/20): *"What cancellation rate is considered normal?"* used to
  pass only because the old scorer counted "cancellation" / "cancellations" twice; removing the double counting (fix
  1) sends it to the sales tool. Not tuned back, as it is a held-out question. The gold set is unchanged at 6/6.
