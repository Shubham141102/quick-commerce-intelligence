# 13. Business Assistant — RAG (Phase 6)

**Status:** 6A–6F complete (all 6 evaluation bars pass) · 6G–6H in progress · **Plan:** [Project_Plan_v2.md §9.9](Project_Plan_v2.md)

## Principles

- **Numbers only from data functions** — retrieval never estimates a number.
- **Works without an LLM** — template answers by default; an optional Claude layer only rephrases the evidence and must cite it.
- **No vector database** — knowledge is a CSV in the app snapshot; BM25 keyword search runs in memory inside Streamlit (no extra service, no model download, fits Streamlit Community Cloud).
- **Role-aware** — the assistant calls only the role-checked data functions the user's role may use.

## 6A Knowledge base

**Sources (11 documents):**

| Type | Documents | Answers questions like |
|---|---|---|
| **Policy** (5) | `policies/inventory.md` (POL-INV), `delivery.md` (POL-DEL), `cancellation.md` (POL-CAN), `refund.md` (POL-REF), `promotion.md` (POL-PRO) | *What should we do when stock is below the reorder level?* |
| **Metric** (1) | `docs/metric_definitions.md` (MET-DEF) | *How is net revenue calculated?* |
| **Method** (5) | Model cards 05 forecasting (MC-FCST), 06 stockout (MC-STOCK), 08 lost sales (MC-LOST), 09 anomalies (MC-ANOM), 10 marketing (MC-MKT) | *How does the stockout risk tier work?* |

**Policies match the data.** Every threshold is the one the generator and the models use, checked against the code:
- 15-minute delivery SLA (median delivery ~14 min)
- reorder level = ⌈3 × daily demand⌉ + 1, checked nightly at 22:00 IST; order-up-to = reorder + 7 days of demand + 2
- supplier lead time 1–2 days; weekly counts Monday 06:00 IST
- cancellations ~7% of orders, ~79% before dispatch, 4 + 3 allowed reasons
- refunds: full for prepaid post-dispatch cancellations (≤ 48 h), item line for damaged / missing / quality / expired (≤ 72 h); statuses pending / completed / rejected; refund ≤ payment
- promotions: category-wide, 5–30% in 5% steps, 3–14 days

**Chunking (`src/rag/chunking.py`):**

| Rule | Detail |
|---|---|
| Policies and model cards | One chunk per section (`##` / `###`); a section over 200 words → windows of 160 words, 30-word overlap |
| Metric definitions | One chunk per metric (table row), so a citation names the exact metric |
| Tables | Flattened to `column: value` text so their words are searchable |
| Clean-up | Markdown markup and links removed; sections under 12 words dropped |
| Metadata | `chunk_id`, `doc_id`, `doc_type`, `doc_title`, `section`, `words`, `source`, `doc_sha` (SHA-256 of the file, to rebuild when it changes) |

**Published as** `data/demo/rag_chunks.csv` by the publish stage (CSV-only rule); lineage edges link each source file to it.

**Result:**

| doc_type | Documents | Chunks | Words |
|---|---|---|---|
| policy | 5 | 30 | 1,652 |
| metric | 1 | 18 | 435 |
| method | 5 | 53 | 4,998 |
| **Total** | **11** | **101** | 7,085 (avg 70 / chunk, max 191) |

**Tests:** `tests/unit/test_chunking.py` — section split, markup clean-up, 30-word overlap, one chunk per metric, table flattening, and the real knowledge base (all 11 sources present, unique ids, under 200 chunks).

**Problems found and fixed:** the file registry had no group for `src/rag` (docs build failed); publish recorded lineage only for Gold / lookup tables (the first publish swapped the snapshot in, then failed while writing lineage — fixed and re-run cleanly); two policy drafts corrected against the code (refund statuses, discount range).

## 6B Retrieval

**Method:** BM25 Okapi, implemented in `src/rag/retrieval.py` (≈25 lines; the `rank-bm25` package was removed from
the requirements, so the cloud app has one dependency fewer). Index built in memory from `rag_chunks.csv`.

```
score(q, d) = Σ_t∈q idf(t) · f(t,d)·(k1+1) / ( f(t,d) + k1·(1 − b + b·|d|/avgdl) )      k1 = 1.5, b = 0.75
idf(t)      = ln( (N − n(t) + 0.5) / (n(t) + 0.5) + 1 )
```

| Piece | Detail |
|---|---|
| Tokenizer | lowercase, letters / digits, common words dropped, light plural stripping (stockouts → stockout, categories → category) |
| Document text | title + section + section + text (headings counted twice: strong signals) |
| Synonyms | 16 fixed groups expand the question (out of stock → stockout, late delivery → SLA, money back → refund, discount → promotion, …); whole phrases only, so "ideal" never matches "deal" |
| Filter | optional doc-type filter (policy / metric / method), used by the router |
| Output | top 3 hits above `MIN_SCORE`, each with chunk id, score and a citation: `POL-INV §3 Reorder rules`, `Metric Definitions › Net revenue`, `Demand Forecasting › Results` |

**Threshold calibration (practice questions only, never the evaluation set).** *(First version — revised in 6F: the threshold is now per √question-word, 3.24.)* `tests/rag/dev_questions.yaml`
holds 10 practice questions (7 in scope, 3 off-topic); `python -m scripts.calibrate_retrieval`:

| Practice question | Top hit | Score | Expected in top 3 |
|---|---|---|---|
| When should a store reorder a SKU? | POL-INV §3 Reorder rules | 15.57 | yes |
| What is our delivery promise to customers? | POL-DEL §1 Delivery promise | 39.76 | yes |
| Can a refund be larger than the payment? | POL-REF §3 Refund amount limit | 14.22 | yes |
| What discount levels are allowed for promotions? | POL-PRO §4 Reviewing and stopping promotions | 22.20 | yes (§2 at rank 2–3) |
| Which cancellation reasons are allowed after dispatch? | POL-CAN §2 Allowed reasons | 24.67 | yes |
| How is average order value calculated? | Metric Definitions › Average order value (AOV) | 26.66 | yes |
| How do we estimate the sales lost to stockouts? | Metric Definitions › Lost sales (estimate) | 11.85 | yes (model card in top 3) |
| What is the capital of France? | — | 0.00 | out of scope |
| Will it rain in Mumbai tomorrow? | POL-DEL §1 Delivery promise | 3.83 | out of scope |
| Who won the cricket match yesterday? | Business workspace › Results | 4.26 | out of scope |

In-scope hit@3 **7/7**. Lowest correct in-scope score **11.31**, highest out-of-scope score **4.26** — cleanly
separable, so **`MIN_SCORE = 7.8`** (the midpoint). Off-topic questions therefore retrieve nothing; the router (6D)
adds a second screen.

**Tests:** `tests/unit/test_retrieval.py` — tokenizer, synonyms, ranking, type filter, threshold, citations, core
questions on the real knowledge base.

**Problems found and fixed:** `rank-bm25` was listed but not installed — replaced by the in-house implementation;
one test assumed a type-filtered search would return nothing, but synonym expansion legitimately matched — the
test now checks what the filter guarantees (only the requested type is returned).

## 6C Data tools and entities

**The only way the assistant gets numbers.** `src/rag/tools.py` holds 16 whitelisted tools; each wraps an existing
role-checked function in `src/serving/queries.py`, so access rules are inherited (a tool outside the user's
workspace raises `AccessDenied`, exactly like the pages). No free-form SQL.

| Workspace | Tools |
|---|---|
| Inventory | `inventory_overview` · `at_risk_skus` · `suggested_orders` · `category_forecast` (+ reliability) · `forecast_reliability` · `lost_sales` |
| Business | `sales_overview` (vs previous period) · `top_products` · `delivery_overview` · `cancellations` · `anomalies` |
| Marketing | `segments` · `bought_with` · `recommendation_accuracy` · `retention` · `promotions` |

**Every tool:**
- validates inputs — Pydantic types (`extra="forbid"`, so an injected parameter such as `sql` is rejected; at most 20
  rows) plus checks against the published lookups (known stores / categories / products) and the data period;
- returns a `ToolResult`: headline numbers (formatted), a table, the source table(s), the as-of date and the scope;
- drops customer-identifier columns — answers are aggregates (a `customers` count is allowed, an id is not).

**Entities (`src/rag/entities.py`)** turn words into parameters: store area or id ("Dwarka", "S01"), city (all its
stores), category (full name or part: "dairy"), product (longest name match, size ignored), risk tier ("high risk";
not "highly"), anomaly type, top-N, and periods **relative to the end of the data (30 Sep 2025)**, not today:
"last week" → 24–30 Sep, "last 30 days", "July", "last month" → September; "May I…" is not a month.

**Shared rule moved:** the forecast reliability bands now live in `src/serving/reliability.py`, used by both the
Inventory page and the assistant (one definition).

**Tests:** `tests/unit/test_entities.py` (every entity type, period edge cases); `tests/app/test_rag_tools.py` on a
real snapshot — each role sees only its tools, all 16 tools run with a source and no customer ids, role refusals,
input validation (dates outside the data, unknown store, missing category, 500 rows, injected `sql`), and **tool
numbers equal the dashboard functions** (High / Medium risk counts, net revenue).

**Problems found and fixed:** one tool's source label named the wrong Gold table (`top_products` reads
`gld_product_performance`); the PII test was too strict (it flagged the aggregate `customers` count) and now checks
identifier columns only.

## 6D Router, composer and assistant

**Flow (`src/rag/assistant.py`):** question → entities → route → tool and / or retrieval → composed answer.

**Router (`src/rag/router.py`) — rules, no LLM, checked in this order:**

| # | Route | Triggered by | Example |
|---|---|---|---|
| 1 | insufficient | "why / what caused / reason for" + a data topic | *Why did revenue drop in September?* → the numbers, explicitly no cause |
| 2 | method | "how is / was … calculated / chosen / estimated / defined", definition, formula | *How is net revenue calculated?* (method even though "revenue" is a data topic) |
| 3 | hybrid | a policy cue + a data topic + a data cue or an entity | *Which SKUs are at high risk and what does the policy say?* |
| 4 | policy | should / allowed / how long / what happens when / rule / policy … | *How long do customers have to ask for a refund?* |
| 5 | method | "what does X mean" with no data topic | *What does lift mean?* |
| 6 | data | a data topic; tool = longest matching trigger phrase | *How many at risk customers?* → `retention` (beats "at risk" → stock) |
| 7 | (search all) | anything else; **out of scope** when no chunk clears the threshold | *Will it rain in Mumbai tomorrow?* |

**Special replies:** greeting / "what can you do" → help with example questions for the user's workspace;
missing detail → clarify ("Please name a category"); "why" with no data topic → clarify.

**Role rule:** a tool from another workspace is refused before it runs — *"Sales performance is part of the
Business & Revenue workspace, which your role cannot open"* — with no number in the reply; the policy part of
the question is still answered. The wrapped query checks the role again (defence in depth).

**Composer (`src/rag/composer.py`) — the house style:** title with scope and as-of; headline numbers as
`Label — value`; the tool's table; for guidance the **1–2 sentences of a chunk that share the most words with the
question**, quoted, each with its citation; an "understood as" line (what was recognised); sources (Gold tables
with as-of, document citations). Numbers only from a ToolResult, quotes only from retrieved chunks.

**Log record per answer:** question, role, route, tool, citations, sources, mode, latency (→ `meta_rag_runs` in 6F).

**Tests:** `tests/unit/test_router.py` (16 phrasings, whole-word matching, quote selection);
`tests/app/test_assistant.py` on a real snapshot — data numbers equal the dashboard, hybrid answers cite the
inventory policy, policy / method answers cite, the inventory role is refused revenue **with no ₹ figure in the
reply**, why-questions, out of scope, help, clarify, log record. These questions are deliberately not reused in
the 6F evaluation set. **141 tests pass** (unit + app).

## 6E Assistant page

`app/views/assistant.py`, registered in `app/Home.py` as the last page of each persona's sidebar group (the admin
has it next to Overview; its breadcrumb reads "All workspaces").

| Element | Detail |
|---|---|
| Header | Breadcrumb, "Assistant", purpose; on the right the active mode — **Template mode (no LLM)** |
| Suggested questions | 3 buttons for the persona (4 mixed for the admin), from `composer.EXAMPLES` |
| Chat | `st.chat_input`; history kept for the session; **Clear conversation** |
| Each answer | Route badge (Data · Data + policy · Policy · How it works · Data (no causal claim) · Not in your workspace · Need more detail · Out of scope · Help), the answer in the house style, its table (tier / severity colours), an **Evidence & sources** expander: understood as · source tables with as-of · citations · response time · mode |
| Speed | `Assistant` (BM25 index + lookups) cached with `st.cache_resource`, keyed by the snapshot's pipeline run, so a new publish rebuilds it |
| Log | one `log_record` per answer kept in the session (`assistant_log`) |

**Tests** (`tests/app/test_app.py`): a hybrid question renders the badge, the High-risk numbers, a `POL-INV §`
citation, a table and the sources expander; the inventory role asking for revenue sees "Not in your workspace" with
no ₹ figure; clicking a suggested question renders a question and an answer. **144 tests pass.**

## 6F Evaluation

**Sets** (written before measuring; never used for tuning):
- `tests/rag/questions.yaml` — the **gold set**: 40 questions (10 policy · 5 method · 10 data · 8 hybrid · 4 ambiguous ·
  3 out-of-scope) + 5 role checks. **The bars apply here.**
- `tests/rag/holdout_questions.yaml` — a **held-out set**: 20 questions + 2 role checks, written after the first run
  and **before** any fix, to show whether the fixes generalise.
- Practice set for calibration only: `tests/rag/dev_questions.yaml`.

**How each bar is checked (`scripts/check_phase6.py`):** routing = right route and, for data / hybrid, right tool;
hit@3 = an acceptable section among the top 3 chunks; **numbers = the figure in the answer equals a separate SQL query
on the Gold tables** (not the assistant's own code); **citations = every quoted sentence literally exists in the chunk
it cites**; refusal = out-of-scope / ambiguous questions refused or clarified with nothing invented; role = another
workspace's data refused with no table and no ₹ figure. Every answer is logged to
`data/metadata/meta_rag_runs/<run>.csv`.

### First measurement (2026-10-08): 4 passed, 2 failed

| Bar | Result |
|---|---|
| 6F.1 Routing ≥ 90% | **85.0%** (34/40) — FAIL |
| 6F.2 Hit@3 ≥ 85% | 87.0% (20/23) |
| 6F.3 Numbers 100% | 18/18 |
| 6F.4 Citations 100% | 33/33 |
| 6F.5 Refusal ≥ 90% | **85.7%** (6/7) — FAIL |
| 6F.6 Role check 100% | 5/5 |

**Root causes (diagnosed with retrieval scores):**
1. *The threshold penalised short questions.* BM25 scores grow with the number of question words; the raw threshold
   (7.8, calibrated on longer practice questions) rejected 3-word questions whose top hit was correct.
2. *Policy questions without cue words went to data tools* ("Which reasons can be recorded for a cancellation…",
   "How quickly is a refund issued…", "How many minutes do we promise for delivery?").
3. *Exact word matching* — "lapsing" ≠ "lapsed", "detected" not a known method verb.

### Approved revision (user decision 2026-10-08) — general fixes only

The user chose **general fixes + a held-out set + one re-measure** (bars unchanged). Nothing was aimed at a specific
question:

| Fix | Change |
|---|---|
| (a) Length-normalised threshold | score ÷ √(question words). Compared on the practice set only: raw (penalises short questions), per word (dilutes two-part questions), per known word (not separable — a one-word off-topic match scored high), **per √word (separable: correct ≥ 4.28, off-topic ≤ 2.21 → `MIN_SCORE = 3.24`)** |
| (b) Stem matching | one stemmer for router and retrieval with -ing / -ed (lapsing ≈ lapsed, detected ≈ detect); router phrases matched on stems, hyphens ignored; passive "how is / are X -ed" = method question |
| (c) Content-based policy detection | a data topic resting on exactly one generic word ("delivery", "cancellation"), with no report cue (show / list / total …) and no entity, is answered from policy when a policy section clears the threshold |

Unit / app tests (not the evaluation) caught two side effects before the re-measure, both refined within the same
fixes: dividing by every word diluted two-part questions, and the generic-word rule must count one word (stems
deduplicated: "delivery" / "deliveries" are one). One two-part practice question was added to the practice set for
the calibration (recorded in the file).

### Re-measure (binding) and held-out result

| Bar | Gold set (bars apply) | Held-out set (generalisation) |
|---|---|---|
| 6F.1 Routing ≥ 90% | **92.5%** (37/40) ✅ | 90.0% (18/20) |
| 6F.2 Hit@3 ≥ 85% | **87.0%** (20/23) ✅ | 100% (12/12) |
| 6F.3 Numbers = independent SQL 100% | **18/18** ✅ | 9/9 |
| 6F.4 Citations correct 100% | **42/42** ✅ | 24/24 |
| 6F.5 Correct refusal ≥ 90% | **100%** (7/7) ✅ | 66.7% (2/3) |
| 6F.6 Role check 100% | **5/5** ✅ | 2/2 |

Latency: median 14 ms (gold), 17 ms (held-out). **All six bars pass on the gold set.**

### Known characteristics (no further changes)

- **Short definition questions** can fall below the threshold and are refused as out of scope (gold M02 "What does
  days of inventory mean?", M04 "How are demand spikes detected?"; held-out HM3 "What does support mean for basket
  rules?"). Never answered wrongly.
- **A "why" question whose topic word is not a data topic** is refused instead of clarified (held-out HA1 "Why did
  stockouts rise at Dwarka?") — the held-out refusal score (2/3) rests on only 3 questions.
- **A rule question naming a data topic plus an entity** is answered as data + policy instead of policy only (gold P03);
  the policy guidance is still included.
- Every remaining miss is conservative — a refusal or extra data — never a wrong number or a false citation.
