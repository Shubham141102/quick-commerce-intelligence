# 15. Viva Preparation — Likely Questions and Answers

**Purpose:** questions a reviewer is likely to ask while the project is demonstrated, with short answers that can be said out loud. Every figure comes from the process documents (01–14); the source is linked so each answer can be backed up on screen.

**How to use it:** answer in 2–3 sentences first, and give the detail only if asked. Questions marked ⚠️ are traps: the honest answer is a limitation, and saying so plainly earns more credit than defending it.

---

## A. The story and the problem

**1. What problem does this project solve?**
Quick-commerce dark stores (10-minute delivery) carry a small stock of each product, so an item runs out within hours, and every empty shelf is a lost sale. Managers need to know what will sell, what will run out, and how much to reorder. The business also needs clean, trusted numbers for sales, delivery and marketing. The platform takes raw, messy operational data all the way to those answers.

**2. Who are the users?**
There are three personas, each with its own workspace, plus an admin who sees everything:
- **Inventory manager:** keep shelves full
- **Business & Growth manager:** revenue, the delivery promise, customers and growth levers
- **Data engineer:** pipeline, data quality and models

Business and Marketing were one persona each at first. I merged them because they shared revenue, promotions and customers ([16](16_persona_consolidation.md)). Each person logs in and lands on their own pages. They can't query another role's data, not even through the assistant.

**3. Why synthetic data instead of a real dataset?**
Synthetic data gives three things real data can't. ([01](01_data_generation.md))
- **Known dirt:** exactly 10% of the rows are dirty, and every issue is logged, so the cleaning can be checked against the truth.
- **Planted signal:** customer personas, product pairs, anomalies and stockouts are planted on purpose, so the models can be scored against ground truth.
- **Reproducibility:** the same seed always gives the same data.

**4. ⚠️ Doesn't synthetic data make the results meaningless?**
It makes them *verifiable*, not meaningless. With real data I can't prove the anomaly detector found the real anomalies, because nobody knows what they were. The limitation is that real data has patterns I didn't simulate, so the exact accuracy figures won't carry over. The **pipeline, checks and methods** do carry over.

**5. How big is the data?**
- 36,000 orders, 122,400 order lines, 6,000 customers, 600 products, 20 categories and 12 dark stores in 3 cities
- April to September 2025, across 19 datasets
- 1,864 landing files and 307,784 Bronze rows

([01](01_data_generation.md), [03](03_silver_transformations.md))

**6. ⚠️ That's small. Why use Spark at all?**
Spark is used where the design must scale: ingestion, cleaning and Gold aggregations. That code works unchanged on 1,000× the data. For the ML step, the tables are small and bounded, so it runs in pandas and scikit-learn. Using Spark there would add overhead and give nothing. Choosing the tool by data size is a deliberate decision, written down in the plan.

---

## B. Architecture and data engineering

**7. What is the Medallion architecture?**
It's a layered pipeline:
- **Bronze** stores the raw data exactly as received.
- **Silver** holds the cleaned, typed, de-duplicated, validated data.
- **Gold** holds the business tables and ML features.

Each layer has one job, so if a number looks wrong you can trace it back layer by layer to the original record.

**8. Why store everything as CSV, not Parquet or Delta?**
It was a project constraint: CSV only. The cost is that CSV has no types, no in-place updates and no transactions. ([02](02_bronze_ingestion.md), [03](03_silver_transformations.md)) I handle each of those:
- Bronze reads every column as text, and Silver applies the types.
- Silver and Gold are **rebuilt in full** each run, which makes re-runs safe.
- Writes go to a staging folder, the rows are counted back, and only then is the folder renamed into place.

In production I'd use Delta or Iceberg.

**9. What happens if the pipeline runs twice? Do you get duplicates?**
No. Bronze records the SHA-256 hash of every file it loads, and a file already in the load log is skipped. Silver is rebuilt from all of Bronze and de-duplicated on business keys, so a re-run gives the same result. ([02](02_bronze_ingestion.md))

**10. What ingestion patterns do you support?**
There are four:
- daily batch files
- a historical backfill (monthly files)
- a **simulated stream** using Spark Structured Streaming: hourly files, `availableNow` trigger, checkpoints
- log files, where malformed rows are kept

**11. How do you handle late-arriving data?**
Bronze never drops late rows; it labels them. `_is_late` marks an event that happened before the hour its file covers. `_beyond_watermark` marks an event older than a 2-hour watermark, which a streaming aggregation would drop. The watermark is deliberately shorter than the maximum lateness (6 hours) to show that trade-off. Silver is a full rebuild, so late rows are still included. ([02](02_bronze_ingestion.md))

**12. Why doesn't Bronze clean anything?**
So no information is ever lost at load time. If a cleaning rule turns out to be wrong, I fix the rule and rebuild Silver from the untouched raw data.

**13. What does Silver do?** ([03](03_silver_transformations.md))
It runs seven steps for every dataset:
1. separate unparseable rows
2. standardise (trimming, case, spelling)
3. parse and cast types; five timestamp formats are accepted
4. de-duplicate on the business key
5. validate: required values, ranges, allowed values, time order, foreign keys
6. split the rows into kept and quarantined
7. derive helper columns

**14. What is quarantine, and why not just delete bad rows?**
A rejected row goes to a quarantine table, together with the rule it failed and its original values. Nothing disappears silently, every rejection can be counted and explained, and the data engineer page shows quarantine by dataset and by reason.

**15. What is a "cascade" rejection?**
If an order is rejected, its order items, payments and deliveries are rejected too, because they point to a parent that no longer exists. They're labelled `cascade`, not `direct`, so the knock-on effect is visible. This keeps Silver's referential integrity perfect.

**16. Do suspicious rows always get rejected?**
No:
- A row that breaks a **hard** rule (bad type, missing key, orphan) is quarantined.
- A real but suspect value (a soft business rule) gets a `dq_*` flag and stays in Silver.
- The one exception is a **refund larger than its payment**, which is financially impossible, so it's quarantined.

**17. What is Gold, and what is "grain"?**
Gold has the business tables. Grain is what one row means, for example "one store × category × day". Fixing the grain prevents double counting. ([04](04_gold.md))

**18. How do you make sure every dashboard shows the same revenue?**
Every metric is defined **once**, in `src/serving/metrics.py`. The dashboards, the models and the assistant all use that one definition, and the metric definitions document is generated from that code, so it can't drift. ([04](04_gold.md))

**19. Define net revenue. Which orders count?**
- Only **completed** orders count: delivered, paid, and not cancelled.
- GMV = Σ quantity × price.
- Net revenue = GMV − promotion discount − completed refunds.
- Refunds are counted on the refund date.

**20. ⚠️ Why does AOV exclude refunds?**
Refunds are dated when they happen, not when the order was placed. Mixing them into a daily average would distort it. Net revenue includes them.

**21. How do you track lineage?**
Every stage writes metadata: pipeline runs, stage runs, table runs (rows read, written, rejected), quality results, and lineage edges (source → table). The Data Engineer workspace shows this as a simple box diagram of the pipeline and of each table's neighbours. ([14](14_data_engineer_workspace.md))

**22. How do you know the pipeline is correct?**
Every phase has completion checks with **pass bars fixed before measuring**. For example:
- Bronze 9/9
- Silver 10/10
- Gold 10/10
- check_phase7 9/9

There are also more than 230 automated tests, and the cleaning is checked against the generator's log of injected issues.

---

## C. Demand forecasting

**23. What does the forecast predict?**
How many units each store will sell in each category over the next 7 days, and also for each focus SKU (each store's 25 most popular products). ([05](05_demand_forecasting.md))

**24. Which model, and why?**
One **HistGradientBoostingRegressor** across all 240 series (12 stores × 20 categories) with **Poisson loss**:
- One global model learns from every store and category at once, which is better than 240 tiny models.
- Poisson loss suits daily unit counts, which are small, non-negative integers.
- Gradient boosting handles mixed features without heavy tuning.

**25. What features?**
There are 17: lags of 1, 7 and 14 days; rolling mean and standard deviation; day of week, weekend, holiday, month and day of month; promotion active; rainfall and temperature; yesterday's stockout share; store and category.

**26. ⚠️ How did you avoid data leakage?**
- Lags and rolling statistics use only **earlier** days; this is verified in Gold and again in ML.
- Training runs April–August and the test is September, which the model never sees during training.
- For days 2–7, the model's **own** forecasts fill the lag features (recursive forecasting), exactly as it would work live.

**27. How did you evaluate it?**
A September backtest: the model forecasts 1–7 days ahead from every day of September, giving 45,360 forecasts. I compared it with two baselines, seasonal naive and a 7-day moving average. The main metric is **WAPE** = Σ|error| ÷ Σ actual.

**28. ⚠️ WAPE 0.62 sounds bad. Is the model any good?**
Yes, in context. An average store-category-day sells only 3.3 units, and 17% of days sell nothing, so much of the error is pure randomness. ([05](05_demand_forecasting.md))
- Seasonal naive: 0.87
- Moving average: 0.69
- Our model: **0.62**
- A *perfect* model on this data: about **0.58**

So the model closes about two-thirds of the gap between the best baseline and the best achievable result.

**29. Why WAPE and not MAPE?**
MAPE divides by each actual value, and 17% of actuals are zero, which means division by zero. WAPE weights errors by volume, which is what matters for stock.

**30. Why down-weight stockout days in training?**
On a stockout day, sales understate real demand because the shelf was empty. Training on those days at full weight would teach the model to under-forecast. Their weight is 1 − 0.5 × stockout share.

**31. Do you give an uncertainty range?**
Yes. Two extra gradient-boosting models with quantile loss give a 10th–90th percentile band.

**32. Why does the app show "reliability" instead of WAPE tables?**
A manager doesn't think in WAPE. Each category is shown as **High, Medium or Low** reliability: below 0.55 is High, 0.55–0.70 is Medium, above 0.70 is Low. That tells the manager how far to trust the number.

---

## D. Stockout risk, replenishment and lost sales

**33. How is stockout risk decided?** ([06](06_stockout_replenishment.md))
With transparent rules, not a black box:
- **High:** stock is 0, or the forecast demand over the next 3 days is at least the stock.
- **Medium:** stock is at or below the reorder level, or there are fewer than 3 days of cover.
- **Low:** everything else.

Every flag lists its reason codes, so it explains itself.

**34. How much does it suggest reordering?**
`suggested_qty = ceil(max(0, forecast demand over lead time + review period + safety stock − current stock))`, with safety stock = 1.65 × standard deviation of daily sales × √lead time. That gives a 95% service level, with a 2-day lead time and a 1-day review period.

**35. ⚠️ Does it place orders automatically?**
No. Everything is advisory, and a human decides. That's deliberate: a wrong automatic order costs money.

**36. How good is the risk flag?**
In the backtest, the High tier has **20.5% precision**. That's about twice the naive "stock ≤ reorder level" rule (11.6%) and 4.7× the base rate (4.4%).

**37. ⚠️ 20% precision means 80% false alarms. Why is that useful?**
A stockout is rare (4.4% base rate), so any short list will contain false alarms. The question is whether it beats the alternatives, and it beats both the naive rule and chance by a wide margin. Recall is limited because many stockouts come from planted **supplier failures**, which no stock or demand signal can predict. The precision/recall trade-off is shown on screen rather than hidden.

**38. How do you estimate lost sales?** ([08](08_inventory_explorer_lost_sales.md))
On each stockout day:
- expected units = average sales on the **in-stock** days of the previous 28 days
- lost units = max(0, expected − actual)
- lost revenue = lost units × catalog price

The estimate came to 811 lost units against the simulator's true 809.

**39. ⚠️ Is that really money lost?**
It's money lost *for that SKU*. Shoppers often bought a substitute, so the store as a whole lost less. The app says so.

---

## E. Business analytics and anomaly detection

**40. How are anomalies detected?** ([09](09_business_workspace.md))
With statistical tests on counts. Each case is compared with what's expected from the previous 28 days, and is flagged when the probability of seeing it by chance is tiny:
- **Demand spikes:** per product × store × day, using a negative binomial model
- **Payment failures:** per hour, network-wide
- **Store outages:** marked experimental

**41. Why negative binomial and not Poisson?**
The first version used Poisson, which assumes variance = mean. The real variance was almost double (mean 15.6, variance 27.7), so ordinary busy days looked "impossible", and precision failed at 23%. The negative binomial allows extra variance. After this approved revision, precision is **64%** (7 of 11).

**42. ⚠️ Why are store outages "experimental"?**
A store gets less than one order an hour, so a 3-hour outage can't be told apart from a quiet spell using orders alone. Real systems use store heartbeats, which this data doesn't have. I said so instead of claiming a detector that doesn't work.

---

## F. Marketing analytics

**43. How did you segment customers?** ([10](10_marketing_analytics.md))
With K-means on standardised behaviour features: spend, frequency, recency, AOV, basket size, night / weekend / promotion share, and category mix.
- k was chosen from 3–8 by silhouette, giving 7 segments.
- Recovery of the planted personas: ARI **0.505** (bar 0.3).
- Stability across seeds: ARI **0.84**.

**44. ⚠️ A silhouette of 0.078 is very low. Are the segments real?**
A low silhouette means customers lie on a continuum rather than in separated clusters, which is normal for behaviour data. That's why I didn't rely on silhouette alone. The segments recover the planted personas (ARI 0.5), are stable across random seeds (0.84), and are named from their measured traits, never from preset names.

**45. How do basket rules work?**
Each product pair becomes rules A → B, where confidence = P(B | A) and lift = confidence ÷ P(B). A rule is kept when confidence ≥ 10% and lift ≥ 2, with thresholds set in advance. That gives 352 rules, which include **23 of the 25** planted pairs.

**46. How do recommendations work?**
A hybrid score per customer: buy-again frequency + co-purchase strength + popularity. The weights were chosen on **August** from a grid fixed in advance, then tested **once** on September. Precision@10 is 14.8%, against 14.0% for popularity alone.

**47. ⚠️ +0.8 points over popularity is tiny. Why bother?**
Accuracy is nearly equal, but the hybrid recommends **17× more distinct products** (coverage 34% against 2%). Popularity shows everyone the same 10 items; the hybrid helps discovery. I also disclose that the first version lost to popularity and was revised once, with approval.

---

## G. The business assistant (RAG)

**48. What is RAG?**
Retrieval-Augmented Generation. Before answering, the system **retrieves** relevant facts and builds the answer from them, instead of letting a language model answer from memory. That's what makes the answers grounded and citable.

**49. How does your assistant work?** ([13](13_rag_assistant.md))
1. A **rule-based router** decides what kind of question it is: data, policy, method, hybrid, "why", help, clarify, out of scope, or not your workspace.
2. Data questions run one of **22 whitelisted tools**, each a role-checked SQL query.
3. Policy and method questions retrieve passages with **BM25** from 101 chunks (5 policies, metric definitions, model cards).
4. A template writes the answer, with numbers only from the tool and quotes only from retrieved chunks, each with a citation.
5. An optional LLM rephrases the answer, and is checked afterwards.

**50. ⚠️ Why no vector database or embeddings?**
- The app runs on Streamlit Community Cloud, which has no room for a separate database service or a model download.
- The knowledge base is small (101 chunks of business text with specific terms), and keyword search does well on that.
- BM25 reaches hit@3 of 87% on the gold set, and 100% on the held-out set.

The design keeps retrieval swappable. Embeddings would be the next step for a larger, more paraphrase-heavy knowledge base.

**51. What is BM25?**
A keyword-ranking formula. A chunk scores higher when it contains the question's words, especially rare words, with diminishing returns for repeats (k1 = 1.5) and a correction for long chunks (b = 0.75). I implemented it myself in about 25 lines, which removed a dependency.

**52. How does it know when it *doesn't* know?**
There's a minimum score threshold, calibrated on practice questions and divided by √(question words) so short questions aren't penalised. Below it, the assistant says it has no relevant information instead of guessing. It also refuses "why did X happen" causal claims: it shows the numbers and states that the data can't prove a cause.

**53. Why not let the LLM write SQL (text-to-SQL)?**
For safety and correctness. Generated SQL can be wrong, can read tables the user isn't allowed to see, or can be manipulated by a prompt. Whitelisted tools mean every query is reviewed and role-checked in advance, and every number is reproducible.

**54. Which LLM, and how do you stop it hallucinating?**
`gpt-oss-120b`, an open-weight model, through Groq's free API. It's **optional**; the assistant works fully without it. Its output is verified before it's shown:
- every number in the text must exist in the tool's evidence (number words too)
- every citation must survive
- the length is limited

If any check fails, the user sees the template answer instead. The LLM can only reword verified facts.

**55. How did you evaluate the assistant?**
With question sets written **before** building, and six bars:

| Bar | Pass level | Gold set result |
|---|---|---|
| Routing | ≥ 90% | 92.5% |
| Retrieval hit@3 | ≥ 85% | 87% |
| Numbers equal independent SQL | 100% | 100% |
| Citations correct | 100% | 100% |
| Refusals | ≥ 90% | pass |
| Role checks | 100% | 100% |

The numbers are checked against **separate SQL** that doesn't use the assistant's code. **Held-out** sets, never used for tuning, measure honest generalisation (for example, held-out routing 90%).

**56. ⚠️ Didn't you just tune until the tests passed?**
No, and that's the rule I followed throughout:
- Bars are fixed before measuring and never lowered.
- When a measurement failed, I made only **general** fixes, wrote a **new held-out set before** fixing, and re-measured once.
- Every miss and revision is disclosed in the docs, including misses that remain as known limitations.

**57. Can a marketing manager get inventory numbers from the assistant?**
No. The role check runs **before** any tool executes, and the same `@workspace` decorator that protects the dashboards protects the tools. Every evaluation set includes role checks, and they all pass at 100%.

**58. ⚠️ What questions does it handle badly?**
- Phrasings far from its trigger phrases can be mis-routed; it's rule-based, not semantic.
- It doesn't hold multi-turn context.
- It doesn't answer schema / column questions; it points to the catalog page instead.

These are documented as known limitations, with held-out misses listed.

---

## H. Application, security and deployment

**59. What's the tech stack?**

| Layer | Technology |
|---|---|
| Pipeline | Python 3.11, PySpark 3.5 (Java 17) |
| ML | pandas, scikit-learn |
| Serving | DuckDB, in memory over the published CSV snapshot |
| App | Streamlit with Plotly charts |
| Assistant | in-house BM25, optional Groq LLM |
| Testing | pytest, ruff |
| Hosting | Streamlit Community Cloud from GitHub |

**60. Why DuckDB in the app?**
The app needs fast SQL over CSV files with no server. DuckDB runs inside the Python process, reads CSV directly, and answers dashboard queries in milliseconds. Spark would take seconds just to start.

**61. Why doesn't the hosted app run Spark?**
The pipeline runs offline and **publishes** a small snapshot (45 tables, about 33 MB) to `data/demo`. The app only reads that snapshot. This separates heavy processing from serving, and it fits the free cloud tier.

**62. How is login handled? Are passwords stored?**
Only **hashed** passwords are stored, in Streamlit secrets, a file that's never committed to GitHub. The LLM API key is kept the same way. Each account maps to a role, and the role decides the pages and the data.

**63. ⚠️ Is the access control just hiding pages?**
No. Hiding pages would be UI only. Every data function is wrapped in a role check, so even a direct call (or an assistant tool) raises an error for the wrong role. Tests confirm that each role is refused outside its workspace.

**64. How do you know the published data doesn't leak anything?**
There's a check (7.9) that scans the published snapshot for local file paths and user names; error messages are scrubbed before publishing. Customer personal data is removed from the assistant's tables.

---

## I. Testing, quality and process

**65. How is the code tested?**
- More than 230 automated tests (unit, app and assistant)
- per-phase completion check scripts with fixed bars
- linting with ruff

A test also fails if a generated document is out of date.

**66. What would you do differently / what's next?**
Phase 8 is planned:
- CI on every push
- one `check_all` command
- pinned dependencies
- a deployment health check
- a privacy pass and measured performance

Beyond the project:
- Delta Lake instead of CSV
- an orchestrator (Airflow) for scheduling
- embeddings once the knowledge base grows
- real store heartbeats for outage detection

**67. What was the hardest problem?**
There are good honest answers to choose from:
- **Inventory drift:** stock rebuilt from weekly counts drifts when Silver quarantines some events, and that limits stockout recall.
- **Anomaly over-dispersion:** Poisson → negative binomial.
- **Assistant routing without an LLM:** phrase overlap between tools needed general rules, not one-off patches.

**68. What is the business impact?**
- A manager sees what will sell, what will run out and how much to order, with reasons.
- Stockouts are turned into money, so effort goes where it pays.
- One set of trusted metrics replaces conflicting spreadsheets.
- A grounded assistant answers in seconds, with citations, inside each person's permissions.
- The data engineer can see every run, rejection and table's lineage.

---

## Rapid-fire definitions

| Term | One-line answer |
|---|---|
| Idempotent | Running it twice gives the same result as running it once |
| Watermark | How late an event may arrive and still be counted by a streaming aggregation |
| Grain | What one row of a table means |
| Referential integrity | Every foreign key points to an existing parent row |
| WAPE | Total absolute error ÷ total actual units |
| Poisson loss | A training objective for count data (non-negative integers) |
| Quantile loss | Trains a model to predict a percentile, e.g. the 90th, instead of the mean |
| Safety stock | Extra stock to cover demand variability during the lead time |
| Precision / recall | Of the flags, how many were right / of the real events, how many were flagged |
| Lift | How much more likely B is when A is in the basket, compared with B in general |
| ARI | Agreement between two groupings, adjusted for chance (1 = identical, 0 = random) |
| Silhouette | How well separated clusters are (−1 to 1) |
| BM25 | Keyword ranking: rare matching words count more; long documents are penalised |
| Held-out set | Questions never used for tuning, used only to measure honest performance |
| Hallucination | A model stating something not supported by its sources |
