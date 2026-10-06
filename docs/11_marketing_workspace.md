# 11. Customer Growth & Marketing Workspace (Phase 5E)

**Status:** complete · **Code:** `app/views/marketing.py`, `src/serving/queries.py` (15 marketing functions), `app/components/charts.py` (`cohort_heatmap`, `segment_bubble_chart`) · **Run:** `streamlit run app/Home.py` · **Login:** the Marketing Manager demo account (and admin)
**Data:** the published snapshot in `data/demo/` (33 tables, 32.8 MB), read through DuckDB. The models behind the page are described in [10_marketing_analytics.md](10_marketing_analytics.md).

## Business questions

- *Who are our customers and how should we talk to each group?* → Segments
- *What do people buy together?* → Basket & affinity
- *What should we show each customer next?* → Recommendations
- *Who is slipping away, and who is worth winning back?* → Retention
- *Did our promotions pay off?* → Promotion effectiveness

## The page

Header KPIs: customers, customers who have ordered, repeat rate (2+ completed orders ÷ 1+), active in the last 14 days, at risk or lapsed.

| Tab | What it shows |
|---|---|
| **Segments** | Bubble chart (avg orders vs AOV, size = share), the profile table with top categories and campaign ideas, the highest-spending customers in a chosen segment, and how k was chosen (silhouette per k, stability) |
| **Basket & affinity** | "What is bought with…" lookup for any product with rules; strongest rules filterable by category and minimum lift; how to use them (placement, bundles, checkout prompts) |
| **Recommendations** | Precision@10 of hybrid vs buy-again vs popularity (September backtest) and the chosen weights; pick a segment and a customer to see their top 10 with reasons; most-recommended new-to-them products per segment (campaign candidates) |
| **Retention** | Descriptive only (stated on the page). Cohort heatmap; status × value tier grid; a win-back list filterable by status, value tier and "overdue" |
| **Promotion effectiveness** | Number of promotions, total discount cost, median uplift; uplift per promotion; full table. Caveat on the page: a before/after comparison, not a controlled experiment |

## Access control

Every function is decorated with `@workspace("marketing")`: it runs for the Marketing Manager and admin and raises `AccessDenied` for every other role, even if the page code were bypassed. Inventory, Business and Data Engineer users get "cannot open" on this page (tested).

## Results (medium run)

| Measure | Value |
|---|---|
| Segments | 7 (5,307 customers with orders) |
| Basket rules | 352 |
| Recommendation Precision@10 | hybrid 14.8% vs popularity 14.0% |
| Retention status | active 1,933 · cooling 1,143 · at risk 1,124 · lapsed 1,107 · never ordered 693 |
| Promotions | 57; discount cost ₹2,15,742; median uplift +40.2% (range −10.5% to +81.5%) |

## Tests

| Test | What it proves |
|---|---|
| `test_marketing_functions_check_the_role` | All 15 functions: marketing and admin allowed, four other roles refused |
| `test_marketing_numbers_are_consistent` | Segment sizes add up to all buyers; shares sum to 1; labels unique; retention covers every customer; ranks 1…10; every rule has lift ≥ 2; three backtest methods present |
| `test_marketing_workspace_renders` | The page runs without errors and shows the key metrics |
| `test_business_role_is_blocked_from_marketing` | Another role sees "cannot open" and no data |
| `scripts/check_phase5.py` 5D.7 | Same role check against the real medium snapshot |
