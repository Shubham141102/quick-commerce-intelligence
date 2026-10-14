# 16. Persona Consolidation: Four Personas → Three

**Status:** code complete; measurements pending (see "Verification") · **Date:** 2026-10-14
**Code:** `src/serving/permissions.py`, `src/serving/queries.py` (`@workspace("growth")`, new `store_scorecard`), `app/views/growth.py` (replaces `business.py` and `marketing.py`), `app/components/personas.py`, `app/Home.py`, `src/rag/tools.py`, `src/rag/composer.py`, `scripts/create_demo_secrets.py`

## Why

Manager feedback was that the app does too many things. Review showed that the **Business & Revenue** and **Customer Growth & Marketing** personas overlapped: revenue, promotions and customers appeared in both, and cancellations appeared on two pages. Quick-commerce companies usually give revenue, customers and campaigns to one growth team. So the two personas were merged into one, with **one question**:

> *Are we growing profitably and keeping customers?*

The merged workspace has **4 pages instead of 8**, so the merge also reduces what is on screen; it does not just move it.

## Before and after

| | Before | After |
|---|---|---|
| Personas (roles) | inventory_manager, business_analyst, marketing_manager, data_engineer (+ admin) | inventory_manager, **growth_manager**, data_engineer (+ admin) |
| Workspaces | inventory, business, marketing, engineering | inventory, **growth**, engineering |
| Demo logins | inventory, business, marketing, engineer, admin | inventory, **growth**, engineer, admin |
| Sidebar pages | 4 + 3 + 5 + 4 = **16** | 4 + **4** + 4 = **12** |

## The Business & Growth pages

| Page | Built from | What changed |
|---|---|---|
| **Performance** | Sales & revenue + Delivery & operations | One KPI row (net revenue, completed orders, AOV, on-time rate, cancellation rate); one revenue trend; **one store scorecard** (revenue, orders, AOV, on-time, delivery minutes, cancel rate; replaces the two store tables); category contribution next to cancellations by reason; top products. Delivery detail (avg / p90 minutes, failed, cancelled in transit, delivery trend) moved into a collapsed section. |
| **Alerts** | Anomaly detection | Demand spikes and payment failures only. The **store-outage detector is no longer shown**: it was labelled experimental, because at about one order an hour a 3–6 hour outage cannot be told apart from chance (docs/09). It still runs in the pipeline and is still measured by check_phase5 (5C.4). |
| **Customers** | Segments + Retention | Customer KPIs; segment bubble chart and profiles; the active / cooling / at-risk / lapsed grid and the win-back list. Customers per segment, the cohort heatmap and how k was chosen moved into collapsed sections. |
| **Growth levers** | Basket & affinity + Recommendations + Promotions | Bought-together rules; per-customer recommendations; promotion results. Recommendation accuracy (precision@10 table) and the full promotion table moved into collapsed sections. |

No formula changed. `store_scorecard` uses the same SQL as `revenue_by_store` and `delivery_by_store`, and a test checks that its numbers equal both of them.

## Assistant

- The 10 business and marketing tools now belong to the `growth` workspace. A Business & Growth manager can ask about revenue **and** customers; inventory and engineering data are still refused.
- Example questions and "see this page" hints point to the four new pages.

## Evaluation sets: what changed (disclosed)

| Change | Sets | Why it is not a lowered bar |
|---|---|---|
| `business_analyst` / `marketing_manager` → `growth_manager` on every question | gold, held-out, engineering, engineering held-out, forecast, forecast held-out | The role names no longer exist. Same questions, same expected routes and answers. |
| **R03** (gold) "What are our customer segments?" asked by `business_analyst` → asked by **`data_engineer`** | gold | The analyst now *owns* segments, so a refusal would be wrong. Same question; the role is one that is still outside the workspace, so it still tests a refusal. |
| **HR2** (held-out) "Which promotions had the best uplift?" asked by `business_analyst` → asked by **`inventory_manager`** | held-out | Same reason as R03. |

Every other role check still asks across a boundary that exists (e.g. inventory → revenue, growth → inventory, growth → engineering).

## Check scripts

- `check_phase5`: 5B.3, 5C.6 and 5D.7 now check that `growth_manager` is allowed and `inventory_manager` and `data_engineer` are refused (the bar is unchanged: every function role-checked). 5B.3 also covers `store_scorecard`.
- `check_ml` #12 and `check_phase7` 7.8 refuse `growth_manager` instead of the two old roles.

## Deployment step (Streamlit Cloud)

The hosted app's **Secrets** still contain the old `business` and `marketing` logins, whose roles no longer exist (they would sign in to an empty app). After pushing:

1. Locally: `python -m scripts.create_demo_secrets --password <demo password>`. This rewrites the `[auth...]` part of `.streamlit/secrets.toml` and keeps the `[llm]` section.
2. Copy the new `[auth.users.growth]` block (and remove the `business` and `marketing` blocks) in the app's **Settings → Secrets** on Streamlit Community Cloud.

## Verification

Measured 2026-10-14:

| Check | Expected | Result |
|---|---|---|
| ruff (classic rules E4, E7, E9, F) | all pass | ✅ All checks passed. *Note: the installed ruff is now 0.16.10, whose broader default rule set reports 199 pre-existing style findings across the codebase (unused `noqa`, `dict()` calls, …). Not caused by this change; to be settled in Phase 8 (pin ruff or adopt the rules deliberately).* |
| pytest | all pass | ✅ 236 passed, 0 failed |
| check_phase5 | 18 pass, 2 known limitations | ✅ 18 passed, 2 known limitations (5A.3, 5C.2), 0 failed — unchanged |
| check_phase7 | 9/9 | ✅ 9 passed, 0 failed |
| check_phase6 gold | 6/6 bars | ✅ 6/6 (routing 37/40, hit@3 20/23, numbers 18/18, citations 42/42, refusal 7/7, roles 5/5) |
| check_phase6 held-out, engineering, engineering held-out | unchanged | ✅ unchanged: held-out routing 17/20 and refusal 2/3 (HP5, HM3, HA1 — known), engineering all pass, engineering held-out EH2 miss (known) |
| Per-question comparison with the previous runs (2026-10-11) | no question changes outcome | ✅ none changed in any of the four sets |
| Demo logins | inventory, growth, engineer, admin | ✅ regenerated; `[llm]` section kept |

## Not done (next candidates)

- Inventory pages 4 → 2 (Stockout risk and Inventory explorer both have a one-SKU drill-down).
- Move the model-proof panels (forecast WAPE detail, stockout backtest, recommendation precision, segmentation selection) to Data Engineer → Model monitoring. This needs those functions opened to the engineering role.
