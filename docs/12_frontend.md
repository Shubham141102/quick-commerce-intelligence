# 12. Frontend: Login, Persona Landing and Page Design

**Status:** complete for the three built workspaces; the Data Engineer workspace is on hold (decision 2026-10-05) · **Code:** `app/Home.py`, `app/components/{auth,ui,personas}.py`, `app/views/{inventory,business,marketing,welcome,engineering}.py`, `.streamlit/config.toml` · **Run:** `streamlit run app/Home.py`

## What the user sees

```
streamlit run app/Home.py
        │
        ▼
 ┌──────────────────────────── Login page ────────────────────────────┐
 │ 🛒 Quick-Commerce Intelligence            │  🔐 Sign in             │
 │ what the platform is                      │  Username / Password    │
 │ one card per persona with its use cases   │  [ Sign in ]            │
 │                                           │  ▸ Demo accounts        │
 └────────────────────────────────────────────────────────────────────┘
        │ sign in
        ▼
 inventory  → 📦 Inventory & Supply Chain     (4 tabs)
 business   → 📈 Business & Revenue           (3 tabs)
 marketing  → 🎯 Customer Growth & Marketing  (5 tabs)
 engineer   → 🛠️ Data Engineer                (on hold: planned use cases only)
 admin      → 🏠 Overview → links to every workspace (sidebar menu)
```

## How "land directly in your persona" works

`app/Home.py` builds the page list **from the signed-in user's role** on every run:

| Situation | Pages registered | Landing (default) page |
|---|---|---|
| Not signed in | the login page only (menu hidden) | Login |
| One workspace (inventory / business / marketing / engineer) | only that workspace (menu hidden) | that workspace |
| Admin (all workspaces) | Overview + all workspaces (sidebar menu) | Overview |

A persona's workspace is registered as Streamlit's **default page**, so after sign-in the app opens it directly. Other workspaces are **not registered at all** for that user, so typing their URL cannot open them. Every data function also checks the role again (`@workspace(...)` in `src/serving/queries.py`).

## Page design

| Element | Where | What it does |
|---|---|---|
| Theme | `.streamlit/config.toml` | Teal primary colour, light background, minimal toolbar (committed, no secrets) |
| Persona header band | `ui.page_header(workspace)` | Icon, title, what the workspace is for, chips: role, data period and run, snapshot time |
| Tab question line | `ui.tab_intro(question)` | Every tab opens with the business question it answers |
| KPI cards | CSS in `ui.inject_css()` | Metrics are shown as bordered cards |
| Sidebar profile | `ui.sidebar_profile()` | Initials avatar, name, role, **Sign out**, "About the data" (pipeline and generation run) |
| "What the chart says" boxes | `app/components/insights.py` | Under the forecast chart: next-7-day total, change vs the last 7 days, busiest / quietest day, September accuracy for that store × category, and a day-by-day forecast table. Under the SKU stock chart: stock on hand vs reorder level, last 28 days' sales and stockout days, expected demand, likely run-out day, tier and suggested order |
| Simpler forecast chart | `charts.forecast_chart` | Sold units as bars, the model's September test forecast as a line, a "today → forecast" divider, the next 7 days with a likely range; last 30 days by default, "Show full history" toggle |
| "Can I trust the forecast?" | `insights.accuracy_headline`, `insights.reliability` | Replaces the WAPE tables on the Inventory forecasting tab (decision 2026-10-06: they answer a data scientist's question, not a manager's). One sentence: how much less error than the best simple rule, and whether accuracy holds over 7 days. Each category gets a forecast reliability label from its September WAPE, with bands fixed before looking at the results: **High < 0.55 ≤ Medium ≤ 0.70 < Low**, coloured green / yellow / red, with an action (order close to the forecast / normal buffer / bigger buffer and manual review). The selected category's label also appears under the chart. The full WAPE tables, baselines and feature importance stay in a collapsed "Technical details (for analysts)" expander and will move to Model monitoring when the Data Engineer workspace is built |
| Tier colours | `insights.style_tiers` | High = red, Medium = yellow, Low = green: risk tier, per-store tier counts, anomaly severity. Not applied to the Marketing value tier (there High is good) |
| Persona descriptions | `app/components/personas.py` | One place for icon, title, tagline and use cases, used by the login page, headers and the admin overview |

### Questions answered per tab

| Workspace | Tab | Question |
|---|---|---|
| Inventory | Overview | Where do we stand today: which SKUs are at risk and what should be reordered? |
| | Demand forecasting | How many units will each category sell in the next 7 days, and how far can we trust it? |
| | Stockout risk & replenishment | Which SKUs will run out before the next delivery, and how much should we order? |
| | Inventory explorer | What did empty shelves cost us, and why did a SKU run out? |
| Business | Sales & revenue performance | How are we selling compared with the previous period, and where does revenue come from? |
| | Delivery & operations | Are we keeping the delivery promise, and why are orders cancelled? |
| | Anomaly detection | Did something unusual happen that needs investigating? |
| Marketing | Segments | Who are our customers, and how should we talk to each group? |
| | Basket & affinity | What do customers buy together? |
| | Recommendations | What should we show each customer next, and does it work? |
| | Retention | Who is still ordering, and who is worth winning back? |
| | Promotion effectiveness | Did our promotions lift sales, and what did they cost? |

## Tests (`tests/app/test_app.py`)

| Test | What it proves |
|---|---|
| `test_login_with_demo_account` | Wrong password is refused; correct password signs in **and lands in the Inventory workspace** |
| `test_each_persona_lands_in_its_own_workspace` | inventory / business / marketing each open their own workspace after sign-in |
| `test_signed_out_visitor_sees_only_the_login_page` | Before sign-in there is only the login form, no data |
| `test_admin_lands_on_overview_and_engineer_on_hold_page` | Admin lands on the overview; the engineer sees the on-hold page |
| existing workspace tests | Each page renders, and other roles get "cannot open" |

## Not in scope yet

- Data Engineer workspace (on hold).
- Business assistant (RAG, Phase 6).
- Production login (SSO, password reset); this is demo-grade access control.
