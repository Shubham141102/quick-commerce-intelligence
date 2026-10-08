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

## Sidebar navigation (2026-10-07)

Each workspace is a **group in the left sidebar** and each of its sections is a page with its own URL
(the former tabs). A persona sees only its own group and lands on its first section; the admin sees the
Overview plus every group.

| Workspace | Sidebar pages (URL) |
|---|---|
| Inventory & Supply Chain | Overview (`/`), Demand forecasting (`/inventory-forecasting`), Stockout risk & replenishment (`/inventory-stockout`), Inventory explorer (`/inventory-explorer`) |
| Business & Revenue | Sales & revenue performance, Delivery & operations, Anomaly detection (`/business-…`) |
| Customer Growth & Marketing | Segments, Basket & affinity, Recommendations, Retention, Promotion effectiveness (`/marketing-…`) |
| Data Engineer | one page (on hold) |
| Every persona | **Assistant** (last item of the group; next to Overview for the admin) — see docs/13_rag_assistant.md |

How it is built: each view file defines one function per section plus a `PAGES` list; `app/Home.py` registers
those functions as `st.Page`s. Every section page shows the breadcrumb (app › workspace) and the section as
its title. The Business period and store filters appear on each Business page and keep their values when
moving between them. Running a view file directly (as the tests do) still renders all its sections.

## How "land directly in your persona" works

`app/Home.py` builds the page list **from the signed-in user's role** on every run:

| Situation | Pages registered | Landing (default) page |
|---|---|---|
| Not signed in | the login page only (menu hidden) | Login |
| One workspace (inventory / business / marketing / engineer) | only that workspace (menu hidden) | that workspace |
| Admin (all workspaces) | Overview + all workspaces (sidebar menu) | Overview |

A persona's workspace is registered as Streamlit's **default page**, so after sign-in the app opens it directly. Other workspaces are **not registered at all** for that user, so typing their URL cannot open them. Every data function also checks the role again (`@workspace(...)` in `src/serving/queries.py`).

## Design system (enterprise dashboard, 2026-10-07)

Standard business-software practice: clean grid, minimal functional colour, high scannability, professional
typography, a clear hierarchy. Theme values are real Streamlit theme settings in `.streamlit/config.toml`;
`app/components/ui.py` adds light CSS for components the theme does not cover.

| Element | Rule |
|---|---|
| Typography | Inter for text and headings; heading scale 1.6 / 1.25 / 1.05 rem; base 14 px; KPI values 1.6 rem, weight 600, **tabular numbers** so digits line up |
| Colour | Slate neutrals for structure (text #0F172A, secondary #475569 / #64748B, borders #E2E8F0, canvas #F5F7FA, panels white). **One accent blue #1D4ED8** for interaction and primary data. Red / amber / green **only for status** (risk tier, severity, reliability, live / on hold) |
| Navigation | Dark navy sidebar (#0F172A) with the QC wordmark (`app/assets/logo.svg`), signed-in profile card, Sign out, "About this snapshot"; Material icons instead of emojis |
| Page header | Breadcrumb (APP › WORKSPACE), title, one-line purpose; right-aligned status: "Snapshot current", data period and profile, publish time; hairline divider |
| Tabs | Underline navigation, medium-weight labels, the active tab in dark text |
| Business question | Each tab opens with a white callout, blue left rule, "BUSINESS QUESTION" overline |
| KPI cards | White, 1 px border, 8 px radius, faint shadow, uppercase grey label, large value |
| Panels | Bordered containers are white on the grey canvas; "KEY TAKEAWAYS" overline above chart summaries |
| Tables | Slate header row (#F1F5F9 / #475569), light borders; status cells as soft badges (High #FEF3F2/#B42318, Medium #FFFAEB/#B54708, Low #ECFDF3/#067647) |
| Charts | One shared style (`charts._base`): white plot, faint horizontal grid only, Inter 12 px, legend top-left. History in light slate, the model's past forecast in slate, forecast / primary series in the accent blue with a light-blue band, thresholds (reorder level, SLA) in amber, problem periods (stockouts, anomaly windows) as a faint red wash |
| Text | Crisp points, not paragraphs (2026-10-07): explanations are 2–3 grey bullets via `ui.notes()`, one idea each, ~10 words; summaries use `Label — value` (e.g. *Next 7 days — 46 units (~6.6/day)*); info and warning boxes lead with a bold headline. Chart notes say how to read the chart, not which colour is which (the legend does that). No information was removed |
| Login | Dark brand panel (facts: 12 stores, 3 cities, 6 months; the four workspaces with codes INV / BIZ / MKT / ENG) beside a plain sign-in form |
| Admin overview | Snapshot KPI strip, then a two-column grid of workspace cards with a Live / On hold badge and an "Open" link |

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
