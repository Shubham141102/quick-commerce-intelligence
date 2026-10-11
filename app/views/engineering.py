"""Data Engineer workspace (Phase 7C): pipeline runs, data quality & quarantine, lineage & tables, model monitoring.

Each section is a page in the sidebar (registered by app/Home.py). Every number comes from the role-checked
functions in src/serving/queries.py over the ops_* tables the publish stage writes (src/orchestration/ops_snapshot.py).
Running this file directly (as the tests do) renders every section on one page.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.components.charts import (  # noqa: E402
    LAYER_LABELS,
    bar_chart,
    lineage_sankey,
    metric_history_chart,
    stage_duration_chart,
)
from app.components.ui import label, notes, page_header, require_workspace, tab_intro  # noqa: E402
from src.common.paths import PROJECT_ROOT  # noqa: E402
from src.serving import queries as q  # noqa: E402
from src.serving.queries import snapshot_info  # noqa: E402

WS = "engineering"
SECTIONS = [  # (function, sidebar title, icon) — each is a page in the sidebar
    ("runs", "Pipeline runs", ":material/play_circle:"),
    ("quality", "Data quality & quarantine", ":material/rule:"),
    ("lineage", "Lineage & tables", ":material/account_tree:"),
    ("models", "Model monitoring", ":material/monitor_heart:"),
]
_TITLES = {key: title for key, title, _ in SECTIONS}
STATUS_BADGE = {"success": ":green-badge[success]", "failed": ":red-badge[failed]"}


def _open(key: str) -> str:
    """Guard + header for one section page; returns the role."""
    role = require_workspace(WS)
    page_header(WS, _TITLES[key])
    return role


def _status_style(df: pd.DataFrame, column: str = "status"):
    colours = {"success": ("#ECFDF3", "#067647"), "failed": ("#FEF3F2", "#B42318")}

    def css(v):
        bg, fg = colours.get(str(v), (None, None))
        return f"background-color: {bg}; color: {fg}; font-weight: 600" if bg else ""
    return df.style.map(css, subset=[column]) if column in df.columns else df


# ------------------------------------------------------------------------------------------- pipeline runs
def runs() -> None:
    role = _open("runs")
    tab_intro("Did the pipeline run, how long did it take, and what did each step write or reject?")
    o = q.ops_overview(role)
    c = st.columns(5)
    c[0].metric("Last run", (o["last_status"] or "—").capitalize(),
                help=f"Started {pd.Timestamp(o['last_started']):%d %b %Y %H:%M} UTC · stages {o['last_stages']}"
                if o["last_started"] is not None else None)
    c[1].metric("Runs recorded", o["runs"])
    c[2].metric("Success rate", f"{o['success_rate']:.0%}" if o["success_rate"] is not None else "—",
                help=f"{o['succeeded']} succeeded, {o['failed']} failed")
    c[3].metric("Median run time", f"{o['median_duration_s']:.0f} s" if o["median_duration_s"] else "—",
                help="Successful runs; a full rebuild takes longer than ml,publish")
    c[4].metric("Last success", f"{pd.Timestamp(o['last_success']):%d %b %H:%M}" if o["last_success"] is not None
                else "—")
    notes("Times are UTC · a run is one execution of scripts.run_pipeline with the stages listed")

    st.subheader("Time per stage (recent runs)")
    st.plotly_chart(stage_duration_chart(q.stage_durations(role)), width="stretch")
    notes("Red outline = the stage that failed")

    st.subheader("Run history")
    history = q.pipeline_runs(role)
    st.dataframe(_status_style(history), hide_index=True, width="stretch",
                 column_config={"duration_s": st.column_config.NumberColumn("duration (s)", format="%.0f")})

    st.subheader("What one run wrote")
    pick = st.selectbox("Run", history["run_id"],
                        format_func=lambda r: f"{r} · {history.set_index('run_id').loc[r, 'stages']} · "
                                              f"{history.set_index('run_id').loc[r, 'status']}")
    tables = q.table_runs_for(role, pick)
    if len(tables):
        t = st.columns(4)
        t[0].metric("Tables written", len(tables))
        t[1].metric("Rows written", f"{int(tables['rows_written'].fillna(0).sum()):,}")
        t[2].metric("Rows rejected", f"{int(tables[['rows_rejected', 'rows_rejected_cascade']].fillna(0).sum().sum()):,}")
        t[3].metric("Duplicates removed", f"{int(tables['rows_deduplicated'].fillna(0).sum()):,}")
        st.dataframe(_status_style(tables), hide_index=True, width="stretch")
    else:
        notes("No table-level records for this run (it failed before writing any table)")

    fails = q.failed_runs(role)
    with st.expander(f"Failed runs and their errors ({len(fails)})"):
        st.dataframe(fails, hide_index=True, width="stretch")
        notes("A failed run never replaces the published data: every layer is written to staging and swapped in "
              "only on success")


# ------------------------------------------------------------------------------------------- quality
def quality() -> None:
    role = _open("quality")
    tab_intro("How clean is the data, what did Silver remove or fix, and what was quarantined and why?")
    ds = q.quality_by_dataset(role)
    qd = q.quarantine_by_dataset(role)
    loads = q.file_loads_summary(role)
    c = st.columns(4)
    c[0].metric("Datasets checked", len(ds))
    c[1].metric("Rows checked", f"{int(ds['rows_checked'].sum()):,}")
    c[2].metric("Rows quarantined", f"{int(qd['total'].sum()):,}",
                help=f"{int(qd['direct'].sum()):,} failed a rule · {int(qd['cascade'].sum()):,} lost their parent row")
    c[3].metric("Lowest clean rate", f"{ds['clean_rate'].min():.1%}" if len(ds) else "—",
                help=ds.iloc[0]["dataset"] if len(ds) else None)
    notes("Clean rate = rows kept ÷ rows checked (duplicates removed and quarantined rows excluded) · latest Silver run")

    left, right = st.columns([3, 2])
    with left:
        label("Clean rate by dataset")
        st.dataframe(ds, hide_index=True, width="stretch",
                     column_config={"clean_rate": st.column_config.ProgressColumn("clean rate", format="%.1f%%",
                                                                                  min_value=0, max_value=1)})
    with right:
        label("Quarantined rows by dataset")
        st.plotly_chart(bar_chart(qd.head(12), "total", "dataset", x_title="rows quarantined"), width="stretch")

    st.subheader("Quarantine detail")
    pick = st.selectbox("Dataset", [None, *qd["dataset"]], format_func=lambda d: "All datasets" if d is None else d,
                        key="qtn_dataset")
    st.dataframe(q.quarantine_summary(role, pick), hide_index=True, width="stretch")
    notes("Direct = the row failed a rule · cascade = its parent row was quarantined (e.g. an order item whose order "
          "was rejected)", "Counts and record ids only — raw rejected rows stay in data/quarantine on the pipeline "
          "machine (they can contain customer fields)")

    st.subheader("Rules that fired")
    st.dataframe(q.quality_rules(role, pick), hide_index=True, width="stretch",
                 column_config={"affected_pct": st.column_config.NumberColumn("affected %", format="%.2f")})
    with st.expander("Bronze file loads"):
        st.dataframe(loads, hide_index=True, width="stretch")
        notes("Files loaded per dataset and slice (historical / batch / stream) · corrupt rows are kept in Bronze "
              "and rejected in Silver")


# ------------------------------------------------------------------------------------------- lineage
def lineage() -> None:
    role = _open("lineage")
    tab_intro("Where does each table come from, what uses it, and what does it contain?")
    edges = q.lineage_edges(role)
    st.subheader("Data flow between layers")
    st.plotly_chart(lineage_sankey(edges), width="stretch")
    notes("Width = number of table-to-table links · recorded automatically by the pipeline code, not drawn by hand")

    st.subheader("Table catalog")
    f1, f2 = st.columns([1, 2])
    layers = ["bronze", "silver", "gold", "ml", "app"]
    layer = f1.selectbox("Layer", [None, *layers], format_func=lambda v: "All layers" if v is None else
                         LAYER_LABELS.get(v, v), key="cat_layer")
    search = f2.text_input("Search", placeholder="table name or description, e.g. revenue", key="cat_search")
    catalog = q.table_catalog(role, layer, search or None)
    st.dataframe(catalog, hide_index=True, width="stretch",
                 column_config={"description": st.column_config.TextColumn(width="large")})

    st.subheader("Upstream and downstream of one table")
    names = sorted(set(edges["source"]) | set(edges["target"]))
    default = names.index("gld_daily_sales") if "gld_daily_sales" in names else 0
    focus = st.selectbox("Table", names, index=default, key="lineage_focus")
    st.plotly_chart(lineage_sankey(edges, focus), width="stretch")
    up = edges.loc[edges["target"] == focus, "source"].tolist()
    down = edges.loc[edges["source"] == focus, "target"].tolist()
    notes(f"**Reads from** — {', '.join(up) or 'nothing (a source)'}", f"**Used by** — {', '.join(down) or 'nothing yet'}")

    with st.expander("Transformation catalog (every Bronze → Silver rule)"):
        path = PROJECT_ROOT / "docs" / "transformation_catalog.md"
        st.markdown(path.read_text(encoding="utf-8") if path.exists() else "docs/transformation_catalog.md not found")


# ------------------------------------------------------------------------------------------- models
def models() -> None:
    role = _open("models")
    tab_intro("Which model versions are live, how accurate are they, and is the assistant still passing its checks?")
    reg = q.model_registry(role)
    hist = q.key_metric_history(role)
    evals = q.assistant_eval_history(role)
    gold = evals[(evals["question_set"] == "gold") & (evals["mode"] == "template")]
    c = st.columns(4)
    c[0].metric("Models in the registry", len(reg))
    latest = hist.sort_values("trained_at").groupby("metric").tail(1).set_index("metric")["value"]
    c[1].metric("Forecast WAPE (latest)", f"{latest.get('Demand forecast — WAPE', float('nan')):.3f}",
                help="Lower is better; a perfect model would score about 0.58 on this data")
    c[2].metric("Assistant evaluations", len(evals))
    c[3].metric("Assistant routing (latest gold)", f"{gold.iloc[0]['routing_rate']:.1%}" if len(gold) else "—",
                help="Bar ≥ 90% (scripts/check_phase6.py)")

    st.subheader("Model registry (latest version of each model)")
    st.dataframe(reg.drop(columns=["metrics"]), hide_index=True, width="stretch")
    with st.expander("Recorded metrics of each model (JSON)"):
        st.dataframe(reg[["model_name", "model_version", "metrics"]], hide_index=True, width="stretch")

    st.subheader("Headline metric per model across runs")
    if len(hist):
        st.plotly_chart(metric_history_chart(hist), width="stretch")
    notes("WAPE: lower is better · precision: higher is better · one point per training run")

    st.subheader("Business assistant evaluations")
    st.dataframe(evals, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(c.replace("_", " "), format="percent")
                                for c in ("routing_rate", "hit_at_3_rate", "numbers_rate")})
    notes("Gold set = 40 questions + 5 role checks (bars apply) · held-out = 20 questions written before the 6F fix",
          "LLM mode: verified = rephrased and checked; fallbacks = rejected by the guardrails or unavailable")
    st.caption(f"Snapshot: pipeline run `{snapshot_info().get('pipeline_run_id', '?')}`")


PAGES = [(key, title, icon, globals()[key]) for key, title, icon in SECTIONS]

if __name__ == "__main__":   # run as a script (tests): every section on one page
    for *_, render in PAGES:
        render()
