"""Plotly charts used by the workspaces."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Enterprise palette: history in neutral slate, model / forecast / primary series in the accent blue,
# thresholds in amber, problems (stockouts, anomaly windows) in a soft red. Nothing decorative.
ACCENT, SLATE, SLATE_LIGHT, AMBER, RED = "#1D4ED8", "#64748B", "#CBD5E1", "#F59E0B", "#DC2626"
ACTUAL = SLATE_LIGHT          # what happened (bars)
MODEL = SLATE                 # model's past forecast / secondary series
FUTURE = ACCENT               # forecast / primary series
BAND = "rgba(29, 78, 216, 0.12)"
STOCK = ACCENT
REORDER = AMBER
PROBLEM_FILL = "rgba(220, 38, 38, 0.08)"
GRID, AXIS, TEXT = "#EEF2F6", "#CBD5E1", "#334155"
FONT = "Inter, Segoe UI, Arial, sans-serif"


def _base(fig: go.Figure, height: int = 380) -> go.Figure:
    """Shared look: white plot, faint horizontal grid, no vertical grid, top-left legend, quiet hover label."""
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=8, r=8, t=36, b=8),
        font=dict(family=FONT, size=12, color=TEXT), paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=11, color=SLATE), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=AXIS, font=dict(family=FONT, size=12, color=TEXT)))
    fig.update_xaxes(showgrid=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS, tickfont=dict(color=SLATE),
                     title_font=dict(size=11, color=SLATE))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=SLATE),
                     title_font=dict(size=11, color=SLATE))
    return fig


def _layout(fig: go.Figure, y_title: str) -> go.Figure:
    fig = _base(fig)
    fig.update_layout(hovermode="x unified", yaxis_title=y_title)
    return fig


def forecast_chart(df: pd.DataFrame, days_back: int | None = 30) -> go.Figure:
    """Sold units (bars), the September test forecast (line), a 'today' divider and the next 7 days (line +
    likely range). `days_back` limits the history shown (None = everything)."""
    hist, future = df.dropna(subset=["actual"]), df.dropna(subset=["future_forecast"])
    if days_back and len(hist):
        hist = hist[hist["day"] > hist["day"].max() - pd.Timedelta(days=days_back)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hist["day"], y=hist["actual"], name="Units sold", marker_color=ACTUAL,
                         hovertemplate="%{x|%a %d %b}: sold %{y}<extra></extra>"))
    test = hist.dropna(subset=["backtest_forecast"])
    fig.add_trace(go.Scatter(x=test["day"], y=test["backtest_forecast"], name="Model's forecast (September test)",
                             line=dict(color=MODEL, width=2), hovertemplate="model said %{y:.1f}<extra></extra>"))
    if not future.empty:
        fig.add_trace(go.Scatter(x=pd.concat([future["day"], future["day"][::-1]]),
                                 y=pd.concat([future["upper"], future["lower"][::-1]]), fill="toself",
                                 fillcolor=BAND, line=dict(width=0), name="Likely range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=future["day"], y=future["future_forecast"], name="Forecast, next 7 days",
                                 mode="lines+markers", line=dict(color=FUTURE, width=3),
                                 hovertemplate="%{x|%a %d %b}: forecast %{y:.1f}<extra></extra>"))
        if len(hist):
            today = hist["day"].max() + pd.Timedelta(hours=12)
            fig.add_vline(x=today, line=dict(color=AXIS, dash="dot", width=1))
            fig.add_annotation(x=today, y=1.02, yref="paper", text="today → forecast", showarrow=False,
                               xanchor="left", font=dict(size=11, color=SLATE))
    fig = _layout(fig, "units per day")
    fig.update_layout(hovermode="x", bargap=0.15)
    return fig


def stock_chart(df: pd.DataFrame) -> go.Figure:
    hist, future = df.dropna(subset=["closing_stock"]), df.dropna(subset=["forecast"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["day"], y=hist["closing_stock"], name="Closing stock",
                             line=dict(color=STOCK, width=2), line_shape="hv"))
    fig.add_trace(go.Bar(x=hist["day"], y=hist["units_sold"], name="Units sold", marker_color=SLATE_LIGHT))
    fig.add_trace(go.Scatter(x=hist["day"], y=hist["reorder_level"], name="Reorder level",
                             line=dict(color=REORDER, dash="dash", width=1.5)))
    if not future.empty:
        fig.add_trace(go.Bar(x=future["day"], y=future["forecast"], name="Forecast demand", marker_color=FUTURE,
                             error_y=dict(type="data", symmetric=False, array=future["upper"] - future["forecast"],
                                          arrayminus=future["forecast"] - future["lower"])))
    return _layout(fig, "units")


def explorer_chart(df: pd.DataFrame) -> go.Figure:
    """One SKU's full history: calculated stock, weekly counted stock, restocks, damage, stockout days."""
    fig = go.Figure()
    out = df[df["is_stockout"] == True]  # noqa: E712  (pandas boolean column may hold NA)
    for day in out["day"]:
        fig.add_vrect(x0=day - pd.Timedelta(hours=12), x1=day + pd.Timedelta(hours=12), fillcolor=PROBLEM_FILL,
                      line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=df["day"], y=df["closing_stock"], name="Closing stock (calculated)",
                             line=dict(color=STOCK, width=2), line_shape="hv"))
    counted = df.dropna(subset=["snapshot_stock"])
    fig.add_trace(go.Scatter(x=counted["day"], y=counted["snapshot_stock"], name="Weekly count", mode="markers",
                             marker=dict(color="#0F172A", size=7, symbol="diamond"),
                             customdata=counted["reconciliation_gap"],
                             hovertemplate="counted %{y}<br>gap (counted − calculated) %{customdata}<extra></extra>"))
    fig.add_trace(go.Bar(x=df["day"], y=df["restocked"], name="Restocked", marker_color=SLATE_LIGHT))
    fig.add_trace(go.Bar(x=df["day"], y=-df["damaged"], name="Damaged (written off)", marker_color="#F87171"))
    fig.add_trace(go.Scatter(x=df["day"], y=df["reorder_level"], name="Reorder level",
                             line=dict(color=REORDER, dash="dash", width=1)))
    fig.add_annotation(text="shaded = stockout day", xref="paper", yref="paper", x=1, y=1.12, showarrow=False,
                       font=dict(size=11, color=RED))
    return _layout(fig, "units")


def revenue_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Net revenue (bars) and completed orders (line, right axis) over time."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["period"], y=df["net_revenue"], name="Net revenue (₹)", marker_color=ACCENT))
    fig.add_trace(go.Scatter(x=df["period"], y=df["orders_completed"], name="Completed orders", yaxis="y2",
                             line=dict(color=SLATE, width=2)))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="orders", showgrid=False,
                                  tickfont=dict(color=SLATE)), bargap=0.3)
    return _layout(fig, "net revenue (₹)")


def delivery_trend_chart(df: pd.DataFrame, sla_minutes: int = 15) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["day"], y=df["on_time_rate"] * 100, name="On-time %", line=dict(color=STOCK, width=2)))
    fig.add_trace(go.Scatter(x=df["day"], y=df["avg_minutes"], name="Avg delivery minutes", yaxis="y2",
                             line=dict(color=MODEL, width=1.5)))
    fig.add_hline(y=sla_minutes, line_dash="dot", line_color=REORDER, yref="y2",
                  annotation_text=f"SLA {sla_minutes} min", annotation_position="top left")
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="minutes", showgrid=False,
                                  tickfont=dict(color=SLATE)))
    return _layout(fig, "on-time %")


def anomaly_chart(df: pd.DataFrame) -> go.Figure:
    """Observed vs expected around one anomaly, with the anomaly window shaded."""
    fig = go.Figure()
    if len(df):
        fig.add_vrect(x0=df["start_ts"].iloc[0], x1=df["end_ts"].iloc[0], fillcolor=PROBLEM_FILL,
                      line_width=0, layer="below")
    shape = "hv" if len(df) and df["detector"].iloc[0] != "demand_spike" else "linear"
    fig.add_trace(go.Scatter(x=df["time"], y=df["observed"], name="Observed", line=dict(color=ACCENT, width=2),
                             line_shape=shape))
    fig.add_trace(go.Scatter(x=df["time"], y=df["expected"], name="Expected (previous 28 days)",
                             line=dict(color=MODEL, width=1.5, dash="dash"), line_shape=shape))
    return _layout(fig, "count")


def bar_chart(df: pd.DataFrame, x: str, y: str, color: str = ACCENT, horizontal: bool = True,
              x_title: str = "") -> go.Figure:
    data = df.sort_values(x) if horizontal else df
    trace = (go.Bar(x=data[x], y=data[y], orientation="h", marker_color=color) if horizontal
             else go.Bar(x=data[y], y=data[x], marker_color=color))
    fig = _base(go.Figure(trace), height=max(260, 26 * len(df) + 60))
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8), xaxis_title=x_title, bargap=0.35)
    if horizontal:
        fig.update_xaxes(showgrid=True, gridcolor=GRID)
        fig.update_yaxes(showgrid=False)
    return fig


def importance_chart(df: pd.DataFrame) -> go.Figure:
    top = df.sort_values("importance_mae").tail(12)
    fig = go.Figure(go.Bar(x=top["importance_mae"], y=top["feature"], orientation="h", marker_color=MODEL,
                           error_x=dict(type="data", array=top["importance_std"])))
    fig = _base(fig)
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8), xaxis_title="increase in error when shuffled (units)")
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return fig


def cohort_heatmap(df: pd.DataFrame) -> go.Figure:
    grid = df.pivot_table(index="cohort_month", columns="month_offset", values="retention_rate")
    sizes = df.drop_duplicates("cohort_month").set_index("cohort_month")["cohort_size"].reindex(grid.index)
    rows = [f"{pd.Timestamp(m):%b %Y} (n={int(n)})" for m, n in zip(grid.index, sizes)]
    fig = go.Figure(go.Heatmap(z=grid.to_numpy(), x=[f"+{int(c)}" for c in grid.columns], y=rows, colorscale=[[0, "#F1F5F9"], [1, ACCENT]],
                               zmin=0, zmax=1, text=grid.map(lambda v: "" if pd.isna(v) else f"{v:.0%}").to_numpy(),
                               texttemplate="%{text}", hovertemplate="%{y}, month %{x}: %{z:.1%}<extra></extra>"))
    fig = _base(fig, height=70 + 42 * len(rows))
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8), xaxis_title="months after first order")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def segment_bubble_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df["avg_orders"], y=df["avg_order_value"], mode="markers+text", text=df["segment_label"],
                               textposition="top center", marker=dict(size=df["share"] * 400 + 10, color=ACCENT,
                                                                      opacity=0.35, line=dict(color=ACCENT, width=1)),
                               customdata=df[["customers"]], hovertemplate="%{text}<br>%{customdata[0]} customers"
                               "<br>avg orders %{x:.1f}<br>AOV ₹%{y:.0f}<extra></extra>"))
    fig = _base(fig, height=440)
    fig.update_layout(xaxis_title="average completed orders", yaxis_title="average order value (₹)")
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    return fig
