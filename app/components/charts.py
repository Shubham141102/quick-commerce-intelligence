"""Plotly charts used by the workspaces."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

ACTUAL, MODEL, FUTURE, BAND, STOCK, REORDER = "#4C78A8", "#F58518", "#E45756", "rgba(228,87,86,0.18)", "#54A24B", "#B279A2"


def _layout(fig: go.Figure, y_title: str) -> go.Figure:
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0), yaxis_title=y_title)
    return fig


def forecast_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["day"], y=df["actual"], name="Actual units", line=dict(color=ACTUAL, width=1.5)))
    fig.add_trace(go.Scatter(x=df["day"], y=df["backtest_forecast"], name="Backtest forecast (1 day ahead)",
                             line=dict(color=MODEL, width=1.5, dash="dot")))
    future = df.dropna(subset=["future_forecast"])
    if not future.empty:
        fig.add_trace(go.Scatter(x=pd.concat([future["day"], future["day"][::-1]]),
                                 y=pd.concat([future["upper"], future["lower"][::-1]]), fill="toself",
                                 fillcolor=BAND, line=dict(width=0), name="10–90% range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=future["day"], y=future["future_forecast"], name="Forecast (next 7 days)",
                                 line=dict(color=FUTURE, width=2.5)))
    return _layout(fig, "units per day")


def stock_chart(df: pd.DataFrame) -> go.Figure:
    hist, future = df.dropna(subset=["closing_stock"]), df.dropna(subset=["forecast"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["day"], y=hist["closing_stock"], name="Closing stock",
                             line=dict(color=STOCK, width=2), line_shape="hv"))
    fig.add_trace(go.Bar(x=hist["day"], y=hist["units_sold"], name="Units sold", marker_color=ACTUAL, opacity=0.45))
    fig.add_trace(go.Scatter(x=hist["day"], y=hist["reorder_level"], name="Reorder level",
                             line=dict(color=REORDER, dash="dash", width=1.5)))
    if not future.empty:
        fig.add_trace(go.Bar(x=future["day"], y=future["forecast"], name="Forecast demand", marker_color=FUTURE,
                             error_y=dict(type="data", symmetric=False, array=future["upper"] - future["forecast"],
                                          arrayminus=future["forecast"] - future["lower"])))
    return _layout(fig, "units")


def importance_chart(df: pd.DataFrame) -> go.Figure:
    top = df.sort_values("importance_mae").tail(12)
    fig = go.Figure(go.Bar(x=top["importance_mae"], y=top["feature"], orientation="h", marker_color=MODEL,
                           error_x=dict(type="data", array=top["importance_std"])))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="increase in error when shuffled (units)")
    return fig
