"""Plain-language summaries of what a chart shows (so the numbers can be read without the chart), and
tier colour-coding for tables."""

from __future__ import annotations

import math

import pandas as pd

from src.serving.reliability import (  # noqa: F401 (re-exported)
    RELIABILITY_ACTION,
    RELIABILITY_BANDS,
    reliability,
)

TIER_COLOURS = {  # background, text — soft status badges; colour is used only for status
    "High": ("#FEF3F2", "#B42318"),
    "Medium": ("#FFFAEB", "#B54708"),
    "Low": ("#ECFDF3", "#067647"),
}


def _tier_css(value) -> str:
    bg, fg = TIER_COLOURS.get(str(value), (None, None))
    return f"background-color: {bg}; color: {fg}; font-weight: 600" if bg else ""


def style_tiers(df: pd.DataFrame, tier_columns: list[str] = (), count_columns: dict[str, str] | None = None):
    """Colour tier cells (High red, Medium yellow, Low green). `count_columns` maps a count column to a tier:
    the cell is coloured when the count is above zero (e.g. the number of High-risk SKUs in a store)."""
    styler = df.style
    cols = [c for c in tier_columns if c in df.columns]
    if cols:
        styler = styler.map(_tier_css, subset=cols)
    for col, tier in (count_columns or {}).items():
        if col in df.columns:
            styler = styler.map(lambda v, t=tier: _tier_css(t) if pd.notna(v) and v > 0 else "", subset=[col])
    return styler


def forecast_summary(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Bullet sentences + a day-by-day table for one store × category forecast (category_forecast output)."""
    hist = df.dropna(subset=["actual"]).copy()
    hist["actual"] = hist["actual"].astype(float)
    future = df.dropna(subset=["future_forecast"]).astype({"future_forecast": float, "lower": float, "upper": float})
    lines: list[str] = []
    if future.empty or hist.empty:
        return ["No forecast available for this selection."], pd.DataFrame()
    nxt = float(future["future_forecast"].sum())
    last7 = float(hist.tail(7)["actual"].sum())
    change = (nxt / last7 - 1) if last7 else None
    peak = future.loc[future["future_forecast"].idxmax()]
    low = future.loc[future["future_forecast"].idxmin()]
    lines.append(f"**Next 7 days** — {nxt:,.0f} units (~{nxt / len(future):.1f}/day, "
                 f"{future['day'].min():%d %b}–{future['day'].max():%d %b})")
    if change is not None:
        lines.append(f"**vs last 7 days** — {change:+.0%} ({last7:,.0f} sold)")
    lines.append(f"**Peak / low** — {peak['day']:%a %d %b} (~{peak['future_forecast']:.1f}) · "
                 f"{low['day']:%a %d %b} (~{low['future_forecast']:.1f})")
    test = hist.dropna(subset=["backtest_forecast"])
    if len(test):
        err = (test["actual"] - test["backtest_forecast"]).abs()
        wape = err.sum() / test["actual"].sum() if test["actual"].sum() else float("nan")
        lines.append(f"**September accuracy** — ±{err.mean():.1f} units/day on ~{test['actual'].mean():.1f} sold "
                     f"(WAPE {wape:.2f})")
    table = pd.DataFrame({
        "date": future["day"].dt.strftime("%a %d %b"),
        "forecast (units)": future["future_forecast"].round(1),
        "likely low": future["lower"].round(1),
        "likely high": future["upper"].round(1),
    })
    return lines, table


def stock_summary(df: pd.DataFrame, risk_row: pd.Series | None = None) -> list[str]:
    """Bullet sentences for one store × SKU (sku_timeline output) and its row in the risk list, if any."""
    hist = df.dropna(subset=["closing_stock"]).astype({"closing_stock": float, "units_sold": float,
                                                       "reorder_level": float})
    future = df.dropna(subset=["forecast"]).astype({"forecast": float})
    if hist.empty:
        return ["No stock history for this SKU."]
    last = hist.iloc[-1]
    stock, reorder = float(last["closing_stock"]), float(last["reorder_level"])
    lines = [f"**On shelf ({last['day']:%d %b})** — {stock:.0f} units · reorder level {reorder:.0f}"
             f"{' · at or below reorder' if stock <= reorder else ''}"]
    recent = hist.tail(28)
    lines.append(f"**Last 28 days** — {float(recent['units_sold'].sum()):.0f} sold "
                 f"(~{float(recent['units_sold'].mean()):.1f}/day) · "
                 f"{int((recent['closing_stock'] <= 0).sum())} days out of stock")
    if len(future):
        f3, f7 = float(future.head(3)["forecast"].sum()), float(future["forecast"].sum())
        lines.append(f"**Expected demand** — {f3:.1f} units (3 days) · {f7:.1f} (7 days)")
        cum = future["forecast"].cumsum().to_numpy()
        hit = next((i for i, c in enumerate(cum) if c >= stock), None)
        if stock <= 0:
            lines.append("**Run-out** — already out of stock · sales lost until delivery")
        elif hit is not None:
            lines.append(f"**Run-out** — around {future['day'].iloc[hit]:%a %d %b} (in ~{hit + 1} days)")
        else:
            daily = f7 / len(future)
            cover = stock / daily if daily > 0 else math.inf
            lines.append(f"**Run-out** — not this week "
                         f"(~{'many weeks' if cover == math.inf else f'{cover:.0f} days'} of cover)")
    if risk_row is not None:
        qty = int(risk_row["suggested_qty"]) if pd.notna(risk_row["suggested_qty"]) else 0
        lines.append(f"**Action** — {risk_row['risk_tier']} risk ({risk_row['reason_codes'] or 'no warning'}) · "
                     f"order {qty} units")
    return lines


# Forecast reliability: one definition in src/serving/reliability.py (shared with the business assistant).
_RELIABILITY_TO_TIER = {"High": "Low", "Medium": "Medium", "Low": "High"}   # High reliability = green


def style_reliability(df: pd.DataFrame, column: str = "reliability"):
    return df.style.map(lambda v: _tier_css(_RELIABILITY_TO_TIER.get(str(v))), subset=[column])


def accuracy_headline(pooled: pd.Series, by_horizon: pd.DataFrame) -> str:
    """One sentence: how much better than the best simple rule, and whether it holds over 7 days."""
    best_rule = min(pooled["baseline_moving_avg"], pooled["baseline_seasonal_naive"])
    gain = 1 - pooled["model"] / best_rule
    spread = by_horizon["model"].max() - by_horizon["model"].min() if len(by_horizon) else 0
    holds = "stays just as accurate up to 7 days ahead" if spread < 0.02 else "gets less accurate further ahead"
    verdict = ":green-badge[Reliable for ordering]" if gain > 0 else ":red-badge[Does not beat simple rules]"
    return (f"{verdict} **{gain:.0%} less error** than the best simple rule · {holds} · "
            "September test, trained on Apr–Aug only")
