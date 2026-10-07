"""Plain-language summaries of what a chart shows (so the numbers can be read without the chart), and
tier colour-coding for tables."""

from __future__ import annotations

import math

import pandas as pd

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
    lines.append(f"**Next 7 days ({future['day'].min():%d %b}–{future['day'].max():%d %b}):** about **{nxt:,.0f} units** "
                 f"in total, roughly **{nxt / len(future):.1f} a day**.")
    if change is not None:
        word = "more" if change >= 0 else "fewer"
        lines.append(f"**Compared with the last 7 days** ({last7:,.0f} units sold): about **{abs(change):.0%} {word}**.")
    lines.append(f"**Busiest day:** {peak['day']:%a %d %b} (~{peak['future_forecast']:.1f} units); "
                 f"**quietest:** {low['day']:%a %d %b} (~{low['future_forecast']:.1f}).")
    test = hist.dropna(subset=["backtest_forecast"])
    if len(test):
        err = (test["actual"] - test["backtest_forecast"]).abs()
        wape = err.sum() / test["actual"].sum() if test["actual"].sum() else float("nan")
        lines.append(f"**How reliable for this store × category:** in September the model's next-day forecast was off by "
                     f"**{err.mean():.1f} units a day** on average (WAPE {wape:.2f}), while the store sold "
                     f"{test['actual'].mean():.1f} a day.")
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
    lines = [f"**On the shelf at the end of {last['day']:%d %b}:** **{stock:.0f} units** "
             f"(reorder level {reorder:.0f}{' — at or below it' if stock <= reorder else ''})."]
    recent = hist.tail(28)
    lines.append(f"**Last 28 days:** sold {float(recent['units_sold'].sum()):.0f} units "
                 f"(~{float(recent['units_sold'].mean()):.1f} a day); out of stock on "
                 f"**{int((recent['closing_stock'] <= 0).sum())}** of those days.")
    if len(future):
        f3, f7 = float(future.head(3)["forecast"].sum()), float(future["forecast"].sum())
        lines.append(f"**Expected demand:** {f3:.1f} units over the next 3 days, {f7:.1f} over 7 days.")
        cum = future["forecast"].cumsum().to_numpy()
        hit = next((i for i, c in enumerate(cum) if c >= stock), None)
        if stock <= 0:
            lines.append("**Already out of stock** — every sale until the next delivery is lost.")
        elif hit is not None:
            lines.append(f"**At that rate it will likely run out around {future['day'].iloc[hit]:%a %d %b}** "
                         f"(about {hit + 1} day(s) from now).")
        else:
            daily = f7 / len(future)
            cover = stock / daily if daily > 0 else math.inf
            lines.append(f"**Enough for the whole week**; at ~{daily:.1f} a day the stock lasts about "
                         f"{'many weeks' if cover == math.inf else f'{cover:.0f} days'}.")
    if risk_row is not None:
        qty = int(risk_row["suggested_qty"]) if pd.notna(risk_row["suggested_qty"]) else 0
        lines.append(f"**Risk tier: {risk_row['risk_tier']}** ({risk_row['reason_codes'] or 'no warning'}); "
                     f"suggested order: **{qty} units**.")
    return lines


# Forecast reliability per category, from its September WAPE. Bands fixed before looking at the results
# (2026-10-06): High < 0.55 <= Medium <= 0.70 < Low.
RELIABILITY_BANDS = (0.55, 0.70)
RELIABILITY_ACTION = {
    "High": "Order close to the forecast.",
    "Medium": "Keep a normal safety buffer.",
    "Low": "Keep a bigger buffer and review these SKUs manually.",
}
_RELIABILITY_TO_TIER = {"High": "Low", "Medium": "Medium", "Low": "High"}   # High reliability = green


def reliability(wape: float) -> str:
    if wape < RELIABILITY_BANDS[0]:
        return "High"
    return "Medium" if wape <= RELIABILITY_BANDS[1] else "Low"


def style_reliability(df: pd.DataFrame, column: str = "reliability"):
    return df.style.map(lambda v: _tier_css(_RELIABILITY_TO_TIER.get(str(v))), subset=[column])


def accuracy_headline(pooled: pd.Series, by_horizon: pd.DataFrame) -> str:
    """One sentence: how much better than the best simple rule, and whether it holds over 7 days."""
    best_rule = min(pooled["baseline_moving_avg"], pooled["baseline_seasonal_naive"])
    gain = 1 - pooled["model"] / best_rule
    spread = by_horizon["model"].max() - by_horizon["model"].min() if len(by_horizon) else 0
    holds = "stays just as accurate up to 7 days ahead" if spread < 0.02 else "gets less accurate further ahead"
    verdict = "✅ **The forecast is reliable for ordering.**" if gain > 0 else "⚠️ **The forecast does not beat simple rules.**"
    return (f"{verdict} In a September test (model trained on April–August only), it made **{gain:.0%} less error "
            f"than the best simple rule** and {holds}.")
