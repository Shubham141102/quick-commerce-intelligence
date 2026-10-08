"""Forecast reliability per category, from its September WAPE — one definition for the app and the assistant.

Bands fixed before looking at the results (2026-10-06): High < 0.55 ≤ Medium ≤ 0.70 < Low.
"""

from __future__ import annotations

RELIABILITY_BANDS = (0.55, 0.70)
RELIABILITY_ACTION = {
    "High": "Order close to the forecast.",
    "Medium": "Keep a normal safety buffer.",
    "Low": "Keep a bigger buffer and review these SKUs manually.",
}


def reliability(wape: float) -> str:
    if wape < RELIABILITY_BANDS[0]:
        return "High"
    return "Medium" if wape <= RELIABILITY_BANDS[1] else "Low"
