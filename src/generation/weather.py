"""3-hourly weather per city, with a monsoon rainfall pattern that drives demand."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.generation.context import IST_OFFSET, GenContext

# Probability of a rainy day and seasonal temperature offset, by month
RAIN_DAY_PROB = {1: 0.05, 2: 0.05, 3: 0.08, 4: 0.12, 5: 0.20, 6: 0.55, 7: 0.70, 8: 0.65, 9: 0.50,
                 10: 0.30, 11: 0.12, 12: 0.06}
TEMP_OFFSET = {1: -6, 2: -4, 3: 0, 4: 2, 5: 3, 6: 0, 7: -2, 8: -2, 9: -1, 10: -1, 11: -3, 12: -5}
MONSOON_MONTHS = {6, 7, 8, 9}


@dataclass
class WeatherData:
    weather: pd.DataFrame        # source columns
    daily_rain_mm: np.ndarray    # [city, day]


def build_weather(ctx: GenContext) -> WeatherData:
    cfg = ctx.cfg
    rng = ctx.rng("weather")
    readings = cfg.counts.weather_readings_per_day
    step_h = 24 // readings
    months = (ctx.days.astype("datetime64[M]").astype(int) % 12) + 1

    frames = []
    daily_rain = np.zeros((len(cfg.cities), ctx.n_days))
    for ci, city in enumerate(cfg.cities):
        for d in range(ctx.n_days):
            m = int(months[d])
            rainy = rng.random() < RAIN_DAY_PROB[m]
            intensity = city.monsoon_rain_mm_day * (1.0 if m in MONSOON_MONTHS else 0.3)
            total = float(rng.gamma(0.9, intensity / 0.9)) if rainy else 0.0
            daily_rain[ci, d] = total
            split = rng.dirichlet(np.full(readings, 0.6)) * total
            hours = np.arange(readings) * step_h
            diurnal = 4.0 * np.sin((hours - 9) / 24 * 2 * np.pi)  # peak mid-afternoon
            temp = city.base_temp_c + TEMP_OFFSET[m] + diurnal - 0.15 * split + rng.normal(0, 0.8, readings)
            humidity = 55 + (22 if m in MONSOON_MONTHS else 0) + 1.5 * split + rng.normal(0, 5, readings)
            ts_ist = ctx.days[d].astype("datetime64[s]") + (hours * 3600).astype("timedelta64[s]")
            frames.append(pd.DataFrame({
                "city_id": city.city_id,
                "observation_ts": ts_ist - IST_OFFSET,
                "temperature_c": np.round(temp, 1),
                "rainfall_mm": np.round(split, 1),
                "humidity_pct": np.round(np.clip(humidity, 25, 99), 1),
            }))
    weather = pd.concat(frames, ignore_index=True)
    return WeatherData(weather, daily_rain)
