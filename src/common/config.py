"""Load and validate run configuration.

`configs/default.yaml` is deep-merged with a profile file (small/medium/large),
then the cities and dirty-data files are attached. The result is a frozen,
validated `Config`. `derived_targets` turns it into per-dataset row targets.
"""

from __future__ import annotations

import copy
import math
from datetime import date, timedelta
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from src.common.paths import resolve

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Calendar(_Model):
    start_date: date
    end_date: date
    storage_timezone: str = "UTC"
    business_timezone: str = "Asia/Kolkata"

    @model_validator(mode="after")
    def _check_order(self) -> Calendar:
        if self.end_date < self.start_date:
            raise ValueError("calendar.end_date is before start_date")
        return self

    @property
    def n_days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    def dates(self) -> list[date]:
        return [self.start_date + timedelta(days=i) for i in range(self.n_days)]


class Slice(_Model):
    start: date
    end: date
    file_grain: str | None = None
    micro_batch_minutes: int | None = None


class IngestionSlices(_Model):
    historical: Slice
    batch: Slice
    stream: Slice


class MlSplit(_Model):
    train_end: date
    test_start: date


class Storage(_Model):
    format: str
    encoding: str
    delimiter: str
    quote: str
    escape: str
    header: bool
    null_value: str
    timestamp_format: str
    date_format: str
    list_separator: str
    single_file_max_rows: int


class Ratios(_Model):
    avg_items_per_order: float
    payment_retry_rate: float
    cancellation_rate: float
    pre_dispatch_cancel_share: float
    refund_rate: float
    order_promotion_rate: float
    review_rate: float
    partners_per_store: int


class Demand(_Model):
    product_popularity_zipf_s: float
    focus_skus_per_store: int
    snapshot_weekday: str
    snapshot_hour_ist: int
    day_of_week_factors: dict[str, float]
    rain_uplift_per_10mm: float
    monthly_trend: float
    restock_lead_time_days: tuple[int, int]
    delivery_minutes_lognormal: dict[str, float]
    delivery_sla_minutes: int

    @model_validator(mode="after")
    def _check(self) -> Demand:
        if set(self.day_of_week_factors) != set(WEEKDAYS):
            raise ValueError(f"day_of_week_factors must define exactly {WEEKDAYS}")
        if self.snapshot_weekday not in WEEKDAYS:
            raise ValueError(f"snapshot_weekday must be one of {WEEKDAYS}")
        return self


class Persona(_Model):
    share: float
    orders_per_month: float
    avg_basket: float
    night_share: float
    promo_sensitivity: float


class Planted(_Model):
    affinity_pairs: int
    affinity_attach_probability: tuple[float, float]
    anomalies: int
    stockout_episodes: int
    outlier_order_rate: float


class Counts(_Model):
    cities: int
    stores_per_city: int
    categories: int
    products: int
    customers: int
    promotions: int
    orders: int
    inventory_events: int
    application_events: int
    application_logs: int
    weather_readings_per_day: int


class Holiday(_Model):
    date: date
    name: str


class City(_Model):
    city_id: str
    city: str
    state: str
    lat: float
    lon: float
    base_temp_c: float
    monsoon_rain_mm_day: float


class Config(_Model):
    profile: str
    project: dict[str, str]
    seed: int
    calendar: Calendar
    ingestion_slices: IngestionSlices
    ml_split: MlSplit
    paths: dict[str, str]
    storage: Storage
    ratios: Ratios
    demand: Demand
    personas: dict[str, Persona]
    planted: Planted
    counts: Counts
    holidays: tuple[Holiday, ...] = ()
    dirty_data_config: str
    cities_config: str
    cities: tuple[City, ...]
    store_radius_km: float
    dirty: dict[str, Any]

    @model_validator(mode="after")
    def _check(self) -> Config:
        share = sum(p.share for p in self.personas.values())
        if not math.isclose(share, 1.0, abs_tol=1e-6):
            raise ValueError(f"persona shares must sum to 1.0, got {share}")
        if len(self.cities) != self.counts.cities:
            raise ValueError(f"profile needs {self.counts.cities} cities, config defines {len(self.cities)}")
        s = self.ingestion_slices
        if s.historical.start != self.calendar.start_date or s.stream.end != self.calendar.end_date:
            raise ValueError("ingestion slices must start and end with the calendar")
        if s.batch.start != s.historical.end + timedelta(days=1) or s.stream.start != s.batch.end + timedelta(days=1):
            raise ValueError("ingestion slices must be contiguous: historical -> batch -> stream")
        if self.focus_skus > self.counts.products:
            raise ValueError("focus_skus_per_store exceeds product count")
        return self

    @property
    def n_stores(self) -> int:
        return self.counts.cities * self.counts.stores_per_city

    @property
    def focus_skus(self) -> int:
        return self.demand.focus_skus_per_store

    def snapshot_dates(self) -> list[date]:
        wd = WEEKDAYS.index(self.demand.snapshot_weekday)
        return [d for d in self.calendar.dates() if d.weekday() == wd]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: str) -> dict:
    with open(resolve(path), encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(profile: str = "small", seed: int | None = None, overrides: dict | None = None) -> Config:
    raw = _deep_merge(_read_yaml("configs/default.yaml"), _read_yaml(f"configs/{profile}.yaml"))
    if overrides:
        raw = _deep_merge(raw, overrides)
    if seed is not None:
        raw["seed"] = seed
    cities_doc = _read_yaml(raw["cities_config"])
    raw["cities"] = cities_doc["cities"][: raw["counts"]["cities"]]
    raw["store_radius_km"] = cities_doc.get("store_radius_km", 10)
    raw["dirty"] = _read_yaml(raw["dirty_data_config"])
    return Config.model_validate(raw)


def derived_targets(cfg: Config) -> dict[str, int]:
    """Clean (pre-duplicate) row targets per source dataset (Project_Plan_v2.md §3.4)."""
    c, r = cfg.counts, cfg.ratios
    orders = c.orders
    cancellations = round(orders * r.cancellation_rate)
    pre_dispatch = round(cancellations * r.pre_dispatch_cancel_share)
    return {
        "categories": c.categories,
        "products": c.products,
        "stores": cfg.n_stores,
        "customers": c.customers,
        "delivery_partners": cfg.n_stores * r.partners_per_store,
        "promotions": c.promotions,
        "orders": orders,
        "order_items": round(orders * r.avg_items_per_order),
        "payments": orders + round(orders * r.payment_retry_rate),
        "deliveries": orders - pre_dispatch,
        "cancellations": cancellations,
        "returns_refunds": round(orders * r.refund_rate),
        "order_promotions": round(orders * r.order_promotion_rate),
        "reviews": round(orders * r.review_rate),
        "inventory_snapshots": cfg.n_stores * cfg.focus_skus * len(cfg.snapshot_dates()),
        "inventory_events": c.inventory_events,
        "application_events": c.application_events,
        "weather": c.cities * cfg.calendar.n_days * c.weather_readings_per_day,
        "application_logs": c.application_logs,
    }


def pre_dispatch_cancellations(cfg: Config) -> int:
    cancellations = round(cfg.counts.orders * cfg.ratios.cancellation_rate)
    return round(cancellations * cfg.ratios.pre_dispatch_cancel_share)
