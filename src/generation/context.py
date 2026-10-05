"""Shared state and helpers for the data generator."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np

from src.common.config import Config, derived_targets

IST_OFFSET = np.timedelta64(19800, "s")  # UTC+05:30
SECONDS_PER_DAY = 86400


@dataclass
class GenContext:
    cfg: Config
    targets: dict[str, int] = field(init=False)
    days: np.ndarray = field(init=False)  # business dates, datetime64[D]
    ground_truth: dict[str, list[dict]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.targets = derived_targets(self.cfg)
        start = np.datetime64(self.cfg.calendar.start_date, "D")
        self.days = start + np.arange(self.cfg.calendar.n_days)

    def rng(self, *names: object) -> np.random.Generator:
        """Independent, reproducible RNG per named component.

        Each module draws from its own stream, so changing one generator does not
        shift the random numbers of another.
        """
        key = "/".join(str(n) for n in names)
        return np.random.default_rng([self.cfg.seed, zlib.crc32(key.encode())])

    def faker_seed(self, name: str) -> int:
        return (self.cfg.seed * 1_000_003 + zlib.crc32(name.encode())) % (2**31)

    @property
    def n_days(self) -> int:
        return len(self.days)

    def day_start_utc(self, day_idx: np.ndarray | int) -> np.ndarray:
        """UTC instant of 00:00 IST on the given business day(s)."""
        return self.days[day_idx].astype("datetime64[s]") - IST_OFFSET

    @property
    def end_utc(self) -> np.datetime64:
        """Last instant of the business period (23:59:59 IST on the end date)."""
        return self.days[-1].astype("datetime64[s]") + np.timedelta64(SECONDS_PER_DAY - 1, "s") - IST_OFFSET


def make_ids(prefix: str, n: int, width: int, start: int = 1) -> np.ndarray:
    nums = np.arange(start, start + n).astype(str)
    return np.char.add(prefix, np.char.zfill(nums, width))


def to_ist_date(ts_utc: np.ndarray) -> np.ndarray:
    return (ts_utc.astype("datetime64[s]") + IST_OFFSET).astype("datetime64[D]")


def largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Split an integer total across keys in proportion to weights, summing exactly to total."""
    keys = list(weights)
    w = np.array([weights[k] for k in keys], dtype=float)
    if total <= 0 or w.sum() <= 0:
        return {k: 0 for k in keys}
    raw = w / w.sum() * total
    base = np.floor(raw).astype(int)
    order = np.argsort(-(raw - base), kind="stable")
    base[order[: total - base.sum()]] += 1
    return dict(zip(keys, base.tolist()))
