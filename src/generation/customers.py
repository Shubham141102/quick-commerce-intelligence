"""Customers with latent personas (persona labels go to ground truth only)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
from faker import Faker

from src.generation.context import GenContext, make_ids

EMAIL_DOMAINS = ["gmail.com", "yahoo.co.in", "outlook.com", "rediffmail.com", "hotmail.com"]


@dataclass
class CustomerData:
    customers: pd.DataFrame   # source columns + persona_idx, city_idx, activity, signup (datetime64[D])
    persona_names: list[str]


def _email_local(name: str, rng: np.random.Generator) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    parts = [p for p in re.split(r"[^a-z]+", ascii_name.lower()) if p]
    base = ".".join(parts[:2]) if parts else "user"
    return f"{base}{int(rng.integers(1, 999))}"


def build_customers(ctx: GenContext) -> CustomerData:
    cfg = ctx.cfg
    rng = ctx.rng("customers")
    fake = Faker("en_IN")
    fake.seed_instance(ctx.faker_seed("customers"))
    n = cfg.counts.customers

    persona_names = list(cfg.personas)
    shares = np.array([cfg.personas[p].share for p in persona_names])
    persona_idx = rng.choice(len(persona_names), size=n, p=shares)
    city_idx = rng.integers(0, cfg.counts.cities, size=n)

    # 75% signed up before the period (since 2023-01-01), 25% during it
    start = ctx.days[0]
    before_span = int((start - np.datetime64("2023-01-01")) // np.timedelta64(1, "D"))
    during = rng.random(n) < 0.25
    offsets = np.where(during, rng.integers(0, ctx.n_days, size=n), -rng.integers(1, before_span + 1, size=n))
    signup = start + offsets.astype("timedelta64[D]")

    # Order propensity: persona frequency x individual heterogeneity (heavy-tailed)
    freq = np.array([cfg.personas[p].orders_per_month for p in persona_names])[persona_idx]
    activity = freq * rng.lognormal(0.0, 0.6, size=n)

    names = [fake.name() for _ in range(n)]
    emails = [f"{_email_local(nm, rng)}@{rng.choice(EMAIL_DOMAINS)}" for nm in names]
    missing_email = rng.random(n) < cfg.dirty["outside_budget"]["missing_optional"]["customers.email"]
    emails = [None if m else e for e, m in zip(emails, missing_email)]

    customers = pd.DataFrame({
        "customer_id": make_ids("CUST", n, 5),
        "name": names,
        "email": emails,
        "city_id": np.array([c.city_id for c in cfg.cities])[city_idx],
        "signup_date": signup,
        "persona_idx": persona_idx,
        "city_idx": city_idx,
        "activity": activity,
    })

    ctx.ground_truth["gt_personas"] = [
        {"customer_id": cid, "persona": persona_names[p]}
        for cid, p in zip(customers["customer_id"], persona_idx)
    ]
    return CustomerData(customers, persona_names)
