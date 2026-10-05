import pytest
from pydantic import ValidationError

from src.common.config import derived_targets, load_config


@pytest.mark.parametrize("profile", ["small", "medium", "large"])
def test_profiles_load(profile):
    cfg = load_config(profile)
    assert cfg.profile == profile
    assert len(cfg.cities) == cfg.counts.cities


def test_medium_targets_match_plan():
    targets = derived_targets(load_config("medium"))
    assert targets["order_items"] == 122_400
    assert targets["payments"] == 37_800
    assert targets["deliveries"] == 34_000
    assert targets["inventory_snapshots"] == 7_800  # 12 stores x 25 focus SKUs x 26 Mondays
    assert targets["weather"] == 4_392
    assert sum(targets.values()) == 298_124


def test_persona_shares_must_sum_to_one():
    with pytest.raises(ValidationError, match="persona shares"):
        load_config("small", overrides={"personas": {"premium": {"share": 0.5}}})


def test_seed_override():
    assert load_config("small", seed=7).seed == 7
