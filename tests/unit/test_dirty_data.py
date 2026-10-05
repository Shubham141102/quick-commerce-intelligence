import numpy as np
import pandas as pd
import pytest

from src.common.config import load_config
from src.common.io import to_source_strings
from src.common.schemas import SOURCE_SCHEMAS
from src.generation.dirty_data import _Mutator, plan_issue_counts
from src.generation.generate_all import build_lookups
from src.generation.landing import SLICE_TS

DIRTY = load_config("small").dirty


def test_budget_sums_to_ten_percent():
    assert sum(DIRTY["budget"].values()) == pytest.approx(0.10)


@pytest.mark.parametrize("dataset", list(SOURCE_SCHEMAS))
def test_plan_keeps_ninety_percent_clean(dataset):
    n = 36_000
    counts = plan_issue_counts(dataset, n, DIRTY)
    final = n + counts.get("DUP", 0)
    dirty = sum(counts.values())
    assert dirty == round(0.10 * final)
    assert set(counts) == set(DIRTY["applicability"][dataset])


def test_dimension_tables_only_get_correctable_issues():
    for ds in ["categories", "stores", "products", "customers", "delivery_partners"]:
        assert set(DIRTY["applicability"][ds]) <= {"DUP", "FMT"}


def test_every_configured_mutation_is_implemented_and_applies(small_run):
    """Each mutation in dirty_data.yaml must succeed on at least one real clean row."""
    lookups = build_lookups(small_run.clean)
    rng = np.random.default_rng(0)
    for dataset, by_code in DIRTY["mutations"].items():
        schema = SOURCE_SCHEMAS[dataset]
        rows = to_source_strings(small_run.clean[dataset], schema, keep=(SLICE_TS,)).to_dict("records")
        mutator = _Mutator(dataset, schema, rng, lookups)
        for code, names in by_code.items():
            assert code in DIRTY["applicability"][dataset], (dataset, code)
            for name in names:
                assert any(mutator.apply(code, name, dict(r)) for r in rows[:500]), (dataset, code, name)


def _normalise(values, dtype):
    if dtype == "decimal":
        return values.astype(float)
    return values.str.strip().str.lower().str.replace("+00:00", "z", regex=False)


def test_conflicting_duplicates_differ_only_in_formatting(small_run):
    for dataset, res in small_run.injected.items():
        schema = SOURCE_SCHEMAS[dataset]
        conflicting = [r for r in res.issue_rows if r["mutation"].startswith("conflicting:")]
        for issue in conflicting[:5]:
            col = issue["mutation"].split(":", 1)[1]
            mask = np.ones(len(res.rows), dtype=bool)
            for kv in issue["business_key"].split(";"):
                k, v = kv.split("=", 1)
                if k == col:  # the varied column can itself be part of the key
                    mask &= (_normalise(res.rows[k], schema.column(k).dtype) == _normalise(pd.Series([v]), schema.column(k).dtype)[0]).to_numpy()
                else:
                    mask &= (res.rows[k] == v).to_numpy()
            values = res.rows.loc[mask, col]
            assert len(values) >= 2, (dataset, issue)
            assert _normalise(values, schema.column(col).dtype).nunique() == 1, (dataset, col, values.tolist())
