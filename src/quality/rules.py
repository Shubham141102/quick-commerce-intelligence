"""Data-quality rule model (Project_Plan_v2.md §6).

A rule is named `<category>:<detail>`, e.g. `range_invalid:quantity`. Its check returns a
boolean Column that is TRUE when the row FAILS (null means "not applicable", never a
failure). Hard rules send the row to quarantine; soft rules keep it and set a
`dq_<detail>` flag column.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import or_
from typing import Callable

from pyspark.sql import Column
from pyspark.sql import functions as F

# Highest priority first: decides `primary_rule` when a row fails several rules.
RULE_PRIORITY = (
    "malformed", "required_missing", "type_invalid", "enum_invalid", "range_invalid",
    "sequence_invalid", "fk_missing", "business", "parent_rejected",
)
# Which injected issue code each rule category detects (for detected-vs-injected reports).
ISSUE_FOR_CATEGORY = {
    "malformed": "MALF", "required_missing": "MISS", "type_invalid": "TYPE", "enum_invalid": "FMT",
    "range_invalid": "NUM", "sequence_invalid": "SEQ", "fk_missing": "FK", "business": "BIZ",
    "parent_rejected": "cascade",
}


@dataclass(frozen=True)
class Rule:
    name: str
    fails: Callable[[], Column]   # built lazily: Columns need an active SparkSession
    hard: bool = True
    # soft rules only: when the check can't be evaluated (null), store the flag as false (default)
    # or keep it empty, meaning "could not verify"
    unknown_is_pass: bool = True
    description: str = ""          # plain-English rule, shown in docs/transformation_catalog.md

    @property
    def category(self) -> str:
        return self.name.split(":", 1)[0]

    @property
    def detail(self) -> str:
        return self.name.split(":", 1)[1]

    @property
    def flag_column(self) -> str:
        return f"dq_{self.detail}"


def range_rule(column: str, lo: float | None = None, hi: float | None = None,
               lo_inclusive: bool = True, hi_inclusive: bool = True) -> Rule:
    def fails() -> Column:
        c = F.col(column)
        conditions = []
        if lo is not None:
            conditions.append(c < lo if lo_inclusive else c <= lo)
        if hi is not None:
            conditions.append(c > hi if hi_inclusive else c >= hi)
        return reduce(or_, conditions)
    bounds = []
    if lo is not None:
        bounds.append(f"{'≥' if lo_inclusive else '>'} {lo:g}")
    if hi is not None:
        bounds.append(f"{'≤' if hi_inclusive else '<'} {hi:g}")
    return Rule(f"range_invalid:{column}", fails, description=f"`{column}` must be {' and '.join(bounds)}")


def sequence_rule(detail: str, later: str, earlier: str, description: str = "") -> Rule:
    """Fails when `later` is before `earlier` (e.g. delivered before pickup)."""
    return Rule(f"sequence_invalid:{detail}", lambda: F.col(later) < F.col(earlier),
                description=description or f"`{later}` must not be before `{earlier}`")


def business_rule(detail: str, fails: Callable[[], Column], description: str, hard: bool = False,
                  unknown_is_pass: bool = True) -> Rule:
    return Rule(f"business:{detail}", fails, hard, unknown_is_pass, description)


def _starts_with(prefix: str) -> Callable[[Column], Column]:
    # a one-argument lambda: PySpark treats a two-argument one as (element, index)
    return lambda x: x.startswith(prefix)


def primary_rule(failures: Column) -> Column:
    """First failure by RULE_PRIORITY."""
    return F.coalesce(*[F.element_at(F.filter(failures, _starts_with(p)), 1) for p in RULE_PRIORITY])
