"""Controlled dirty-data injection (Project_Plan_v2.md §4.2, configs/dirty_data.yaml).

Works on the CSV string form of one clean dataset:

- Final rows F = clean rows / (1 - duplicate share); duplicates are added as copies.
- Exactly round(10% x F) rows are dirty; each dirty row gets one issue.
- Issue types that don't apply to the dataset hand their share to the ones that do.
- Every injected issue is recorded (dataset, code, mutation, business key) for the
  manifest; the rows themselves carry no marker.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.common.schemas import TableSchema
from src.generation.context import largest_remainder

ISO = "%Y-%m-%dT%H:%M:%SZ"
DATE = "%Y-%m-%d"
ALT_TS_FORMATS = ["%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"]
BAD_NUMBERS = ["N/A", "two", "--", "unknown", "12a"]
ID_FORMATS = {  # column -> (prefix, digits); unknown ids use a leading 9, outside the generated range
    "customer_id": ("CUST", 5), "store_id": ("S", 2), "product_id": ("P", 4), "order_id": ("ORD", 7),
    "partner_id": ("DP", 4), "payment_id": ("PAY", 7), "promotion_id": ("PR", 3),
}
CASING_COLUMNS = {
    "status_casing": "status", "method_casing": "method", "reason_casing": "reason",
    "event_type_casing": "event_type", "event_name_casing": "event_name", "log_level_casing": "log_level",
    "name_casing": "category_name", "city_casing": "city", "email_casing": "email", "brand_casing": "brand",
    "availability_casing": "availability_status",
}
WHITESPACE_COLUMNS = {
    "name_whitespace": {"products": "product_name", "stores": "store_name", "customers": "name", "promotions": "name"},
    "comment_whitespace": {"reviews": "comment"},
    "status_whitespace": {"orders": "status"},
}
TEXT_COLUMNS = {
    "total_amount_text": "total_amount", "quantity_text": "quantity", "amount_text": "amount",
    "stock_quantity_text": "stock_quantity", "temperature_text": "temperature_c",
}
NEGATIVE_COLUMNS = {
    "negative_total_amount": "total_amount", "negative_quantity": "quantity", "negative_unit_price": "unit_price",
    "negative_amount": "amount", "negative_stock": "stock_quantity", "negative_discount_amount": "discount_amount",
}
# How a "conflicting" duplicate differs: a formatting-only change that Silver standardisation removes
CONFLICT_VARIANT = {
    "categories": ("category_name", "casing"), "stores": ("city", "casing"), "products": ("brand", "casing"),
    "customers": ("name", "whitespace"), "delivery_partners": ("availability_status", "casing"),
    "orders": ("status", "casing"), "order_items": ("unit_price", "decimal"), "payments": ("method", "casing"),
    "deliveries": ("status", "casing"), "cancellations": ("reason", "casing"),
    "inventory_snapshots": ("snapshot_ts", "ts_offset"), "inventory_events": ("event_type", "casing"),
    "promotions": ("name", "whitespace"), "order_promotions": ("discount_amount", "decimal"),
    "returns_refunds": ("amount", "decimal"), "reviews": ("review_ts", "ts_offset"),
    "application_events": ("event_name", "casing"), "weather": ("observation_ts", "ts_offset"),
    "application_logs": ("log_level", "casing"),
}


@dataclass
class Lookups:
    """Clean reference values some mutations need (built from the clean typed frames)."""
    valid_ids: dict[str, set[str]] = field(default_factory=dict)
    order_ts: dict[str, datetime] = field(default_factory=dict)
    order_date_ist: dict[str, str] = field(default_factory=dict)
    payment_amount: dict[str, float] = field(default_factory=dict)
    payment_ts: dict[str, datetime] = field(default_factory=dict)
    promotions: list[tuple[str, str, str]] = field(default_factory=list)   # (id, start, end) ISO dates
    partner_city: dict[str, str] = field(default_factory=dict)
    partners_by_city: dict[str, list[str]] = field(default_factory=dict)
    customer_products: dict[str, set[str]] = field(default_factory=dict)
    product_ids: list[str] = field(default_factory=list)


@dataclass
class InjectionResult:
    rows: pd.DataFrame
    raw_lines: dict[int, str]
    issue_counts: dict[str, int]
    issue_rows: list[dict]
    clean_rows: int
    original_rows: int
    duplicate_rows: int

    @property
    def final_rows(self) -> int:
        return len(self.rows)


def plan_issue_counts(dataset: str, n_original: int, dirty_cfg: dict) -> dict[str, int]:
    """Exact issue counts for one dataset; DUP is the number of duplicate copies to add."""
    applicable = dirty_cfg["applicability"][dataset]
    base = {**dirty_cfg["budget"], **dirty_cfg.get("substitute_shares", {})}
    dirty_ratio = 1 - dirty_cfg["clean_ratio"]
    weights = {code: base[code] for code in applicable}
    total_w = sum(weights.values())
    shares = {code: w / total_w * dirty_ratio for code, w in weights.items()}
    s_dup = shares.get("DUP", 0.0)
    final = round(n_original / (1 - s_dup))
    dup = final - n_original
    other = max(0, round(dirty_ratio * final) - dup)
    counts = largest_remainder(other, {k: v for k, v in shares.items() if k != "DUP"})
    if "DUP" in shares:
        counts["DUP"] = dup
    return counts


class _Mutator:
    def __init__(self, dataset: str, schema: TableSchema, rng: np.random.Generator, lookups: Lookups):
        self.ds, self.schema, self.rng, self.lk = dataset, schema, rng, lookups

    # -- helpers ---------------------------------------------------------------------------
    def _unknown_id(self, column: str) -> str:
        prefix, width = ID_FORMATS[column]
        valid = self.lk.valid_ids.get(column, set())
        while True:
            value = prefix + "9" + "".join(str(d) for d in self.rng.integers(0, 10, size=width - 1))
            if value not in valid:
                return value

    def _casing(self, value: str) -> str | None:
        options = [v for v in {value.upper(), value.title(), value.lower(), value.swapcase()} if v != value]
        return str(self.rng.choice(sorted(options))) if value and options else None

    def _pad(self, value: str) -> str | None:
        return str(self.rng.choice([f"  {value}", f"{value}  ", f" {value} "])) if value else None

    @staticmethod
    def _ts(value: str) -> datetime | None:
        try:
            return datetime.strptime(value, ISO)
        except (TypeError, ValueError):
            return None

    def _minutes(self, lo: int, hi: int) -> timedelta:
        return timedelta(minutes=int(self.rng.integers(lo, hi + 1)))

    # -- mutations -------------------------------------------------------------------------
    def apply(self, code: str, name: str, row: dict) -> bool:
        if code == "MISS":
            col = "delivered_ts" if name == "delivered_ts_on_delivered" else name
            if name == "delivered_ts_on_delivered" and row["status"] != "delivered":
                return False
            if row.get(col, "") == "":
                return False
            row[col] = ""
            return True
        handler = getattr(self, f"_m_{name}", None)
        if handler is not None:
            return handler(row)
        for table, fn in (
            (CASING_COLUMNS, self._casing), (TEXT_COLUMNS, None), (NEGATIVE_COLUMNS, None),
        ):
            if name in table:
                col = table[name]
                value = row.get(col, "")
                if fn is not None:
                    new = fn(value)
                elif table is TEXT_COLUMNS:
                    new = str(self.rng.choice(BAD_NUMBERS)) if value else None
                else:
                    new = "-" + value if value and _is_number(value) and float(value) > 0 else None
                if new is None:
                    return False
                row[col] = new
                return True
        if name in WHITESPACE_COLUMNS:
            col = WHITESPACE_COLUMNS[name][self.ds]
            new = self._pad(row.get(col, ""))
            if new is None:
                return False
            row[col] = new
            return True
        if name.endswith("_alt_format"):
            col = name.removesuffix("_alt_format")
            ts = self._ts(row.get(col, ""))
            if ts is None:
                return False
            row[col] = ts.strftime(str(self.rng.choice(ALT_TS_FORMATS)))
            return True
        if name.startswith("unknown_"):
            col = name.removeprefix("unknown_")
            if not row.get(col):
                return False
            row[col] = self._unknown_id(col)
            return True
        raise KeyError(f"unknown mutation {name!r} for {self.ds}")

    def _scale(self, row: dict, col: str, lo: float, hi: float) -> bool:
        if not _is_number(row.get(col, "")) or float(row[col]) <= 0:
            return False
        row[col] = f"{float(row[col]) * self.rng.uniform(lo, hi):.2f}"
        return True

    def _m_zero_quantity(self, row):
        if row["quantity"] in ("", "0"):
            return False
        row["quantity"] = "0"
        return True

    def _m_rating_out_of_range(self, row):
        row["rating"] = str(self.rng.choice(["0", "6", "7", "8", "10"]))
        return True

    def _m_discount_over_100(self, row):
        row["discount_pct"] = f"{int(self.rng.integers(120, 251))}.00"
        return True

    def _m_humidity_over_100(self, row):
        row["humidity_pct"] = f"{self.rng.uniform(101, 160):.1f}"
        return True

    def _m_quantity_sign_mismatch(self, row):
        if row["event_type"] not in ("restock", "damage") or not _is_number(row["quantity"]) or int(row["quantity"]) == 0:
            return False
        row["quantity"] = str(-int(row["quantity"]))
        return True

    def _m_total_amount_mismatch(self, row):
        return self._scale(row, "total_amount", 1.1, 1.5)

    def _m_unit_price_mismatch_vs_catalog(self, row):
        lo, hi = (0.5, 0.8) if self.rng.random() < 0.5 else (1.2, 1.6)
        return self._scale(row, "unit_price", lo, hi)

    def _m_amount_mismatch_vs_order(self, row):
        return self._scale(row, "amount", 1.1, 1.4)

    def _m_delivery_partner_wrong_city(self, row):
        city = self.lk.partner_city.get(row["partner_id"])
        others = sorted(c for c in self.lk.partners_by_city if c != city)
        if city is None or not others:
            return False
        pool = self.lk.partners_by_city[str(self.rng.choice(others))]
        row["partner_id"] = str(self.rng.choice(pool))
        return True

    def _m_promotion_not_active_on_order_date(self, row):
        day = self.lk.order_date_ist.get(row["order_id"])
        pool = [pid for pid, start, end in self.lk.promotions if not (start <= day <= end)] if day else []
        if not pool:
            return False
        row["promotion_id"] = str(self.rng.choice(pool))
        return True

    def _m_refund_exceeds_payment(self, row):
        amount = self.lk.payment_amount.get(row["payment_id"])
        if not amount:
            return False
        row["amount"] = f"{amount * self.rng.uniform(1.2, 2.0):.2f}"
        return True

    def _m_review_for_unpurchased_product(self, row):
        bought = self.lk.customer_products.get(row["customer_id"], set())
        for _ in range(20):
            candidate = str(self.rng.choice(self.lk.product_ids))
            if candidate not in bought:
                row["product_id"] = candidate
                return True
        return False

    def _m_delivered_before_pickup(self, row):
        pickup = self._ts(row["pickup_ts"])
        if pickup is None or not row["delivered_ts"]:
            return False
        row["delivered_ts"] = (pickup - self._minutes(1, 30)).strftime(ISO)
        return True

    def _m_cancelled_before_order(self, row):
        placed = self.lk.order_ts.get(row["order_id"])
        if placed is None:
            return False
        row["cancelled_ts"] = (placed - self._minutes(5, 120)).strftime(ISO)
        return True

    def _m_end_before_start(self, row):
        if not row["start_date"] or row["start_date"] == row["end_date"]:
            return False
        row["start_date"], row["end_date"] = row["end_date"], row["start_date"]
        return True

    def _m_refund_before_payment(self, row):
        paid = self.lk.payment_ts.get(row["payment_id"])
        if paid is None:
            return False
        row["refund_ts"] = (paid - timedelta(hours=int(self.rng.integers(1, 49)))).strftime(ISO)
        return True

    def _m_invalid_json_metadata(self, row):
        if len(row["metadata"]) < 3:
            return False
        row["metadata"] = row["metadata"][: -int(self.rng.integers(1, 4))]
        return True

    def _m_wrong_column_count(self, row):
        message = row["message"].replace('"', "")
        if "," not in message:
            message += ", retry=1"
        row["__raw__"] = ",".join([row[c] for c in self.schema.names[:-1]] + [message])
        return True

    def _m_truncated_row(self, row):
        buf = io.StringIO()
        csv.writer(buf, lineterminator="").writerow([row[c] for c in self.schema.names[:3]])
        row["__raw__"] = buf.getvalue()
        return True

    # -- conflicting duplicates ------------------------------------------------------------
    def conflict_variant(self, row: dict) -> str | None:
        col, kind = CONFLICT_VARIANT[self.ds]
        value = row.get(col, "")
        if not value:
            return None
        if kind == "casing":
            new = self._casing(value)
        elif kind == "whitespace":
            new = self._pad(value)
        elif kind == "decimal":
            new = (value[:-1] if value.endswith("0") and "." in value else None) if _is_number(value) else None
        else:  # ts_offset
            new = value[:-1] + "+00:00" if value.endswith("Z") else None
        if new is None:
            return None
        row[col] = new
        return col


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def inject(dataset: str, rows: pd.DataFrame, schema: TableSchema, dirty_cfg: dict,
           lookups: Lookups, rng: np.random.Generator) -> InjectionResult:
    n = len(rows)
    if not dirty_cfg.get("enabled", True) or n == 0:
        return InjectionResult(rows.reset_index(drop=True), {}, {}, [], n, n, 0)

    counts = plan_issue_counts(dataset, n, dirty_cfg)
    mutations = dirty_cfg["mutations"].get(dataset, {})
    records = rows.to_dict("records")
    mutator = _Mutator(dataset, schema, rng, lookups)
    key_cols = schema.business_key
    used: set[int] = set()
    issue_rows: list[dict] = []
    achieved: dict[str, int] = {}

    def key_of(rec: dict) -> str:
        return ";".join(f"{c}={rec[c]}" for c in key_cols)

    for code in sorted(c for c in counts if c != "DUP"):
        wanted, done = counts[code], 0
        names = list(mutations.get(code, []))
        for pos in rng.permutation(n).tolist():
            if done >= wanted:
                break
            if pos in used:
                continue
            rec = records[pos]
            original_key = key_of(rec)
            for name in rng.permutation(names).tolist():
                trial = dict(rec)
                if mutator.apply(code, name, trial):
                    records[pos] = trial
                    used.add(pos)
                    issue_rows.append({"dataset": dataset, "issue_code": code, "mutation": name,
                                       "business_key": original_key})
                    done += 1
                    break
        achieved[code] = done

    # Duplicates: copies of untouched rows, half exact and half formatting-only variants
    order_keys = np.arange(n, dtype=float).tolist()
    n_dup = min(counts.get("DUP", 0), n - len(used))
    clean_pool = np.array([i for i in range(n) if i not in used], dtype=int)
    copies: list[dict] = []
    if n_dup:
        exact_share = dirty_cfg.get("duplicate_split", {}).get("exact", 0.5)
        for j, pos in enumerate(np.sort(rng.choice(clean_pool, size=n_dup, replace=False)).tolist()):
            copy = dict(records[pos])
            changed = None if j < round(n_dup * exact_share) else mutator.conflict_variant(copy)
            copies.append(copy)
            order_keys.append(pos + float(rng.uniform(0.01, 50)))
            issue_rows.append({"dataset": dataset, "issue_code": "DUP",
                               "mutation": f"conflicting:{changed}" if changed else "exact",
                               "business_key": key_of(records[pos])})
    achieved["DUP"] = len(copies)

    all_records = records + copies
    order = np.argsort(np.array(order_keys), kind="stable")
    final = [all_records[i] for i in order]
    raw_lines = {i: rec.pop("__raw__") for i, rec in enumerate(final) if "__raw__" in rec}
    frame = pd.DataFrame(final, columns=list(rows.columns))
    return InjectionResult(frame, raw_lines, achieved, issue_rows,
                           clean_rows=n - len(used), original_rows=n, duplicate_rows=len(copies))
