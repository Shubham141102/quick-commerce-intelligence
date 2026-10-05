"""Split each source dataset into landing files by ingestion pattern (Project_Plan_v2.md §3.1, §3.9).

- historical: one file per month (dimension tables: one file at the start)
- batch:      one file per business day (IST)
- stream:     micro-batch files for STREAM_DATASETS; other datasets stay daily

Late arrivals and out-of-order rows are applied here. They are valid data, so
they don't count against the dirty-data budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io import write_csv
from src.common.schemas import DIMENSION_DATASETS, STREAM_DATASETS, TableSchema
from src.generation.context import GenContext

SLICE_TS = "_slice_ts"   # IST-naive datetime64[s] of the row's business event
RAW = "_raw"             # pre-built malformed line, if any


@dataclass
class LandingStats:
    files: list[dict]
    late_rows: int
    out_of_order_rows: int


def write_landing(ctx: GenContext, dataset: str, frame: pd.DataFrame, schema: TableSchema, root: Path) -> LandingStats:
    cfg = ctx.cfg
    rng = ctx.rng("landing", dataset)
    slices = cfg.ingestion_slices
    late_cfg = cfg.dirty["outside_budget"]["late_arrival"]
    ooo_share = cfg.dirty["outside_budget"]["out_of_order_arrival"]["stream_share"]
    streamed = dataset in STREAM_DATASETS
    mb = np.timedelta64(slices.stream.micro_batch_minutes or 60, "m")

    start = np.datetime64(cfg.calendar.start_date, "D")
    end = np.datetime64(cfg.calendar.end_date, "D")
    hist_end = np.datetime64(slices.historical.end, "D")
    stream_start = np.datetime64(slices.stream.start, "D")
    last_ts = end.astype("datetime64[s]") + np.timedelta64(86399, "s")

    ts = np.clip(frame[SLICE_TS].to_numpy().astype("datetime64[s]"), start.astype("datetime64[s]"), last_ts)
    n = len(frame)
    late = np.zeros(n, dtype=bool)

    if dataset not in DIMENSION_DATASETS:
        day = ts.astype("datetime64[D]")
        in_batch = (day > hist_end) & ~((day >= stream_start) & streamed)
        in_stream = (day >= stream_start) & streamed
        # Batch: a share of rows arrives in the next day's file
        move = in_batch & (rng.random(n) < late_cfg["batch_next_day_share"]) & (day < end)
        ts = np.where(move, (day + 1).astype("datetime64[s]"), ts)
        late |= move
        # Stream: a share of rows is delayed by 1..max hours into a later micro-batch
        delay = rng.integers(1, late_cfg["stream_max_delay_hours"] + 1, size=n).astype("timedelta64[h]")
        move = in_stream & (rng.random(n) < late_cfg["stream_share"])
        ts = np.where(move, np.minimum(ts + delay.astype("timedelta64[s]"), last_ts), ts)
        late |= move

    keys = np.empty(n, dtype=object)
    slice_name = np.empty(n, dtype=object)
    day = ts.astype("datetime64[D]")
    for i in range(n):
        d = day[i]
        if dataset in DIMENSION_DATASETS or d <= hist_end:
            slice_name[i] = "historical"
            keys[i] = str(start.astype("datetime64[M]")) if dataset in DIMENSION_DATASETS else str(d.astype("datetime64[M]"))
        elif streamed and d >= stream_start:
            slice_name[i] = "stream"
            minutes = (ts[i] - d.astype("datetime64[s]")) // np.timedelta64(60, "s")
            floor = d.astype("datetime64[m]") + (int(minutes) // int(mb / np.timedelta64(1, "m"))) * mb
            keys[i] = str(floor).replace(":", "-")
        else:
            slice_name[i] = "batch"
            keys[i] = str(d)

    out = frame.copy()
    out["_late"] = late
    out["_slice"] = slice_name
    out["_key"] = keys
    out["_order"] = frame[SLICE_TS].to_numpy()

    files, ooo = [], 0
    for (sl, key), part in out.groupby(["_slice", "_key"], sort=True):
        part = part.sort_values("_order", kind="stable").reset_index(drop=True)
        if sl == "stream" and len(part) > 2:
            k = int(round(len(part) * ooo_share))
            if k >= 2:
                pos = rng.choice(len(part), size=k, replace=False)
                idx = np.arange(len(part))
                idx[pos] = rng.permutation(idx[pos])  # shuffle a subset in place
                part = part.iloc[idx].reset_index(drop=True)
                ooo += k
        raw = {i: v for i, v in enumerate(part[RAW].tolist()) if isinstance(v, str)}
        path = root / sl / dataset / f"{dataset}__{key}.csv"
        write_csv(part, path, schema.names, raw)
        files.append({"dataset": dataset, "slice": sl, "file": path.relative_to(root.parent).as_posix(),
                      "rows": len(part)})
    return LandingStats(files, int(late.sum()), ooo)
