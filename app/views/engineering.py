"""Data Engineer workspace — ON HOLD (user decision 2026-10-05). The engineer account lands here so it has a
page; the four use cases (pipeline runs, data quality & quarantine, lineage & tables, model monitoring) come later."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components.personas import PERSONAS  # noqa: E402
from app.components.ui import page_header, require_workspace  # noqa: E402

require_workspace("engineering")
page_header("engineering")
st.info("This workspace is **on hold** and will be built in a later phase. The pipeline already records "
        "everything it will show.", icon="🚧")
st.markdown("**Planned use cases**")
for uc in PERSONAS["engineering"]["use_cases"]:
    st.markdown(f"- {uc}")
st.caption("Meanwhile the generated references in docs/ cover the same ground: data_dictionary.md, "
           "transformation_catalog.md, gold_catalog.md, file_registry.md.")
