"""Regenerate every generated document from the code and configs.

    python -m scripts.build_docs

- docs/data_dictionary.md        source datasets + Bronze columns   (src/common/data_dictionary.py)
- docs/transformation_catalog.md Bronze -> Silver rules per dataset (src/transformations/silver/catalog.py)
- docs/gold_catalog.md           Gold tables: grain, columns, lineage    (src/transformations/gold/catalog.py)
- docs/metric_definitions.md     shared business metric definitions      (src/serving/metrics.py)

Tests fail when a generated document is out of date.
"""

from __future__ import annotations

from src.common.config import load_config
from src.common.data_dictionary import render_markdown
from src.common.paths import PROJECT_ROOT

DOCS = PROJECT_ROOT / "docs"


def generated_docs() -> dict[str, str]:
    from src.common.file_registry import render_file_registry
    from src.transformations.gold.catalog import render_gold_catalog, render_metric_definitions
    from src.transformations.silver.catalog import render_catalog

    cfg = load_config("medium")
    return {"data_dictionary.md": render_markdown(cfg) + "\n", "transformation_catalog.md": render_catalog(cfg) + "\n",
            "gold_catalog.md": render_gold_catalog() + "\n", "metric_definitions.md": render_metric_definitions() + "\n",
            "file_registry.md": render_file_registry() + "\n"}


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    for name, text in generated_docs().items():
        (DOCS / name).write_text(text, encoding="utf-8")
        print(f"wrote docs/{name}")


if __name__ == "__main__":
    main()
