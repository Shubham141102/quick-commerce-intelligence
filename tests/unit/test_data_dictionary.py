import pytest

from scripts.build_docs import DOCS, generated_docs
from src.common.data_dictionary import BRONZE_COLUMNS_DOC, COLUMNS, DATASETS
from src.common.schemas import BRONZE_METADATA_COLUMNS, CORRUPT_RECORD, SOURCE_SCHEMAS


def test_every_dataset_and_column_is_documented():
    assert set(DATASETS) == set(SOURCE_SCHEMAS) == set(COLUMNS)
    for name, schema in SOURCE_SCHEMAS.items():
        assert set(COLUMNS[name]) == set(schema.names), name
    assert set(BRONZE_COLUMNS_DOC) == {CORRUPT_RECORD, *BRONZE_METADATA_COLUMNS}


def test_foreign_keys_point_at_real_columns():
    for name, cols in COLUMNS.items():
        for col, doc in cols.items():
            if doc.fk:
                table, column = doc.fk.split(".")
                assert column in SOURCE_SCHEMAS[table].names, (name, col, doc.fk)


def test_every_silver_rule_and_derived_column_is_described():
    pytest.importorskip("pyspark")
    from src.transformations.silver.catalog import DERIVED_DOCS
    from src.transformations.silver.specs import SPECS

    assert set(SPECS) == set(SOURCE_SCHEMAS)
    for ds, spec in SPECS.items():
        for rule in spec.rules:
            assert rule.description, (ds, rule.name)
        for col, _ in spec.derived:
            assert col == "business_date" or col in DERIVED_DOCS, (ds, col)


@pytest.mark.parametrize("name", ["data_dictionary.md", "transformation_catalog.md", "gold_catalog.md",
                                  "metric_definitions.md"])
def test_generated_docs_are_up_to_date(name):
    pytest.importorskip("pyspark")
    expected = generated_docs()[name]
    assert (DOCS / name).read_text(encoding="utf-8") == expected, "run: python -m scripts.build_docs"
