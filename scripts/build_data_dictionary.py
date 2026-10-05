"""Kept for compatibility: regenerates all generated docs (same as `python -m scripts.build_docs`)."""

from scripts.build_docs import DOCS, main

OUTPUT = DOCS / "data_dictionary.md"

if __name__ == "__main__":
    main()
