"""Landing-file discovery and load tracking (Project_Plan_v2.md §3.9).

A file's identity is (dataset, file name, SHA-256 of its contents). A file whose
identity already appears in `meta_file_loads` with status `loaded` is skipped, so
re-running ingestion never duplicates Bronze rows.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.common.schemas import SOURCE_SCHEMAS
from src.orchestration.tracking import read_meta

SLICES = ("historical", "batch", "stream")


@dataclass(frozen=True)
class LandingFile:
    dataset: str
    slice: str
    path: Path
    rel: str          # "<generation_run_id>/landing/<slice>/<dataset>/<file>.csv"
    size: int
    sha256: str

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.dataset, self.path.name, self.sha256


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(generation_dir: Path) -> list[LandingFile]:
    landing = generation_dir / "landing"
    files = []
    for sl in SLICES:
        for dataset in SOURCE_SCHEMAS:
            for path in sorted((landing / sl / dataset).glob("*.csv")):
                files.append(LandingFile(dataset, sl, path, path.relative_to(generation_dir.parent).as_posix(),
                                         path.stat().st_size, sha256_file(path)))
    return files


def loaded_identities(metadata_root: Path) -> tuple[set[tuple[str, str, str]], set[str]]:
    """Identities of files already loaded, and the generation runs they came from."""
    loads = read_meta(metadata_root, "meta_file_loads")
    loads = loads[loads["status"] == "loaded"]
    files = set(zip(loads["dataset"], loads["file"].map(lambda f: Path(f).name), loads["sha256"]))
    return files, set(loads["generation_run_id"])
