"""Every project file must be in the file registry, and the registry must not point to deleted files."""

from src.common.file_registry import REGISTRY, tracked_files
from src.common.paths import PROJECT_ROOT


def test_every_project_file_is_registered():
    registered = {e.path for e in REGISTRY}
    missing = sorted(tracked_files(PROJECT_ROOT) - registered)
    assert not missing, f"add these to src/common/file_registry.py: {missing}"


def test_no_registry_entry_points_to_a_missing_file():
    stale = sorted(e.path for e in REGISTRY if not (PROJECT_ROOT / e.path).exists())
    assert not stale, f"remove or fix these registry entries: {stale}"


def test_each_entry_explains_itself():
    for e in REGISTRY:
        assert e.purpose and e.phase, e.path
    assert len({e.path for e in REGISTRY}) == len(REGISTRY), "duplicate registry entries"
