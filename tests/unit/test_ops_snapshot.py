"""Ops snapshot helpers (Phase 7): error text is scrubbed of local paths before it is published."""

from src.common.paths import PROJECT_ROOT
from src.orchestration.ops_snapshot import layer_of, scrub


def test_scrub_removes_project_and_home_paths():
    root = str(PROJECT_ROOT)
    doubled = root.replace("\\", "\\\\")                        # how Python reprs a Windows path in an error
    text = (f'File "{root}\\src\\x.py", line 3 - PermissionError: Access is denied: '
            f"'{doubled}\\\\data\\\\gold' - C:\\Users\\someone\\AppData\\tmp - /home/ana/x")
    out = scrub(text)
    assert "<project>" in out and "<home>" in out
    assert "Users\\someone" not in out and "/home/ana" not in out and root not in out and doubled not in out


def test_scrub_keeps_ordinary_text():
    assert scrub("ValueError: unknown stages ['x']") == "ValueError: unknown stages ['x']"
    assert scrub("") == "" and scrub(None) == ""


def test_layers():
    ml = {"gld_demand_predictions"}
    assert layer_of("landing/orders", ml) == "source" and layer_of("brz_orders", ml) == "bronze"
    assert layer_of("gld_demand_predictions", ml) == "ml" and layer_of("gld_daily_sales", ml) == "gold"
    assert layer_of("demo/gld_daily_sales", ml) == "app" and layer_of("policies/refund.md", ml) == "knowledge"
