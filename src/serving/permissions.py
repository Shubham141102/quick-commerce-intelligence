"""Roles and workspaces (Project_Plan_v2.md §10.1). Checked inside every data function, not only in the menu.

Demo-grade access control: it separates personas for the demonstration; it is not production security.
"""

from __future__ import annotations

# Three personas (2026-10-14): Business & Revenue and Customer Growth & Marketing were merged into one Business &
# Growth workspace — they shared revenue, promotions and customers, and one owner matches how quick-commerce
# growth teams work. See docs/16_persona_consolidation.md.
WORKSPACES = {
    "inventory": "Inventory & Supply Chain",
    "growth": "Business & Growth",
    "engineering": "Data Engineer",
}
ROLES = {
    "inventory_manager": {"inventory"},
    "growth_manager": {"growth"},
    "data_engineer": {"engineering"},
    "admin": set(WORKSPACES),
}
ROLE_LABELS = {
    "inventory_manager": "Inventory & Supply Chain Manager", "growth_manager": "Business & Growth Manager",
    "data_engineer": "Data Engineer", "admin": "Demo administrator",
}


class AccessDenied(PermissionError):
    pass


def allowed_workspaces(role: str) -> list[str]:
    return [w for w in WORKSPACES if w in ROLES.get(role, set())]


def require(role: str | None, workspace: str) -> None:
    if not role or workspace not in ROLES.get(role, set()):
        raise AccessDenied(f"role {role!r} may not open the {WORKSPACES.get(workspace, workspace)} workspace")
