"""Create `.streamlit/secrets.toml` with demo users (bcrypt-hashed passwords). The file is git-ignored.

    python -m scripts.create_demo_secrets                 # default demo passwords (printed)
    python -m scripts.create_demo_secrets --password X    # same password for every demo user

For Streamlit Community Cloud, paste the generated file's contents into the app's Secrets settings.
Demo-grade login: it separates personas for the demonstration; it is not production security.
"""

from __future__ import annotations

import argparse

import bcrypt

from src.common.paths import PROJECT_ROOT

DEMO_USERS = {  # username: (display name, role, default demo password)
    "inventory": ("Inventory Manager (demo)", "inventory_manager", "inventory-demo"),
    "marketing": ("Marketing Manager (demo)", "marketing_manager", "marketing-demo"),
    "business": ("Business Analyst (demo)", "business_analyst", "business-demo"),
    "engineer": ("Data Engineer (demo)", "data_engineer", "engineer-demo"),
    "admin": ("Demo Administrator", "admin", "admin-demo"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write .streamlit/secrets.toml with demo users.")
    parser.add_argument("--password", default=None, help="use this password for every demo user")
    args = parser.parse_args()
    lines = ["# Demo users for the Quick-Commerce Intelligence app (generated; do not commit)", ""]
    print("Demo logins:")
    for username, (name, role, default) in DEMO_USERS.items():
        password = args.password or default
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        lines += [f"[auth.users.{username}]", f'name = "{name}"', f'role = "{role}"', f'password_hash = "{hashed}"', ""]
        print(f"  {username:10s} / {password:16s} -> {role}")
    path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    path.parent.mkdir(exist_ok=True)
    kept = _other_sections(path.read_text(encoding="utf-8")) if path.exists() else []
    if kept:   # e.g. the optional [anthropic] key (Phase 6G) survives a password change
        lines += kept
        print("kept existing non-login sections: " + ", ".join(ln for ln in kept if ln.startswith("[")))
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")


def _other_sections(text: str) -> list[str]:
    """Every section of an existing secrets file except the demo logins ([auth...]), kept verbatim."""
    out, keep = [], False
    for line in text.splitlines():
        if line.strip().startswith("["):
            keep = not line.strip().startswith("[auth")
        if keep:
            out.append(line)
    return out + ([""] if out else [])


if __name__ == "__main__":
    main()
