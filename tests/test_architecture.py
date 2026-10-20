"""Enforce the one-way dependency rule between subpackages (see docs/interfaces.md)."""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "edtsim"

# subpackage -> subpackages it is allowed to import from
ALLOWED = {
    "core": set(),
    "inertia": {"core"},
    "environment": {"core"},
    "edt_current": {"core"},
    "tether": {"core", "inertia"},
    "viz": {"core"},
    "coupled": {"core", "inertia", "environment", "edt_current", "tether"},
}


def imported_subpackages(py_file: Path, own: str):
    found = set()
    for node in ast.walk(ast.parse(py_file.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and node.module.split(".")[0] == "edtsim":
                parts = node.module.split(".")
                if len(parts) > 1:
                    found.add(parts[1])
                else:
                    found.update(a.name for a in node.names)           # "from edtsim import core"
            elif node.level >= 2 and node.module:                      # "from ..core import x"
                found.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] == "edtsim" and len(parts) > 1:
                    found.add(parts[1])
    found.discard(own)
    return found


def existing_subpackages():
    return sorted(p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").exists())


@pytest.mark.parametrize("sub", existing_subpackages())
def test_subpackage_only_imports_allowed_siblings(sub):
    assert sub in ALLOWED, f"New subpackage '{sub}': add it to ALLOWED in tests/test_architecture.py"
    for py in (SRC / sub).rglob("*.py"):
        bad = imported_subpackages(py, sub) - ALLOWED[sub]
        assert not bad, f"{py.relative_to(SRC)} imports {sorted(bad)}, allowed: {sorted(ALLOWED[sub])}"
