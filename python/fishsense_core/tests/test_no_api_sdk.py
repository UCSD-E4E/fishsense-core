"""Nothing in fishsense_core may import fishsense_api_sdk.

fishsense-services (v2) retires the v1 API SDK at cutover, so a core that
imports it cannot be installed there. Two halves: a static scan that sees every
import, including lazy and ``TYPE_CHECKING`` ones; and a runtime import of every
module with the SDK blocked, which catches anything the scan cannot see.
"""

import ast
import subprocess
import sys
from pathlib import Path

import fishsense_core

_PACKAGE_DIR = Path(fishsense_core.__file__).parent


def _sdk_imports(path: Path) -> list[str]:
    """Every import of ``fishsense_api_sdk`` in ``path``, wherever it sits —
    module level, inside a function, or under ``if TYPE_CHECKING``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module or ""]
        else:
            continue
        for name in names:
            if name == "fishsense_api_sdk" or name.startswith("fishsense_api_sdk."):
                found.append(f"{path.relative_to(_PACKAGE_DIR.parent)}:{node.lineno}")
    return found


def test_no_module_imports_the_api_sdk():
    sources = sorted(_PACKAGE_DIR.rglob("*.py"))
    assert sources, "found no sources to check"
    offenders = [hit for source in sources for hit in _sdk_imports(source)]
    assert not offenders, f"fishsense_api_sdk imported at: {offenders}"


def _module_names() -> list[str]:
    """Dotted names of every source module. Derived from the files, not from
    ``pkgutil.walk_packages``: ``image/`` has no ``__init__.py``, and pkgutil
    silently skips namespace packages — which would skip ``rectified_image``."""
    names = []
    for source in sorted(_PACKAGE_DIR.rglob("*.py")):
        parts = source.relative_to(_PACKAGE_DIR.parent).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


def test_every_module_imports_with_the_api_sdk_blocked():
    """The runtime half of the guard: with ``fishsense_api_sdk`` made
    unimportable, every module in the package still imports. Run in a fresh
    interpreter so this process's already-imported modules can't mask it."""
    names = _module_names()
    assert "fishsense_core.image.rectified_image" in names
    script = """
import importlib, sys
sys.modules["fishsense_api_sdk"] = None  # any import of it now raises
failed = []
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except ImportError as exc:
        failed.append(f"{name}: {exc}")
assert not failed, failed
"""
    result = subprocess.run(
        [sys.executable, "-c", script, *names],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
