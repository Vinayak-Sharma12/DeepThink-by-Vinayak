"""Smoke tests: every Project LOGOS package must import cleanly."""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

import pytest

# Top-level packages from phases/01_phase1_project_setup.md (excluding tests/).
PACKAGE_NAMES: tuple[str, ...] = (
    "app",
    "configs",
    "data",
    "datasets",
    "docs",
    "evaluation",
    "experiments",
    "inference",
    "logs",
    "model",
    "models",
    "scripts",
    "tokenizer",
    "training",
)


@pytest.mark.parametrize("package_name", PACKAGE_NAMES)
def test_package_imports(package_name: str) -> None:
    """Each scaffold package loads without ImportError."""
    module = importlib.import_module(package_name)
    assert isinstance(module, ModuleType)
    assert module.__name__ == package_name


def test_all_packages_enumerated() -> None:
    """Smoke test list stays in sync with discoverable top-level packages."""
    discovered = {
        name
        for _finder, name, is_pkg in pkgutil.iter_modules()
        if is_pkg and name in PACKAGE_NAMES
    }
    assert discovered == set(PACKAGE_NAMES)
