"""Resolve shared template modules in either supported package layout.

Environment code runs from ``local_play`` in this repository and from ``sandbox`` after template
composition. This module keeps that layout choice in one place while preserving the imported module
objects for callers that re-export their helpers.
"""

from __future__ import annotations

import importlib
from types import ModuleType

_PACKAGES = ("local_play", "sandbox")


def resolve(*names: str) -> tuple[ModuleType, ...]:
    """Return the requested modules from the first available shared package.

    A missing candidate package or one of its requested submodules selects the next layout. Missing
    dependencies from elsewhere are real import failures and must reach the caller unchanged.
    """
    for package in _PACKAGES:
        candidates = tuple(f"{package}.{name}" for name in names)
        try:
            return tuple(importlib.import_module(candidate) for candidate in candidates)
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if missing == package or any(
                candidate == missing or candidate.startswith(f"{missing}.") for candidate in candidates
            ):
                continue
            raise

    requested = ", ".join(names)
    packages = ", ".join(_PACKAGES)
    raise ModuleNotFoundError(f"could not resolve shared module(s) {requested} from: {packages}")
