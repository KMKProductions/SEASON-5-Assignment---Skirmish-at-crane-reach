"""Shared isolation for tests that run inside a student's template repository."""

from __future__ import annotations

import pytest
from sandbox import evaluate, play


@pytest.fixture(autouse=True)
def disable_local_season_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep command tests from loading a contributor's local ``season.json`` file."""
    monkeypatch.setattr(play, "load_season_settings", lambda root, meta: None)
    monkeypatch.setattr(evaluate, "load_season_settings", lambda root, meta: None)
