"""The ruleset's fixed per-type numbers: footman, archer, and cavalry.

One table, read by both sides of the codebase. ``engine.py`` uses it to run real matches, and the
student template's ``sandbox.crane.units`` helper reads the very same table, so a unit's stats can
never drift between the rules the engine enforces and the numbers a student agent sees. Stdlib-only
at runtime, so importing it never drags anything heavier along.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnitStats:
    hit_points: int
    movement_points: int
    attack_range: int
    damage: int
    vision: int


UNIT_STATS = {
    "footman": UnitStats(12, 2, 1, 3, 4),
    "archer": UnitStats(6, 2, 6, 2, 6),
    "cavalry": UnitStats(10, 4, 1, 3, 6),
}
