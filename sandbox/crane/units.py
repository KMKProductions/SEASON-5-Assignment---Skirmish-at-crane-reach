"""Fixed per-type numbers, as the ruleset sets them.

The observation tells you a visible unit's type, but not its movement points or attack range.
This table fills that gap: standing knowledge for both sides, the same for a unit of a given
type whether it is yours or the enemy's. These are the values a unit starts with; a live unit's
``hit_points`` in the observation is what it has left, not this table's starting number.

``STATS`` is the same table of numbers the rules engine plays by, placed here when the template
is composed, so the values cannot drift from what a match enforces. The composed copy defines
its own ``UnitStats`` class, so compare the numbers themselves; do not rely on object identity
or an ``isinstance`` check against the engine's class.
"""

from __future__ import annotations

from sandbox.unit_stats import UNIT_STATS as STATS
from sandbox.unit_stats import UnitStats

__all__ = ["STATS", "UnitStats"]
