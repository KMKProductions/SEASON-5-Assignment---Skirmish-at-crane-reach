"""Both sides' starting rosters: who is on the field, whether or not you can see them.

A roster entry carries a unit's owning ``player``, its ``unit_id``, its ``side``, and its
``type``. Rosters are standing knowledge, identical for every unit and constant for the whole
match, so they are how you address an ally you cannot see and how you plan around an enemy who
has not shown up yet. Dead units stay listed: the roster records who started, not who is left.

The enemy roster's order is also the order target slots count in, which is why
``action.possible_targets`` returns its ids in this order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import me

if TYPE_CHECKING:
    from sandbox.observation_types import RosterEntry, SkirmishObservation

__all__ = ["allies", "enemies"]


def allies(observation: SkirmishObservation) -> tuple[RosterEntry, ...]:
    """Return your own side's starting roster, your unit included."""
    rosters = observation["observation"]["rosters"]
    return rosters["red"] if me.side(observation) == "red" else rosters["blue"]


def enemies(observation: SkirmishObservation) -> tuple[RosterEntry, ...]:
    """Return the other side's starting roster, in the order target slots count."""
    rosters = observation["observation"]["rosters"]
    return rosters["blue"] if me.side(observation) == "red" else rosters["red"]
