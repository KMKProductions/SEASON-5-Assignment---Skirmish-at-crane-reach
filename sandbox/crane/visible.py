"""What your unit can see right now, split into the two sides.

Both readers keep the observation's own order. Units outside your vision are simply absent, and
nothing tells you how many are missing. Every enemy you can see is also one you may name this
turn, so ``visible.enemies`` and ``action.possible_targets`` describe the same units.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import me

if TYPE_CHECKING:
    from sandbox.observation_types import SkirmishObservation, VisibleUnit

__all__ = ["allies", "enemies"]


def enemies(observation: SkirmishObservation) -> list[VisibleUnit]:
    """Return the units of the other side your unit can see."""
    own_side = me.side(observation)
    return [unit for unit in observation["observation"]["visible_units"] if unit["side"] != own_side]


def allies(observation: SkirmishObservation) -> list[VisibleUnit]:
    """Return the units of your own side your unit can see. Your own unit is never among them."""
    own_side = me.side(observation)
    return [unit for unit in observation["observation"]["visible_units"] if unit["side"] == own_side]
