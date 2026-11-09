"""Your own unit: the fields under the observation's ``self``, one reader each.

These save you from reaching through ``observation["observation"]["self"]`` every time.
``direction`` is the one that is easy to miss: it is the digit that heads toward the enemy side,
``2`` (east) for red and ``5`` (west) for blue, and it stays the same all match.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sandbox.observation_types import AxialPosition, SkirmishObservation

__all__ = [
    "direction",
    "hit_points",
    "movement_points",
    "position",
    "side",
    "unit_id",
    "unit_type",
]


def unit_id(observation: SkirmishObservation) -> str:
    """Return this unit's id, such as ``"red_archer_0"``, stable for the whole match."""
    return observation["observation"]["self"]["unit_id"]


def side(observation: SkirmishObservation) -> str:
    """Return the side this unit fights for, ``"red"`` or ``"blue"``."""
    return unit_id(observation).split("_", 1)[0]


def unit_type(observation: SkirmishObservation) -> str:
    """Return this unit's type: ``"footman"``, ``"archer"``, or ``"cavalry"``."""
    return observation["observation"]["self"]["type"]


def position(observation: SkirmishObservation) -> AxialPosition:
    """Return the position this unit is standing on."""
    return observation["observation"]["self"]["position"]


def direction(observation: SkirmishObservation) -> int:
    """Return the direction digit that heads toward the enemy side.

    A one-step path id is its direction digit, so ``action.move(me.direction(observation))``
    walks a single step that way, as long as the mask allows it.
    """
    return observation["observation"]["self"]["direction"]


def hit_points(observation: SkirmishObservation) -> int:
    """Return the hit points this unit has left."""
    return observation["observation"]["self"]["hit_points"]


def movement_points(observation: SkirmishObservation) -> int:
    """Return the movement points this unit has for this activation, always its full stat."""
    return observation["observation"]["self"]["movement_points"]
