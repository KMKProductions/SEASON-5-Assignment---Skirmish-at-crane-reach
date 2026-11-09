"""Hex geometry and the ground itself: where tiles are, and what is on them.

Every position here is an axial dictionary shaped like ``{"q": 8, "r": 5}``, the shape the
observation uses everywhere. This is all plain coordinate math and plain battlefield reading: it
says nothing about whether a tile is walkable from where you stand. The action mask stays the
sole legality authority, through ``action.legal_paths``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import paths

if TYPE_CHECKING:
    from sandbox.observation_types import AxialPosition, SkirmishObservation, Tile

__all__ = [
    "DIRECTIONS",
    "at_center",
    "at_mirror",
    "at_path_end",
    "distance",
    "neighbors",
    "terrain_at",
]

# Clockwise from northeast. These digits are part of the student contract.
DIRECTIONS: dict[int, tuple[int, int]] = {
    1: (1, -1),
    2: (1, 0),
    3: (0, 1),
    4: (-1, 1),
    5: (-1, 0),
    6: (0, -1),
}

_VOID: Tile = {"terrain": "void", "feature": "none"}


def distance(first: AxialPosition, second: AxialPosition) -> int:
    """Return the hex distance between two positions, counted in steps."""
    dq = first["q"] - second["q"]
    dr = first["r"] - second["r"]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def neighbors(position: AxialPosition) -> dict[int, AxialPosition]:
    """Return the six positions around ``position``, keyed by the direction digit reaching each."""
    return {
        digit: {"q": position["q"] + dq, "r": position["r"] + dr} for digit, (dq, dr) in DIRECTIONS.items()
    }


def at_path_end(position: AxialPosition, path_id: int) -> AxialPosition:
    """Return the position a path ends on, walked from ``position`` one direction digit at a time.

    This answers "where would this path put me" without walking the digits yourself. Raises
    ``ValueError`` for a path id outside the encoding.
    """
    q, r = position["q"], position["r"]
    for digit in paths.decode(path_id):
        dq, dr = DIRECTIONS[digit]
        q, r = q + dq, r + dr
    return {"q": q, "r": r}


def at_center(observation: SkirmishObservation) -> AxialPosition:
    """Return the middle position of the battlefield, the landmark both sides share.

    The field is point-symmetric about this position, so it sits equally far from both sides'
    starting ground.
    """
    middle = (observation["observation"]["battlefield"]["side"] - 1) // 2
    return {"q": middle, "r": middle}


def at_mirror(position: AxialPosition, observation: SkirmishObservation) -> AxialPosition:
    """Return the position opposite ``position``, reflected through the middle of the field.

    Because the field is point-symmetric, the mirror of one of your own starting tiles always
    lies in enemy ground.
    """
    side = observation["observation"]["battlefield"]["side"]
    return {"q": side - 1 - position["q"], "r": side - 1 - position["r"]}


def terrain_at(observation: SkirmishObservation, position: AxialPosition) -> Tile:
    """Return the ``{"terrain", "feature"}`` pair standing on one position.

    The battlefield grid is stored row first, so this indexes it for you. A position off the
    field reads as void, the terrain that fills the square grid's corners outside the hexagon.
    """
    grid = observation["observation"]["battlefield"]["tiles"]
    q, r = position["q"], position["r"]
    if not (0 <= r < len(grid) and 0 <= q < len(grid[r])):
        return _VOID
    return grid[r][q]
