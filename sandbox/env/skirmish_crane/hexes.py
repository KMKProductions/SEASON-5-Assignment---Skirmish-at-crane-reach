"""Hexagonal field geometry shared by every tactical rules module."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .tile_types import FEATURES, TERRAINS

Position = tuple[int, int]

# Clockwise from northeast.  These digits are part of the student contract.
DIRECTIONS: dict[int, Position] = {
    1: (1, -1),
    2: (1, 0),
    3: (0, 1),
    4: (-1, 1),
    5: (-1, 0),
    6: (0, -1),
}


@dataclass(frozen=True)
class Tile:
    """A field cell. Terrain and feature properties both come from the tile-type source."""

    terrain: str = "grass"
    feature: str = "none"

    @property
    def passable(self) -> bool:
        return TERRAINS[self.terrain].passable

    @property
    def move_cost(self) -> int:
        if not self.passable:
            raise ValueError("an impassable tile has no movement cost")
        return TERRAINS[self.terrain].move_cost + FEATURES[self.feature].move_cost_delta

    @property
    def entry_damage(self) -> int:
        return FEATURES[self.feature].entry_damage


VOID = Tile("void")


def on_field(position: Position, extent: int) -> bool:
    q, r = position
    return 0 <= q <= 2 * extent and 0 <= r <= 2 * extent and extent <= q + r <= 3 * extent


def field_positions(extent: int) -> Iterator[Position]:
    for q in range(2 * extent + 1):
        for r in range(2 * extent + 1):
            if on_field((q, r), extent):
                yield q, r


def neighbor(position: Position, direction: int) -> Position:
    if type(direction) is not int or direction not in DIRECTIONS:
        raise ValueError(f"direction must be 1 through 6, got {direction!r}")
    dq, dr = DIRECTIONS[direction]
    return position[0] + dq, position[1] + dr


def neighbors(position: Position, extent: int) -> tuple[Position, ...]:
    candidates = (neighbor(position, digit) for digit in DIRECTIONS)
    return tuple(candidate for candidate in candidates if on_field(candidate, extent))


def distance(first: Position, second: Position) -> int:
    dq = first[0] - second[0]
    dr = first[1] - second[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def opposite(direction: int) -> int:
    if type(direction) is not int or direction not in DIRECTIONS:
        raise ValueError(f"direction must be 1 through 6, got {direction!r}")
    return ((direction + 2) % 6) + 1


def rotate_position(position: Position, extent: int) -> Position:
    return 2 * extent - position[0], 2 * extent - position[1]


def rotate_path(path: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    return tuple(opposite(direction) for direction in path)


def retrace_path(path: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    return tuple(opposite(direction) for direction in reversed(path))


def path_positions(start: Position, path: tuple[int, ...] | list[int]) -> tuple[Position, ...]:
    """Return every tile a walked path enters, in order, including the final tile."""
    positions: list[Position] = []
    position = start
    for direction in path:
        position = neighbor(position, direction)
        positions.append(position)
    return tuple(positions)


def tile_array(extent: int, tiles: dict[Position, Tile]) -> tuple[tuple[Tile, ...], ...]:
    """Freeze a position-keyed field into the square grid, indexed row then column as ``grid[r][q]``.

    This ordering is the shape participants receive, so it is part of the student contract.
    """
    return tuple(
        tuple(tiles.get((q, r), VOID) if on_field((q, r), extent) else VOID for q in range(2 * extent + 1))
        for r in range(2 * extent + 1)
    )
