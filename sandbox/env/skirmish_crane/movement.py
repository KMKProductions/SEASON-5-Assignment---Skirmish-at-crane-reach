"""Movement validation and walking for one tactical activation."""

from __future__ import annotations

from .battlefield import Battlefield
from .hexes import Position, neighbor
from .paths import MAX_PATH_STEPS


def walk(
    battlefield: Battlefield,
    start: Position,
    movement_points: int,
    path: tuple[int, ...] | list[int],
    occupied: set[Position],
) -> Position:
    """Validate and complete a path, returning its final position.

    The caller must remove the acting unit's current position from ``occupied``.
    """
    try:
        directions = tuple(path)
    except TypeError as error:
        raise ValueError("a path must be an iterable of directions") from error
    if len(directions) > MAX_PATH_STEPS:
        raise ValueError("a path may contain at most four steps")
    position = start
    remaining = movement_points
    for index, direction in enumerate(directions):
        candidate = neighbor(position, direction)
        tile = battlefield.tile_at(candidate)
        if not tile.passable:
            raise ValueError("a path cannot enter an impassable tile")
        if candidate in occupied:
            raise ValueError("a path cannot enter an occupied tile")
        cost = tile.move_cost
        # At full movement points a unit may always take one step, whatever it costs.
        first_step = index == 0
        if not first_step and remaining < cost:
            raise ValueError("a path spends more movement than is available")
        remaining -= cost
        if remaining < 0 and index != len(directions) - 1:
            raise ValueError("a negative movement balance must end the path")
        position = candidate
    return position


def legal_paths(
    battlefield: Battlefield,
    start: Position,
    movement_points: int,
    occupied: set[Position],
) -> tuple[tuple[int, ...], ...]:
    """Enumerate legal paths without recreating their legality in consumers."""
    paths: list[tuple[int, ...]] = [()]
    frontier = [()]
    for _ in range(MAX_PATH_STEPS):
        next_frontier: list[tuple[int, ...]] = []
        for prefix in frontier:
            for direction in range(1, 7):
                candidate = (*prefix, direction)
                try:
                    walk(battlefield, start, movement_points, candidate, occupied)
                except ValueError:
                    continue
                paths.append(candidate)
                next_frontier.append(candidate)
        frontier = next_frontier
    return tuple(paths)
