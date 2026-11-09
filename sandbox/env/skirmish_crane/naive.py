"""A small seeded baseline agent for Skirmish at Crane Reach."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .observation_types import SkirmishAction, SkirmishObservation


def _decode_path(path_id: int) -> tuple[int, ...]:
    """Decode the stable path id without depending on the environment package."""
    if not isinstance(path_id, int) or not 0 <= path_id <= 1554:
        raise ValueError("path id must be an integer from 0 through 1554")
    if path_id == 0:
        return ()
    remaining = path_id - 1
    length = 1
    while remaining >= 6**length:
        remaining -= 6**length
        length += 1
    digits = []
    for power in range(length - 1, -1, -1):
        digit, remaining = divmod(remaining, 6**power)
        digits.append(digit + 1)
    return tuple(digits)


def _distance(first: tuple[int, int], second: tuple[int, int]) -> int:
    dq, dr = first[0] - second[0], first[1] - second[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def _end(position: tuple[int, int], path_id: int) -> tuple[int, int]:
    q, r = position
    directions = ((1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1))
    for direction in _decode_path(path_id):
        dq, dr = directions[direction - 1]
        q, r = q + dq, r + dr
    return q, r


class Agent:
    """Walk toward visible enemies, otherwise toward the point-reflected starting tile."""

    def reset(self, seed, observation) -> None:
        self._rng = random.Random(seed)
        self._start: tuple[int, int] | None = None

    def act(self, observation: SkirmishObservation) -> SkirmishAction:
        state = observation["observation"]
        mask = observation["action_mask"]
        position = (state["self"]["position"]["q"], state["self"]["position"]["r"])
        if self._start is None:
            self._start = position
        legal_paths = [index for index, allowed in enumerate(mask["path"]) if allowed]
        own_side = state["self"]["unit_id"].split("_", 1)[0]
        enemies = [unit for unit in state["visible_units"] if unit["side"] != own_side]
        if enemies:
            targets = [(unit["position"]["q"], unit["position"]["r"]) for unit in enemies]
            current = min(_distance(position, target) for target in targets)
            distances = {
                path: min(_distance(_end(position, path), target) for target in targets)
                for path in legal_paths
            }
            best = min(distances.values())
            if best < current:
                return {
                    "path": self._rng.choice([path for path in legal_paths if distances[path] == best]),
                    "target": 0,
                }
            return {"path": 0, "target": 0}
        side = state["battlefield"]["side"]
        goal = (side - 1 - self._start[0], side - 1 - self._start[1])
        one_steps = [path for path in legal_paths if len(_decode_path(path)) == 1]
        improving = [
            path for path in one_steps if _distance(_end(position, path), goal) < _distance(position, goal)
        ]
        choices = improving or one_steps
        return {"path": self._rng.choice(choices) if choices else 0, "target": 0}
