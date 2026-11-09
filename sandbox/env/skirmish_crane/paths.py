"""The stable compact codec for tactical movement paths."""

from __future__ import annotations

from .hexes import DIRECTIONS

MAX_PATH_STEPS = 4
# One id per direction sequence of each length, so 6 + 36 + 216 + 1296 = 1554 alongside id 0.
MAX_PATH_ID = sum(len(DIRECTIONS) ** length for length in range(1, MAX_PATH_STEPS + 1))


def encode_path(path: tuple[int, ...] | list[int]) -> int:
    """Encode zero through four direction digits in length then lexical order."""
    try:
        directions = tuple(path)
    except TypeError as error:
        raise ValueError("a path must be an iterable of directions") from error
    if len(directions) > MAX_PATH_STEPS or any(
        type(direction) is not int or direction not in DIRECTIONS for direction in directions
    ):
        raise ValueError("a path must contain zero through four directions numbered 1 through 6")
    if not directions:
        return 0
    return (
        sum(6**power for power in range(1, len(directions)))
        + sum(
            (direction - 1) * 6 ** (len(directions) - offset - 1)
            for offset, direction in enumerate(directions)
        )
        + 1
    )


def decode_path(path_id: int) -> tuple[int, ...]:
    """Decode a stable path id."""
    if isinstance(path_id, bool) or not isinstance(path_id, int) or not 0 <= path_id <= MAX_PATH_ID:
        raise ValueError(f"path id must be an integer from 0 through {MAX_PATH_ID}")
    if path_id == 0:
        return ()
    remaining = path_id - 1
    length = 1
    while remaining >= 6**length:
        remaining -= 6**length
        length += 1
    digits: list[int] = []
    for power in range(length - 1, -1, -1):
        digit, remaining = divmod(remaining, 6**power)
        digits.append(digit + 1)
    return tuple(digits)
