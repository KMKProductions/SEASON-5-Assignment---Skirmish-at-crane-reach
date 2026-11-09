"""The stable path encoding: direction digits packed into the path id an order carries.

A path is a sequence of zero through four direction digits, each digit one hex step.
:func:`encode` packs a digit sequence into its id and :func:`decode` unpacks it again. The empty
path (id 0) means stand still. Direction digits run clockwise from northeast: ``1`` northeast,
``2`` east, ``3`` southeast, ``4`` southwest, ``5`` west, ``6`` northwest, the same numbering
``tile.DIRECTIONS`` maps to coordinate offsets.

Ids are ordered first by path length and then lexically by digit, so the six single steps take
ids 1 through 6: a one-step path id is simply its direction digit.
"""

from __future__ import annotations

__all__ = ["MAX_ID", "MAX_STEPS", "decode", "encode"]

MAX_STEPS = 4
# The six hex directions. tile.DIRECTIONS maps each digit to its coordinate offset.
_DIGITS = 6
# One id per direction sequence of each length, so 6 + 36 + 216 + 1296 = 1554 alongside id 0.
MAX_ID = sum(_DIGITS**length for length in range(1, MAX_STEPS + 1))


def encode(directions: tuple[int, ...] | list[int]) -> int:
    """Encode zero through four direction digits into their stable path id.

    ``encode([])`` is 0 and ``encode([6, 6, 6, 6])`` is the largest id, 1554. Raises
    ``ValueError`` for anything that is not zero through four digits, each one of the six
    direction numbers.
    """
    try:
        digits = tuple(directions)
    except TypeError as error:
        raise ValueError("a path must be an iterable of directions") from error
    if len(digits) > MAX_STEPS or any(
        type(digit) is not int or not 1 <= digit <= _DIGITS for digit in digits
    ):
        raise ValueError("a path must contain zero through four directions numbered 1 through 6")
    if not digits:
        return 0
    return (
        sum(_DIGITS**power for power in range(1, len(digits)))
        + sum((digit - 1) * _DIGITS ** (len(digits) - offset - 1) for offset, digit in enumerate(digits))
        + 1
    )


def decode(path_id: int) -> tuple[int, ...]:
    """Decode a stable path id back into its direction digits, the inverse of :func:`encode`.

    Raises ``ValueError`` unless ``path_id`` is an integer from 0 through :data:`MAX_ID`.
    """
    if isinstance(path_id, bool) or not isinstance(path_id, int) or not 0 <= path_id <= MAX_ID:
        raise ValueError(f"path id must be an integer from 0 through {MAX_ID}")
    if path_id == 0:
        return ()
    remaining = path_id - 1
    length = 1
    while remaining >= _DIGITS**length:
        remaining -= _DIGITS**length
        length += 1
    digits: list[int] = []
    for power in range(length - 1, -1, -1):
        digit, remaining = divmod(remaining, _DIGITS**power)
        digits.append(digit + 1)
    return tuple(digits)
