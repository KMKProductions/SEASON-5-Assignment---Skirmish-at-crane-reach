"""Reading what is legal this turn, and building the order your ``act`` returns.

An order is one path choice and one target choice. :func:`move` walks an encoded path and
:func:`stay` holds position; either accepts an optional ``target_id`` naming the enemy your
strike should prefer over the automatic in-range draw.

:func:`legal_paths`, :func:`legal_steps`, and :func:`possible_targets` read this turn's action
mask, the sole authority on what the environment accepts right now. Neither :func:`move` nor
:func:`stay` checks the mask itself, so pick a path from :func:`legal_paths` and a target from
:func:`possible_targets` and your order is legal by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import paths, roster

if TYPE_CHECKING:
    from sandbox.observation_types import SkirmishAction, SkirmishObservation

__all__ = ["legal_paths", "legal_steps", "move", "possible_targets", "stay"]


def legal_paths(observation: SkirmishObservation) -> list[int]:
    """Return every path id legal this turn, ascending, including 0 for stay."""
    mask = observation["action_mask"]["path"]
    return [path_id for path_id, allowed in enumerate(mask) if allowed]


def legal_steps(observation: SkirmishObservation) -> list[int]:
    """Return the legal single-step path ids, which are their direction digits, 1 through 6.

    The short way to ask "which of the six tiles around me can I walk onto right now", without
    reading the path encoding first. Longer paths are in :func:`legal_paths`.
    """
    return [path_id for path_id in legal_paths(observation) if 1 <= path_id <= 6]


def possible_targets(observation: SkirmishObservation) -> list[str]:
    """Return the ids of enemies you may name as a target this turn, in enemy roster order."""
    mask = observation["action_mask"]["target"]
    return [entry["unit_id"] for index, entry in enumerate(roster.enemies(observation)) if mask[index + 1]]


def _resolve_target(target_id: str | None, observation: SkirmishObservation | None) -> int:
    if target_id is None:
        return 0
    if observation is None:
        raise ValueError("naming a target requires the observation it was read from")
    for index, entry in enumerate(roster.enemies(observation)):
        if entry["unit_id"] == target_id:
            return index + 1
    raise ValueError(f"{target_id!r} is not an enemy unit id in this observation's roster")


def stay(target_id: str | None = None, observation: SkirmishObservation | None = None) -> SkirmishAction:
    """Return an order that holds this activation's position, optionally naming a target.

    Your unit still strikes from where it stands if an enemy ends up in range. Naming a target
    requires the ``observation`` it came from: pass one of the ids :func:`possible_targets`
    returns, otherwise this raises ``ValueError``.
    """
    return {"path": 0, "target": _resolve_target(target_id, observation)}


def move(
    path_id: int, target_id: str | None = None, observation: SkirmishObservation | None = None
) -> SkirmishAction:
    """Return an order that walks the encoded path, optionally naming a target.

    ``path_id`` only needs to be a valid encoded id (0 through ``paths.MAX_ID``); it is not
    checked against the mask here, since legality depends on the live battlefield. Use
    :func:`legal_paths` or :func:`legal_steps` to choose one that is actually walkable this turn.
    Naming a target requires the ``observation`` it came from: pass one of the ids
    :func:`possible_targets` returns, otherwise this raises ``ValueError``.
    """
    paths.decode(path_id)
    return {"path": path_id, "target": _resolve_target(target_id, observation)}
