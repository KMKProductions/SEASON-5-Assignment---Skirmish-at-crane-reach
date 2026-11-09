"""Static types for the semantic Skirmish at Crane Reach observation.

TypedDicts mirroring the runtime shapes ``env.py`` builds in ``_observe_living``: the nested
position, unit, tile, zone, and roster shapes, plus the top-level ``SkirmishObservation`` that
``env.observe()`` returns and the ``SkirmishAction`` an agent's ``act()`` returns. Stdlib-only at
runtime (numpy is imported only for type checking), so ``sandbox.observation_types`` may be
imported without dragging in the engine. Ships into the student template as
``sandbox.observation_types``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class AxialPosition(TypedDict):
    """One axial hex coordinate."""

    q: int
    r: int


class SelfUnit(TypedDict):
    """Your own unit: the fields only you can see (its kind, movement, and forward direction)."""

    unit_id: str
    type: str  # "footman", "archer", or "cavalry"
    position: AxialPosition
    hit_points: int
    movement_points: int
    direction: int  # the digit heading toward the enemy side: 2 east for red, 5 west for blue


class UnitSummary(TypedDict):
    """Unit details shared by visible records and the zone occupant helper."""

    unit_id: str
    side: str  # "red" or "blue"
    type: str  # "footman", "archer", or "cavalry"
    position: AxialPosition
    hit_points: int


class VisibleUnit(UnitSummary):
    """One friendly or enemy unit currently visible to your unit."""

    has_acted: bool  # whether this unit has completed its activation in the current round


class Capture(TypedDict):
    """Running capture-zone scores. All zero when capture play is disabled."""

    red: int
    blue: int
    target: int


class Tile(TypedDict):
    """One battlefield tile's terrain and optional feature."""

    terrain: str
    feature: str


class Zone(TypedDict):
    """One capture zone: its center tile and the tiles it covers."""

    center: AxialPosition
    tiles: tuple[AxialPosition, ...]


class Battlefield(TypedDict):
    """The hex field: its side length, every tile row by row, and its capture zones."""

    side: int
    tiles: tuple[tuple[Tile, ...], ...]
    zones: tuple[Zone, ...]


class RosterEntry(TypedDict):
    """One starting unit's owning player, identity, and side, stable for the whole match."""

    player: str
    unit_id: str
    side: str  # "red" or "blue"
    type: str  # "footman", "archer", or "cavalry"


class Rosters(TypedDict):
    """Each side's starting roster, keyed by side name."""

    red: tuple[RosterEntry, ...]
    blue: tuple[RosterEntry, ...]


class MatchParameters(TypedDict):
    """The resolved match settings this episode is playing under."""

    seat_plan: str  # "skirmish" or "army"
    field_extent: int
    terrain: int  # 0 or 1
    wasteland: int  # 0 or 1
    unit_abilities: int  # 0 or 1
    capture_zones: int
    capture_target: int
    round_cap: int


class SkirmishObservationData(TypedDict):
    """The semantic Skirmish at Crane Reach state under the observation's "observation" key."""

    self: SelfUnit
    visible_units: tuple[VisibleUnit, ...]
    round: int
    capture: Capture
    battlefield: Battlefield
    rosters: Rosters
    parameters: MatchParameters


class ActionMask(TypedDict):
    """Legal path and target choices for this turn."""

    path: NDArray[np.int8]  # shape (1555,), 1 = legal encoded path id
    target: NDArray[np.int8]  # shape (enemy roster size + 1,), 1 = legal target slot


class SkirmishObservation(TypedDict):
    """The full dict a Skirmish at Crane Reach agent's act() receives."""

    observation: SkirmishObservationData
    action_mask: ActionMask


class SkirmishAction(TypedDict):
    """The order an agent's act() returns: a path choice and a target choice."""

    path: int  # encoded path id; 0 always means "stay in place"
    target: int  # 1-based slot into the enemy roster; 0 always means "no named target"
