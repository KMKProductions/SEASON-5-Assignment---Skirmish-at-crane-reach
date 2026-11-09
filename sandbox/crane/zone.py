"""Capture zones: the ground worth holding, fixed for the whole match.

Zones are set once when the battlefield is generated and never move, resize, or appear
mid-match; a match not playing capture simply has none. These helpers describe geometry and who
is standing where. What that presence is worth, the running score and target, lives at
``observation["observation"]["capture"]``, and these helpers never touch it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import me

if TYPE_CHECKING:
    from sandbox.observation_types import AxialPosition, SkirmishObservation, UnitSummary, Zone

__all__ = ["at", "occupants", "zones"]


def zones(observation: SkirmishObservation) -> tuple[Zone, ...]:
    """Return the capture zones on the battlefield, empty when this match is not playing capture."""
    return observation["observation"]["battlefield"]["zones"]


def _covers(area: Zone, position: AxialPosition) -> bool:
    return any(spot["q"] == position["q"] and spot["r"] == position["r"] for spot in area["tiles"])


def at(observation: SkirmishObservation, position: AxialPosition) -> Zone | None:
    """Return the zone covering ``position``, or None when no zone does."""
    for zone in zones(observation):
        if _covers(zone, position):
            return zone
    return None


def occupants(observation: SkirmishObservation, area: Zone) -> list[UnitSummary]:
    """Return the units standing in ``area``, your own unit first when it is one of them.

    A unit you cannot see is simply absent, so an empty result here is not proof the zone is
    free.
    """
    own = observation["observation"]["self"]
    found: list[UnitSummary] = []
    if _covers(area, own["position"]):
        found.append(
            {
                "unit_id": own["unit_id"],
                "side": me.side(observation),
                "type": own["type"],
                "position": own["position"].copy(),
                "hit_points": own["hit_points"],
            }
        )
    visible = observation["observation"]["visible_units"]
    found.extend(unit for unit in visible if _covers(area, unit["position"]))
    return found
