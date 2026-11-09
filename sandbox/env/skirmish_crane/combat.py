"""Visibility, target selection, and damage for tactical strikes."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING

from .battlefield import Battlefield
from .hexes import Position, distance, neighbors

if TYPE_CHECKING:
    from .engine import Unit


@dataclass(frozen=True)
class Strike:
    attacker_id: str
    target_id: str
    damage: int
    automatic: bool


def vision(unit: Unit, battlefield: Battlefield) -> int:
    return unit.stats.vision + (battlefield.tile_at(unit.position).terrain == "hill")


def visible_units(unit: Unit, units: dict[str, Unit], battlefield: Battlefield) -> tuple[Unit, ...]:
    radius = vision(unit, battlefield)
    return tuple(
        other
        for other in units.values()
        if other.unit_id != unit.unit_id and distance(unit.position, other.position) <= radius
    )


def damage(
    attacker: Unit,
    defender: Unit,
    battlefield: Battlefield,
    units: dict[str, Unit],
    *,
    abilities: bool,
    start: Position,
) -> int:
    """Apply modifiers in the pinned ruleset order and return at least one."""
    attacker_tile = battlefield.tile_at(attacker.position)
    defender_tile = battlefield.tile_at(defender.position)
    total = attacker.stats.damage
    adjacent = neighbors(defender.position, battlefield.extent)
    shielded = (
        abilities
        and defender.kind == "footman"
        and any(
            other.side == defender.side and other.kind == "footman" and other.unit_id != defender.unit_id
            for other in units.values()
            if other.position in adjacent
        )
    )
    # Forest cover and a shield wall each deny the charge outright.
    charging = (
        abilities
        and attacker.kind == "cavalry"
        and distance(start, attacker.position) >= 3
        and defender_tile.feature != "forest"
        and not shielded
    )
    if charging:
        total += 2
    if attacker_tile.terrain == "hill" and defender_tile.terrain != "hill":
        total += 1
    if attacker_tile.terrain != "hill" and defender_tile.terrain == "hill":
        total -= 1
    if defender_tile.feature == "forest" and distance(attacker.position, defender.position) > 1:
        total -= 1
    if shielded:
        total -= 1
    return max(1, total)


def resolve_strike(
    attacker: Unit,
    units: dict[str, Unit],
    battlefield: Battlefield,
    rng: Random,
    *,
    named_target: str | None,
    visible_at_activation: set[str],
    abilities: bool,
    start: Position,
) -> Strike | None:
    """Resolve the mandatory strike, consuming the RNG only for an auto target."""
    enemies = [
        unit
        for unit in units.values()
        if unit.side != attacker.side
        and distance(attacker.position, unit.position) <= attacker.stats.attack_range
    ]
    if not enemies:
        return None
    named = units.get(named_target) if named_target else None
    automatic = not (
        named is not None
        and named.side != attacker.side
        and named.unit_id in visible_at_activation
        and distance(attacker.position, named.position) <= attacker.stats.attack_range
    )
    if named is not None and not automatic:
        target = named
    else:
        closest = min(distance(attacker.position, unit.position) for unit in enemies)
        target = rng.choice(
            [unit for unit in enemies if distance(attacker.position, unit.position) == closest]
        )
    hit = damage(attacker, target, battlefield, units, abilities=abilities, start=start)
    target.hit_points -= hit
    return Strike(attacker.unit_id, target.unit_id, hit, automatic)
