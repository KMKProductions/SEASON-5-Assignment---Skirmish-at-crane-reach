"""The single tile-type source, shared by the rules engine and the renderer.

``tile_types.json`` sits beside this module and is the only place terrain and feature
properties, the generation scatter tables, and the compact overlay wire codes are written
down. The TypeScript renderer imports the same file, so adding a tile type is one JSON
entry plus a color and mark row in the renderer's own style table.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from typing import Any


@dataclass(frozen=True)
class Terrain:
    passable: bool
    move_cost: int


@dataclass(frozen=True)
class Feature:
    move_cost_delta: int
    entry_damage: int


@dataclass(frozen=True)
class ScatterRoll:
    roll: int
    value: str
    requires_parameter: str | None


@dataclass(frozen=True)
class ScatterTable:
    """One die rolled per generated tile pair. A roll matching no entry takes the default."""

    die: int
    default: str
    rolls: tuple[ScatterRoll, ...]

    def value_for(self, roll: int, flags: Mapping[str, bool]) -> str:
        for entry in self.rolls:
            if entry.roll != roll:
                continue
            if entry.requires_parameter is not None and not flags[entry.requires_parameter]:
                return self.default
            return entry.value
        return self.default


@dataclass(frozen=True)
class TileTypes:
    terrains: Mapping[str, Terrain]
    features: Mapping[str, Feature]
    terrain_scatter: ScatterTable
    feature_scatter: ScatterTable
    codes: Mapping[tuple[str, str], str]


def _section(data: Any, name: str) -> dict[str, Any]:
    section = data.get(name) if isinstance(data, dict) else None
    if not isinstance(section, dict) or not section:
        raise ValueError(f"tile types: {name} must be a non-empty object")
    return section


def _field(entry: Any, name: str, kind: type, owner: str) -> Any:
    if not isinstance(entry, dict) or name not in entry or type(entry[name]) is not kind:
        raise ValueError(f"tile types: {owner} needs a {kind.__name__} {name}")
    return entry[name]


def _scatter_table(data: Any, name: str, declared: Mapping[str, Any]) -> ScatterTable:
    table = _section(data, "scatter").get(name)
    die = _field(table, "die", int, f"the {name} scatter")
    default = _field(table, "default", str, f"the {name} scatter")
    entries = _field(table, "rolls", list, f"the {name} scatter")
    if die < 1:
        raise ValueError(f"tile types: the {name} scatter die must be positive")
    if default not in declared:
        raise ValueError(f"tile types: the {name} scatter default {default!r} is not declared")
    rolls: list[ScatterRoll] = []
    for entry in entries:
        owner = f"a {name} scatter roll"
        roll = _field(entry, "roll", int, owner)
        value = _field(entry, "value", str, owner)
        parameter = entry.get("requires_parameter")
        if parameter is not None and not isinstance(parameter, str):
            raise ValueError(f"tile types: {owner} needs a parameter name or null")
        if not 0 <= roll < die:
            raise ValueError(f"tile types: {name} scatter roll {roll} is outside its die")
        if value not in declared:
            raise ValueError(f"tile types: {name} scatter value {value!r} is not declared")
        rolls.append(ScatterRoll(roll, value, parameter))
    if len({entry.roll for entry in rolls}) != len(rolls):
        raise ValueError(f"tile types: the {name} scatter repeats a roll")
    return ScatterTable(die, default, tuple(rolls))


def _codes(
    data: Any, terrains: Mapping[str, Terrain], features: Mapping[str, Feature]
) -> dict[tuple[str, str], str]:
    codes: dict[tuple[str, str], str] = {}
    for terrain, row in _section(data, "tile_codes").items():
        if terrain not in terrains:
            raise ValueError(f"tile types: tile codes name an undeclared terrain {terrain!r}")
        if not isinstance(row, dict):
            raise ValueError(f"tile types: the {terrain} tile codes must be an object")
        for feature, code in row.items():
            if feature not in features:
                raise ValueError(f"tile types: tile codes name an undeclared feature {feature!r}")
            if not isinstance(code, str) or len(code) != 1:
                raise ValueError(f"tile types: the {terrain} {feature} code must be one character")
            codes[terrain, feature] = code
    if len(set(codes.values())) != len(codes):
        raise ValueError("tile types: tile codes repeat a character")
    # Generation scatters features onto passable tiles only, so the table must cover every
    # passable pairing and leave impassable terrain featureless.
    expected = {
        (terrain, feature)
        for terrain, properties in terrains.items()
        for feature in (features if properties.passable else ("none",))
    }
    if set(codes) != expected:
        raise ValueError("tile types: tile codes must cover every terrain and feature pairing")
    return codes


def load(data: Any) -> TileTypes:
    """Validate a decoded tile-type document, rejecting anything the engine cannot render."""
    terrains = {
        name: Terrain(_field(entry, "passable", bool, name), _field(entry, "move_cost", int, name))
        for name, entry in _section(data, "terrains").items()
    }
    features = {
        name: Feature(_field(entry, "move_cost_delta", int, name), _field(entry, "entry_damage", int, name))
        for name, entry in _section(data, "features").items()
    }
    if "none" not in features:
        raise ValueError("tile types: a none feature must be declared")
    terrain_scatter = _scatter_table(data, "terrain", terrains)
    feature_scatter = _scatter_table(data, "feature", features)
    if not terrains[terrain_scatter.default].passable or any(
        not terrains[entry.value].passable for entry in terrain_scatter.rolls
    ):
        raise ValueError("tile types: the terrain scatter may only produce passable terrain")
    return TileTypes(terrains, features, terrain_scatter, feature_scatter, _codes(data, terrains, features))


TILE_TYPES = load(
    json.loads(resources.files(__package__).joinpath("tile_types.json").read_text(encoding="utf-8"))
)
TERRAINS = TILE_TYPES.terrains
FEATURES = TILE_TYPES.features
TERRAIN_SCATTER = TILE_TYPES.terrain_scatter
FEATURE_SCATTER = TILE_TYPES.feature_scatter
TILE_CODES = TILE_TYPES.codes
