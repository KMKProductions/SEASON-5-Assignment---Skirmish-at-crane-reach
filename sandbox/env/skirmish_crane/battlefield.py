"""Constructive, point-symmetric tactical battlefield generation."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from random import Random
from types import MappingProxyType

from .hexes import (
    VOID,
    Position,
    Tile,
    distance,
    field_positions,
    neighbors,
    rotate_position,
    tile_array,
)
from .tile_types import FEATURE_SCATTER, TERRAIN_SCATTER

MAX_REDRAWS = 12
# Inclusive bounds this generator enforces. The environment package reuses them for its
# metadata declarations and observation spaces, so they are stated only here.
FIELD_EXTENT_BOUNDS = (5, 22)
CAPTURE_ZONES_BOUNDS = (0, 5)


@dataclass(frozen=True)
class CaptureZone:
    center: Position
    tiles: tuple[Position, ...]


@dataclass(frozen=True)
class Battlefield:
    """A generated field. Every field is immutable once built.

    ``tiles`` is the square grid indexed row then column as ``tiles[r][q]``, with void
    outside the hexagon. That is the same shape participants receive through perception.
    """

    extent: int
    tiles: tuple[tuple[Tile, ...], ...]
    spawns: Mapping[str, tuple[Position, ...]]
    zones: tuple[CaptureZone, ...]
    passage_tiles: tuple[tuple[Position, ...], ...]
    redraw_count: int = 0

    @property
    def side(self) -> int:
        return 2 * self.extent + 1

    def tile_at(self, position: Position) -> Tile:
        q, r = position
        if 0 <= q < self.side and 0 <= r < self.side:
            return self.tiles[r][q]
        return VOID

    def snapshot(self) -> BattlefieldSnapshot:
        """Return a deeply immutable view suitable for participant perception."""
        return BattlefieldSnapshot(
            self.extent,
            self.side,
            self.tiles,
            self.spawns,
            self.zones,
            self.passage_tiles,
        )

    def connected(self) -> bool:
        passable = {(q, r) for r, row in enumerate(self.tiles) for q, tile in enumerate(row) if tile.passable}
        if not passable:
            return False
        seen = {next(iter(passable))}
        queue = deque(seen)
        while queue:
            position = queue.popleft()
            for candidate in neighbors(position, self.extent):
                if candidate in passable and candidate not in seen:
                    seen.add(candidate)
                    queue.append(candidate)
        return seen == passable


@dataclass(frozen=True)
class BattlefieldSnapshot:
    extent: int
    side: int
    tiles: tuple[tuple[Tile, ...], ...]
    spawns: Mapping[str, tuple[Position, ...]]
    zones: tuple[CaptureZone, ...]
    passage_tiles: tuple[tuple[Position, ...], ...]


def _passage_ranges(extent: int, count: int, rng: Random) -> tuple[range, ...]:
    """Return reflected seam gaps. Ranges are separated by at least one water tile."""
    side = 2 * extent + 1
    if count == 2:
        width = rng.randint(2, 4)
        lower_start = 1
        return range(lower_start, lower_start + width), range(side - lower_start - width, side - lower_start)
    # A three-wide central gap is the only self-reflecting contiguous gap. It
    # also leaves room for two reflected outer passages at every supported size.
    width = rng.randint(2, min(4, extent - 3))
    central = range(extent - 1, extent + 2)
    return range(1, 1 + width), central, range(side - 1 - width, side - 1)


def _terrain_tiles(
    extent: int, rng: Random, terrain: bool, zone_count: int, wasteland: bool
) -> tuple[dict[Position, Tile], tuple[tuple[Position, ...], ...]]:
    tiles = {position: Tile() for position in field_positions(extent)}
    if not terrain:
        return tiles, ()
    # Capture zones force three passages because only the three-wide central gap keeps
    # the middle tile passable, and an odd zone count always needs a central zone there.
    passage_count = 3 if zone_count else rng.choice((2, 3))
    passages = _passage_ranges(extent, passage_count, rng)
    gap_rows = {r for passage in passages for r in passage}
    for r in range(2 * extent + 1):
        position = (extent, r)
        if position in tiles and r not in gap_rows:
            tiles[position] = Tile("water")

    # Only select one representative from each reflected pair. The seam was
    # already made symmetric and is left alone so passage metadata stays exact.
    flags = {"wasteland": wasteland}
    for position in sorted(tiles):
        mirror = rotate_position(position, extent)
        if position > mirror or position[0] == extent:
            continue
        if not tiles[position].passable:
            continue
        # Terrain and feature draw independently, so a feature may land on any passable
        # terrain and each kind keeps its own density.
        drawn_terrain = TERRAIN_SCATTER.value_for(rng.randrange(TERRAIN_SCATTER.die), flags)
        drawn_feature = FEATURE_SCATTER.value_for(rng.randrange(FEATURE_SCATTER.die), flags)
        tile = Tile(drawn_terrain, drawn_feature)
        tiles[position] = tile
        tiles[mirror] = tile
    passage_tiles = tuple(tuple((extent, r) for r in passage) for passage in passages)
    return tiles, passage_tiles


def _zone_tiles(center: Position, extent: int) -> tuple[Position, ...]:
    return (center, *neighbors(center, extent))


def _zones(extent: int, tiles: dict[Position, Tile], count: int) -> tuple[CaptureZone, ...]:
    if count == 0:
        return ()
    valid = [
        position
        for position in sorted(tiles)
        if len(_zone_tiles(position, extent)) == 7
        and all(tiles[candidate].passable for candidate in _zone_tiles(position, extent))
    ]
    valid_set = set(valid)
    chosen: list[Position] = []
    center = (extent, extent)
    if count % 2:
        if center not in valid_set:
            raise ValueError("battlefield has no passable central capture zone")
        chosen.append(center)
    for candidate in valid:
        if len(chosen) >= count:
            break
        mirror = rotate_position(candidate, extent)
        if candidate >= mirror or mirror not in valid_set:
            continue
        # Centers three apart keep the seven-tile footprints disjoint, so no single
        # unit can ever score two zones in the same round.
        if distance(candidate, mirror) < 3:
            continue
        if any(distance(candidate, taken) < 3 or distance(mirror, taken) < 3 for taken in chosen):
            continue
        chosen.extend((candidate, mirror))
    if len(chosen) != count:
        raise ValueError("battlefield has insufficient passable capture zones")
    return tuple(CaptureZone(position, _zone_tiles(position, extent)) for position in chosen)


def _spawns(extent: int, tiles: dict[Position, Tile], units_per_side: int) -> dict[str, tuple[Position, ...]]:
    left = [position for position in sorted(tiles) if position[0] < extent and tiles[position].passable]
    # Spread a roster across the half deterministically, avoiding any dependence
    # on match-play randomness.
    if len(left) < units_per_side:
        raise ValueError("battlefield does not have enough spawn tiles")
    selected = tuple(left[index * len(left) // units_per_side] for index in range(units_per_side))
    return {"red": selected, "blue": tuple(rotate_position(position, extent) for position in selected)}


def generate_battlefield(
    extent: int,
    rng: Random,
    *,
    terrain: bool = False,
    wasteland: bool = False,
    capture_zones: int = 0,
    units_per_side: int = 3,
) -> Battlefield:
    """Build a connected field using only the supplied battlefield RNG stream."""
    extent_low, extent_high = FIELD_EXTENT_BOUNDS
    if not extent_low <= extent <= extent_high:
        raise ValueError(f"field extent must be from {extent_low} through {extent_high}")
    zones_low, zones_high = CAPTURE_ZONES_BOUNDS
    if not zones_low <= capture_zones <= zones_high:
        raise ValueError(f"capture zones must be from {zones_low} through {zones_high}")
    for redraw in range(MAX_REDRAWS):
        tiles, passages = _terrain_tiles(extent, rng, terrain, capture_zones, wasteland)
        try:
            zones = _zones(extent, tiles, capture_zones)
            field = Battlefield(
                extent,
                tile_array(extent, tiles),
                MappingProxyType(_spawns(extent, tiles, units_per_side)),
                zones,
                passages,
                redraw,
            )
        except ValueError:
            continue
        if field.connected():
            return field
    raise RuntimeError("could not construct a connected battlefield within the redraw limit")
