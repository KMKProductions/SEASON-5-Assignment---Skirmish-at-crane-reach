"""Compact, JSON-safe render state for Skirmish at Crane Reach."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

from .combat import visible_units
from .engine import COMPOSITIONS
from .hexes import neighbors, on_field, path_positions
from .paths import decode_path, encode_path
from .tile_types import TILE_CODES

OVERLAY_VERSION = 1

_BASE64 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_"
_TILE_FROM_CODE = {code: tile for tile, code in TILE_CODES.items()}


def _base64(value: int) -> str:
    if value == 0:
        return "0"
    digits: list[str] = []
    while value:
        value, digit = divmod(value, 64)
        digits.append(_BASE64[digit])
    return "".join(reversed(digits))


def _base36(value: int, width: int) -> str:
    digits = []
    for _ in range(width):
        value, digit = divmod(value, 36)
        digits.append(_BASE64[digit])
    return "".join(reversed(digits))


def _decode_number(value: str, alphabet: str) -> int:
    if not value or any(digit not in alphabet for digit in value):
        raise ValueError("compact overlay contains an invalid encoded number")
    total = 0
    for digit in value:
        total = total * len(alphabet) + alphabet.index(digit)
    return total


def _roster(seat_plan: str) -> tuple[dict[str, Any], ...]:
    if seat_plan not in COMPOSITIONS:
        raise ValueError("compact overlay has an unknown seat plan")
    return tuple(
        {
            "player": f"player_{player}",
            "unit_id": f"{side}_{kind}_{index}",
            "side": side,
            "type": kind,
        }
        for player, (side, kind, index) in enumerate(
            (side, kind, index)
            for side in ("red", "blue")
            for kind, count in COMPOSITIONS[seat_plan].items()
            for index in range(count)
        )
    )


def _player_index(env: Any, unit_id: str) -> int:
    """Convert an internal unit id to the compact wire-format player index."""
    return int(env.agent_by_unit[unit_id].removeprefix("player_"))


def _event(env: Any) -> list[Any] | None:
    activation = env.last_activation
    if activation is None:
        return None
    strike = activation.strike
    return [
        _player_index(env, activation.unit_id),
        activation.start[0],
        activation.start[1],
        activation.end[0],
        activation.end[1],
        -1 if strike is None else _player_index(env, strike.target_id),
        0 if strike is None else strike.damage,
        False if strike is None else strike.automatic,
        -1 if activation.killed_id is None else _player_index(env, activation.killed_id),
        env.last_capture_changes["red"],
        env.last_capture_changes["blue"],
        encode_path(activation.path),
    ]


def _battlefield(field: Any) -> dict[str, Any]:
    """Pack battlefield data that stays fixed for one episode."""
    return {
        "s": field.side,
        "t": ["".join(TILE_CODES[tile.terrain, tile.feature] for tile in row) for row in field.tiles],
        "z": ["".join((_base36(zone.center[0], 2), _base36(zone.center[1], 2))) for zone in field.zones],
    }


def extract_overlay_static(env: Any) -> dict[str, Any]:
    """Extract the immutable compact layout captured in a recording header after reset."""
    return {"k": OVERLAY_VERSION, "p": env.config.seat_plan, "b": _battlefield(env.match.battlefield)}


def extract_overlay(env: Any) -> dict[str, Any]:
    """Extract one dynamic compact render state containing only JSON-native values.

    Tile rows, zones, and the roster plan live in the recording header. Units and activations use
    canonical player indexes, and every visibility entry is a roster-order base-64 bitmask.
    """
    match = env.match
    field = match.battlefield
    live = tuple(match.units.values())
    visibility: list[str | None] = [None] * len(env.possible_agents)
    for unit in live:
        seen_ids = {seen.unit_id for seen in visible_units(unit, match.units, field)}
        bits = sum(
            1 << index
            for index, player in enumerate(env.possible_agents)
            if env.unit_by_agent[player] in seen_ids
        )
        visibility[_player_index(env, unit.unit_id)] = _base64(bits)
    current = None if match.result is not None else match.current_unit_id
    acted = sum(
        1 << _player_index(env, unit_id) for unit_id in match.activation_order[: match.activation_index]
    )
    return {
        "k": OVERLAY_VERSION,
        "r": match.round,
        "c": [
            match.capture_scores["red"],
            match.capture_scores["blue"],
            env.config.capture_target if env.config.capture else 0,
        ],
        "u": [
            "".join(
                (
                    _base36(_player_index(env, unit.unit_id), 2),
                    _base36(unit.position[0], 2),
                    _base36(unit.position[1], 2),
                    _base36(unit.hit_points, 1),
                )
            )
            for unit in live
        ],
        "a": None if current is None else _player_index(env, current),
        "d": _base64(acted),
        "v": visibility,
        "e": _event(env),
        "x": match.result is not None,
        "o": None if match.result is None else [match.result.red, match.result.blue],
    }


def decode_overlay(compact: Mapping[str, Any], static: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate and decode one dynamic replay frame with its static recording-header layout."""
    if isinstance(compact, Mapping) and ({"p", "b"} & set(compact)):
        raise ValueError("compact overlay dynamic frame must not contain static layout data")
    if static is None:
        raise ValueError("compact overlay static data is required")
    if not isinstance(static, Mapping) or set(static) != {"k", "p", "b"}:
        raise ValueError("compact overlay static data has unexpected fields")
    version = static["k"]
    if type(version) is not int or version != OVERLAY_VERSION:
        raise ValueError("compact overlay static data has an unsupported version")
    required_fields = {"k", "r", "c", "u", "a", "v", "e", "x", "o"}
    if not isinstance(compact, Mapping) or set(compact) not in (required_fields, required_fields | {"d"}):
        raise ValueError("compact overlay dynamic frame has unexpected fields")
    if type(compact["k"]) is not int or compact["k"] != OVERLAY_VERSION:
        raise ValueError("compact overlay dynamic frame has an unsupported version")

    seat_plan = static["p"]
    if not isinstance(seat_plan, str):
        raise ValueError("compact overlay seat plan must be text")
    roster = _roster(seat_plan)
    player_count = len(roster)

    battlefield = static["b"]
    if not isinstance(battlefield, Mapping) or set(battlefield) != {"s", "t", "z"}:
        raise ValueError("compact overlay battlefield is malformed")
    side = battlefield["s"]
    rows = battlefield["t"]
    zone_records = battlefield["z"]
    if type(side) is not int or side < 1 or side % 2 == 0:
        raise ValueError("compact overlay battlefield side must be a positive odd integer")
    if not isinstance(rows, list) or len(rows) != side:
        raise ValueError("compact overlay tile rows do not match the battlefield side")
    tiles = []
    for row in rows:
        if not isinstance(row, str) or len(row) != side or any(code not in _TILE_FROM_CODE for code in row):
            raise ValueError("compact overlay contains an invalid tile row")
        tiles.append(
            [{"terrain": _TILE_FROM_CODE[code][0], "feature": _TILE_FROM_CODE[code][1]} for code in row]
        )
    if not isinstance(zone_records, list):
        raise ValueError("compact overlay zones must be a list")
    extent = (side - 1) // 2
    zones = []
    for record in zone_records:
        if not isinstance(record, str) or len(record) != 4:
            raise ValueError("compact overlay zone records must be four characters")
        q = _decode_number(record[:2], _BASE64[:36])
        r = _decode_number(record[2:], _BASE64[:36])
        center = (q, r)
        zone_tiles = (center, *neighbors(center, extent))
        if len(zone_tiles) != 7:
            raise ValueError("compact overlay zone center does not have six field neighbors")
        zones.append(
            {
                "center": {"q": q, "r": r},
                "tiles": [{"q": tile_q, "r": tile_r} for tile_q, tile_r in zone_tiles],
            }
        )

    unit_records = compact["u"]
    if not isinstance(unit_records, list):
        raise ValueError("compact overlay units must be a list")
    units = []
    living_players: set[int] = set()
    for record in unit_records:
        if not isinstance(record, str) or len(record) != 7:
            raise ValueError("compact overlay unit records must be seven characters")
        player = _decode_number(record[:2], _BASE64[:36])
        q = _decode_number(record[2:4], _BASE64[:36])
        r = _decode_number(record[4:6], _BASE64[:36])
        hit_points = _decode_number(record[6], _BASE64[:36])
        if player >= player_count or player in living_players or q >= side or r >= side or hit_points < 1:
            raise ValueError("compact overlay unit record is out of range or duplicated")
        living_players.add(player)
        units.append({**roster[player], "position": {"q": q, "r": r}, "hit_points": hit_points})

    acted_record = compact.get("d", "0")
    if not isinstance(acted_record, str):
        raise ValueError("compact overlay acted state must be a bitmask string")
    acted_bits = _decode_number(acted_record, _BASE64)
    if acted_bits >= 1 << player_count:
        raise ValueError("compact overlay acted state has bits outside the roster")
    for unit in units:
        unit["has_acted"] = bool(acted_bits & (1 << int(unit["player"].removeprefix("player_"))))

    visibility_records = compact["v"]
    if not isinstance(visibility_records, list) or len(visibility_records) != player_count:
        raise ValueError("compact overlay visibility must follow full roster order")
    visible_units: dict[str, tuple[str, ...]] = {}
    for player, record in enumerate(visibility_records):
        if player not in living_players:
            if record is not None:
                raise ValueError("compact overlay gives visibility to a dead player")
            continue
        if not isinstance(record, str):
            raise ValueError("compact overlay living visibility must be a bitmask string")
        bits = _decode_number(record, _BASE64)
        if bits >= 1 << player_count:
            raise ValueError("compact overlay visibility has bits outside the roster")
        visible_units[roster[player]["player"]] = tuple(
            roster[index]["unit_id"] for index in range(player_count) if bits & (1 << index)
        )

    activation = compact["a"]
    if activation is not None and (type(activation) is not int or activation not in living_players):
        raise ValueError("compact overlay activation must name a living roster position")

    event_record = compact["e"]
    event = None
    if event_record is not None:
        if not isinstance(event_record, list) or len(event_record) != 12:
            raise ValueError("compact overlay version 1 event must have 12 values")
        actor, start_q, start_r, end_q, end_r, target, damage, automatic, death, red_delta, blue_delta = (
            event_record[:11]
        )
        indexes = (actor, target, death)
        if any(type(value) is not int for value in indexes) or not 0 <= actor < player_count:
            raise ValueError("compact overlay event player indexes are malformed")
        if target != -1 and not 0 <= target < player_count:
            raise ValueError("compact overlay event target is out of range")
        if death != -1 and not 0 <= death < player_count:
            raise ValueError("compact overlay event death is out of range")
        if any(
            type(value) is not int
            for value in (start_q, start_r, end_q, end_r, damage, red_delta, blue_delta)
        ):
            raise ValueError("compact overlay event numeric fields are malformed")
        if not on_field((start_q, start_r), extent) or not on_field((end_q, end_r), extent):
            raise ValueError("compact overlay event coordinates are outside the battlefield")
        if not isinstance(automatic, bool):
            raise ValueError("compact overlay event automatic flag must be boolean")
        try:
            path = decode_path(event_record[11])
        except ValueError as error:
            raise ValueError("compact overlay event path id is malformed") from error
        entered = path_positions((start_q, start_r), path)
        if any(not on_field(position, extent) for position in entered):
            raise ValueError("compact overlay event path leaves the battlefield")
        path_end = entered[-1] if entered else (start_q, start_r)
        if path_end != (end_q, end_r):
            raise ValueError("compact overlay event path does not reach its endpoint")
        event = {
            "unit_id": roster[actor]["unit_id"],
            "from": {"q": start_q, "r": start_r},
            "to": {"q": end_q, "r": end_r},
            "path": path,
            "attack": None
            if target == -1
            else {
                "target_id": roster[target]["unit_id"],
                "damage": damage,
                "automatic": automatic,
            },
            "death": None if death == -1 else roster[death]["unit_id"],
            "capture": {"red": red_delta, "blue": blue_delta},
        }

    capture = compact["c"]
    if not isinstance(capture, list) or len(capture) != 3 or any(type(value) is not int for value in capture):
        raise ValueError("compact overlay capture must contain three integers")
    round_number = compact["r"]
    if type(round_number) is not int or round_number < 1:
        raise ValueError("compact overlay round must be a positive integer")
    terminal = compact["x"]
    outcome = compact["o"]
    if not isinstance(terminal, bool):
        raise ValueError("compact overlay terminal flag must be boolean")
    if terminal:
        if not isinstance(outcome, list) or len(outcome) != 2:
            raise ValueError("terminal compact overlay must contain two outcome scores")
        if any(
            isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(float(score))
            for score in outcome
        ):
            raise ValueError("terminal compact overlay outcome scores must be finite real numbers")
    elif outcome is not None:
        raise ValueError("nonterminal compact overlay cannot contain outcome scores")

    return {
        "version": version,
        "seat_plan": seat_plan,
        "battlefield": {"side": side, "tiles": tiles, "zones": zones},
        "round": round_number,
        "capture": {"red": capture[0], "blue": capture[1], "target": capture[2]},
        "units": units,
        "current_activation": None if activation is None else roster[activation]["player"],
        "visible_units": visible_units,
        "event": event,
        "terminal": terminal,
        "outcome": outcome,
    }
