"""Load the packaged schema and validate payloads against it.

The canonical schema lives under ``schema/`` at the repository root. ``scripts/generate.py``
copies byte-identical bytes into ``schema_data/`` in this package, so editable dev
installs, built wheels inside session containers, and CI all read the same files. We
validate directly against those files rather than against generated models, so there is
zero drift between the contract and the validator by construction.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

#: The single integer schema version this harness produces and accepts.
SCHEMA_VERSION = 1

_STEP_STATE_FILE = "step-state.schema.json"
_RECORDING_HEADER_FILE = "recording-header.schema.json"
_ENVIRONMENT_META_FILE = "environment-meta.schema.json"


class SchemaValidationError(ValueError):
    """Raised when a payload fails validation against its schema."""


def _load_schema(filename: str) -> dict[str, Any]:
    data = resources.files(__package__).joinpath("schema_data", filename).read_text(encoding="utf-8")
    return json.loads(data)


def _build_validator(filename: str) -> Draft202012Validator:
    schema = _load_schema(filename)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


# Validators are compiled once per process. If per-step validation ever shows up in a
# profile, fastjsonschema can replace the implementation behind these same functions.
_step_validator = _build_validator(_STEP_STATE_FILE)
_header_validator = _build_validator(_RECORDING_HEADER_FILE)
_environment_meta_validator = _build_validator(_ENVIRONMENT_META_FILE)


def _validate(validator: Draft202012Validator, payload: Any, label: str) -> None:
    # jsonschema types iter_errors as an untyped overload, so pyright cannot narrow it;
    # we re-assert the documented return type here.
    errors: list[ValidationError] = sorted(
        validator.iter_errors(payload),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        key=lambda e: list(e.absolute_path),
    )
    if errors:
        first = errors[0]
        location = "/".join(str(p) for p in first.absolute_path) or "<root>"
        raise SchemaValidationError(f"{label} invalid at {location}: {first.message}")


def validate_step(payload: Any) -> None:
    """Validate a per-step state object, raising :class:`SchemaValidationError` on failure."""
    _validate(_step_validator, payload, "step state")


def validate_environment_meta(payload: Any) -> None:
    """Validate one environment's public metadata, raising :class:`SchemaValidationError` on failure.

    This is a conformance check beside :meth:`EnvironmentMeta.__post_init__` in
    ``environment.py``, not a replacement for it. The dataclass gives environment authors
    immediate, readable, ``env_id``-specific errors at import time; this validates the same
    ``to_json()`` shape against the schema TypeScript and the frontend also validate against, so a
    drift between the two languages' rules shows up as a failure here rather than silently.
    """
    _validate(_environment_meta_validator, payload, "environment metadata")


def validate_header(payload: Any) -> None:
    """Validate a recording header, raising :class:`SchemaValidationError` on failure."""
    _validate(_header_validator, payload, "recording header")
    if not isinstance(payload, dict):
        raise SchemaValidationError("recording header must be an object")
    header = cast("dict[str, object]", payload)
    players = header.get("players")
    seats = header.get("seats")
    seat_plan = header.get("seat_plan")
    if not isinstance(players, dict) or not isinstance(seats, dict) or not isinstance(seat_plan, str):
        raise SchemaValidationError("recording header must contain players, seats, and seat_plan")
    player_map = cast("dict[str, object]", players)
    seat_map = cast("dict[str, object]", seats)
    members: list[str] = []
    for _seat_id, seat_players in seat_map.items():
        if not isinstance(seat_players, list):
            raise SchemaValidationError("recording header seats must map seat ids to player lists")
        player_ids = cast("list[object]", seat_players)
        if not player_ids or not all(isinstance(player_id, str) for player_id in player_ids):
            raise SchemaValidationError("recording header seats must contain non-empty player lists")
        members.extend(cast("list[str]", player_ids))
    if len(members) != len(set(members)):
        raise SchemaValidationError("recording header seats must not assign a player to multiple seats")
    if set(player_map) != set(members):
        raise SchemaValidationError("recording header players must exactly match the seat partition")
