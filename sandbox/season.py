"""Read the optional local season settings file beside ``manifest.json``."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sandbox.harness.environment import EnvParameter, ParameterValue, effective_parameters, resolve_parameters


@dataclass(frozen=True)
class SeasonSettings:
    """The locally reproducible part of a season downloaded from Game Sandbox."""

    label: str
    parameters: Mapping[str, ParameterValue]
    decision_limit_ms: int | None
    game_limit_ms: int | None


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"season.json {name} must be an object")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"season.json {name} must be a positive integer")
    return value


def _season_label(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("season.json season must be a nonempty string")
    return value


def load_season_settings(repo_root: Path, meta: Any) -> SeasonSettings | None:
    """Load and validate ``season.json``, returning ``None`` when it is not present."""
    path = repo_root / "season.json"
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"could not read season.json: {error}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"season.json is not valid JSON: {error.msg}") from None

    data = _object(document, "root")
    known_fields = {
        "env_id",
        "season",
        "parameters",
        "decision_limit_ms",
        "game_limit_ms",
    }
    unknown_fields = set(data) - known_fields
    if unknown_fields:
        raise ValueError(f"season.json has unknown field {min(unknown_fields)!r}")
    env_id = data.get("env_id")
    if env_id != meta.env_id:
        raise ValueError(f"season.json is for environment {env_id!r}, not {meta.env_id!r}")
    label = _season_label(data.get("season"))
    raw_parameters = data.get("parameters", {})
    if not isinstance(raw_parameters, dict):
        raise ValueError("season.json parameters must be an object")
    try:
        parameters = resolve_parameters(meta, raw_parameters)
    except ValueError as error:
        raise ValueError(f"season.json parameters: {error}") from None
    decision_limit_ms = (
        None
        if "decision_limit_ms" not in data
        else _positive_int(data["decision_limit_ms"], "decision_limit_ms")
    )
    game_limit_ms = (
        None if "game_limit_ms" not in data else _positive_int(data["game_limit_ms"], "game_limit_ms")
    )
    return SeasonSettings(label, parameters, decision_limit_ms, game_limit_ms)


def parse_parameter_overrides(meta: Any, raw_parameters: list[str]) -> dict[str, ParameterValue]:
    """Parse repeatable ``--parameter NAME=VALUE`` flags through effective declarations."""
    declarations = {parameter.name: parameter for parameter in effective_parameters(meta)}
    values: dict[str, ParameterValue] = {}
    for raw in raw_parameters:
        name, separator, raw_value = raw.partition("=")
        if not separator or not name:
            raise ValueError("--parameter must use NAME=VALUE")
        declaration = declarations.get(name)
        if declaration is None:
            raise ValueError(f"unknown environment parameter {name!r}")
        values[name] = _parse_parameter_value(declaration, raw_value)
    return values


def _parse_parameter_value(declaration: EnvParameter, raw: str) -> ParameterValue:
    if declaration.type in {"string", "choice"}:
        value: object = raw
    else:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"--parameter {declaration.name} needs a valid {declaration.type} value"
            ) from error
    return declaration.validate_value(value)


def announce(settings: SeasonSettings | None) -> None:
    """Print one concise acknowledgement when local settings are active."""
    if settings is not None:
        print(f"Using {settings.label} settings from season.json.")
