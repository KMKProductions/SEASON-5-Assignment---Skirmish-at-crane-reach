"""The live session runner: ``python -m game_sandbox_harness.live``.

This is the container side of a Stage 3 live session. It reads a single JSON config from argv,
resolves the environment from the in-image registry, binds each player to either the transport
(an external/human player) or a built-in agent, and drives the Stage 2 :class:`Episode` machinery
under wall-clock pacing with pause and stop. State lines stream out on the protocol stdout and
are simultaneously written to the recording on the mounted volume; one ``result`` envelope is
emitted at the end.

The live loop and headless ``run_episode`` both call :meth:`Episode.advance`, which dispatches to
one sequential AEC step or one simultaneous parallel tick. Sequential pacing retains its target
cadence. Simultaneous pacing starts ticks one interval apart, and a tick that runs past its next
boundary restarts the cadence from its completion without a catch-up burst. Pause is a cooperative
wait shared by both, and because the injected :class:`PausableClock` freezes while paused, cadence
and measured durations freeze with it.

Module-level imports stay free of environment packages so :func:`main` can claim stdout
*before* anything imports a game; the environment is loaded only inside :func:`main`,
after that redirection is in place.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import sys
import traceback
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, cast

from .clock import SystemClock
from .environment import (
    EnvironmentEntry,
    ParameterValue,
    ResolvedLayout,
    load_environment,
    resolve_layout,
    validate_complete_parameters,
)
from .live_io import (
    PausableClock,
    ProtocolStream,
    RealSleeper,
    SessionControl,
    Sleeper,
    TransportSource,
    build_tee_store,
    result_envelope,
    start_command_pump,
)
from .manifest import load_agent
from .recording import RecordingStore
from .session import REASON_STOPPED, AgentPlayer, Episode, ExternalPlayer, Player
from .state import PlayerAttribution

#: Where the session base image stages the built-in agents the watch-style runs load, one
#: per-environment and per-name directory beneath this base
#: (``/opt/agents/builtin/<env_id>/<name>``). A player with no explicit overlay path takes its named
#: agent from the session's own environment.
DEFAULT_BUILTIN_AGENT_BASE = "/opt/agents/builtin"
_BUILTIN_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
#: Cooperative wait granularity. Small enough that stop and resume stay responsive.
_SLICE_MS = 5
#: Tick markers are local control-plane requests and must never hold the participant loop open for
#: an unbounded period when the proxy is unavailable.
_MARKER_TIMEOUT_SECONDS = 2.0
#: The longest exception message kept in a failure description, so one runaway message cannot bloat
#: the result envelope or the rows that store it.
_FAILURE_MESSAGE_LIMIT = 300
_HARNESS_PACKAGE_DIR = Path(__file__).resolve().parent


class LiveConfigError(ValueError):
    """Raised when the session config argument is missing, malformed, or self-inconsistent."""


class UnsetTimeout:
    """The absent timeout wire state, distinct from JSON null (which disables it)."""


UNSET_TIMEOUT = UnsetTimeout()


@dataclass(frozen=True)
class PlayerBinding:
    """How one PettingZoo player is driven: transport or a loaded agent."""

    kind: str
    path: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class LlmConfig:
    """OpenAI-compatible endpoint and per-agent credentials for an official LLM session."""

    base_url: str
    tick_url: str
    inflight_url: str
    keys: dict[str, str]


@dataclass(frozen=True)
class LiveConfig:
    """The parsed session config carried in argv.

    Environment facts — pace interval, time limits, the default human timeout — come from the
    in-image registry, so this config carries only the session's own choices and overrides:
    which environment and seed, how each player is driven, the resolved human-player timeout (an
    override, or ``None`` to take the metadata default), and where to record.
    """

    env_id: str
    seed: int
    player_bindings: dict[str, PlayerBinding]
    human_timeout_ms: int | None | UnsetTimeout
    recording_dir: str
    recording_id: str | None
    #: Complete resolved parameter map required by every launch path.
    parameters: dict[str, ParameterValue]
    #: The layout resolved from the installed environment metadata and complete parameters.
    layout: ResolvedLayout | None = None
    #: Per-player attribution copied verbatim into the recording header (player id -> attribution
    #: object). It exactly covers the configured players and agrees with each binding kind.
    players: dict[str, PlayerAttribution] | None = None
    #: Optional per-step/per-episode time-limit overrides (the Stage 6 season overrides).
    #: ``None`` takes the environment's metadata default, as a session with no override does.
    step_timeout_ms: int | None = None
    episode_timeout_ms: int | None = None
    #: Effective session-level messaging config from the backend (metadata AND override). ``None``
    #: leaves the environment metadata to decide; the harness combines defensively, so a value here
    #: can only disable or tighten, never enable messaging on an environment that opted out.
    messaging_enabled: bool | None = None
    message_cap: int | None = None
    #: The one external binding authorized to send chat. ``None`` disables human chat acceptance.
    external_chat_player: str | None = None
    #: Workflow containers set this to run as fast as the agents compute, without live pacing.
    headless: bool = False
    #: Absent for ordinary sessions. The key map covers agent players exactly and excludes humans.
    llm: LlmConfig | None = None
    #: Emit the header and opening state, then wait for a resume command before the first step.
    start_paused: bool = False
    #: Optional local-play step cap. Production launch configs omit it.
    max_steps: int | None = None


def parse_config(
    argv: list[str],
    *,
    entry: EnvironmentEntry | None = None,
) -> LiveConfig:
    """Parse and validate one JSON config, using an injected environment when supplied."""
    if len(argv) != 1:
        raise LiveConfigError(f"expected exactly one JSON config argument, got {len(argv)} argument(s)")
    try:
        raw: object = json.loads(argv[0])
    except json.JSONDecodeError as error:
        raise LiveConfigError(f"config is not valid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise LiveConfigError(f"config must be a JSON object, got {type(raw).__name__}")
    config = cast("dict[str, Any]", raw)

    env_id = config.get("env_id")
    if not isinstance(env_id, str) or not env_id:
        raise LiveConfigError("config 'env_id' must be a non-empty string")

    seed = config.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise LiveConfigError("config 'seed' must be an integer")

    if entry is None:
        try:
            entry = load_environment(env_id)
        except Exception as error:  # noqa: BLE001 - normalize registry failures at the config boundary
            raise LiveConfigError(f"config environment {env_id!r} could not be loaded: {error}") from error
    elif entry.meta.env_id != env_id:
        raise LiveConfigError(
            f"config environment {env_id!r} does not match injected environment {entry.meta.env_id!r}"
        )

    raw_player_bindings = config.get("player_bindings")
    if not isinstance(raw_player_bindings, dict) or not raw_player_bindings:
        raise LiveConfigError("config 'player_bindings' must be a non-empty object keyed by player id")
    player_bindings: dict[str, PlayerBinding] = {}
    for player_id, raw_binding in cast("dict[str, Any]", raw_player_bindings).items():
        if not isinstance(raw_binding, dict):
            raise LiveConfigError(f"config player binding {player_id!r} must be an object")
        binding = cast("dict[str, Any]", raw_binding)
        kind = binding.get("kind")
        if kind not in ("external", "builtin-agent"):
            raise LiveConfigError(
                f"config player binding {player_id!r} has kind {kind!r}; expected "
                "'external' or 'builtin-agent'"
            )
        path = binding.get("path")
        if path is not None and not isinstance(path, str):
            raise LiveConfigError(f"config player binding {player_id!r} 'path' must be a string when present")
        name = binding.get("name")
        if kind == "external":
            if "name" in binding:
                raise LiveConfigError(f"config external binding {player_id!r} must not name an agent")
        else:
            if name is not None and (not isinstance(name, str) or _BUILTIN_NAME.fullmatch(name) is None):
                raise LiveConfigError(
                    f"config player binding {player_id!r} 'name' must be a snake_case built-in name"
                )
            if path is None and name is None:
                raise LiveConfigError(
                    f"config built-in binding {player_id!r} without a path must name a built-in agent"
                )
            declared_names = {builtin.name for builtin in entry.meta.builtin_agents}
            if name is not None and name not in declared_names:
                raise LiveConfigError(
                    f"config built-in binding {player_id!r} names {name!r}, "
                    f"which environment {env_id!r} does not declare"
                )
        player_bindings[player_id] = PlayerBinding(kind=kind, path=path, name=cast("str | None", name))

    if entry.meta.stepping == "simultaneous" and "human_timeout_ms" in config:
        raise LiveConfigError("simultaneous environments must not supply 'human_timeout_ms'")
    if "human_timeout_ms" not in config:
        human_timeout_ms: int | None | UnsetTimeout = UNSET_TIMEOUT
    else:
        human_timeout_ms = config["human_timeout_ms"]
        if human_timeout_ms is not None and (
            not isinstance(human_timeout_ms, int) or isinstance(human_timeout_ms, bool)
        ):
            raise LiveConfigError("config 'human_timeout_ms' must be an integer or null")

    recording_dir = config.get("recording_dir")
    if not isinstance(recording_dir, str) or not recording_dir:
        raise LiveConfigError("config 'recording_dir' must be a non-empty string")

    recording_id = config.get("recording_id")
    if recording_id is not None and not isinstance(recording_id, str):
        raise LiveConfigError("config 'recording_id' must be a string or null")

    if "parameters" not in config:
        raise LiveConfigError("config 'parameters' must be an object")
    raw_parameters = config["parameters"]
    parameter_items = cast("dict[object, object]", raw_parameters)
    if not isinstance(raw_parameters, dict) or not all(
        isinstance(name, str) and _is_parameter_value(value) for name, value in parameter_items.items()
    ):
        raise LiveConfigError(
            "config 'parameters' must contain booleans, finite numbers, strings, or string lists"
        )
    try:
        parameters = validate_complete_parameters(
            entry.meta, cast("dict[str, ParameterValue]", raw_parameters)
        )
        layout = resolve_layout(entry.meta, parameters)
    except ValueError as error:
        raise LiveConfigError(
            f"config 'parameters' do not resolve for environment {env_id!r}: {error}"
        ) from error
    expected_players = {player_id for seat in layout.seats for player_id in seat.players}
    _validate_player_ids("player_bindings", set(player_bindings), expected_players)

    external_chat_player = config.get("external_chat_player")
    if external_chat_player is not None and not isinstance(external_chat_player, str):
        raise LiveConfigError("config 'external_chat_player' must be a player id string or null")
    if external_chat_player is not None:
        binding = player_bindings.get(external_chat_player)
        if binding is None or binding.kind != "external":
            raise LiveConfigError("config 'external_chat_player' must name an external player binding")

    players = _parse_players(config.get("players"), player_bindings, expected_players)
    step_timeout_ms = _parse_optional_int(config, "step_timeout_ms")
    episode_timeout_ms = _parse_optional_int(config, "episode_timeout_ms")
    message_cap = _parse_optional_int(config, "message_cap")
    max_steps = _parse_max_steps(config.get("max_steps"))
    messaging_enabled = config.get("messaging_enabled")
    if messaging_enabled is not None and not isinstance(messaging_enabled, bool):
        raise LiveConfigError("config 'messaging_enabled' must be a boolean or null")
    headless = config.get("headless", False)
    if not isinstance(headless, bool):
        raise LiveConfigError("config 'headless' must be a boolean")
    start_paused = config.get("start_paused", False)
    if not isinstance(start_paused, bool):
        raise LiveConfigError("config 'start_paused' must be a boolean")
    llm = _parse_llm(config.get("llm"), player_bindings)

    return LiveConfig(
        env_id=env_id,
        seed=seed,
        player_bindings=player_bindings,
        human_timeout_ms=human_timeout_ms,
        recording_dir=recording_dir,
        recording_id=recording_id,
        parameters=parameters,
        layout=layout,
        players=players,
        step_timeout_ms=step_timeout_ms,
        episode_timeout_ms=episode_timeout_ms,
        messaging_enabled=messaging_enabled,
        message_cap=message_cap,
        external_chat_player=external_chat_player,
        headless=headless,
        llm=llm,
        start_paused=start_paused,
        max_steps=max_steps,
    )


def _is_parameter_value(value: object) -> bool:
    """Check the shallow JSON shape before environment-aware resolution in ``Episode``."""
    if isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return isinstance(value, list) and all(isinstance(item, str) for item in cast("list[object]", value))


def _parse_optional_int(config: dict[str, Any], key: str) -> int | None:
    """Validate an optional positive-integer config value, defaulting to ``None`` when absent."""
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise LiveConfigError(f"config {key!r} must be an integer or null")
    return value


def _parse_max_steps(value: object) -> int | None:
    """Parse an optional positive local-play step cap."""
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LiveConfigError("config 'max_steps' must be a positive integer or null")
    return value


def _validate_player_ids(name: str, actual: set[str], expected: set[str]) -> None:
    """Require one config map to exactly cover the layout's canonical player ids."""
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing players {missing!r}")
    if unknown:
        details.append(f"unknown players {unknown!r}")
    raise LiveConfigError(f"config {name!r} must exactly cover resolved players ({'; '.join(details)})")


def _parse_players(
    raw: object,
    player_bindings: dict[str, PlayerBinding],
    expected_players: set[str],
) -> dict[str, PlayerAttribution]:
    """Validate the required, binding-aligned per-player ``players`` attribution map.

    Each entry must name a ``kind`` of ``human`` or ``agent`` and a non-empty ``label``; ``user``
    and ``submission_id`` are optional strings. The validated entries are passed through verbatim
    into the recording header, so the harness keeps no opinion on their meaning beyond being
    well-formed.
    """
    if not isinstance(raw, dict):
        raise LiveConfigError("config 'players' must be an object keyed by player id")
    raw_players = cast("dict[str, object]", raw)
    _validate_player_ids("players", set(raw_players), expected_players)
    players: dict[str, PlayerAttribution] = {}
    for player_id, raw_entry in raw_players.items():
        if not isinstance(raw_entry, dict):
            raise LiveConfigError(f"config player {player_id!r} must be an object")
        entry = cast("dict[str, object]", raw_entry)
        kind = entry.get("kind")
        if kind not in ("human", "agent"):
            raise LiveConfigError(
                f"config player {player_id!r} has kind {kind!r}; expected 'human' or 'agent'"
            )
        label = entry.get("label")
        if not isinstance(label, str) or not label:
            raise LiveConfigError(f"config player {player_id!r} 'label' must be a non-empty string")
        if set(entry) - {"kind", "label", "user", "submission_id", "builtin_name"}:
            raise LiveConfigError(f"config player {player_id!r} contains an unknown attribution field")
        for optional in ("user", "submission_id", "builtin_name"):
            value = entry.get(optional)
            if optional in entry and not isinstance(value, str):
                raise LiveConfigError(
                    f"config player {player_id!r} {optional!r} must be a string when present"
                )
        expected_kind = "human" if player_bindings[player_id].kind == "external" else "agent"
        if kind != expected_kind:
            raise LiveConfigError(
                f"config player {player_id!r} has kind {kind!r}; expected {expected_kind!r} for its binding"
            )
        submission_id = entry.get("submission_id")
        builtin_name = entry.get("builtin_name")
        if kind == "human":
            if submission_id is not None or builtin_name is not None:
                raise LiveConfigError(f"config human player {player_id!r} cannot carry agent identity")
        elif (submission_id is None) == (builtin_name is None):
            raise LiveConfigError(
                f"config agent player {player_id!r} must carry exactly one of submission_id or builtin_name"
            )
        elif submission_id == "":
            raise LiveConfigError(f"config player {player_id!r} submission_id must be non-empty")
        elif builtin_name is not None and _BUILTIN_NAME.fullmatch(cast("str", builtin_name)) is None:
            raise LiveConfigError(
                f"config player {player_id!r} builtin_name must be a snake_case built-in name"
            )
        elif builtin_name is not None and "user" in entry:
            raise LiveConfigError(f"config built-in agent player {player_id!r} cannot carry user")
        elif player_bindings[player_id].name is None and builtin_name is not None:
            raise LiveConfigError(
                f"config player {player_id!r} with a path-only binding must carry submission_id"
            )
        elif player_bindings[player_id].name is not None and builtin_name != player_bindings[player_id].name:
            raise LiveConfigError(
                f"config player {player_id!r} builtin_name must match its named built-in binding"
            )
        players[player_id] = cast("PlayerAttribution", entry)
    return players


def _parse_llm(raw: object, player_bindings: dict[str, PlayerBinding]) -> LlmConfig | None:
    """Validate the optional LLM launch block and its agent-player key coverage.

    Keys arrive either inline (``keys`` — the local-play path) or out-of-band through a
    read-only mounted file (``keys_file`` — the container path, which keeps the key material out
    of the container's argv/process metadata). Exactly one of the two must be present.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise LiveConfigError("config 'llm' must be an object or null")
    llm = cast("dict[str, Any]", raw)
    required_fields = {"base_url", "tick_url", "inflight_url"}
    known_fields = required_fields | {"keys", "keys_file"}
    if not required_fields <= set(llm) or not set(llm) <= known_fields:
        missing = sorted(required_fields - set(llm))
        unknown = sorted(set(llm) - known_fields)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing!r}")
        if unknown:
            details.append(f"unknown {unknown!r}")
        raise LiveConfigError(
            "config 'llm' must contain base_url, tick_url, inflight_url, and exactly one of "
            f"'keys' or 'keys_file' ({'; '.join(details)})"
        )

    for field_name in ("base_url", "tick_url", "inflight_url"):
        value = llm[field_name]
        if not isinstance(value, str) or not value:
            raise LiveConfigError(f"config 'llm' {field_name!r} must be a non-empty string")

    has_keys = "keys" in llm
    has_keys_file = "keys_file" in llm
    if has_keys == has_keys_file:
        raise LiveConfigError(
            "config 'llm' must provide exactly one of 'keys' or 'keys_file' "
            f"(currently {'both' if has_keys else 'neither'})"
        )
    if has_keys_file:
        keys = _read_keys_file(llm["keys_file"])
    else:
        raw_keys = llm["keys"]
        if not isinstance(raw_keys, dict):
            raise LiveConfigError("config 'llm' 'keys' must be an object keyed by agent player id")
        keys = cast("dict[str, Any]", raw_keys)
        for player_id, key in keys.items():
            if not isinstance(key, str) or not key:
                raise LiveConfigError(f"config 'llm' key for {player_id!r} must be a non-empty string")

    agent_players = {
        player_id for player_id, binding in player_bindings.items() if binding.kind == "builtin-agent"
    }
    supplied_players = set(keys)
    if supplied_players != agent_players:
        missing = sorted(agent_players - supplied_players)
        unknown = sorted(supplied_players - agent_players)
        details: list[str] = []
        if missing:
            details.append(f"missing agent players {missing!r}")
        if unknown:
            details.append(f"unknown or non-agent players {unknown!r}")
        raise LiveConfigError(
            f"config 'llm' keys must exactly cover configured agent players ({'; '.join(details)})"
        )

    return LlmConfig(
        base_url=cast("str", llm["base_url"]),
        tick_url=cast("str", llm["tick_url"]),
        inflight_url=cast("str", llm["inflight_url"]),
        keys=keys,
    )


def _read_keys_file(path: object) -> dict[str, str]:
    """Read a per-session LLM keys file (the out-of-band container handoff) into a player-id map."""
    if not isinstance(path, str) or not path:
        raise LiveConfigError("config 'llm' 'keys_file' must be a non-empty path")
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except OSError as error:
        raise LiveConfigError(f"config 'llm' 'keys_file' could not be read: {error}") from error
    except json.JSONDecodeError as error:
        raise LiveConfigError("config 'llm' 'keys_file' must contain a JSON object") from error
    if not isinstance(raw, dict):
        raise LiveConfigError("config 'llm' 'keys_file' must contain a JSON object")
    raw_keys = cast("dict[object, object]", raw)
    keys: dict[str, str] = {}
    for player_id, key in raw_keys.items():
        if not isinstance(player_id, str) or not isinstance(key, str) or not key:
            raise LiveConfigError("config 'llm' 'keys_file' must map player ids to non-empty key strings")
        keys[player_id] = key
    return keys


class _LlmExecutionScope:
    """Select a player credential and best-effort marker immediately before participant work."""

    def __init__(self, config: LlmConfig) -> None:
        self._config = config
        # The backend retains one marker per bearer key. Keep the last successful post for each
        # player so interleaved action, chat, and learning hooks do not repeat the same boundary.
        self._posted_markers: dict[str, dict[str, str | int]] = {}

    def setup(self, player_id: str) -> None:
        self._activate(player_id)
        self._post_marker(player_id, {"phase": "setup"})

    def turn(self, player_id: str, tick: int) -> None:
        self._activate(player_id)
        self._post_marker(player_id, {"tick": tick})

    def inflight_ms(self, player_id: str) -> int | None:
        """Read the player's proxy time, including a capped active partial, without making it fatal."""
        try:
            request = urllib.request.Request(
                self._config.inflight_url,
                headers={"Authorization": f"Bearer {self._config.keys[player_id]}"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=_MARKER_TIMEOUT_SECONDS) as response:
                raw_payload: object = json.loads(response.read())
            if not isinstance(raw_payload, dict):
                raise ValueError("response must contain one non-negative integer inflight_ms")
            payload = cast("dict[str, object]", raw_payload)
            inflight_ms = payload.get("inflight_ms")
            if (
                set(payload) != {"inflight_ms"}
                or not isinstance(inflight_ms, int)
                or isinstance(inflight_ms, bool)
                or inflight_ms < 0
            ):
                raise ValueError("response must contain one non-negative integer inflight_ms")
            return inflight_ms
        except Exception as error:  # noqa: BLE001 - timing discount must never stop agent lifecycle
            print(
                f"live: LLM in-flight snapshot failed for player {player_id!r}: {error}",
                file=sys.stderr,
                flush=True,
            )
            return None

    def _activate(self, player_id: str) -> None:
        os.environ["OPENAI_BASE_URL"] = self._config.base_url
        os.environ["OPENAI_API_KEY"] = self._config.keys[player_id]

    def _post_marker(self, player_id: str, payload: dict[str, str | int]) -> None:
        if self._posted_markers.get(player_id) == payload:
            return
        try:
            request = urllib.request.Request(
                self._config.tick_url,
                data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self._config.keys[player_id]}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            # The response body is unused. Closing after the headers arrive avoids draining a slow
            # body beyond the local marker timeout.
            with urllib.request.urlopen(request, timeout=_MARKER_TIMEOUT_SECONDS):
                pass
            self._posted_markers[player_id] = payload
        except Exception as error:  # noqa: BLE001 - marker telemetry is deliberately best-effort
            print(
                f"live: LLM marker failed for player {player_id!r}: {error}",
                file=sys.stderr,
                flush=True,
            )


def build_players(
    config: LiveConfig,
    entry: EnvironmentEntry,
    control: SessionControl,
    clock: PausableClock,
    sleeper: Sleeper,
) -> dict[str, Player]:
    """Bind every configured player to a harness :class:`Player`.

    External players get a :class:`TransportSource` over the command pump and carry the resolved
    human-player timeout — the config override when given, otherwise the environment metadata
    default. Built-in-agent players are loaded through the same manifest loader Stage 5 uses for
    submissions, from ``path`` or the image's default built-in agent location.
    """
    paced = not config.headless and entry.meta.pace_interval_ms is not None
    configured_timeout = config.human_timeout_ms
    resolved_timeout = (
        entry.meta.human_timeout_ms if isinstance(configured_timeout, UnsetTimeout) else configured_timeout
    )
    players: dict[str, Player] = {}
    execution_scope = _LlmExecutionScope(config.llm) if config.llm is not None else None
    for player_id, binding in config.player_bindings.items():
        if binding.kind == "external":
            source = TransportSource(control, clock=clock, paced=paced, sleeper=sleeper)
            players[player_id] = ExternalPlayer(
                source,
                timeout_ms=resolved_timeout,
                message_source=source,
            )
        else:  # "builtin-agent" — parse_config rejects any other kind.
            if binding.path is not None:
                agent_path = binding.path
            else:
                assert binding.name is not None
                agent_path = f"{DEFAULT_BUILTIN_AGENT_BASE}/{config.env_id}/{binding.name}"
            if execution_scope is not None:
                # Manifest loading imports the participant module and constructs its agent, so this
                # boundary must be activated before either operation can capture a client.
                execution_scope.setup(player_id)
            agent = load_agent(agent_path)
            players[player_id] = AgentPlayer(agent, execution_scope=execution_scope)
    return players


def run_live_loop(
    episode: Episode,
    *,
    pace_interval_ms: int | None,
    control: SessionControl,
    clock: PausableClock,
    sleeper: Sleeper,
    slice_ms: int = _SLICE_MS,
) -> None:
    """Drive ``episode`` to its end under pacing, pause, and stop.

    A thin loop over :meth:`Episode.advance`. Sequential environments retain their target-based
    scheduler. Both simultaneous modes wait cooperatively while paused. A paced simultaneous session
    starts ticks on a fixed cadence one interval apart. A tick that completes at or past its next
    boundary restarts the cadence one interval after that completion, so no catch-up tick ever runs.
    Headless mode advances back to back. A ``stop`` command ends the run with reason ``stopped``.
    """
    if episode.stepping == "simultaneous":
        # Only read when paced: the first tick waits one full input window before advancing.
        next_instant = clock.now_ms() + (pace_interval_ms or 0)
        while not episode.done:
            while control.paused and not control.stopping:
                sleeper.sleep_ms(slice_ms)
            if pace_interval_ms is not None:
                while clock.now_ms() < next_instant and not control.stopping:
                    sleeper.sleep_ms(slice_ms)
            if control.stopping:
                episode.stop(REASON_STOPPED)
                break
            episode.advance()
            if pace_interval_ms is not None:
                # Tick work and sleep overshoot must not stretch the period, or every frame lands late.
                next_instant += pace_interval_ms
                completed = clock.now_ms()
                if completed >= next_instant:
                    # A slow tick or a late wakeup passed the boundary. Restart the cadence instead of
                    # running a catch-up tick at once.
                    next_instant = completed + pace_interval_ms
        return

    # Preserve the original AEC target-based scheduler byte-for-byte apart from dispatching through
    # Episode.advance(), which selects step_once() for every sequential declaration.
    next_instant = clock.now_ms()
    while not episode.done:
        # Pause is a cooperative wait shared by both cadences; the frozen clock does the rest.
        while control.paused and not control.stopping:
            sleeper.sleep_ms(slice_ms)
        if pace_interval_ms is not None:
            next_instant += pace_interval_ms
            while clock.now_ms() < next_instant and not control.stopping:
                sleeper.sleep_ms(slice_ms)
        if control.stopping:
            episode.stop(REASON_STOPPED)
            break
        episode.advance()


def _claim_stdout() -> IO[str]:
    """Hand the real stdout to the protocol writer and redirect fd 1 onto stderr.

    Protocol lines must own stdout, while imported game or agent code may print to it. Done as the
    first thing in :func:`main`, before any environment import: ``dup`` the real stdout for the
    protocol writer, then ``dup2`` stderr over fd 1 so later diagnostic chatter lands in stderr
    instead of corrupting the protocol.
    """
    sys.stdout.flush()
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    return os.fdopen(protocol_fd, "w", encoding="utf-8", newline="\n")


def run(
    entry: EnvironmentEntry,
    config: LiveConfig,
    *,
    protocol: ProtocolStream,
    control: SessionControl,
    clock: PausableClock,
    sleeper: Sleeper,
    store: RecordingStore,
    command_lines: Iterable[str] | None = None,
) -> int:
    """Run one injected live episode without discovery or stdout ownership.

    When ``command_lines`` is supplied, its blocking command pump starts after :class:`Episode`
    has entered, emitted the recording header, and streamed any live-only opening state, but before
    the live loop begins. Keeping that boundary here leaves the loop directly testable without a
    stdin reader and lets local clients attach before Windows starts the blocking reader thread.
    """
    if config.start_paused:
        control.pause()
    episode: Episode | None = None
    try:
        players = build_players(config, entry, control, clock, sleeper)
        episode = Episode(
            entry,
            players,
            seed=config.seed,
            store=store,
            recording_id=config.recording_id,
            clock=clock,
            step_limit_ms=config.step_timeout_ms,
            episode_limit_ms=config.episode_timeout_ms,
            player_attribution=config.players,
            messaging=config.messaging_enabled,
            message_cap=config.message_cap,
            max_steps=config.max_steps,
            parameters=config.parameters,
            layout=config.layout,
            external_chat_player=config.external_chat_player,
        )
        # The effective messaging decision (metadata AND config) is resolved once inside the episode;
        # reuse it to gate the human chat queue, so a frame is accepted only when the loop will route
        # it. The command pump starts only after this gate has been configured.
        control.configure_chat(episode.external_chat_sender)
        with episode:
            # Stream the opening deal frame (turn-based envs only) so a human who must act first sees
            # the table before the loop blocks for their move. It is streamed, never recorded.
            if not config.headless:
                opening = episode.opening_state()
                if opening is not None:
                    protocol.emit_state(opening)
            if command_lines is not None:
                start_command_pump(command_lines, control)
            run_live_loop(
                episode,
                pace_interval_ms=None if config.headless else entry.meta.pace_interval_ms,
                control=control,
                clock=clock,
                sleeper=sleeper,
            )
        protocol.emit_envelope(result_envelope(episode.result()))
    except Exception as error:  # noqa: BLE001 - surfaced to diagnostics; the orchestrator records it
        print(f"live: session failed: {error!r}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        # Emit the partial result, marked as an error with a short description of the exception, so the
        # orchestrator stores why the game ended and can charge a crashing agent to its own seat
        # (episode.failed_player) instead of to every competitor sharing the container. Best-effort: this
        # advisory note must never mask the original error or change the exit code. The close is
        # idempotent belt-and-suspenders (Episode.start and the `with` already flush the recording on
        # their own failures) and guarantees the writer is closed before result() reads it back.
        if episode is not None:
            with contextlib.suppress(Exception):
                episode.fail(describe_failure(error))
            with contextlib.suppress(Exception):
                episode.close()
            with contextlib.suppress(Exception):
                protocol.emit_envelope(result_envelope(episode.result()))
        return 1
    return 0


def describe_failure(error: BaseException) -> str:
    """Describe an exception in one line: ``"<Type>: <message> (<file>:<line> in <function>)"``.

    The location is the innermost traceback frame outside the harness package, which for an agent crash
    is the line in the agent's own code. Only the file's base name is kept, so no server path leaks into
    a public replay. The location is omitted when every frame belongs to the harness (an illegal action,
    whose message already names the player and the problem).
    """
    message = " ".join(str(error).split())
    if len(message) > _FAILURE_MESSAGE_LIMIT:
        message = message[: _FAILURE_MESSAGE_LIMIT - 3] + "..."
    description = f"{type(error).__name__}: {message}" if message else type(error).__name__
    location = None
    for frame in traceback.extract_tb(error.__traceback__):
        if not Path(frame.filename).resolve().is_relative_to(_HARNESS_PACKAGE_DIR):
            location = frame
    if location is not None:
        description += f" ({Path(location.filename).name}:{location.lineno} in {location.name})"
    return description


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Claim stdout before anything can import a game and print to it.
    protocol = ProtocolStream(_claim_stdout())
    try:
        config = parse_config(argv)
    except LiveConfigError as error:
        print(f"live: invalid config: {error}", file=sys.stderr, flush=True)
        return 2

    clock = PausableClock(SystemClock())
    control = SessionControl(clock)
    sleeper = RealSleeper()
    try:
        entry = load_environment(config.env_id)
    except Exception as error:  # noqa: BLE001 - keeps startup failures on diagnostics like run failures
        print(f"live: session failed: {error!r}", file=sys.stderr, flush=True)
        return 1
    return run(
        entry,
        config,
        protocol=protocol,
        control=control,
        clock=clock,
        sleeper=sleeper,
        store=build_tee_store(config.recording_dir, protocol),
        command_lines=sys.stdin,
    )


if __name__ == "__main__":
    sys.exit(main())
