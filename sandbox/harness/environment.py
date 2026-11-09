"""Public-facing environment metadata and the registry entry.

These types live in the harness, not the environments package, because the harness loop and
the Stage 3 container consume them while the environments package already depends on the
harness; putting them the other way round would be an import cycle. The harness itself never
imports the environments package or PettingZoo — environments are discovered through Python
entry points and their AEC or parallel envs are used duck-typed (hence the ``Any`` factory return).

:class:`EnvironmentMeta` is pure, serialisable data — the layer the backend serves to the
frontend verbatim. :class:`EnvironmentEntry` adds the non-serialisable hooks (the factory,
the default-action provider, the overlay extractor) and is what an environment registers.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import InitVar, dataclass
from importlib.metadata import entry_points
from typing import Any, Literal, Protocol, TypeGuard, cast, runtime_checkable

#: The entry-point group every environment registers its ``ENTRY`` under.
ENTRY_POINT_GROUP = "game_sandbox.environments"

# JSON numbers lose integer precision beyond this range in JavaScript, which is part of every
# environment parameter's public contract.
_MAX_JSON_SAFE_INTEGER = 2**53 - 1
_PARAMETER_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_RESERVED_PARAMETER_NAMES = frozenset({"players", "seat_plan"})

type ParameterValue = bool | int | float | str | list[str]
"""A JSON-safe environment parameter value."""

type SteppingMode = Literal["sequential", "simultaneous"]
"""The PettingZoo stepping contract an environment declares."""

type HumanPause = Literal["session", "playback"]
"""What the browser's Pause button does for a human session of the environment."""


class EnvParameterValueError(ValueError):
    """Raised when a parameter value does not satisfy its declaration."""


class EnvironmentContractError(ValueError):
    """Raised when a constructed environment contradicts its declared contract."""

    def __init__(self, environment_id: str, stepping: SteppingMode, fact: str) -> None:
        self.environment_id = environment_id
        self.stepping = stepping
        self.fact = fact
        super().__init__(f"environment {environment_id!r} declares {stepping} stepping, but {fact}")


@dataclass(frozen=True)
class EnvParameterChoice:
    """One friendly-labelled option for a choice parameter."""

    value: str
    label: str

    def __post_init__(self) -> None:
        if not _is_nonempty_string(self.value):
            raise ValueError("parameter choice value must be a non-empty string")
        if not _is_nonempty_string(self.label):
            raise ValueError("parameter choice label must be a non-empty string")
        if _has_blank_line(self.label):
            raise ValueError("parameter choice label must not contain a blank line")

    def to_json(self) -> dict[str, str]:
        """Return the public wire representation."""
        return {"value": self.value, "label": self.label}


@dataclass(frozen=True)
class EnvParameter:
    """A typed, player-facing gameplay parameter declared by an environment."""

    name: str
    title: str
    description: str
    type: Literal["int", "float", "string", "bool", "choice", "multi_choice"]
    default: ParameterValue
    min: int | float | None = None
    max: int | float | None = None
    choices: tuple[EnvParameterChoice, ...] = ()
    _allow_reserved: InitVar[bool] = False

    def __post_init__(self, _allow_reserved: bool) -> None:
        if not _is_parameter_name(self.name):
            raise ValueError("parameter name must be a snake_case identifier")
        if self.name in _RESERVED_PARAMETER_NAMES and not _allow_reserved:
            raise ValueError(f"{self.name!r} is reserved for the synthesized layout parameter")
        if not _is_nonempty_string(self.title):
            raise ValueError("parameter title must be a non-empty string")
        if _has_blank_line(self.title):
            raise ValueError("parameter title must not contain a blank line")
        if not _is_nonempty_string(self.description):
            raise ValueError("parameter description must be a non-empty string")
        if self.type not in {"int", "float", "string", "bool", "choice", "multi_choice"}:
            raise ValueError(f"unsupported parameter type {self.type!r}")

        numeric = self.type in {"int", "float"}
        if numeric:
            if self.min is None or self.max is None:
                raise ValueError(f"{self.type} parameters require min and max")
            self._validate_numeric_bound(self.min, "min")
            self._validate_numeric_bound(self.max, "max")
            if self.min > self.max:
                raise ValueError("parameter min must be no greater than max")
        elif self.min is not None or self.max is not None:
            raise ValueError("only numeric parameters may declare min or max")

        if self.type in {"choice", "multi_choice"}:
            if not self.choices:
                raise ValueError("choice parameters require at least one choice")
            if not all(_is_parameter_choice(choice) for choice in self.choices):
                raise ValueError("parameter choices must be EnvParameterChoice instances")
            values = [choice.value for choice in self.choices]
            if len(values) != len(set(values)):
                raise ValueError("parameter choice values must be unique")
        elif self.choices:
            raise ValueError("only choice parameters may declare choices")

        # Validate the declaration eagerly, including bounds and choice membership.
        self.validate_value(self.default)

    def _validate_numeric_bound(self, value: int | float, field: str) -> None:
        if self.type == "int":
            if not is_json_safe_integer(value):
                raise ValueError(f"int parameter {field} must be a JSON-safe integer")
        elif not _is_finite_number(value):
            raise ValueError(f"float parameter {field} must be finite")

    def validate_value(self, value: object) -> ParameterValue:
        """Validate and normalize one supplied value for this declaration."""
        if self.type == "int":
            if not is_json_safe_integer(value):
                raise EnvParameterValueError(f"{self.name} must be a JSON-safe integer")
            assert isinstance(self.min, int) and isinstance(self.max, int)
            if not self.min <= value <= self.max:
                raise EnvParameterValueError(f"{self.name} must be between {self.min} and {self.max}")
            return value
        if self.type == "float":
            if not _is_finite_number(value):
                raise EnvParameterValueError(f"{self.name} must be a finite number")
            assert self.min is not None and self.max is not None
            normalized = float(value)
            if not self.min <= normalized <= self.max:
                raise EnvParameterValueError(f"{self.name} must be between {self.min} and {self.max}")
            return normalized
        if self.type == "string":
            if not isinstance(value, str):
                raise EnvParameterValueError(f"{self.name} must be a string")
            return value
        if self.type == "bool":
            if not isinstance(value, bool):
                raise EnvParameterValueError(f"{self.name} must be a boolean")
            return value
        values = {choice.value for choice in self.choices}
        if self.type == "choice":
            if not isinstance(value, str) or value not in values:
                raise EnvParameterValueError(f"{self.name} must be a declared choice value")
            return value
        if not _is_string_list(value):
            raise EnvParameterValueError(f"{self.name} must be a list of declared choice values")
        if len(value) != len(set(value)) or any(item not in values for item in value):
            raise EnvParameterValueError(f"{self.name} must be unique declared choice values")
        return [choice.value for choice in self.choices if choice.value in value]

    def to_json(self) -> dict[str, Any]:
        """Return the snake_case wire representation served by the registry.

        Only the keys a declaration's type actually uses are emitted. ``__post_init__`` already rejects
        bounds on a non-numeric type and choices on a non-choice type, so serialising them as ``null``
        and ``[]`` would put fields on the wire that the declaration is not allowed to have, and that
        the consuming TypeScript type does not model.
        """
        payload: dict[str, Any] = {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "type": self.type,
            "default": self.validate_value(self.default),
        }
        if self.type in {"int", "float"}:
            payload["min"] = self.min
            payload["max"] = self.max
        if self.type in {"choice", "multi_choice"}:
            payload["choices"] = [choice.to_json() for choice in self.choices]
        return payload


@dataclass(frozen=True)
class EnvPreset:
    """One named partial parameter configuration an environment recommends.

    ``llm`` records whether the season this preset stands up should enable the LLM API. It
    defaults to ``False`` and is validated against the environment's own ``llm`` flag in
    ``EnvironmentMeta.__post_init__``, so a template can never ask for LLM access the
    environment cannot serve.

    ``example`` optionally names one of the environment's published worked examples. A season
    seeded from this preset gives students that example's template branch as their starting
    point. The template generator checks that the example is published.
    """

    name: str
    title: str
    values: Mapping[str, ParameterValue]
    llm: bool = False
    example: str | None = None

    def __post_init__(self) -> None:
        if not _is_parameter_name(self.name):
            raise ValueError("preset name must be a snake_case identifier")
        if not _is_nonempty_string(self.title):
            raise ValueError("preset title must be a non-empty string")
        if _has_blank_line(self.title):
            raise ValueError("preset title must not contain a blank line")
        if not isinstance(cast("object", self.values), Mapping):
            raise ValueError("preset values must be a parameter-value mapping")
        if not isinstance(cast("object", self.llm), bool):
            raise ValueError("preset llm must be a bool")
        if self.example is not None and not _is_nonempty_string(self.example):
            raise ValueError("preset example must be a non-empty string")

    def to_json(self) -> dict[str, Any]:
        """Return the public wire representation."""
        payload: dict[str, Any] = {
            "name": self.name,
            "title": self.title,
            "values": dict(self.values),
            "llm": self.llm,
        }
        if self.example is not None:
            payload["example"] = self.example
        return payload


def _is_nonempty_string(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value)


def _has_blank_line(value: str) -> bool:
    """Whether a short human-facing string would split a description into paragraphs.

    The backend's generated season descriptions (the template and Playground prose) must stay one
    paragraph, so any of the strings they embed, such as a preset title or a parameter title, must
    not contain a blank line. Matches the API's ``multiple_paragraphs`` rule (`\\n[ \\t]*\\n`).
    """
    return re.search(r"\n[ \t]*\n", value) is not None


def _is_parameter_name(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and _PARAMETER_NAME.fullmatch(value) is not None


def _is_parameter_choice(value: object) -> TypeGuard[EnvParameterChoice]:
    return isinstance(value, EnvParameterChoice)


def _is_string_list(value: object) -> TypeGuard[list[str]]:
    return isinstance(value, list) and all(isinstance(item, str) for item in cast("list[object]", value))


def is_json_safe_integer(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and abs(value) <= _MAX_JSON_SAFE_INTEGER


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


@dataclass(frozen=True)
class PlayerBounds:
    """A player-count range where every player receives one assignable seat."""

    min: int
    max: int


@dataclass(frozen=True)
class BuiltinAgent:
    """One named built-in agent an environment stages and exposes."""

    name: str
    label: str

    def __post_init__(self) -> None:
        if not _is_parameter_name(self.name):
            raise ValueError("builtin agent name must be a snake_case identifier")
        if not _is_nonempty_string(self.label):
            raise ValueError("builtin agent label must be a non-empty string")

    def to_json(self) -> dict[str, str]:
        """Return the public wire representation."""
        return {"name": self.name, "label": self.label}


@dataclass(frozen=True)
class SeatDeclaration:
    """One declared seat, its player indexes, and any designated built-in agent."""

    players: tuple[int, ...]
    restricted_builtin: str | None = None


@dataclass(frozen=True)
class SeatPlan:
    """One named, complete assignment of PettingZoo players to seats."""

    key: str
    title: str
    seats: tuple[SeatDeclaration, ...]


@dataclass(frozen=True)
class SeatPlans:
    """The ordered layouts an environment may select through ``seat_plan``."""

    plans: tuple[SeatPlan, ...]


type EnvironmentLayout = PlayerBounds | SeatPlans


@dataclass(frozen=True)
class ResolvedSeat:
    """One canonical seat and its ordered PettingZoo player members."""

    seat_id: str
    players: tuple[str, ...]
    restricted_builtin: str | None = None


@dataclass(frozen=True)
class ResolvedLayout:
    """The complete, canonical seat-to-player layout for resolved parameters."""

    plan_key: str
    seats: tuple[ResolvedSeat, ...]
    player_count: int
    seat_count: int

    @property
    def players(self) -> tuple[str, ...]:
        """Every resolved player id in canonical order, however the seats group them."""
        return canonical_player_order(player for seat in self.seats for player in seat.players)


def canonical_player_order(player_ids: Iterable[str]) -> tuple[str, ...]:
    """Sort player ids into their canonical numeric order.

    :func:`resolve_layout` names every player ``player_<index>``, but callers reach them through
    orderings that do not preserve that numbering: a mapping's key order, or a wide seat plan whose
    seats group non-adjacent players. Anything that presents players to a human or an environment
    sorts through here so one convention is applied in one place.
    """
    return tuple(sorted(player_ids, key=lambda player_id: int(player_id.removeprefix("player_"))))


@dataclass(frozen=True)
class ChatPolicy:
    """One acting player's live direct-message choices.

    ``target_recipients`` is ordered for presentation and limits direct messages only. Broadcast is
    always allowed. ``default_recipient`` is ``None`` for broadcast or one member of that tuple. The
    harness validates an environment hook's returned policy against the resolved players before
    using it.
    """

    target_recipients: tuple[str, ...]
    default_recipient: str | None


@runtime_checkable
class ChatPolicySource(Protocol):
    """A running environment that supplies live direct-message choices."""

    def chat_policy(self, sender: str) -> object: ...


@runtime_checkable
class BroadcastRecipientsSource(Protocol):
    """A running environment that supplies the live audience for one broadcast."""

    def broadcast_recipients(self, sender: str) -> object: ...


def _player_count_parameter(bounds: PlayerBounds) -> EnvParameter:
    """Build the reserved declaration derived from player bounds."""
    return EnvParameter(
        name="players",
        title="Players",
        description="Number of PettingZoo players in each game.",
        type="int",
        default=bounds.max,
        min=bounds.min,
        max=bounds.max,
        _allow_reserved=True,
    )


def _seat_plan_parameter(layout: SeatPlans) -> EnvParameter:
    """Build the reserved declaration selecting one declared seat plan."""
    return EnvParameter(
        name="seat_plan",
        title="Seat plan",
        description="Seat-to-player layout for each game.",
        type="choice",
        default=layout.plans[0].key,
        choices=tuple(EnvParameterChoice(plan.key, plan.title) for plan in layout.plans),
        _allow_reserved=True,
    )


@dataclass(frozen=True)
class EnvironmentMeta:
    """The serialisable public-facing metadata for one environment.

    Field-for-field the layer described in the environment spec. ``human_timeout_ms`` is
    ``None`` when a ``pace_interval_ms`` is set, because a set pace interval is itself the
    human deadline; ``pace_interval_ms`` is ``None`` for turn-based environments.
    """

    env_id: str
    display_name: str
    description: str
    stepping: SteppingMode
    builtin_agents: tuple[BuiltinAgent, ...]
    layout: EnvironmentLayout
    human_players: tuple[str, ...]
    human_timeout_ms: int | None
    recommended_episode_ticks: int
    pace_interval_ms: int | None
    step_limit_ms: int
    episode_limit_ms: int
    messaging: bool
    message_cap: int | None
    llm: bool
    renderer: str
    #: Whether two agents swapping seats produce a genuinely different game. ``True`` for a
    #: positional game (Hearts: seat order is part of play), ``False`` for a symmetric one
    #: where only the participant set matters. The Stage 6/7 multi-seat scheduler reads this to
    #: choose ordered (permutation) versus unordered (combination) seat expansion; single-player
    #: environments leave it ``False``. Defaulted so additive declaration never breaks a caller.
    seat_order_matters: bool = False
    #: Optional viewing cadence (ms) for watch/replay playback, independent of ``pace_interval_ms`` so a
    #: turn-based game (Hearts) can slow its playback without becoming realtime. ``None`` falls back to
    #: the frontend's default viewing cadence; it never affects live human stepping or scoring.
    view_interval_ms: int | None = None
    #: Optional cadence (ms) at which a *live human* turn-based session plays out the other players'
    #: moves, so a burst of fast AI replies animates one at a time instead of snapping together. The
    #: human's own move still renders the instant it arrives. ``None`` (the default, and what a realtime
    #: env like Flappy Bird keeps) means "render every frame on arrival" — today's behaviour. Distinct
    #: from ``view_interval_ms`` (spectator/replay pace, typically slower) and never affects scoring.
    live_interval_ms: int | None = None
    #: Which pause a *human* session of this environment uses when the browser's Pause button is
    #: pressed. ``"session"`` (the default) pauses the actual harness session: stepping, cadence, and
    #: every in-harness timer, including the human move clock, stop advancing. ``"playback"`` pauses
    #: only the viewer's frame playout, while the session and the move clock keep running underneath.
    #: A *watch* session always pauses playout regardless of this value.
    human_pause: HumanPause = "session"
    #: Explicit gameplay parameters. The public layout parameter is synthesized from ``layout``.
    parameters: tuple[EnvParameter, ...] = ()
    #: Named partial parameter configurations an environment recommends.
    presets: tuple[EnvPreset, ...] = ()

    def __post_init__(self) -> None:
        if self.stepping not in {"sequential", "simultaneous"}:
            raise ValueError(f"environment {self.env_id!r} stepping must be 'sequential' or 'simultaneous'")
        if self.human_pause not in {"session", "playback"}:
            raise ValueError(f"environment {self.env_id!r} human_pause must be 'session' or 'playback'")
        if self.pace_interval_ms is not None and not is_json_safe_integer(self.pace_interval_ms):
            raise ValueError(
                f"environment {self.env_id!r} pace_interval_ms must be a JSON-safe integer or None"
            )
        if self.stepping == "simultaneous":
            if self.pace_interval_ms is None or self.pace_interval_ms <= 0:
                raise ValueError(
                    f"simultaneous environment {self.env_id!r} pace_interval_ms must be a positive integer"
                )
            if self.human_timeout_ms is not None:
                raise ValueError(f"simultaneous environment {self.env_id!r} human_timeout_ms must be None")
        if not self.builtin_agents:
            raise ValueError(f"environment {self.env_id!r} must declare at least one builtin agent")
        if any(type(agent) is not BuiltinAgent for agent in self.builtin_agents):
            raise ValueError(f"environment {self.env_id!r} builtin agents must be BuiltinAgent entries")
        builtin_names = [agent.name for agent in self.builtin_agents]
        if builtin_names[0] != "naive":
            raise ValueError(f"environment {self.env_id!r} first builtin agent must be 'naive'")
        if len(builtin_names) != len(set(builtin_names)):
            raise ValueError(f"environment {self.env_id!r} builtin agent names must be unique")
        _validate_layout(self.env_id, self.layout, frozenset(builtin_names))
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("environment parameter names must be unique")
        collisions = sorted(set(names) & _RESERVED_PARAMETER_NAMES)
        if collisions:
            raise ValueError(f"{collisions[0]!r} is synthesized from the environment layout")
        if any(type(preset) is not EnvPreset for preset in self.presets):
            raise ValueError("environment presets must be EnvPreset entries")
        preset_names = [preset.name for preset in self.presets]
        if len(preset_names) != len(set(preset_names)):
            raise ValueError("environment preset names must be unique")
        for preset in self.presets:
            try:
                resolve_parameters(self, preset.values)
            except EnvParameterValueError as error:
                raise ValueError(
                    f"environment preset {preset.name!r} has invalid parameter values: {error}"
                ) from error
            if preset.llm and not self.llm:
                raise ValueError(
                    f"environment {self.env_id!r} preset {preset.name!r} enables the LLM "
                    "but the environment declares llm=False"
                )

    def to_json(self) -> dict[str, Any]:
        """Return the snake_case JSON-serialisable dict the backend serves verbatim."""
        return {
            "env_id": self.env_id,
            "display_name": self.display_name,
            "description": self.description,
            "stepping": self.stepping,
            "builtin_agents": [agent.to_json() for agent in self.builtin_agents],
            "layout": _layout_to_json(self.layout),
            "human_players": list(self.human_players),
            "human_timeout_ms": self.human_timeout_ms,
            "recommended_episode_ticks": self.recommended_episode_ticks,
            "pace_interval_ms": self.pace_interval_ms,
            "step_limit_ms": self.step_limit_ms,
            "episode_limit_ms": self.episode_limit_ms,
            "messaging": self.messaging,
            "message_cap": self.message_cap,
            "llm": self.llm,
            "renderer": self.renderer,
            "seat_order_matters": self.seat_order_matters,
            "view_interval_ms": self.view_interval_ms,
            "live_interval_ms": self.live_interval_ms,
            "human_pause": self.human_pause,
            "parameters": [parameter.to_json() for parameter in effective_parameters(self)],
            "presets": [preset.to_json() for preset in self.presets],
        }


@dataclass(frozen=True)
class EnvironmentEntry:
    """A full environment registration: metadata plus the harness-facing hooks.

    - ``meta`` is the pure data above.
    - ``make`` receives a fully resolved parameter map and returns a fresh AEC or parallel env
      selected by ``meta.stepping``. The seed arrives at ``reset``, not here, so a factory can
      be called once per episode.
    - ``default_action(env, player_id)`` returns the concrete legal action, in that env's action
      space, the loop applies on every timeout path. Passing the live env lets a provider read
      current state (Hearts' lowest legal card, Spades' suggested bid) so the recording holds the
      action actually played; Flappy Bird just returns its noop (idle).
    - ``overlay`` optionally extracts the per-step overlay dict from a live env instance.
    - ``overlay_static`` optionally extracts episode-static overlay data for the recording header.
      A recorded episode calls it once after ``env.reset``, just before the header is written. Its
      result must remain constant for the episode.
    """

    meta: EnvironmentMeta
    make: Callable[[Mapping[str, ParameterValue]], Any]
    default_action: Callable[[Any, str], Any]
    overlay: Callable[[Any], dict[str, Any]] | None = None
    overlay_static: Callable[[Any], dict[str, Any]] | None = None


def validate_configured_environment(
    entry: EnvironmentEntry,
    env: object,
    expected_players: Sequence[str],
    reset_result: object,
) -> None:
    """Validate a reset environment against its declaration and resolved canonical roster.

    The one contract boundary an episode crosses. An environment's declared stepping mode is taken
    at its word, so this checks the reset surface the declared episode path actually uses rather
    than proving which PettingZoo protocol the object implements. That still catches either
    mislabelled direction: a sequential declaration needs the AEC mappings a parallel environment
    never populates, and a simultaneous one needs the observation and info mappings an AEC
    ``reset()`` never returns. The checks stay duck-typed so the harness keeps its one-way
    dependency on environment packages and does not import PettingZoo in production.
    """
    meta = entry.meta
    if meta.stepping == "sequential":
        _validate_possible_agents(meta, env, expected_players)
        _validate_aec_reset_surface(meta, env)
        return
    validate_parallel_reset(meta, env, expected_players, reset_result)


def validate_parallel_reset(
    meta: EnvironmentMeta,
    env: object,
    expected_players: Sequence[str],
    reset_result: object,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Validate the strict Game Sandbox parallel reset subset and return its two mappings."""
    expected = list(expected_players)
    _validate_possible_agents(meta, env, expected)
    agents = getattr(env, "agents", None)
    if agents != expected:
        raise EnvironmentContractError(
            meta.env_id, meta.stepping, f"reset agents is {agents!r}, expected {expected!r}"
        )
    if not isinstance(reset_result, tuple) or len(cast("tuple[object, ...]", reset_result)) != 2:
        raise EnvironmentContractError(
            meta.env_id, meta.stepping, "reset() must return an observations and infos tuple"
        )
    observations, infos = cast("tuple[object, object]", reset_result)
    _validate_player_mapping(meta, "reset observations", observations, expected, ordered=True)
    _validate_player_mapping(meta, "reset infos", infos, expected, ordered=True)
    return cast("Mapping[str, object]", observations), cast("Mapping[str, object]", infos)


def validate_parallel_step(
    meta: EnvironmentMeta,
    env: object,
    active_players: Sequence[str],
    actions: object,
    step_result: object,
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
]:
    """Validate one strict parallel transition for the later simultaneous tick path.

    The parallel API permits looser mappings. Game Sandbox requires exact pre-step active-player
    coverage in the action mapping and every returned mapping so terminal rewards and final
    observations remain recordable. Roster sequences remain canonical; mapping insertion order
    is not part of the contract.
    """
    expected = list(active_players)
    if tuple(expected) != canonical_player_order(expected):
        raise EnvironmentContractError(
            meta.env_id, meta.stepping, "pre-step active players are not canonical"
        )
    _validate_player_mapping(meta, "actions", actions, expected, ordered=False)
    if not isinstance(step_result, tuple) or len(cast("tuple[object, ...]", step_result)) != 5:
        raise EnvironmentContractError(
            meta.env_id,
            meta.stepping,
            "step(actions) must return observations, rewards, terminations, truncations, and infos",
        )
    observations, rewards, terminations, truncations, infos = cast(
        "tuple[object, object, object, object, object]", step_result
    )
    for label, mapping in (
        ("step observations", observations),
        ("step rewards", rewards),
        ("step terminations", terminations),
        ("step truncations", truncations),
        ("step infos", infos),
    ):
        _validate_player_mapping(meta, label, mapping, expected, ordered=False)
    expected_agents = [
        player_id
        for player_id in expected
        if not cast("Mapping[str, object]", terminations)[player_id]
        and not cast("Mapping[str, object]", truncations)[player_id]
    ]
    agents = getattr(env, "agents", None)
    if agents != expected_agents:
        raise EnvironmentContractError(
            meta.env_id,
            meta.stepping,
            f"post-step agents is {agents!r}, expected nonterminal players {expected_agents!r}",
        )
    return (
        cast("Mapping[str, object]", observations),
        cast("Mapping[str, object]", rewards),
        cast("Mapping[str, object]", terminations),
        cast("Mapping[str, object]", truncations),
        cast("Mapping[str, object]", infos),
    )


#: The subspace types a composite action space may declare, each with a mask entry the platform
#: can read. A type outside this set is named here with the permitted shape to reach for, so one
#: mask vocabulary covers every environment: a 1 always marks a legal choice.
_COMPOSITE_ALTERNATIVES = {
    "MultiBinary": "declare a MultiDiscrete with two values per dimension instead",
    "Tuple": "declare a Dict with a name per component instead",
}
_COMPOSITE_SUBSPACES = frozenset({"Discrete", "MultiDiscrete", "Dict", "Box"})


def action_mask_problems(space: Any, mask: Any) -> list[str]:
    """Describe every disagreement between a declared action space and one published mask.

    An empty list means the two agree, which includes an environment that publishes no mask at
    all. Problems are collected rather than raised so one conformance run reports all of them.
    Spaces are recognised by type name, keeping the harness free of a Gymnasium dependency.
    """
    return _mask_problems(space, mask, "action_mask")


def _mask_problems(space: Any, mask: Any, label: str) -> list[str]:
    kind = type(space).__name__
    if kind == "Dict":
        return _composite_mask_problems(space, mask, label)
    if kind == "Box":
        return [] if mask is None else [f"{label} must be null, because a continuous range cannot be masked"]
    if mask is None:
        return []
    if kind == "Discrete":
        return _binary_vector_problems(mask, int(space.n), label)
    if kind == "MultiDiscrete":
        return _multi_discrete_mask_problems(space, mask, label)
    return [f"{label} is published for a {kind} action space, which cannot carry a mask"]


def _composite_mask_problems(space: Any, mask: Any, label: str) -> list[str]:
    """Check a Dict action space's declared children, then its mask object against them."""
    subspaces = cast("Mapping[str, Any]", space.spaces)
    problems: list[str] = []
    for key, subspace in subspaces.items():
        kind = type(subspace).__name__
        if kind in _COMPOSITE_SUBSPACES:
            continue
        advice = _COMPOSITE_ALTERNATIVES.get(kind)
        problems.append(
            f"action space component {key!r} is a {kind}, which a composite action may not declare"
            + (f": {advice}" if advice is not None else "")
        )
    if mask is None:
        return problems
    if not isinstance(mask, Mapping):
        problems.append(f"{label} is not an object, but the action space is a Dict")
        return problems
    entries = cast("Mapping[str, Any]", mask)
    if set(entries) != set(subspaces):
        problems.append(
            f"{label} carries the keys {sorted(entries)}, but the action space declares {sorted(subspaces)}"
        )
    for key in sorted(set(entries) & set(subspaces)):
        if type(subspaces[key]).__name__ in _COMPOSITE_SUBSPACES:
            problems.extend(_mask_problems(subspaces[key], entries[key], f"{label}[{key!r}]"))
    return problems


def _multi_discrete_mask_problems(space: Any, mask: Any, label: str) -> list[str]:
    """Check one tuple of binary vectors, one per dimension, against a MultiDiscrete subspace."""
    if getattr(space.nvec, "ndim", 1) != 1:
        return [f"{label} covers a MultiDiscrete whose nvec is not one-dimensional"]
    lengths = [int(size) for size in space.nvec]
    if not isinstance(mask, tuple):
        return [f"{label} is not a tuple of one binary vector per dimension"]
    values = cast("tuple[Any, ...]", mask)
    if len(values) != len(lengths):
        return [f"{label} carries {len(values)} vectors, but the subspace has {len(lengths)} dimensions"]
    problems: list[str] = []
    for position, (entry, length) in enumerate(zip(values, lengths, strict=True)):
        problems.extend(_binary_vector_problems(entry, length, f"{label}[{position}]"))
    return problems


def _binary_vector_problems(mask: Any, length: int, label: str) -> list[str]:
    """Check one mask entry is a vector of ``length`` values, each of them 0 or 1."""
    try:
        values = [int(value) for value in mask]
    except (TypeError, ValueError):
        return [f"{label} is not a vector of {length} binary values"]
    if len(values) != length:
        return [f"{label} carries {len(values)} values, but the subspace has {length}"]
    if any(value not in (0, 1) for value in values):
        return [f"{label} carries a value other than 0 and 1"]
    return []


def _validate_aec_reset_surface(meta: EnvironmentMeta, env: object) -> None:
    for name in ("agents", "rewards", "terminations", "truncations", "agent_selection"):
        if not hasattr(env, name):
            raise EnvironmentContractError(meta.env_id, meta.stepping, f"reset leaves out AEC {name}")


def _validate_possible_agents(meta: EnvironmentMeta, env: object, expected_players: Sequence[str]) -> None:
    possible_agents = getattr(env, "possible_agents", None)
    expected = list(expected_players)
    if possible_agents != expected:
        raise EnvironmentContractError(
            meta.env_id,
            meta.stepping,
            f"possible_agents is {possible_agents!r}, expected {expected!r} from resolved layout",
        )


def _validate_player_mapping(
    meta: EnvironmentMeta,
    label: str,
    value: object,
    expected_players: Sequence[str],
    *,
    ordered: bool,
) -> None:
    if not isinstance(value, Mapping):
        raise EnvironmentContractError(meta.env_id, meta.stepping, f"{label} is not a player-keyed mapping")
    actual = list(cast("Mapping[str, object]", value))
    expected = list(expected_players)
    # Sorted rather than set comparison so a caller passing a duplicated player id fails here, with
    # the mapping named, instead of surviving into the active-set arithmetic below.
    matches = actual == expected if ordered else sorted(actual) == sorted(expected)
    if not matches:
        raise EnvironmentContractError(
            meta.env_id,
            meta.stepping,
            f"{label} keys are {actual!r}, expected {expected!r}",
        )


def effective_parameters(meta: EnvironmentMeta) -> tuple[EnvParameter, ...]:
    """Return the environment declarations with its synthesized layout parameter first."""
    if isinstance(meta.layout, PlayerBounds):
        return (_player_count_parameter(meta.layout), *meta.parameters)
    return (_seat_plan_parameter(meta.layout), *meta.parameters)


def resolve_layout(meta: EnvironmentMeta, parameters: Mapping[str, ParameterValue]) -> ResolvedLayout:
    """Resolve the canonical seat-to-player layout from a complete validated parameter map.

    This deliberately performs no defaulting. A missing, malformed, or unknown reserved value means an
    upstream caller broke the complete-parameter contract and must not silently run a different layout.
    """
    if isinstance(meta.layout, PlayerBounds):
        value = parameters.get("players")
        if not is_json_safe_integer(value) or not meta.layout.min <= value <= meta.layout.max:
            raise ValueError("resolved parameters carry no valid players value")
        seats = tuple(
            ResolvedSeat(seat_id=f"seat_{index}", players=(f"player_{index}",)) for index in range(value)
        )
        return ResolvedLayout("solo", seats, value, value)

    selected = parameters.get("seat_plan")
    if not isinstance(selected, str):
        raise ValueError("resolved parameters carry no valid seat_plan value")
    plan = next((plan for plan in meta.layout.plans if plan.key == selected), None)
    if plan is None:
        raise ValueError(f"resolved parameters select unknown seat plan {selected!r}")
    seats = tuple(
        ResolvedSeat(
            seat_id=f"seat_{seat_index}",
            players=tuple(f"player_{player_index}" for player_index in declaration.players),
            restricted_builtin=declaration.restricted_builtin,
        )
        for seat_index, declaration in enumerate(plan.seats)
    )
    player_count = sum(len(declaration.players) for declaration in plan.seats)
    return ResolvedLayout(plan.key, seats, player_count, len(seats))


def _layout_to_json(layout: EnvironmentLayout) -> dict[str, Any]:
    if isinstance(layout, PlayerBounds):
        return {"kind": "player_bounds", "min": layout.min, "max": layout.max}
    return {
        "kind": "seat_plans",
        "plans": [
            {
                "key": plan.key,
                "title": plan.title,
                "seats": [
                    {"players": list(declaration.players)}
                    if declaration.restricted_builtin is None
                    else {
                        "players": list(declaration.players),
                        "restricted_builtin": declaration.restricted_builtin,
                    }
                    for declaration in plan.seats
                ],
            }
            for plan in layout.plans
        ],
    }


def _validate_layout(env_id: str, layout: object, builtin_names: frozenset[str]) -> None:
    if isinstance(layout, PlayerBounds):
        if not is_json_safe_integer(layout.min) or layout.min <= 0:
            raise ValueError(f"environment {env_id!r} player bounds min must be a positive integer")
        if not is_json_safe_integer(layout.max) or layout.max <= 0:
            raise ValueError(f"environment {env_id!r} player bounds max must be a positive integer")
        if layout.min > layout.max:
            raise ValueError(f"environment {env_id!r} player bounds min must be no greater than max")
        return
    if not isinstance(layout, SeatPlans):
        raise ValueError(f"environment {env_id!r} layout must be PlayerBounds or SeatPlans")
    if not layout.plans:
        raise ValueError(f"environment {env_id!r} seat plans must not be empty")
    keys: set[str] = set()
    for plan in layout.plans:
        if not _is_parameter_name(plan.key):
            raise ValueError(f"environment {env_id!r} plan {plan.key!r} key must be snake_case")
        if plan.key in keys:
            raise ValueError(f"environment {env_id!r} plan {plan.key!r} key is duplicated")
        keys.add(plan.key)
        if not _is_nonempty_string(plan.title):
            raise ValueError(f"environment {env_id!r} plan {plan.key!r} title must be non-empty")
        if _has_blank_line(plan.title):
            raise ValueError(f"environment {env_id!r} plan {plan.key!r} title must not contain a blank line")
        if not plan.seats:
            raise ValueError(f"environment {env_id!r} plan {plan.key!r} must contain at least one seat")
        indices: list[int] = []
        restricted_seats = 0
        for declaration in plan.seats:
            if type(declaration) is not SeatDeclaration:
                raise ValueError(f"environment {env_id!r} plan {plan.key!r} seats must be declarations")
            if not declaration.players:
                raise ValueError(f"environment {env_id!r} plan {plan.key!r} has an empty seat")
            if not all(is_json_safe_integer(index) and index >= 0 for index in declaration.players):
                raise ValueError(
                    f"environment {env_id!r} plan {plan.key!r} player indices must be non-negative integers"
                )
            indices.extend(declaration.players)
            if declaration.restricted_builtin is not None:
                if declaration.restricted_builtin not in builtin_names:
                    raise ValueError(
                        f"environment {env_id!r} plan {plan.key!r} restriction names an undeclared builtin"
                    )
                restricted_seats += 1
        if restricted_seats > 1:
            raise ValueError(f"environment {env_id!r} plan {plan.key!r} has more than one restricted seat")
        if restricted_seats == len(plan.seats):
            raise ValueError(f"environment {env_id!r} plan {plan.key!r} must have an unrestricted seat")
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(
                f"environment {env_id!r} plan {plan.key!r} must partition players from index 0 without gaps"
            )


def resolve_parameters(
    meta: EnvironmentMeta, *layers: Mapping[str, ParameterValue]
) -> dict[str, ParameterValue]:
    """Fill defaults and apply ordered parameter override layers, raising on the first bad value.

    Use this where a partial layer is expected and a bad one should stop the caller: local play, the
    CLI, and tests. A boundary that receives an already complete map wants
    :func:`validate_complete_parameters` instead, so a missing name fails rather than silently
    resolving to a default nobody chose.
    """
    declarations = effective_parameters(meta)
    by_name = {parameter.name: parameter for parameter in declarations}
    values = {parameter.name: parameter.validate_value(parameter.default) for parameter in declarations}
    for layer in layers:
        for name, value in layer.items():
            try:
                declaration = by_name[name]
            except KeyError:
                raise EnvParameterValueError(f"unknown environment parameter {name!r}") from None
            values[name] = declaration.validate_value(value)
    return values


def preset_values(meta: EnvironmentMeta, name: str) -> Mapping[str, ParameterValue]:
    """Return the named preset's partial parameter layer for :func:`resolve_parameters`."""
    for preset in meta.presets:
        if preset.name == name:
            return preset.values
    available = ", ".join(sorted(preset.name for preset in meta.presets)) or "none"
    raise ValueError(f"unknown environment preset {name!r}; available: {available}")


def validate_complete_parameters(
    meta: EnvironmentMeta, parameters: Mapping[str, ParameterValue]
) -> dict[str, ParameterValue]:
    """Validate and normalize a map that must already carry every effective parameter.

    Unlike :func:`resolve_parameters`, this applies no defaults. A launch configuration is produced by
    a caller that resolved the values already, so filling a missing name here would let an upstream bug
    run a game on a value nobody chose and then record it as though it had been chosen.
    """
    declarations = effective_parameters(meta)
    names = {declaration.name for declaration in declarations}
    for name in parameters:
        if name not in names:
            raise EnvParameterValueError(f"unknown environment parameter {name!r}")
    values: dict[str, ParameterValue] = {}
    for declaration in declarations:
        if declaration.name not in parameters:
            raise EnvParameterValueError(f"missing environment parameter {declaration.name!r}")
        values[declaration.name] = declaration.validate_value(parameters[declaration.name])
    return values


def int_parameter(parameters: Mapping[str, ParameterValue], name: str) -> int:
    """Read one integer parameter from a resolved map, narrowing its type for the caller.

    Environment factories use this rather than an ``assert``. An assert is stripped under ``python -O``,
    so the narrowing it performs would quietly disappear from an optimized run.
    """
    if name not in parameters:
        raise EnvParameterValueError(f"missing environment parameter {name!r}")
    value = parameters[name]
    if not is_json_safe_integer(value):
        raise EnvParameterValueError(f"{name} must be a JSON-safe integer")
    return value


class EnvironmentLookupError(LookupError):
    """Raised when an environment id is not registered as an installed entry point."""


def discover_environments() -> dict[str, EnvironmentEntry]:
    """Return every installed environment keyed by id, via the entry-point registry.

    Uses ``importlib.metadata`` so the harness never imports the environments package: the
    dependency arrow points one way (environments -> harness) and stays there.
    """
    found: dict[str, EnvironmentEntry] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        entry = ep.load()
        if not isinstance(entry, EnvironmentEntry):
            raise TypeError(
                f"entry point {ep.name!r} in group {ENTRY_POINT_GROUP!r} loaded a "
                f"{type(entry).__name__}, expected EnvironmentEntry"
            )
        # Lookups key on the entry-point name while recordings stamp meta.env_id; a mismatch
        # would silently disagree between the registry and the recorded environment id, so it
        # is rejected at the source instead.
        if ep.name != entry.meta.env_id:
            raise ValueError(
                f"entry point {ep.name!r} in group {ENTRY_POINT_GROUP!r} registers an "
                f"environment whose meta.env_id is {entry.meta.env_id!r}; the entry-point "
                f"name and env_id must match"
            )
        found[ep.name] = entry
    return found


def load_environment(env_id: str) -> EnvironmentEntry:
    """Return the installed environment registered under ``env_id``.

    Raises :class:`EnvironmentLookupError` naming the available ids when none matches.
    """
    found = discover_environments()
    try:
        return found[env_id]
    except KeyError:
        available = ", ".join(sorted(found)) or "(none installed)"
        raise EnvironmentLookupError(
            f"no environment registered as {env_id!r}; available: {available}"
        ) from None
