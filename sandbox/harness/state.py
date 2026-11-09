"""Typed shapes and builders for the per-step state object and the recording header.

These TypedDicts mirror the JSON Schema field-for-field (snake_case throughout) and the
builders produce schema-valid payloads by construction. They are a thin construction
ergonomics layer; the schema in :mod:`game_sandbox_harness.schema` remains the single
source of truth, and the recording store validates on write regardless.
"""

from __future__ import annotations

import json
from typing import Any, NotRequired, TypedDict

from .environment import ParameterValue, ResolvedLayout
from .schema import SCHEMA_VERSION


def json_default(item: Any) -> Any:
    """Convert a NumPy-like scalar or array leaf, leaving every other type to fail normally.

    An environment may hand the harness values that are not JSON types. A Gymnasium space's
    ``sample()`` returns NumPy scalars, and an observation or overlay routinely carries arrays,
    so those normalize to the plain JSON values they already represent. Anything else keeps
    ``json``'s ordinary ``TypeError``, because a set or a game object in a recorded payload is an
    environment defect rather than a shape the wire can carry.

    Duck-typed on ``tolist`` and ``item`` so the harness needs no NumPy dependency.
    """
    for method_name in ("tolist", "item"):
        method = getattr(item, method_name, None)
        if callable(method):
            return method()
    return json.JSONEncoder().default(item)


class StepTiming(TypedDict):
    """Wall-clock timing for a whole step."""

    started_at: int  # epoch milliseconds UTC
    duration_ms: float


class AgentTiming(TypedDict, total=False):
    """Per-agent chargeable timing. Official LLM sessions compare proxy-time readings around each
    hook and discount their non-negative change, with calling-thread CPU time as a lower bound. A
    valid post-hook reading is reused as the next baseline. A missing or invalid reading charges the
    full wall-clock hook time. ``decision_ms`` is ``act`` time, ``learn_ms`` is optional learn-hook
    time, and ``chat_ms`` is optional chat-hook time. The leaderboard compute column sums all three."""

    decision_ms: float
    learn_ms: float
    chat_ms: float


class AgentStep(TypedDict):
    """Per-agent data for one step. ``observation`` and ``action`` are environment-specific."""

    observation: NotRequired[Any]
    action: NotRequired[Any]
    reward: float
    score: float
    timing: NotRequired[AgentTiming]


# ``from`` is a Python keyword, so Message must use the functional TypedDict form.
# ``recipients`` is the live-only audience annotation for a broadcast the environment bounded:
# the players the broadcast was delivered to. It appears only on the live stream (so relays can
# filter a controller's view), never in a recording, and every relay strips it before a browser.
Message = TypedDict(
    "Message",
    {
        "from": str,
        "to": str | None,  # None means broadcast.
        "text": str,
        "recipients": NotRequired[list[str]],
    },
)


class ChatOptions(TypedDict):
    """The live messaging choices for the external player who may act on this state."""

    sender: str
    target_recipients: list[str]
    default_recipient: str | None


class StepState(TypedDict):
    """One per-step state object."""

    schema_version: int
    tick: int
    agents: dict[str, AgentStep]
    overlay: NotRequired[dict[str, Any]]
    messages: NotRequired[list[Message]]
    chat_options: NotRequired[ChatOptions]
    timing: StepTiming


class PlayerAttribution(TypedDict, total=False):
    """Who or what drove one player: a connected human or an agent. ``kind`` and ``label`` are
    always present on a written entry (the schema requires them). A submitted agent carries
    ``submission_id``; a built-in agent carries ``builtin_name``. ``total=False`` keeps the builders
    ergonomic; the store validates on write."""

    kind: str  # "human" or "agent"
    label: str
    user: str
    submission_id: str
    builtin_name: str


class RecordingHeader(TypedDict):
    """Line 1 of a recording / first frame of a live stream."""

    schema_version: int
    environment: str
    parameters: dict[str, ParameterValue]
    overlay_static: NotRequired[dict[str, Any]]
    created_at: NotRequired[str]
    seed: NotRequired[int]
    sidecars: NotRequired[list[Sidecar]]
    players: dict[str, PlayerAttribution]
    seats: dict[str, list[str]]
    seat_plan: str


class Sidecar(TypedDict):
    """A declared auxiliary file alongside a recording."""

    name: str
    path: str


def build_agent_step(
    *,
    reward: float,
    score: float,
    observation: Any = None,
    action: Any = None,
    decision_ms: float | None = None,
    learn_ms: float | None = None,
    chat_ms: float | None = None,
) -> AgentStep:
    """Build a per-agent step. ``observation``/``action`` default to absent when None.

    ``learn_ms`` and ``chat_ms`` are emitted only when supplied, so a non-learning, non-chatting
    agent carries no learn or chat field.
    """
    step: AgentStep = {"reward": reward, "score": score}
    if observation is not None:
        step["observation"] = observation
    if action is not None:
        step["action"] = action
    if decision_ms is not None or learn_ms is not None or chat_ms is not None:
        timing: AgentTiming = {}
        if decision_ms is not None:
            timing["decision_ms"] = decision_ms
        if learn_ms is not None:
            timing["learn_ms"] = learn_ms
        if chat_ms is not None:
            timing["chat_ms"] = chat_ms
        step["timing"] = timing
    return step


def build_step_state(
    *,
    tick: int,
    agents: dict[str, AgentStep],
    started_at: int,
    duration_ms: float,
    overlay: dict[str, Any] | None = None,
    messages: list[Message] | None = None,
    chat_options: ChatOptions | None = None,
) -> StepState:
    """Build a per-step state object stamped with the current schema version."""
    state: StepState = {
        "schema_version": SCHEMA_VERSION,
        "tick": tick,
        "agents": agents,
        "timing": {"started_at": started_at, "duration_ms": duration_ms},
    }
    if overlay is not None:
        state["overlay"] = overlay
    if messages:
        state["messages"] = messages
    if chat_options is not None:
        state["chat_options"] = chat_options
    return state


def build_header(
    *,
    environment: str,
    parameters: dict[str, ParameterValue],
    overlay_static: dict[str, Any] | None = None,
    created_at: str | None = None,
    seed: int | None = None,
    sidecars: list[Sidecar] | None = None,
    players: dict[str, PlayerAttribution],
    layout: ResolvedLayout,
) -> RecordingHeader:
    """Build a recording header stamped with the current schema version and resolved layout."""
    seat_members = [player_id for seat in layout.seats for player_id in seat.players]
    if set(players) != set(seat_members) or len(seat_members) != len(set(seat_members)):
        raise ValueError("recording players must exactly match the resolved seat partition")
    header: RecordingHeader = {
        "schema_version": SCHEMA_VERSION,
        "environment": environment,
        "parameters": dict(parameters),
        "players": dict(players),
        "seats": {seat.seat_id: list(seat.players) for seat in layout.seats},
        "seat_plan": layout.plan_key,
    }
    if created_at is not None:
        header["created_at"] = created_at
    if seed is not None:
        header["seed"] = seed
    if overlay_static is not None:
        header["overlay_static"] = overlay_static
    if sidecars is not None:
        header["sidecars"] = sidecars
    return header
