"""Game Sandbox session harness.

Stage 1 shipped the cross-boundary contract pieces: schema validation, typed state builders,
and the recording store. Stage 2 adds the session loop and its surrounding machinery — the
agent interface, the manifest loader, the environment metadata and registry, the injectable
clock, and ``run_episode`` with its player bindings and action sources. The loop is factored into
an ``Episode`` whose ``advance`` dispatch is shared by ``run_episode`` and the live container runner.
Environments live in a separate package discovered through entry points; the harness never
imports them.
"""

from .agent import (
    AgentBase,
    has_chat,
    has_learn,
    is_agent,
)
from .clock import Clock, ManualClock, SystemClock
from .environment import (
    ENTRY_POINT_GROUP,
    BuiltinAgent,
    ChatPolicy,
    ChatPolicySource,
    EnvironmentContractError,
    EnvironmentEntry,
    EnvironmentLookupError,
    EnvironmentMeta,
    EnvParameter,
    EnvParameterChoice,
    EnvParameterValueError,
    ParameterValue,
    PlayerBounds,
    ResolvedLayout,
    ResolvedSeat,
    SeatDeclaration,
    SeatPlan,
    SeatPlans,
    SteppingMode,
    canonical_player_order,
    discover_environments,
    effective_parameters,
    load_environment,
    resolve_layout,
    resolve_parameters,
    validate_configured_environment,
    validate_parallel_reset,
    validate_parallel_step,
)
from .manifest import (
    Manifest,
    ManifestError,
    load_agent,
    load_manifest,
)
from .schema import (
    SCHEMA_VERSION,
    SchemaValidationError,
    validate_header,
    validate_step,
)
from .session import (
    ActionSource,
    AgentPlayer,
    Episode,
    EpisodeResult,
    ExternalPlayer,
    MessageSource,
    NoopSource,
    ScriptedSource,
    run_episode,
)
from .state import (
    AgentStep,
    ChatOptions,
    Message,
    RecordingHeader,
    StepState,
    StepTiming,
    build_agent_step,
    build_header,
    build_step_state,
)

__all__ = [
    # schema
    "SCHEMA_VERSION",
    "SchemaValidationError",
    "validate_header",
    "validate_step",
    # state
    "AgentStep",
    "ChatOptions",
    "Message",
    "RecordingHeader",
    "StepState",
    "StepTiming",
    "build_agent_step",
    "build_header",
    "build_step_state",
    # agent
    "AgentBase",
    "is_agent",
    "has_learn",
    "has_chat",
    # clock
    "Clock",
    "SystemClock",
    "ManualClock",
    # environment
    "ENTRY_POINT_GROUP",
    "SteppingMode",
    "BuiltinAgent",
    "ParameterValue",
    "EnvParameterChoice",
    "EnvParameter",
    "EnvParameterValueError",
    "EnvironmentMeta",
    "PlayerBounds",
    "SeatPlan",
    "SeatDeclaration",
    "SeatPlans",
    "ResolvedSeat",
    "ResolvedLayout",
    "ChatPolicy",
    "ChatPolicySource",
    "EnvironmentEntry",
    "EnvironmentContractError",
    "EnvironmentLookupError",
    "canonical_player_order",
    "discover_environments",
    "load_environment",
    "effective_parameters",
    "resolve_parameters",
    "resolve_layout",
    "validate_configured_environment",
    "validate_parallel_reset",
    "validate_parallel_step",
    # manifest
    "Manifest",
    "ManifestError",
    "load_manifest",
    "load_agent",
    # session
    "ActionSource",
    "MessageSource",
    "AgentPlayer",
    "ExternalPlayer",
    "NoopSource",
    "ScriptedSource",
    "Episode",
    "EpisodeResult",
    "run_episode",
]
