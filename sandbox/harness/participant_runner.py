"""Mode-neutral participant work for one harness episode.

The AEC and parallel orchestrators keep their different PettingZoo lifecycles in
``session.py``. This module owns the common participant boundary: action collection,
messaging, learning, timing, credential activation, and per-player accounting.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, cast, runtime_checkable

from .agent import has_chat, has_learn
from .chat import ChatRouter
from .clock import Clock
from .environment import ChatPolicy, EnvironmentEntry
from .state import ChatOptions, Message, build_agent_step


class IllegalAgentActionError(RuntimeError):
    """Raised when an agent returns an action the environment would reject."""


@runtime_checkable
class ActionSource(Protocol):
    """A source of actions for an external player.

    ``window_ms`` is the move budget, not a deadline instant: a source decides for itself when that
    budget runs, which is what lets a live human's clock start with their controls.
    """

    def get_action(self, player_id: str, observation: Any, window_ms: int | None) -> Any: ...


@runtime_checkable
class MessageSource(Protocol):
    """A source of queued outgoing messages for an external player."""

    def take_messages(self, player_id: str) -> list[ExternalChatFrame]: ...


class ExternalChatFrame(TypedDict):
    """One human chat frame queued by the live transport."""

    to: str | None
    text: str


@runtime_checkable
class AgentExecutionScope(Protocol):
    """Activate an agent's credentials and expose its official proxy timing."""

    def setup(self, player_id: str) -> None: ...

    def turn(self, player_id: str, tick: int) -> None: ...

    def inflight_ms(self, player_id: str) -> int | None: ...


class NoopSource:
    """An :class:`ActionSource` that never supplies input."""

    def get_action(self, player_id: str, observation: Any, window_ms: int | None) -> Any:
        return None


class ScriptedSource:
    """An :class:`ActionSource` that replays fixed actions, then yields ``None``."""

    def __init__(self, actions: list[Any]) -> None:
        self._actions = list(actions)
        self._index = 0

    def get_action(self, player_id: str, observation: Any, window_ms: int | None) -> Any:
        if self._index >= len(self._actions):
            return None
        action = self._actions[self._index]
        self._index += 1
        return action


@dataclass(frozen=True)
class AgentPlayer:
    """A player driven by a loaded agent under agent-timeout accounting."""

    agent: Any
    execution_scope: AgentExecutionScope | None = None


@dataclass(frozen=True)
class ExternalPlayer:
    """A player fed from outside the harness, usually a human."""

    source: ActionSource
    timeout_ms: int | None = None
    message_source: MessageSource | None = None


Player = AgentPlayer | ExternalPlayer


@dataclass
class PlayerState:
    """Per-player mutable accounting for one episode."""

    score: float = 0.0
    budget_used_ms: float = 0.0
    step_timeouts: int = 0


@dataclass
class StepContext:
    """Mutable values carried through one participant's completed state entry."""

    env: Any
    player_id: str
    observation: Any
    info: Any
    binding: Player
    player: PlayerState
    started_at: int
    action: Any = None
    decision_ms: float | None = None
    agent_compute_ms: float = 0.0
    chat_ms: float | None = None
    messages: list[Message] = field(default_factory=list[Message])
    chat_options: ChatOptions | None = None
    reward: float = 0.0
    learn_ms: float | None = None


class ParticipantRunner:
    """Run shared participant phases while the episode owns environment lifecycle."""

    def __init__(
        self,
        entry: EnvironmentEntry,
        players: Mapping[str, Player],
        state: Mapping[str, PlayerState],
        *,
        clock: Clock,
        cpu_clock_ms: Callable[[], float],
        step_limit_ms: int,
        chat: ChatRouter | None,
        external_chat_sender: str | None,
        active_players: Callable[[], tuple[str, ...]],
        tick: Callable[[], int],
        failed: Callable[[str], None],
    ) -> None:
        self._entry = entry
        self._players = players
        self._state = state
        self._clock = clock
        self._cpu_clock_ms = cpu_clock_ms
        self._step_limit = step_limit_ms
        self._chat = chat
        self._external_chat_sender = external_chat_sender
        self._active_players = active_players
        self._tick = tick
        self._failed = failed
        self._inflight_snapshots: dict[str, int] = {}
        self._human_chat_policy: ChatPolicy | None = None

    @property
    def human_chat_policy(self) -> ChatPolicy | None:
        """The policy advertised in the last live state."""
        return self._human_chat_policy

    def reset_agents(self, seed: int, observations: Mapping[str, object]) -> None:
        """Reset every agent with its initial observation under its credential boundary."""
        for player_id, binding in self._players.items():
            if not isinstance(binding, AgentPlayer):
                continue
            try:
                if binding.execution_scope is not None:
                    binding.execution_scope.setup(player_id)
                observation = observations[player_id]
                _, elapsed_ms = self._timed_llm_hook(
                    binding.execution_scope,
                    player_id,
                    lambda binding=binding, observation=observation: binding.agent.reset(seed, observation),
                )
                self._state[player_id].budget_used_ms += elapsed_ms
            except Exception:  # noqa: BLE001 - preserve the participant exception and attribution
                self._failed(player_id)
                raise

    def select_action(self, context: StepContext) -> None:
        """Obtain and validate an action, defaulting according to the existing policy."""
        binding = context.binding
        if isinstance(binding, AgentPlayer):
            try:
                self._activate_agent_hook(binding, context.player_id)
                context.action, context.decision_ms = self._timed_llm_hook(
                    binding.execution_scope,
                    context.player_id,
                    lambda: binding.agent.act(context.observation),
                )
            except Exception:  # noqa: BLE001 - preserve the participant exception and attribution
                self._failed(context.player_id)
                raise
            context.agent_compute_ms += context.decision_ms
            context.player.budget_used_ms += context.decision_ms
            if context.decision_ms > self._step_limit:
                context.action = self._entry.default_action(context.env, context.player_id)
                return
            reason = illegal_action_reason(
                context.env, context.player_id, context.observation, context.info, context.action
            )
            if reason is not None:
                self._failed(context.player_id)
                raise IllegalAgentActionError(f"{context.player_id} returned an illegal action: {reason}")
            return

        window_ms = external_window(self._entry, binding)
        context.action = binding.source.get_action(context.player_id, context.observation, window_ms)
        if context.action is None:
            print(
                f"human player {context.player_id} defaulted due to no input in time",
                file=sys.stderr,
                flush=True,
            )
            context.action = self._entry.default_action(context.env, context.player_id)
            return
        reason = illegal_action_reason(
            context.env, context.player_id, context.observation, context.info, context.action
        )
        if reason is not None:
            print(
                f"human player {context.player_id} defaulted due to illegal input: {reason}",
                file=sys.stderr,
                flush=True,
            )
            context.action = self._entry.default_action(context.env, context.player_id)

    def drain_human_messages(self) -> list[Message]:
        """Atomically drain and validate the designated human's queued messages."""
        sender = self._external_chat_sender
        if self._chat is None or sender is None:
            return []
        binding = self._players[sender]
        if not isinstance(binding, ExternalPlayer) or binding.message_source is None:
            return []
        queued = binding.message_source.take_messages(sender)
        return self._chat.validate_outgoing(sender, queued, self._human_chat_policy)

    def collect_messages(self, context: StepContext) -> None:
        """Run the pre-step chat phases after action selection."""
        if self._chat is None:
            return
        context.messages.extend(self.drain_human_messages())
        self.collect_agent_messages(context, context.messages)

    def collect_agent_messages(self, context: StepContext, messages: list[Message]) -> None:
        """Drain one inbox and append its validated chat output to the pending batch."""
        if self._chat is None:
            return
        inbox = self._chat.drain(context.player_id)
        binding = context.binding
        if isinstance(binding, AgentPlayer) and has_chat(binding.agent):
            policy = self._chat.policy_from(context.env, context.player_id)
            try:
                self._activate_agent_hook(binding, context.player_id)
                outgoing, context.chat_ms = self._timed_llm_hook(
                    binding.execution_scope,
                    context.player_id,
                    lambda: binding.agent.chat(inbox),
                )
            except Exception:  # noqa: BLE001 - preserve the participant exception and attribution
                self._failed(context.player_id)
                raise
            context.agent_compute_ms += context.chat_ms
            context.player.budget_used_ms += context.chat_ms
            messages.extend(self._chat.validate_outgoing(context.player_id, outgoing, policy))

    def refresh_chat_state(self, env: Any) -> ChatOptions | None:
        """Adopt active players and publish the designated human's current policy."""
        if self._chat is None:
            return None
        active_players = self._active_players()
        self._chat.set_active(active_players)
        sender = self._external_chat_sender
        if sender is None or sender not in active_players:
            self._human_chat_policy = None
            return None
        policy = self._chat.policy_from(env, sender)
        self._human_chat_policy = policy
        return {
            "sender": sender,
            "target_recipients": list(policy.target_recipients),
            "default_recipient": policy.default_recipient,
        }

    def bounded_broadcast_audiences(
        self, env: Any, messages: list[Message]
    ) -> dict[str, tuple[str, ...]] | None:
        """Resolve the environment-bounded broadcast audiences for one recorded batch, or ``None``."""
        if self._chat is None or not messages:
            return None
        return self._chat.bounded_broadcast_audiences(env, messages)

    def deliver_messages(
        self,
        env: Any,
        messages: list[Message],
        audiences: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Deliver a recorded batch at the end of its sending tick, reusing resolved audiences."""
        if self._chat is not None and messages:
            self._chat.deliver(messages, tick=self._tick(), env=env, audiences=audiences)

    def apply_environment_step(self, context: StepContext) -> ChatOptions | None:
        """Apply one AEC action, credit all rewards, and refresh chat policy."""
        context.env.step(context.action)
        context.reward = float(context.env.rewards[context.player_id])
        for player_id, reward in context.env.rewards.items():
            if player_id in self._state:
                self._state[player_id].score += float(reward)
        return self.refresh_chat_state(context.env)

    def run_learning(self, context: StepContext, *, terminated: bool | None = None) -> None:
        """Run a post-step learning hook and finish per-step compute accounting."""
        binding = context.binding
        if isinstance(binding, AgentPlayer) and has_learn(binding.agent):
            terminated_now = (
                terminated
                if terminated is not None
                else bool(
                    context.env.terminations[context.player_id] or context.env.truncations[context.player_id]
                )
            )
            try:
                self._activate_agent_hook(binding, context.player_id)
                _, context.learn_ms = self._timed_llm_hook(
                    binding.execution_scope,
                    context.player_id,
                    lambda: binding.agent.learn(
                        context.observation, context.action, context.reward, terminated_now
                    ),
                )
            except Exception:  # noqa: BLE001 - preserve the participant exception and attribution
                self._failed(context.player_id)
                raise
            context.agent_compute_ms += context.learn_ms
            context.player.budget_used_ms += context.learn_ms
        if isinstance(binding, AgentPlayer) and context.agent_compute_ms > self._step_limit:
            context.player.step_timeouts += 1

    @staticmethod
    def build_agent_step(context: StepContext):
        """Build the ordinary action-bearing state entry for one participant."""
        return build_agent_step(
            reward=context.reward,
            score=context.player.score,
            action=context.action,
            decision_ms=context.decision_ms,
            learn_ms=context.learn_ms,
            chat_ms=context.chat_ms,
        )

    def _activate_agent_hook(self, binding: AgentPlayer, player_id: str) -> None:
        if binding.execution_scope is not None:
            binding.execution_scope.turn(player_id, self._tick())

    def _timed_llm_hook(
        self, scope: AgentExecutionScope | None, player_id: str, callback: Callable[[], Any]
    ) -> tuple[Any, float]:
        if scope is None:
            started = self._clock.now_ms()
            return callback(), self._clock.now_ms() - started
        before = self._inflight_snapshots.pop(player_id, None)
        if before is None:
            before = scope.inflight_ms(player_id)
        started = self._clock.now_ms()
        cpu_started = self._cpu_clock_ms()
        try:
            value = callback()
        except Exception:
            self._inflight_snapshots.pop(player_id, None)
            raise
        cpu_ms = max(0.0, self._cpu_clock_ms() - cpu_started)
        raw_ms = self._clock.now_ms() - started
        after = scope.inflight_ms(player_id)
        if after is not None:
            self._inflight_snapshots[player_id] = after
        else:
            self._inflight_snapshots.pop(player_id, None)
        if before is None or after is None:
            return value, max(cpu_ms, raw_ms)
        return value, max(cpu_ms, raw_ms - max(0, after - before))


def illegal_action_reason(env: Any, player_id: str, observation: Any, info: Any, action: Any) -> str | None:
    """Return why an action is illegal, or ``None`` when the common checks accept it."""
    space = _declared_action_space(env, player_id)
    if space is not None:
        try:
            contained = bool(space.contains(action))
        except Exception:  # noqa: BLE001 - a space that cannot judge does not veto an action
            contained = True
        if not contained:
            return f"action {action!r} is outside the player's action space"
    mask = action_mask(info, observation)
    if mask is None:
        return None
    if isinstance(mask, Mapping):
        # A Dict action space publishes one sub-mask per subspace key, each judged on its own
        # against the subspace that declares it. A mask whose shape disagrees with the action is
        # the environment's defect, not the agent's, so it withholds a verdict rather than
        # charging the acting player.
        if not isinstance(action, Mapping):
            return None
        components = cast("Mapping[str, Any]", action)
        key = _first_masked_out_key(space, components, cast("Mapping[str, Any]", mask))
        if key is not None:
            return f"action component {key!r}={components[key]!r} is not in the legal-move mask"
        return None
    if isinstance(action, Mapping):
        return None
    if _masked_out(space, action, mask):
        return f"action {action!r} is not in the legal-move mask"
    return None


def _declared_action_space(env: Any, player_id: str) -> Any:
    """The player's declared action space, or ``None`` when the environment cannot supply one."""
    space_fn: Any = getattr(env, "action_space", None)
    if space_fn is None:
        return None
    try:
        return space_fn(player_id)
    except Exception:  # noqa: BLE001 - a space that cannot be fetched simply does not judge
        return None


def _subspaces(space: Any) -> Mapping[str, Any]:
    """The declared per-key subspaces of a Dict space, empty when none are available."""
    declared: Any = getattr(space, "spaces", None)
    return cast("Mapping[str, Any]", declared) if isinstance(declared, Mapping) else {}


def _first_masked_out_key(space: Any, components: Mapping[str, Any], mask: Mapping[str, Any]) -> str | None:
    """The first action component an object mask positively rejects, or ``None`` when none is."""
    subspaces = _subspaces(space)
    for key, entry in mask.items():
        # ``None`` is the spelling for an unrestricted subspace, and the only entry a Box subspace
        # may carry. A key the action omits is the space check's to reject.
        if entry is None or key not in components:
            continue
        if _masked_out(subspaces.get(key), components[key], entry):
            return key
    return None


def _masked_out(space: Any, component: Any, entry: Any) -> bool:
    """Whether one mask entry positively rejects one action component.

    The entry is read against the subspace that declares it: a Discrete entry is a binary vector
    whose position ``i`` covers the action ``start + i``, a MultiDiscrete entry is one such vector
    per dimension, and a nested Dict entry is an object judged key by key. An entry this cannot
    read, or a subspace shape it does not cover, withholds the verdict for that component alone,
    because a mask the platform cannot interpret is the environment's defect and not the agent's.
    """
    try:
        if isinstance(entry, Mapping):
            if not isinstance(component, Mapping):
                return False
            nested = _first_masked_out_key(
                space, cast("Mapping[str, Any]", component), cast("Mapping[str, Any]", entry)
            )
            return nested is not None
        nvec: Any = getattr(space, "nvec", None)
        if nvec is not None:
            starts: Any = getattr(space, "start", None)
            if starts is None:
                starts = [0] * len(nvec)
            if not len(component) == len(entry) == len(starts) == len(nvec):
                return False
            return any(
                _index_masked_out(value, vector, start)
                for value, vector, start in zip(component, entry, starts, strict=True)
            )
        if space is None or hasattr(space, "n"):
            return _index_masked_out(component, entry, getattr(space, "start", 0))
        return False
    except Exception:  # noqa: BLE001 - an unreadable entry withholds this component's verdict
        return False


def _index_masked_out(component: Any, mask: Any, start: Any) -> bool:
    """Whether one binary vector rejects one integer component, counting positions from ``start``."""
    index = int(component) - int(start)
    return 0 <= index < len(mask) and not mask[index]


def action_mask(info: Any, observation: Any) -> Any:
    """Return the published action mask from info or observation when available.

    A flat action space publishes one binary mask indexed by action. A Dict action space
    publishes a mapping of one sub-mask per subspace key. The value is returned as the
    environment wrote it; :func:`illegal_action_reason` is what interprets either shape.
    """
    if isinstance(info, Mapping):
        mask = cast("Mapping[str, Any]", info).get("action_mask")
        if mask is not None:
            return mask
    if isinstance(observation, Mapping):
        return cast("Mapping[str, Any]", observation).get("action_mask")
    return None


def external_window(entry: EnvironmentEntry, binding: ExternalPlayer) -> int | None:
    """An external player's move budget in milliseconds, or ``None`` when they have none.

    A duration rather than an instant: the source spends it only while the person holds the controls,
    so when the budget starts running is the source's business, not this function's.
    """
    if binding.timeout_ms is not None:
        return binding.timeout_ms
    return entry.meta.human_timeout_ms
