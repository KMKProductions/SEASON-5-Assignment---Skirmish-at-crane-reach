"""The single session loop and the step machinery beneath it.

One loop serves sequential and simultaneous environments alike. A sequential step asks its
acting player for an action, while a parallel tick snapshots every active player before it
collects their actions. Live pacing remains outside this module, and headless callers advance
without a wall-clock cadence.

The machinery is exposed as an :class:`Episode`: it owns the reset env and recording writer,
delegates participant accounting to a private runner, and advances one declared unit through
:meth:`Episode.advance`. :func:`run_episode` and the live runner share that dispatch point.
The Stage 2 sequential determinism fixtures remain the regression gate for the AEC path.

A player is bound either to a loaded agent (:class:`AgentPlayer`, governed by the cooperative
agent-timeout machinery) or to an external action source (:class:`ExternalPlayer`, which is
what "human" means to the harness). The two paths are deliberately separate: external players
never consult the per-step agent limit and their ``None`` fallback involves no measurement
or overage accounting.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Real
from typing import Any, cast

from .chat import ChatRouter
from .clock import Clock, SystemClock
from .environment import (
    EnvironmentEntry,
    ParameterValue,
    ResolvedLayout,
    canonical_player_order,
    resolve_layout,
    validate_complete_parameters,
    validate_configured_environment,
    validate_parallel_step,
)
from .participant_runner import (
    ActionSource,
    AgentExecutionScope,
    AgentPlayer,
    ExternalChatFrame,
    ExternalPlayer,
    IllegalAgentActionError,
    MessageSource,
    NoopSource,
    ParticipantRunner,
    Player,
    PlayerState,
    ScriptedSource,
    StepContext,
)
from .recording import RecordingStore
from .state import (
    ChatOptions,
    Message,
    PlayerAttribution,
    StepState,
    build_agent_step,
    build_header,
    build_step_state,
)

__all__ = [
    "ActionSource",
    "AgentExecutionScope",
    "AgentPlayer",
    "Episode",
    "EpisodeResult",
    "ExternalChatFrame",
    "ExternalPlayer",
    "IllegalAgentActionError",
    "MessageSource",
    "NoopSource",
    "Player",
    "REASON_EPISODE_LIMIT",
    "REASON_ERROR",
    "REASON_STOPPED",
    "REASON_TERMINATED",
    "REASON_TRUNCATED",
    "ScriptedSource",
    "run_episode",
]

# Termination reasons reported in EpisodeResult.
REASON_TERMINATED = "terminated"
REASON_TRUNCATED = "truncated"
REASON_EPISODE_LIMIT = "episode_limit"
#: Reported when a live session is ended by an external ``stop`` command rather than by the
#: episode reaching its own end. Only the live loop sets it (via :meth:`Episode.stop`); the
#: headless ``run_episode`` never stops early this way.
REASON_STOPPED = "stopped"
#: Reported when the episode is cut short by an exception (an agent crash, an illegal action, or an
#: environment fault). Only the live runner sets it (via :meth:`Episode.fail`), so a crash can never
#: read as a natural ending.
REASON_ERROR = "error"


@dataclass(frozen=True)
class EpisodeResult:
    """The outcome of one episode."""

    ticks: int
    scores: dict[str, float]
    reason: str
    step_timeouts: dict[str, int]
    recording_id: str | None = None
    #: The one player a failure is chargeable to: the player whose agent raised, or whose own per-episode
    #: budget overran. ``None`` for a clean episode, or a container-level fault no single player owns.
    #: The orchestrator maps it to that player's seat instead of charging every competitor.
    failed_player: str | None = None
    #: A one-line description of the exception that ended the episode, or ``None`` when nothing raised.
    error: str | None = None


def _iso_utc(ms: int) -> str:
    """Render epoch milliseconds as an ISO-8601 UTC date-time string."""
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat()


def _annotated_live_state(
    state: StepState,
    messages: list[Message],
    audiences: Mapping[str, tuple[str, ...]] | None,
) -> StepState | None:
    """Build the live-stream variant of one recorded state, or ``None`` when they are identical.

    A broadcast the environment bounded gains a ``recipients`` list naming its delivered audience,
    so a relay can decide whether a controller heard it. An unbounded broadcast stays unannotated
    and remains visible to everyone. The recorded ``state`` is never touched.
    """
    if audiences is None:
        return None
    live_messages: list[Message] = [
        cast("Message", {**message, "recipients": list(audiences[message["from"]])})
        if message["to"] is None and message["from"] in audiences
        else message
        for message in messages
    ]
    return cast("StepState", {**state, "messages": live_messages})


class Episode:
    """One seeded episode's worth of step machinery, advanced one cycle at a time.

    Construct, :meth:`start` (reset the env and the agents, open the recording), then drive
    ``while not episode.done: episode.advance()`` and read :meth:`result`. :meth:`close`
    flushes the recording and closes the env; the context-manager form pairs ``start`` with it.

    :meth:`step_once` runs one AEC agent-environment cycle. :meth:`step_tick` collects one action
    for every active parallel player and applies one joint transition. Both paths share participant
    accounting and state construction, while :meth:`advance` selects the declared path.
    """

    def __init__(
        self,
        entry: EnvironmentEntry,
        players: Mapping[str, Player],
        *,
        seed: int,
        parameters: Mapping[str, ParameterValue],
        store: RecordingStore | None = None,
        recording_id: str | None = None,
        clock: Clock | None = None,
        cpu_clock_ms: Callable[[], float] | None = None,
        step_limit_ms: int | None = None,
        episode_limit_ms: int | None = None,
        max_steps: int | None = None,
        player_attribution: Mapping[str, PlayerAttribution] | None = None,
        messaging: bool | None = None,
        message_cap: int | None = None,
        layout: ResolvedLayout | None = None,
        external_chat_player: str | None = None,
    ) -> None:
        self._entry = entry
        self._players = players
        self._seed = seed
        self._store = store
        self._recording_id = recording_id
        self._player_attribution = player_attribution
        self._clock = clock or SystemClock()
        self._cpu_clock_ms = cpu_clock_ms or (lambda: time.thread_time_ns() / 1_000_000)
        self._step_limit = step_limit_ms if step_limit_ms is not None else entry.meta.step_limit_ms
        self._episode_limit = (
            episode_limit_ms if episode_limit_ms is not None else entry.meta.episode_limit_ms
        )
        self._max_steps = max_steps
        # The launch configuration is produced by a caller that already resolved every value, so a
        # missing name is an upstream bug rather than a request to substitute a default.
        self._parameters = validate_complete_parameters(entry.meta, parameters)
        resolved_layout = resolve_layout(entry.meta, self._parameters)
        if layout is not None and layout != resolved_layout:
            raise ValueError("episode layout does not match the environment's resolved parameters")
        self._layout = resolved_layout

        # Messaging is enabled only when the environment metadata AND the session config agree, and
        # the effective cap is the minimum of the two, so a config override can disable or tighten but
        # never enable messaging on an environment that opted out. Combined once here, the single
        # authority; live.py reads the result back through ``messaging_enabled``. With messaging off no
        # router exists and the loop is byte-identical to a pre-chat run.
        self._messaging = entry.meta.messaging and (messaging if messaging is not None else True)
        caps = [c for c in (entry.meta.message_cap, message_cap) if c is not None]
        self._message_cap = min(caps) if caps else None
        self._chat: ChatRouter | None = (
            ChatRouter(players.keys(), self._message_cap) if self._messaging else None
        )
        # Action control may span several external players, but chat authority is always explicit.
        if external_chat_player is not None and not isinstance(
            players.get(external_chat_player), ExternalPlayer
        ):
            raise ValueError("external_chat_player must name an external player binding")
        self._external_chat_sender = external_chat_player
        self._state = {player_id: PlayerState() for player_id in players}
        self._env: Any = None
        self._writer: Any = None
        self._writer_cm: Any = None
        self._reason = REASON_TERMINATED
        self._tick = 0
        self._stopped = False
        self._failed_player: str | None = None
        self._error: str | None = None
        self._opening_chat_options: ChatOptions | None = None
        self._parallel_observations: Mapping[str, object] | None = None
        self._parallel_infos: Mapping[str, object] | None = None
        self._saw_natural_truncation = False
        self._complete_result_scores: dict[str, float] | None = None
        self._result_scores_checked = False
        self._participant_runner = ParticipantRunner(
            entry,
            players,
            self._state,
            clock=self._clock,
            cpu_clock_ms=self._cpu_clock_ms,
            step_limit_ms=self._step_limit,
            chat=self._chat,
            external_chat_sender=self._external_chat_sender,
            active_players=self._logical_active_players,
            tick=lambda: self._tick,
            failed=self._set_failed_player,
        )

    def start(self) -> None:
        """Reset the environment, open the recording, then reset the agents.

        The env is created here, not in ``__init__``, so a failure in ``env.reset`` or an
        agent's ``reset`` still leaves a constructed :class:`Episode` whose :meth:`close` can
        run; callers using the context-manager form get that for free.

        The recording header is opened *after* the environment resets but *before* the participants
        reset, and each participant ``reset`` is charged to its own player. So an agent whose ``reset``
        raises is attributed to that player (:attr:`failed_player`) over a readable recording, rather
        than looking like an unowned infrastructure fault that yields no recording at all. A failure
        in ``env.reset`` itself, before any player has been touched, stays unowned by design.

        Any startup failure closes the half-opened recording writer and the constructed env before
        re-raising: a context-manager caller (``run_episode``) never reaches ``__exit__`` when
        ``__enter__`` raises, so without this the writer's file handle and the env would leak.
        :meth:`close` is idempotent and leaves :attr:`failed_player` intact, so the live runner's own
        best-effort close and a charged reset crash both keep working.
        """
        try:
            env = self._entry.make(self._parameters)
            self._env = env
            reset_result = env.reset(seed=self._seed)
            validate_configured_environment(self._entry, env, self._layout.players, reset_result)
            if self._entry.meta.stepping == "simultaneous":
                # The reset mapping is the first pre-step snapshot. Later ticks replace it only after
                # their joint transition has passed the strict parallel contract check.
                self._parallel_observations, self._parallel_infos = cast(
                    "tuple[Mapping[str, object], Mapping[str, object]]", reset_result
                )
            self._opening_chat_options = self._participant_runner.refresh_chat_state(env)

            if self._store is not None:
                created_at_ms = self._clock.now_ms()
                if self._recording_id is None:
                    self._recording_id = f"{self._entry.meta.env_id}-seed{self._seed}-{created_at_ms}"
                overlay_static = (
                    self._entry.overlay_static(env) if self._entry.overlay_static is not None else None
                )
                header = build_header(
                    environment=self._entry.meta.env_id,
                    seed=self._seed,
                    created_at=_iso_utc(created_at_ms),
                    parameters=self._parameters,
                    overlay_static=overlay_static,
                    players=self._recording_players(),
                    layout=self._layout,
                )
                self._writer_cm = self._store.create(self._recording_id, header)
                self._writer = self._writer_cm.__enter__()

            if self._entry.meta.stepping == "simultaneous":
                if self._parallel_observations is None:
                    raise RuntimeError("parallel reset supplied no observations")
                reset_observations = self._parallel_observations
            else:
                reset_observations = {
                    player_id: env.observe(player_id)
                    for player_id, binding in self._players.items()
                    if isinstance(binding, AgentPlayer)
                }
            self._participant_runner.reset_agents(self._seed, reset_observations)
            self._stop_if_over_budget(self._players)
        except Exception:  # noqa: BLE001 - release the half-opened recording/env, then re-raise as-is
            # Suppress any close fault so it never masks the original startup error (which the headless
            # caller still receives and which carries the player attribution set just above).
            with contextlib.suppress(Exception):
                self.close()
            raise

    def _recording_players(self) -> dict[str, PlayerAttribution]:
        """Return the supplied attributions, or a human-only default for direct harness callers.

        A replay names who drove each player from the recording alone, so the header has to be true.
        An external player is fully described by the fact that a person drove it, which is why the
        default covers that case. An agent player is not: the harness holds a loaded callable and
        cannot tell which submission or built-in it came from, and guessing would write a false
        identity into an artifact that is later read as authoritative. Recording one therefore
        requires the caller to say what it is.
        """
        if self._player_attribution is not None:
            return dict(self._player_attribution)
        players: dict[str, PlayerAttribution] = {}
        for player_id, player in self._players.items():
            if not isinstance(player, ExternalPlayer):
                raise ValueError(
                    f"recording agent player {player_id!r} requires player_attribution naming it"
                )
            players[player_id] = {"kind": "human", "label": "Human"}
        return players

    @property
    def done(self) -> bool:
        """Whether the loop should stop: the env has no acting agents, or a check tripped."""
        return self._stopped or not self._env.agents

    @property
    def tick(self) -> int:
        """The number of steps recorded so far."""
        return self._tick

    @property
    def stepping(self) -> str:
        """The entry's already validated stepping declaration."""
        return self._entry.meta.stepping

    @property
    def failed_player(self) -> str | None:
        """The player at fault, or ``None``: the player whose agent raised, or whose budget overran.

        Set the instant a player is to blame so the live runner can name it in the result envelope even
        while a crashing agent's exception is propagating out of the loop. Workflow reduction maps the
        failure to that player's seat instead of to every competitor sharing the container.
        """
        return self._failed_player

    @property
    def messaging_enabled(self) -> bool:
        """Whether messaging is effectively on (metadata AND config), resolved once in ``__init__``.

        The live runner reuses this to gate the human chat queue, so the AND/min combination has a
        single authority rather than being recomputed.
        """
        return self._messaging

    @property
    def external_chat_sender(self) -> str | None:
        """The one external player authorized to submit chat, when messaging is effective."""
        return self._external_chat_sender if self._chat is not None else None

    def opening_state(self) -> StepState | None:
        """The pre-action "opening" frame: the dealt overlay with no agent having acted yet.

        A turn-based environment can require the first acting player, possibly a connected human
        (Hearts' 2♣ leader), to act before any :meth:`step_once` has produced a frame, leaving the
        client with an empty table and nothing to render. The live runner streams this one frame
        right after :meth:`start`, so the table (and the human's own hand) is visible immediately and
        the human can play. Returns ``None`` for a paced environment, which steps on its own cadence
        and renders its first frame within an interval, or one with neither an overlay nor chat
        options to publish.

        This is a live-presentation aid only and is never written through the recording: recordings
        and the headless path are byte-for-byte unchanged, so a replay still begins at the first play.
        Valid only after :meth:`start` (the env must be reset); the live runner calls it there.
        """
        simultaneous = self._entry.meta.stepping == "simultaneous"
        if not simultaneous and self._entry.meta.pace_interval_ms is not None:
            return None
        overlay = self._entry.overlay(self._env) if self._entry.overlay is not None else None
        if not simultaneous and overlay is None and self._opening_chat_options is None:
            return None
        return build_step_state(
            tick=0,
            agents={},
            started_at=self._clock.now_ms(),
            duration_ms=0,
            overlay=overlay,
            chat_options=self._opening_chat_options,
        )

    def stop(self, reason: str = REASON_STOPPED) -> None:
        """Mark the episode finished from outside (the live ``stop`` command).

        Sets ``done`` so the next loop check ends the run, and records ``reason`` for the
        result. Headless ``run_episode`` never calls this; only the live loop does.
        """
        self._reason = reason
        self._stopped = True

    def fail(self, description: str) -> None:
        """Mark the episode as ended by an exception, keeping any player already charged for it."""
        self._reason = REASON_ERROR
        self._error = description
        self._stopped = True

    def advance(self) -> None:
        """Advance one declared AEC step or one declared parallel tick."""
        if self._entry.meta.stepping == "simultaneous":
            self.step_tick()
        else:
            self.step_once()

    def _set_failed_player(self, player_id: str) -> None:
        """Record the participant at fault before its original exception propagates."""
        self._failed_player = player_id

    def step_once(self) -> None:
        """Advance the acting player by exactly one PettingZoo cycle. See the class docstring."""
        env = self._env
        player_id = env.agent_selection
        observation, _reward, termination, truncation, info = env.last()

        if termination or truncation:
            self._saw_natural_truncation = self._saw_natural_truncation or truncation
            env.step(None)
            self._finish_without_recorded_step()
            return

        binding = self._players[player_id]
        context = StepContext(
            env=env,
            player_id=player_id,
            observation=observation,
            info=info,
            binding=binding,
            player=self._state[player_id],
            started_at=self._clock.now_ms(),
        )
        self._participant_runner.select_action(context)

        self._participant_runner.collect_messages(context)
        context.chat_options = self._participant_runner.apply_environment_step(context)
        self._saw_natural_truncation = self._saw_natural_truncation or any(
            bool(truncated) for truncated in env.truncations.values()
        )
        self._participant_runner.run_learning(context)

        audiences = self._participant_runner.bounded_broadcast_audiences(env, context.messages)
        self._record_step((context,), context.started_at, context.messages, context.chat_options, audiences)
        self._participant_runner.deliver_messages(env, context.messages, audiences)
        self._finish_step((context.player_id,))

    def step_tick(self) -> None:
        """Advance every active parallel player from one saved pre-step snapshot."""
        env = self._env
        active_players = canonical_player_order(env.agents)
        observations = self._parallel_observations
        infos = self._parallel_infos
        if observations is None or infos is None:
            raise RuntimeError("parallel episode has no saved reset or step snapshot")
        started_at = self._clock.now_ms()
        contexts = [
            StepContext(
                env=env,
                player_id=player_id,
                observation=observations[player_id],
                info=infos[player_id],
                binding=self._players[player_id],
                player=self._state[player_id],
                started_at=started_at,
            )
            for player_id in active_players
        ]

        # Consume every external latch at the boundary before an agent can begin work. Agent callbacks
        # remain canonical and sequential, while the action map below keeps canonical key order.
        for context in contexts:
            if isinstance(context.binding, ExternalPlayer):
                self._participant_runner.select_action(context)
        for context in contexts:
            if isinstance(context.binding, AgentPlayer):
                self._participant_runner.select_action(context)

        messages = self._participant_runner.drain_human_messages()
        for context in contexts:
            self._participant_runner.collect_agent_messages(context, messages)

        actions = {context.player_id: context.action for context in contexts}
        step_result = env.step(actions)
        observations, rewards, terminations, truncations, infos = validate_parallel_step(
            self._entry.meta, env, active_players, actions, step_result
        )
        self._parallel_observations = observations
        self._parallel_infos = infos
        self._saw_natural_truncation = self._saw_natural_truncation or any(
            bool(truncations[player_id]) for player_id in active_players
        )
        for context in contexts:
            context.reward = float(cast("float", rewards[context.player_id]))
            context.player.score += context.reward
        chat_options = self._participant_runner.refresh_chat_state(env)
        for context in contexts:
            terminated = bool(terminations[context.player_id] or truncations[context.player_id])
            self._participant_runner.run_learning(context, terminated=terminated)

        audiences = self._participant_runner.bounded_broadcast_audiences(env, messages)
        self._record_step(tuple(contexts), started_at, messages, chat_options, audiences)
        self._participant_runner.deliver_messages(env, messages, audiences)
        self._finish_step(active_players)

    def _logical_active_players(self) -> tuple[str, ...]:
        """Return players in ``env.agents`` that are not marked terminal, including live AEC players."""
        if self._entry.meta.stepping == "simultaneous":
            return canonical_player_order(self._env.agents)
        return tuple(
            player_id
            for player_id in canonical_player_order(self._env.agents)
            if not self._env.terminations.get(player_id, False)
            and not self._env.truncations.get(player_id, False)
        )

    def _record_step(
        self,
        contexts: tuple[StepContext, ...],
        started_at: int,
        messages: list[Message],
        chat_options: ChatOptions | None,
        audiences: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Persist one completed AEC action or parallel tick before message delivery.

        ``audiences`` carries the environment-bounded broadcast audiences resolved for this batch.
        When present, the live stream's copy of the state annotates each bounded broadcast with its
        delivered ``recipients`` so relays can filter a controller's view; the persisted state stays
        exactly ``{from, to, text}``.
        """
        if self._writer is None:
            return
        env = contexts[0].env
        overlay = self._entry.overlay(env) if self._entry.overlay is not None else None
        agents = {
            context.player_id: self._participant_runner.build_agent_step(context) for context in contexts
        }
        if self._entry.meta.stepping == "sequential":
            # AEC can publish rewards and terminal flags for non-actors. Snapshot those deltas before
            # its required dead steps clear the mappings, but never invent a participant hook or action.
            actor = contexts[0].player_id
            for player_id in canonical_player_order(env.rewards):
                if player_id == actor or player_id not in self._state:
                    continue
                if (
                    float(env.rewards[player_id]) != 0.0
                    or bool(env.terminations.get(player_id, False))
                    or bool(env.truncations.get(player_id, False))
                ):
                    agents[player_id] = build_agent_step(
                        reward=float(env.rewards[player_id]), score=self._state[player_id].score
                    )
        state = build_step_state(
            tick=self._tick,
            agents=agents,
            started_at=started_at,
            duration_ms=self._clock.now_ms() - started_at,
            overlay=overlay,
            messages=messages or None,
            chat_options=chat_options,
        )
        self._writer.write_step(state, _annotated_live_state(state, messages, audiences))

    def _finish_without_recorded_step(self) -> None:
        """Resolve a natural AEC ending reached while consuming a dead step."""
        if not self._env.agents:
            self._mark_natural_end()

    def _finish_step(self, acted_players: tuple[str, ...]) -> None:
        """Advance once, then apply budget, natural-ending, and tick-cap precedence."""
        self._tick += 1
        if self._stop_if_over_budget(acted_players):
            return
        if not self._logical_active_players():
            self._mark_natural_end()
            # An AEC env still holds agents queued for their required dead steps; only an env with
            # nothing left to consume stops the loop here.
            self._stopped = not self._env.agents
            return
        if self._max_steps is not None and self._tick >= self._max_steps:
            self._reason = REASON_TRUNCATED
            self._stopped = True

    def _stop_if_over_budget(self, players: Mapping[str, Player] | tuple[str, ...]) -> bool:
        """Stop for the first canonical supplied player beyond the episode compute budget."""
        over_budget = tuple(
            player_id
            for player_id in canonical_player_order(players)
            if self._state[player_id].budget_used_ms > self._episode_limit
        )
        if not over_budget:
            return False
        self._reason = REASON_EPISODE_LIMIT
        self._failed_player = over_budget[0]
        self._stopped = True
        return True

    def _mark_natural_end(self) -> None:
        """Record the natural ending reason and its complete scores in one place."""
        self._reason = REASON_TRUNCATED if self._saw_natural_truncation else REASON_TERMINATED
        self._capture_complete_result_scores()

    def _capture_complete_result_scores(self) -> None:
        """Cache an environment's optional complete scores at natural completion, before close."""
        if self._result_scores_checked:
            return
        self._result_scores_checked = True
        complete_scores = getattr(self._env, "result_scores", None)
        if not callable(complete_scores):
            return
        reported = complete_scores()
        if reported is None:
            return
        if not isinstance(reported, Mapping):
            raise TypeError("environment result_scores must return a player-keyed mapping or None")
        result_scores = cast("Mapping[object, object]", reported)
        expected_players = set(self._players)
        if set(result_scores) != expected_players:
            raise ValueError("environment result_scores must cover exactly the episode players")
        normalized: dict[str, float] = {}
        for player_id in self._players:
            value = result_scores[player_id]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("environment result_scores values must be real numbers")
            score = float(value)
            if not math.isfinite(score):
                raise ValueError("environment result_scores values must be finite")
            normalized[player_id] = score
        self._complete_result_scores = normalized

    def close(self) -> None:
        """Flush and close the recording, then close the environment. Idempotent."""
        if self._writer_cm is not None:
            self._writer_cm.__exit__(None, None, None)
            self._writer_cm = None
            self._writer = None
        if self._env is not None:
            close = getattr(self._env, "close", None)
            if callable(close):
                close()

    def result(self) -> EpisodeResult:
        """The outcome accumulated so far. Valid after the loop ends; safe after :meth:`close`."""
        scores = {player_id: self._state[player_id].score for player_id in self._players}
        if self._complete_result_scores is not None:
            scores = dict(self._complete_result_scores)
        return EpisodeResult(
            ticks=self._tick,
            scores=scores,
            reason=self._reason,
            step_timeouts={player_id: self._state[player_id].step_timeouts for player_id in self._players},
            recording_id=self._recording_id if self._store is not None else None,
            failed_player=self._failed_player,
            error=self._error,
        )

    def __enter__(self) -> Episode:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def run_episode(
    entry: EnvironmentEntry,
    players: Mapping[str, Player],
    *,
    seed: int,
    parameters: Mapping[str, ParameterValue],
    store: RecordingStore | None = None,
    recording_id: str | None = None,
    clock: Clock | None = None,
    cpu_clock_ms: Callable[[], float] | None = None,
    step_limit_ms: int | None = None,
    episode_limit_ms: int | None = None,
    max_steps: int | None = None,
    player_attribution: Mapping[str, PlayerAttribution] | None = None,
    messaging: bool | None = None,
    message_cap: int | None = None,
    external_chat_player: str | None = None,
) -> EpisodeResult:
    """Play one seeded episode of ``entry`` with the given player bindings.

    A thin headless loop over :class:`Episode`: reset seeds everything (the environment via
    ``reset(seed=seed)`` and every agent via its own ``reset(seed, observation)``), then call
    :meth:`Episode.advance` until the declared AEC or parallel environment ends, recording through
    the store when one is given.
    Recording is optional so the evaluation pattern (run many seeds, keep scores, store
    nothing) shares this exact code path. This loop never paces and never pauses; that is the
    live runner's job, layered around the same :meth:`Episode.advance`.
    """
    with Episode(
        entry,
        players,
        seed=seed,
        store=store,
        recording_id=recording_id,
        clock=clock,
        cpu_clock_ms=cpu_clock_ms,
        step_limit_ms=step_limit_ms,
        episode_limit_ms=episode_limit_ms,
        max_steps=max_steps,
        player_attribution=player_attribution,
        messaging=messaging,
        message_cap=message_cap,
        external_chat_player=external_chat_player,
        parameters=parameters,
    ) as episode:
        while not episode.done:
            episode.advance()
    return episode.result()
