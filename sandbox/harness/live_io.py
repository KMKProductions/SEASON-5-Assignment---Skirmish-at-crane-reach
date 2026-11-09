"""Transport primitives for the live session runner.

These are the container-side halves of the Stage 3 transport: the inbound command pump and
its shared control state, the :class:`PausableClock` that freezes time on pause, the
:class:`TransportSource` that feeds an external player from latched inputs, the protocol output
stream, and the tee that mirrors recording bytes onto that stream. The runner that wires them
together lives in :mod:`game_sandbox_harness.live`; everything here is independently testable
on a :class:`~game_sandbox_harness.clock.ManualClock`.

The line shapes are defined in the Stage 3 transport plan. Outbound: recording lines
(header + per-step states, never carrying a top-level ``kind``) and event envelopes (a
top-level ``kind``; this stage emits one, ``result``). Inbound command envelopes carry a
``kind`` and, where applicable, a ``player`` and ``action`` or ``text``: ``input``, ``clock``
(``player`` + ``running``, whether a human holds that player's controls), ``pause``, ``resume``,
``stop``, and ``chat`` (a human message, ``player`` + ``to`` + ``text``). Unknown
kinds and malformed lines are logged and ignored, so the container never dies because a
client sent garbage. Human ``chat`` frames enter one bounded designated-sender FIFO, not the input
latch: inputs coalesce to the latest value, but messages must not swallow each other.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Iterable, Mapping
from typing import IO, TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from .clock import Clock
from .recording.local import FolderRecordingStore, dump_line
from .session import EpisodeResult, ExternalChatFrame

#: The single outbound event-envelope kind this stage defines.
RESULT_KIND = "result"
_DEFAULT_SLICE_MS = 5
#: The most human ``chat`` frames the designated sender may have queued at once. When it is full the
#: incoming frame is dropped with a diagnostic, so a client flooding the socket costs at most a
#: fixed amount of memory, the same drop-with-diagnostic rule every other rejection follows.
CHAT_QUEUE_LIMIT = 16


def _diag(message: str) -> None:
    """Write a diagnostic line to stderr; it is never part of the protocol stream."""
    print(message, file=sys.stderr, flush=True)


class PausableClock:
    """A :class:`~game_sandbox_harness.clock.Clock` that subtracts accumulated paused time.

    Wraps any base clock, including ``ManualClock``, and reports ``base - total_paused``.
    Wall-clock timing and realtime cadence freeze while paused. The official LLM calling-thread CPU
    floor uses a separate clock and remains chargeable during a pause. Thread-safe: the stepping
    thread reads ``now_ms`` while the command pump calls :meth:`pause`/:meth:`resume`.
    """

    def __init__(self, base: Clock) -> None:
        self._base = base
        self._lock = threading.Lock()
        self._paused_total_ms = 0
        self._paused_at: int | None = None

    def now_ms(self) -> int:
        with self._lock:
            base = self._base.now_ms()
            paused = self._paused_total_ms
            if self._paused_at is not None:
                paused += base - self._paused_at
            return base - paused

    def pause(self) -> None:
        with self._lock:
            if self._paused_at is None:
                self._paused_at = self._base.now_ms()

    def resume(self) -> None:
        with self._lock:
            if self._paused_at is not None:
                self._paused_total_ms += self._base.now_ms() - self._paused_at
                self._paused_at = None


class SessionControl:
    """Thread-safe control state updated by inbound commands and read by the stepping thread.

    The command pump calls :meth:`handle_line` for each inbound line; the stepping thread reads
    :attr:`paused`, :attr:`stopping`, :meth:`take`, and :meth:`clock_running`. Pause and resume
    drive the injected :class:`PausableClock` so the freeze takes effect the instant the command
    arrives, not on the next step boundary.
    """

    def __init__(self, clock: PausableClock | None = None) -> None:
        self._lock = threading.Lock()
        self._latched: dict[str, Any] = {}
        self._chat_queue: list[ExternalChatFrame] = []
        self._chat_sender: str | None = None
        self._paused = False
        self._stopping = False
        self._clock = clock
        # Whether a human currently holds each player's controls, latched from ``clock`` commands.
        self._clock_running: dict[str, bool] = {}

    def configure_chat(self, sender: str | None) -> None:
        """Configure the one external sender authorized to enqueue human ``chat`` frames.

        Called once by the live runner after the effective messaging config is known (the metadata
        AND the session config, resolved in :class:`~game_sandbox_harness.session.Episode`). With
        messaging off or without a designated sender, ``chat`` frames are dropped with a diagnostic.
        """
        with self._lock:
            self._chat_sender = sender
            self._chat_queue = []

    def handle_line(self, raw: str) -> None:
        """Parse and dispatch the inbound command(s) on one line; malformed/unknown lines are ignored.

        Normally a line is exactly one JSON command. The Docker attach hijack is the exception: it
        prepends its options object to the very first command with no separating newline. docker-modem
        carries a hijacked attach by writing the attach options as the request body, and that body is
        also the only thing that flushes the upgrade, so it cannot be suppressed; on the hijacked
        stream those bytes land at the head of the container's stdin (see ``DockerSessionProcess``).
        A line may therefore carry more than one JSON value. Decode them all and dispatch each instead
        of dropping the whole line. This stops the first real input from being lost with the
        transport preamble. A multi-value line is, in practice, only ever that preamble fused with the
        real command, so its leading kind-less object is skipped quietly; a lone kind-less line is a
        client error and still earns a diagnostic.
        """
        for command in parse_commands(raw):
            self._dispatch_command(command, raw)

    def _dispatch_command(self, command: dict[str, Any], source: str) -> None:
        """Apply one already-validated command envelope."""
        kind = command["kind"]
        if kind == "input":
            with self._lock:
                self._latched[command["player"]] = command.get("action")
        elif kind == "chat":
            self._dispatch_chat(command, source)
        elif kind == "clock":
            with self._lock:
                self._clock_running[command["player"]] = cast("bool", command["running"])
        elif kind == "pause":
            self.pause()
        elif kind == "resume":
            self.resume()
        else:  # stop
            with self._lock:
                self._stopping = True

    def _dispatch_chat(self, command: dict[str, Any], source: str) -> None:
        """Queue a human ``chat`` frame into its player's bounded FIFO, or drop it with a diagnostic.

        :func:`parse_commands` is the shared shape gate for stdin and the local relay. This method only
        applies the sender and queue-capacity checks. Recipient legality, the text cap, and per-boundary
        limits are enforced once at the harness validation point shared with agent messages.
        """
        player_id = cast("str", command["player"])
        with self._lock:
            # One check covers both rejections: with messaging off the designated sender is None, so
            # every frame fails it, and the diagnostic names who may send instead of guessing why.
            if player_id != self._chat_sender:
                _diag(
                    f"live: dropping chat command from {player_id!r}; the designated sender is "
                    f"{self._chat_sender!r}: {source!r}"
                )
                return
            if len(self._chat_queue) >= CHAT_QUEUE_LIMIT:
                _diag(f"live: dropping chat command (queue full for {player_id!r}): {source!r}")
                return
            self._chat_queue.append(
                {
                    "to": cast("str | None", command["to"]),
                    "text": cast("str", command["text"]),
                }
            )

    def take(self, player_id: str) -> Any | None:
        """Return and clear the latest input latched for ``player_id`` since the last call.

        Latching is per step, per [interaction.md]: a step uses the most recent input that
        arrived since the previous step, or ``None`` (the loop applies the default action) when
        none did. Clearing on read is what makes an unrepeated Flappy Bird flap a single flap.
        """
        with self._lock:
            return self._latched.pop(player_id, None)

    def take_chat(self, player_id: str) -> list[ExternalChatFrame]:
        """Return and clear the human ``chat`` frames queued for ``player_id`` in FIFO order.

        Unlike :meth:`take`, this is a queue, not a latch: every queued frame is returned in the
        order it arrived, because messages must not swallow each other. Drained at each completed
        boundary by the session loop and passed through the same validator as agent messages.
        """
        with self._lock:
            if player_id != self._chat_sender or not self._chat_queue:
                return []
            queue, self._chat_queue = self._chat_queue, []
            return queue

    def clock_running(self, player_id: str) -> bool:
        """Whether a human currently holds ``player_id``'s controls.

        False until a client says otherwise, so a move budget is never spent on a person who has not
        been given the controls yet. See :meth:`TransportSource.get_action` for what that buys.
        """
        with self._lock:
            return self._clock_running.get(player_id, False)

    def pause(self) -> None:
        with self._lock:
            if not self._paused:
                self._paused = True
                if self._clock is not None:
                    self._clock.pause()

    def resume(self) -> None:
        with self._lock:
            if self._paused:
                self._paused = False
                if self._clock is not None:
                    self._clock.resume()

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def stopping(self) -> bool:
        with self._lock:
            return self._stopping


def _chat_command_error(command: dict[str, Any]) -> str | None:
    """Return the reason a ``chat`` command envelope is malformed, or ``None`` when it is well-shaped.

    Checked in one place so the three distinct shape rules (player, text, ``to``) each
    keep their own diagnostic without repeating the ``kind == "chat"`` guard three times. This is
    only the shape gate shared by stdin and the local relay: recipient legality, the text cap, and
    per-boundary limits are enforced once at the harness validation point shared with agent messages.
    """
    if not isinstance(command.get("player"), str):
        return "a string player"
    if not isinstance(command.get("text"), str):
        return "string text"
    to = command.get("to")
    if "to" not in command or (to is not None and not isinstance(to, str)):
        return "a string or null to"
    return None


def parse_commands(raw: str) -> list[dict[str, Any]]:
    """Decode and cheaply validate inbound commands shared by stdin and the local relay."""
    text = raw.strip()
    if not text:
        return []
    decoder = json.JSONDecoder()
    values: list[object] = []
    index = 0
    while index < len(text):
        try:
            parsed, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            _diag(f"live: ignoring malformed command line: {text[index:]!r}")
            break
        values.append(parsed)
        index = end
        while index < len(text) and text[index].isspace():
            index += 1
    accepted: list[dict[str, Any]] = []
    quiet_missing_kind = len(values) > 1
    for value in values:
        if not isinstance(value, dict) or "kind" not in value:
            if not quiet_missing_kind:
                _diag(f"live: ignoring command without a kind: {text!r}")
            continue
        command = cast("dict[str, Any]", value)
        kind = command.get("kind")
        if kind not in ("input", "chat", "clock", "pause", "resume", "stop"):
            _diag(f"live: ignoring unknown command kind {kind!r}")
            continue
        if kind == "input" and not isinstance(command.get("player"), str):
            _diag(f"live: ignoring input command without a string player: {text!r}")
            continue
        if kind == "clock":
            if not isinstance(command.get("player"), str):
                _diag(f"live: ignoring clock command without a string player: {text!r}")
                continue
            if not isinstance(command.get("running"), bool):
                _diag(f"live: ignoring clock command without a boolean running: {text!r}")
                continue
        if kind == "chat":
            reason = _chat_command_error(command)
            if reason is not None:
                _diag(f"live: ignoring chat command without {reason}: {text!r}")
                continue
        accepted.append(command)
    return accepted


@runtime_checkable
class Sleeper(Protocol):
    """A cooperative sleep, sliced so pause and stop stay responsive while waiting."""

    def sleep_ms(self, ms: int) -> None: ...


class RealSleeper:
    """A :class:`Sleeper` backed by ``time.sleep`` for live container runs."""

    def sleep_ms(self, ms: int) -> None:
        if ms > 0:
            time.sleep(ms / 1000)


class TransportSource:
    """An :class:`~game_sandbox_harness.session.ActionSource` over latched transport inputs.

    With a pace interval set, the cadence is the world clock, so the source returns the latched
    input (or ``None``) immediately and the loop's pacing does the waiting. With no pace interval
    the player is turn-based: the source blocks in short slices until an input arrives, the move
    budget is spent, or a stop is requested. Either way, a ``None`` return routes through
    ``ExternalPlayer``'s existing default, a noop for Flappy Bird, with no agent-timeout accounting.

    A turn-based budget is spent only while the person actually holds the controls, which the client
    reports with ``clock`` commands. The browser plays agent turns through an animation queue, so the
    loop can reach a human seconds before their controls open, and none of that gap is theirs to lose.
    Waiting on a person who never takes the controls is bounded by the session's idle and duration
    limits rather than by this source.
    """

    def __init__(
        self,
        control: SessionControl,
        *,
        clock: Clock,
        paced: bool,
        sleeper: Sleeper | None = None,
        slice_ms: int = _DEFAULT_SLICE_MS,
    ) -> None:
        self._control = control
        self._clock = clock
        self._paced = paced
        self._sleeper = sleeper or RealSleeper()
        self._slice_ms = slice_ms

    def get_action(self, player_id: str, observation: Any, window_ms: int | None) -> Any:
        if self._paced:
            return self._control.take(player_id)
        # Held time accumulates per call, so every turn starts with the full budget while the latched
        # running flag itself persists across turns.
        held_ms = 0
        last_ms = self._clock.now_ms()
        while True:
            if self._control.stopping:
                return None
            if self._control.paused:
                self._sleeper.sleep_ms(self._slice_ms)
                last_ms = self._clock.now_ms()
                continue
            # An input counts whether or not the controls are reported as held, so a dropped or late
            # clock command can never swallow a real action.
            value = self._control.take(player_id)
            if value is not None:
                return value
            now_ms = self._clock.now_ms()
            if self._control.clock_running(player_id):
                held_ms += now_ms - last_ms
            last_ms = now_ms
            if window_ms is not None and held_ms >= window_ms:
                return None
            self._sleeper.sleep_ms(self._slice_ms)

    def take_messages(self, player_id: str) -> list[ExternalChatFrame]:
        """Return the human ``chat`` frames queued for ``player_id`` since the last drain.

        Detected by presence on the source, like the action-source protocol: the session loop finds
        this method with ``getattr`` and drains it at each completed boundary, so it never needs to know
        about :class:`SessionControl`. Non-transport sources (noop, scripted) have no such method and
        are skipped, so a message can only originate from a real external player.
        """
        return self._control.take_chat(player_id)


class ProtocolStream:
    """The single-writer outbound protocol sink, owned by the stepping thread.

    Recording lines reach it pre-serialized through :meth:`emit_raw` (the tee mirror), and event
    envelopes through :meth:`emit_envelope`. Only the stepping thread writes here. The command pump
    reads stdin and never touches this, so no lock is needed.
    """

    def __init__(self, handle: IO[str]) -> None:
        self._handle = handle

    def emit_raw(self, line: str) -> None:
        """Write an already-serialized line (the recording serializer's exact bytes)."""
        self._handle.write(line if line.endswith("\n") else line + "\n")
        self._handle.flush()

    def emit_envelope(self, obj: Mapping[str, Any]) -> None:
        """Serialize and write one event envelope (for example the ``result`` line)."""
        self.emit_raw(json.dumps(obj, separators=(",", ":"), sort_keys=True))

    def emit_state(self, state: Mapping[str, Any]) -> None:
        """Serialize and write one recording-shaped state line that bypasses the recording.

        Used for the live-only opening frame (see :meth:`Episode.opening_state`): it is streamed to
        the client but never persisted, so it goes through the recording store's own serializer and
        is byte-for-byte a recorded state line, which the client parses identically. That shared call
        is also what lets the frame carry an overlay holding NumPy leaves, as recorded states do.
        It carries no top-level ``kind``, so the classification rule routes it to the renderer as a state.
        """
        self.emit_raw(dump_line(state))


def build_tee_store(recordings_root: str, protocol: ProtocolStream) -> FolderRecordingStore:
    """A recording store that mirrors every serialized line onto ``protocol``.

    Wraps :class:`FolderRecordingStore` by handing it the protocol's
    :meth:`~ProtocolStream.emit_raw` as the store's ``on_line`` mirror: each header and state
    line is serialized once and written to both the recording on the mounted volume and the
    protocol stream, so the streamed bytes and the stored bytes cannot differ.
    """
    return FolderRecordingStore(recordings_root, on_line=protocol.emit_raw)


def result_envelope(result: EpisodeResult) -> dict[str, Any]:
    """Build the outbound ``result`` envelope from an :class:`EpisodeResult`.

    Emitted once at session end. Never written to the recording: it carries a top-level
    ``kind``, which the state schema forbids, so the classification rule keeps it out.
    """
    return {
        "kind": RESULT_KIND,
        "ticks": result.ticks,
        "scores": result.scores,
        "reason": result.reason,
        "step_timeouts": result.step_timeouts,
        "recording_id": result.recording_id,
        "failed_player": result.failed_player,
        "error": result.error,
    }


def session_envelope(
    status: str,
    reason: str | None = None,
    awaiting_start: bool = False,
) -> dict[str, Any]:
    """Build a shared local-relay lifecycle envelope without touching recording bytes.

    A running session always carries the start-gate state. Other lifecycle states keep their
    existing shape because ``awaiting_start`` only describes a running session.
    """
    envelope: dict[str, Any] = {"kind": "session", "status": status}
    if status == "running":
        envelope["awaiting_start"] = awaiting_start
    if reason is not None:
        envelope["reason"] = reason
    return envelope


def start_command_pump(lines: Iterable[str], control: SessionControl) -> threading.Thread:
    """Start a daemon thread that pumps ``lines`` (typically stdin) into ``control``.

    The thread owns the only blocking read in the runner, so the stepping thread never stalls on
    input. It swallows its own errors to a diagnostic line: a broken inbound channel must not
    take the session down, since the episode can still run to its natural end.
    """

    def run() -> None:
        try:
            for raw in lines:
                control.handle_line(raw)
        except Exception as error:  # noqa: BLE001 - the pump must never crash the container
            _diag(f"live: command pump stopped: {error!r}")

    thread = threading.Thread(target=run, name="command-pump", daemon=True)
    thread.start()
    return thread


# Structural conformance: TransportSource must satisfy the ActionSource protocol. Checked by
# pyright under TYPE_CHECKING only (this block never runs).
if TYPE_CHECKING:
    from .clock import SystemClock
    from .session import ActionSource, MessageSource

    _source: ActionSource = TransportSource(SessionControl(), clock=SystemClock(), paced=True)
    _message_source: MessageSource = TransportSource(
        SessionControl(),
        clock=SystemClock(),
        paced=True,
    )
