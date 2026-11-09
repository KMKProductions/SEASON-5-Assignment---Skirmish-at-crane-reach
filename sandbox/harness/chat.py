"""Message routing and validation for the optional ``chat`` hook.

This is the correctness core of agent communication, deliberately separate from the session
loop: every outbound rule in the communication spec is enforced here at one point, so the
session loop only ever hands a validated batch to the recording and the router only ever
decides delivery. The module imports nothing from :mod:`session` or :mod:`live_io` so it stays
cycle-free; it needs only the :class:`~game_sandbox_harness.state.Message` shape.

A :class:`ChatRouter` exists only when messaging is effectively enabled for the episode (the
environment metadata and the session config agree). With messaging off, no router is created and
the loop is byte-identical to a pre-chat run.

Validation is a drop-with-diagnostic discipline, never a raise: a chatty agent's malformed
message must not take down the session, exactly as an illegal *action* is charged and refused
without smearing the fault across every seat. Rejections print one stderr diagnostic (the
harness never configures logging, so an INFO record would be silently dropped) and continue.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from .environment import (
    BroadcastRecipientsSource,
    ChatPolicy,
    ChatPolicySource,
    canonical_player_order,
)
from .state import Message


def diag(message: str) -> None:
    """Write a chat diagnostic to stderr (the harness never configures logging)."""
    print(message, file=sys.stderr, flush=True)


def _policy_fields(policy: object) -> tuple[object, object] | None:
    """Read the two policy fields from a :class:`ChatPolicy` or an equivalent mapping.

    Returns ``None`` when the hook returned something that is neither, which is the one defect that
    is about shape rather than about the values themselves.
    """
    if isinstance(policy, ChatPolicy):
        return policy.target_recipients, policy.default_recipient
    if isinstance(policy, Mapping):
        fields = cast("Mapping[str, object]", policy)
        if "target_recipients" in fields and "default_recipient" in fields:
            return fields["target_recipients"], fields["default_recipient"]
    return None


class ChatRouter:
    """Routes and validates messages for one episode's players.

    Owns the pending inboxes only. The participant runner owns accepted batches and delivery, while
    the episode records a batch before asking the runner to deliver it at the end of the step. The
    cap counts Unicode code points (``len(text)`` on a Python ``str``), pinned here so both sides
    of every boundary agree.
    """

    def __init__(self, player_ids: Iterable[str], cap: int | None) -> None:
        self._player_order = canonical_player_order(player_ids)
        self._players = frozenset(self._player_order)
        self._cap = cap
        self._active = self._players
        self._inboxes: dict[str, list[dict[str, Any]]] = {player_id: [] for player_id in self._players}

    def set_active(self, active_players: Iterable[str]) -> None:
        """Adopt the logically active players and discard the inboxes of everyone else.

        Every rule below is stated against this set, so the episode updates it once per completed
        transition and the router never has to be told again. A player that left can no longer act,
        so it will never drain its own inbox; clearing here is what keeps that inbox bounded.
        """
        self._active = frozenset(active_players)
        for player_id in self._players - self._active:
            self._inboxes[player_id] = []

    def default_policy(self, sender: str) -> ChatPolicy:
        """Allow every other logically active player in canonical order, defaulting to broadcast."""
        return ChatPolicy(
            target_recipients=tuple(
                player for player in self._player_order if player != sender and player in self._active
            ),
            default_recipient=None,
        )

    def default_broadcast_audience(self, sender: str) -> tuple[str, ...]:
        """Return every other active player in canonical order for one ordinary broadcast."""
        return tuple(
            player_id for player_id in self._player_order if player_id != sender and player_id in self._active
        )

    def policy_from(self, env: Any, sender: str) -> ChatPolicy:
        """Resolve one sender's policy from the live environment, validated and never raising.

        An environment without the hook, a hook that raises, and a hook that returns something
        unusable all resolve to :meth:`default_policy`, so a policy bug never ends the episode.
        """
        if not isinstance(env, ChatPolicySource):
            return self.default_policy(sender)
        try:
            declared = env.chat_policy(sender)
        except Exception as error:  # noqa: BLE001 - a policy bug must not end the game
            diag(f"chat: {sender} environment policy failed ({error!r}); using the default policy")
            return self.default_policy(sender)
        return self.validate_policy(sender, declared)

    def validate_policy(self, sender: str, policy: object) -> ChatPolicy:
        """Validate an environment policy, falling back to the generic contract on any defect.

        The diagnostic names the specific defect, because the reader is an environment author
        debugging their own hook. Any defect at all yields :meth:`default_policy`, so a broken hook
        widens the recipient list rather than ending the episode.
        """
        try:
            fields = _policy_fields(policy)
            checked = "has the wrong shape" if fields is None else self._checked_policy(sender, *fields)
        except Exception as error:  # noqa: BLE001 - a hook may return an object that fails on access
            checked = f"could not be read ({error!r})"
        if isinstance(checked, str):
            diag(f"chat: {sender} environment policy {checked}; using the default policy")
            return self.default_policy(sender)
        return checked

    def validate_broadcast_recipients(self, sender: str, declared: object) -> tuple[str, ...]:
        """Validate one broadcast hook result, falling back to the default audience on defects."""
        try:
            checked = self._checked_recipients(sender, declared, "broadcast_recipients")
        except Exception as error:  # noqa: BLE001 - a hook may return an unreadable object
            checked = f"could not be read ({error!r})"
        if isinstance(checked, str):
            diag(f"chat: {sender} environment broadcast recipients {checked}; using the default audience")
            return self.default_broadcast_audience(sender)
        return tuple(player_id for player_id in checked if player_id in self._active)

    def broadcast_audience_from(self, env: Any, sender: str) -> tuple[str, ...]:
        """Resolve a live broadcast audience without letting an environment hook stop delivery."""
        if not isinstance(env, BroadcastRecipientsSource):
            return self.default_broadcast_audience(sender)
        try:
            declared = env.broadcast_recipients(sender)
        except Exception as error:  # noqa: BLE001 - a policy bug must not end the game
            diag(
                f"chat: {sender} environment broadcast recipients failed ({error!r}); "
                "using the default audience"
            )
            return self.default_broadcast_audience(sender)
        return self.validate_broadcast_recipients(sender, declared)

    def bounded_broadcast_audiences(
        self, env: Any, messages: list[Message]
    ) -> dict[str, tuple[str, ...]] | None:
        """Resolve each broadcasting sender's bounded audience once for one recorded batch.

        Returns ``None`` when the environment does not bound broadcasts (no hook) or the batch
        carries no broadcast, so callers can tell "unbounded, visible to everyone" apart from an
        explicit audience. The result feeds both the live-state annotation and :meth:`deliver`,
        so the hook runs at most once per sender and boundary and the two views cannot disagree.
        """
        if not isinstance(env, BroadcastRecipientsSource):
            return None
        audiences: dict[str, tuple[str, ...]] = {}
        for message in messages:
            sender = message["from"]
            if message["to"] is None and sender not in audiences:
                audiences[sender] = self.broadcast_audience_from(env, sender)
        return audiences or None

    def _checked_policy(
        self,
        sender: str,
        recipients: object,
        default_recipient: object,
    ) -> ChatPolicy | str:
        """Return the validated policy, or a phrase naming the first defect that makes it unusable."""
        named = self._checked_recipients(sender, recipients, "target_recipients")
        if isinstance(named, str):
            return named
        if default_recipient is not None and (
            not isinstance(default_recipient, str) or default_recipient not in named
        ):
            return f"defaults to {default_recipient!r}, which it does not offer"
        # A recipient who left mid-episode is dropped from the declared list rather than voiding it:
        # treating that as a defect would widen a narrow policy to the default at the very moment a
        # player goes inactive. A default that leaves with them falls back to broadcast, always legal.
        offered = tuple(recipient for recipient in named if recipient in self._active)
        return ChatPolicy(offered, default_recipient if default_recipient in offered else None)

    def _checked_recipients(self, sender: str, recipients: object, label: str) -> tuple[str, ...] | str:
        """Validate one hook's recipient sequence before inactive players are filtered."""
        if isinstance(recipients, (str, bytes)) or not isinstance(recipients, Sequence):
            return f"declares {label} that is not a sequence"
        listed = tuple(cast("Sequence[object]", recipients))
        if not all(isinstance(recipient, str) for recipient in listed):
            return "declares a non-string recipient"
        named = cast("tuple[str, ...]", listed)
        if len(named) != len(set(named)):
            return "declares the same recipient twice"
        for recipient in named:
            if recipient == sender:
                return "declares the sender as one of its own recipients"
            if recipient not in self._players:
                return f"declares unknown recipient {recipient!r}"
        return named

    def validate_outgoing(
        self,
        sender: str,
        batch: object,
        policy: ChatPolicy | None = None,
    ) -> list[Message]:
        """Validate one batch an agent or human returned, stamping ``from`` with ``sender``.

        Returns the accepted messages in order. Every rejected message is dropped with a stderr
        diagnostic and never raises. Enforced, in order: the sender is a known, active player; the
        recipient is a known active player other than the sender or ``None`` for broadcast; a direct
        recipient is permitted by ``policy``; the text is a ``str`` within the code-point cap; and the
        batch carries at most one message per distinct recipient plus one broadcast.

        ``policy`` is the one resolved for this sender and boundary. The human path passes the policy
        its last live state published, which can name a recipient who has since left, so recipient
        activity is checked here as well as when the policy was built.
        """
        # "returns messages or nothing": None or an empty batch is silent, not an error.
        if batch is None:
            return []
        if isinstance(batch, (str, bytes)) or not isinstance(batch, Sequence):
            diag(f"chat: {sender} returned a non-list batch {type(batch).__name__}; dropping it")
            return []
        if sender not in self._players:
            diag(f"chat: dropping unknown sender {sender!r}")
            return []
        if sender not in self._active:
            diag(f"chat: dropping inactive sender {sender!r}")
            return []

        allowed_direct = frozenset((policy or self.default_policy(sender)).target_recipients)
        accepted: list[Message] = []
        seen: set[str | None] = set()
        for raw_item in cast("Sequence[object]", batch):
            if not isinstance(raw_item, Mapping):
                diag(f"chat: {sender} sent a non-object message {raw_item!r}; dropping it")
                continue
            item = cast("Mapping[str, object]", raw_item)
            to = item.get("to")
            # A bool is an int, never a str, so a bool recipient falls through the str check below.
            if to is not None and (not isinstance(to, str) or to not in self._players):
                diag(f"chat: {sender} sent to unknown recipient {to!r}; dropping it")
                continue
            if to is not None and to not in self._active:
                diag(f"chat: {sender} sent to inactive recipient {to!r}; dropping it")
                continue
            if to == sender:
                diag(f"chat: {sender} sent to itself; dropping it")
                continue
            if to is not None and to not in allowed_direct:
                diag(f"chat: {sender} sent to policy-disallowed recipient {to!r}; dropping it")
                continue
            text = item.get("text")
            # A non-str text (including a bool, which is an int, not a str) is dropped here.
            if not isinstance(text, str):
                diag(f"chat: {sender} sent non-string text {text!r}; dropping it")
                continue
            if self._cap is not None and len(text) > self._cap:
                diag(f"chat: {sender} sent {len(text)} code points over the cap of {self._cap}; dropping it")
                continue
            if to in seen:
                where = "broadcast" if to is None else f"message to {to}"
                diag(f"chat: {sender} sent a second {where} in one boundary; dropping it")
                continue
            seen.add(to)
            accepted.append({"from": sender, "to": to, "text": text})
        return accepted

    def deliver(
        self,
        messages: list[Message],
        tick: int,
        env: Any = None,
        audiences: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Deliver a tick's accepted messages to pending inboxes, stamping each with ``tick``.

        A targeted message reaches its recipient's inbox; a broadcast reaches every other active
        player, or the audience the environment bounded. ``audiences`` carries audiences already
        resolved by :meth:`bounded_broadcast_audiences` so the hook is not invoked a second time;
        senders it does not cover are resolved here. A recipient that went inactive on this very
        transition is skipped, because it has no later acting opportunity to read on. Called at the
        end of the sending tick, after the acting agent's inbox was drained, so a message sent on
        tick T is first seen strictly after T.
        """
        resolved: dict[str, tuple[str, ...]] = dict(audiences or {})
        for message in messages:
            sender = message["from"]
            recipient = message["to"]
            item = {**message, "tick": tick}
            if recipient is None:
                if sender not in resolved:
                    resolved[sender] = self.broadcast_audience_from(env, sender)
                audience = resolved[sender]
                for player_id in audience:
                    if player_id in self._active:
                        self._inboxes[player_id].append(dict(item))
            elif recipient in self._active:
                self._inboxes[recipient].append(item)

    def drain(self, player_id: str) -> list[dict[str, Any]]:
        """Return and clear a player's pending inbox.

        Called on every acting player's turn, chat-less agents included, so an inbox can never grow
        without bound behind an agent that will never read it.
        """
        inbox = self._inboxes[player_id]
        self._inboxes[player_id] = []
        return inbox
