"""Static types for the semantic card observations.

TypedDicts mirroring the runtime shapes declared in card_spaces.py: Card and TrickEntry match the
shared CARD and TRICK spaces, and the per-game observation types match each env's observe() output.
Stdlib-only at runtime (numpy is imported only for type checking), so sandbox.cards may import it
without dragging in the engine. Ships into the student template as sandbox.card_types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class Card(TypedDict):
    """One semantic card: suit 0..3 and face-value rank 2..14 (11=J, 12=Q, 13=K, 14=A)."""

    suit: int
    rank: int


class TrickEntry(TypedDict):
    """One card played to the current trick, with the seat that played it."""

    player: int
    card: Card


class HeartsObservationData(TypedDict):
    """The semantic Hearts state under the observation's "observation" key."""

    player: int
    hand: tuple[Card, ...]
    current_trick: tuple[TrickEntry, ...]
    trick_leader: int
    led_suit: int  # 0..3, or 4 when no card has been led this trick
    hearts_broken: int  # 0 or 1
    scores: NDArray[np.int64]  # shape (4,), running penalty points by player


class HeartsObservation(TypedDict):
    """The full dict a Hearts agent's act() receives."""

    observation: HeartsObservationData
    action_mask: NDArray[np.int8]  # shape (52,), 1 = legal card id


class SpadesObservationData(TypedDict):
    """The semantic Spades state under the observation's "observation" key."""

    player: int
    partner_player: int
    phase: int  # 0 = bidding round, 1 = play
    hand: tuple[Card, ...]
    bids: tuple[int, ...]  # length 4, indexed by player; 14 = not yet bid (0..13 once placed)
    team_scores: NDArray[np.int64]  # shape (2,), [team of players 0/2, team of players 1/3]
    current_trick: tuple[TrickEntry, ...]
    last_trick: tuple[TrickEntry, ...]
    last_trick_winner: int  # 0..3, or 4 when no trick has completed yet
    trick_leader: int
    led_suit: int  # 0..3, or 4 when no card has been led this trick
    spades_broken: int  # 0 or 1
    tricks_won: NDArray[np.int64]  # shape (4,), tricks taken so far by player


class SpadesObservation(TypedDict):
    """The full dict a Spades agent's act() receives."""

    observation: SpadesObservationData
    action_mask: NDArray[np.int8]  # shape (66,), 1 = legal action (cards 0..51, bids 52..65)
