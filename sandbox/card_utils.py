"""The shared, dependency-free playing-card codec for the trick-taking games.

Single source of truth for how a 52-card deck is numbered and how a card crosses between the integer
rules engines and the semantic object an agent, overlay, or renderer reads. It imports only the
standard library, so the Hearts and Spades engines share one encoding without a third-party
dependency, and it syncs verbatim into the student template as ``sandbox.card_utils``.

Two rank conventions live side by side and must never be interchanged:

* **Engine index** (``0..12``). A card is an int ``0..51`` with ``card = suit * 13 + rank_index``,
  suits ``0=clubs, 1=diamonds, 2=spades, 3=hearts``, ranks ``0=2 .. 9=J, 10=Q, 11=K, 12=A``.
  :func:`suit_of` / :func:`rank_of` decode it; the engines compare on this index (the queen is ``10``).
* **Face value** (``2..14``). The agent's object ``{"suit": <0..3>, "rank": <2..14>}`` uses familiar
  values (jack ``11``, queen ``12``, ace ``14``). :func:`card_to_obj` / :func:`card_from_obj` are the
  only place the offset is applied, and only at the ``env.py`` / ``overlay.py`` boundary.

The queen of spades pins both at once: card id ``36``, :func:`rank_of` ``10`` (engine index),
semantic ``{"suit": 2, "rank": 12}`` (face value). Using ``card_to_obj(c)["rank"]`` where the engine
expects ``rank_of(c)`` shifts every rank by two and silently corrupts the engine.
"""

from __future__ import annotations

from .card_types import Card

#: Number of suits, ranks, and cards in a standard deck.
NUM_SUITS = 4
NUM_RANKS = 13
NUM_CARDS = NUM_SUITS * NUM_RANKS

#: Suit ids (also the high part of the ``card = suit * 13 + rank_index`` encoding).
CLUBS = 0
DIAMONDS = 1
SPADES = 2
HEARTS = 3

#: The face value of the lowest rank (engine index ``0`` is the 2), so ``face = rank_index + 2`` and
#: the semantic rank runs ``2..14``. Applied only by the codec below.
RANK_OFFSET = 2

#: Human-readable suit names indexed by suit id, for helpers and status lines.
SUIT_NAMES: tuple[str, str, str, str] = ("clubs", "diamonds", "spades", "hearts")


def suit_of(card: int) -> int:
    """Return the suit id (``0..3``) of ``card`` under the suit-major encoding."""
    return card // NUM_RANKS


def rank_of(card: int) -> int:
    """Return the engine rank index (``0..12``, ``0`` is the 2 and ``12`` the ace) of ``card``.

    This is the number the engines compare on — the queen is ``10`` here, not its face value.
    """
    return card % NUM_RANKS


def card_to_obj(card: int) -> Card:
    """Return the semantic object ``{"suit": <0..3>, "rank": <2..14>}`` for engine ``card``.

    The rank is the face value (``rank_of(card) + 2``); this offset lives here alone, never in the engine.
    """
    return {"suit": suit_of(card), "rank": rank_of(card) + RANK_OFFSET}


def card_from_obj(obj: Card) -> int:
    """Return the engine card id ``0..51`` for a semantic ``{"suit", "rank"}`` object.

    The inverse of :func:`card_to_obj`, stripping the face-value offset back to the engine index.
    """
    return obj["suit"] * NUM_RANKS + (obj["rank"] - RANK_OFFSET)
