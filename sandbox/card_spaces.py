"""The shared Gymnasium spaces for a semantic playing-card observation.

A card is ``{"suit": <0..3>, "rank": <2..14>}``, a hand is a sequence of cards, and a trick is a
play-ordered sequence of ``{"player", "card"}`` records. Declaring :data:`CARD`, :data:`HAND`, and
:data:`TRICK` once means every card environment publishes the same shapes. It imports its sibling
:mod:`card_utils` (never a rules engine) and, like the codec, syncs verbatim into the student
template as ``sandbox.card_spaces``.
"""

from __future__ import annotations

from gymnasium import spaces

from . import card_utils as _cu

#: Players at the table. A trick is inherently four-player for both card games, so the player category is
#: fixed here rather than pulled from a per-game rules engine (which this module must not import).
NUM_PLAYERS = 4

#: One card: a suit category ``0..3`` and a face-value rank ``2..14`` (jack ``11``, queen ``12``, ace
#: ``14``), matching :func:`card_utils.card_to_obj`.
CARD = spaces.Dict(
    {
        "suit": spaces.Discrete(_cu.NUM_SUITS),
        "rank": spaces.Discrete(_cu.NUM_RANKS, start=_cu.RANK_OFFSET),
    }
)

#: A hand: a variable-length sequence of cards (an empty hand is a valid, exhausted hand).
HAND = spaces.Sequence(CARD)

#: A trick: a play-ordered sequence of ``{"player", "card"}`` records, one per card played so far, so
#: the order carries who led and who followed. Empty between tricks.
TRICK = spaces.Sequence(
    spaces.Dict(
        {
            "player": spaces.Discrete(NUM_PLAYERS),
            "card": CARD,
        }
    )
)
