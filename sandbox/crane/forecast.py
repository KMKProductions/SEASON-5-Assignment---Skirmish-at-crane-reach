"""Build an independent tactical simulation using only what your unit observes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sandbox.env.skirmish_crane.env import SkirmishCraneEnv
    from sandbox.observation_types import SkirmishObservation

__all__ = ["from_observation"]


def from_observation(observation: SkirmishObservation) -> SkirmishCraneEnv:
    """Return an initialized environment containing your unit and the units it can see.

    Your unit acts next. Visible units that have already acted wait for the next round;
    other activation orders and automatic targets use a separate local random stream.
    Unseen units are absent, so forecast outcomes can differ from the live battle.

    Use ``step``, ``last``, and ``observe`` as usual. Calling ``reset`` starts a new match
    instead of returning to this observation. Construct another forecast to try another order.
    """
    from sandbox.env.skirmish_crane.env import SkirmishCraneEnv

    return SkirmishCraneEnv.from_observation(observation)
