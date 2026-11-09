"""A worked Season 4 side: every unit plays one tactical block toward one goal.

This file is deliberately thin, because in Season 4 the interesting code is not here. A unit
does three things: ask :func:`blocks.assign` which job it holds, run that job's block, and fall
back to plain advancing on the activations when the block has nothing to say. Everything about
how a unit actually moves and fights lives in ``blocks.py``, and everything about who should be
doing what lives in that one call to ``assign``.

That call is the seam worth pulling on. The shipped ``assign`` reads only standing knowledge and
commits every unit to one job in round one, so the side you see is coherent but deaf: archers
keep kiting a fight that has already moved, and a cavalry wing keeps swinging at a wing that is
no longer there. Replacing it with a policy that re-reads the battle, weighs a few rounds of
lookahead over the blocks and goals available, and changes a unit's mind as the battle turns, is
the assignment.

The blocks are yours to change too. Edit them, add your own, or drop the whole library and hand
``assign`` the tactical blocks you wrote in Season 3.
"""

from __future__ import annotations

import blocks
from sandbox.observation_types import AxialPosition, SkirmishAction, SkirmishObservation


class Agent:
    """One unit of the side, playing the single block its assignment handed it."""

    def __init__(self) -> None:
        self.memory: dict = {}
        self.block: blocks.Block | None = None
        self.goal: AxialPosition | None = None

    def reset(self, seed, observation) -> None:
        """Forget the last match. This side draws no random numbers, so the seed goes unused."""
        self.memory = {}
        self.block = None
        self.goal = None

    def act(self, observation: SkirmishObservation) -> SkirmishAction:
        """Run this unit's block, advancing on the activations where it has nothing to say."""
        if self.block is None:
            self.block, self.goal = blocks.assign(observation, self.memory)
        order = self.block(observation, self.memory, self.goal)
        if order is None:
            return blocks.advance(observation, self.memory, self.goal)
        return order
