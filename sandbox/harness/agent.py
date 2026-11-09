"""The agent interface, as an abstract base class.

Participants implement four methods, two required and two optional. The required pair is
declared here as ``@abstractmethod``; the optional pair (``learn`` and ``chat``) is **not**
declared, not even as a no-op default, because the harness detects the optional hooks by
presence (:func:`has_learn`, :func:`has_chat`). A default implementation would make every
agent look like it learns and chats, and the harness would spend a hook call and clock time
(hook time counts against the limits, see :mod:`game_sandbox_harness.session`) on agents
that do nothing. The optional hooks are documented in this class's docstring instead.

Each environment's template layer carries its own plain-class copy of this interface, because
participants develop against vanilla PettingZoo and never install the harness
(``environments/<env>/template/agent.py``). Detection is therefore structural — :func:`is_agent`
checks that ``reset`` and ``act`` exist and are callable, never ``isinstance`` — and a test
asserts the two copies agree method-for-method so they cannot drift.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AgentBase(ABC):
    """Base class for in-repo agents and tests; participants need not subclass it.

    Required methods:

    - ``reset(seed, observation)`` prepares the agent for a new episode. The same seed the
      environment receives is passed here, together with that player's initial observation, so a
      stochastic agent can be made reproducible and can initialize from the first game state. Its
      time is charged only to the episode budget, with no per-call limit, so it can precompute.
    - ``act(observation)`` returns an action in the environment's action space.

    Optional methods, detected by presence (do not declare them unless you implement them):

    - ``learn(observation, action, reward, terminated)`` is called after each environment
      step with that step's transition, so a reinforcement-learning agent can keep updating
      during play. Its time counts against the per-step and per-episode limits.
    - ``chat(inbox)`` is called on the agent's turn with the messages addressed to its player;
      it returns messages to send, or nothing to stay silent. Defined and detected in Stage
      2 but never called until messaging routing arrives in Stage 8.
    """

    @abstractmethod
    def reset(self, seed: int, observation: Any) -> None:
        """Prepare for ``seed`` and the initial ``observation``, charged only to episode budget."""

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """Return an action in the environment's action space for ``observation``."""


def is_agent(obj: object) -> bool:
    """Return whether ``obj`` structurally satisfies the required agent interface.

    Duck-typed on purpose: a template-derived agent never imports :class:`AgentBase`, so the
    check is that ``reset`` and ``act`` are present and callable, not an ``isinstance``.
    """
    return callable(getattr(obj, "reset", None)) and callable(getattr(obj, "act", None))


def has_learn(obj: object) -> bool:
    """Return whether ``obj`` provides an optional, callable ``learn`` hook."""
    return callable(getattr(obj, "learn", None))


def has_chat(obj: object) -> bool:
    """Return whether ``obj`` provides an optional, callable ``chat`` hook."""
    return callable(getattr(obj, "chat", None))
