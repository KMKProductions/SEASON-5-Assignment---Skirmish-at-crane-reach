"""Environment-layer full-episode coverage for the shared multi-player play loop."""

from __future__ import annotations

import time
from typing import Any

from sandbox.env import META, make_env
from sandbox.env.skirmish_crane import naive
from sandbox.harness.environment import resolve_parameters
from sandbox.observation_types import SkirmishAction, SkirmishObservation
from sandbox.play import play_episode, rival_player_ids, run_headless

SEED = 4
RUNTIME_LIMIT_S = 20.0


class FirstLegalAgent:
    """A deterministic policy that trusts the environment-owned action mask."""

    def reset(self, seed, observation) -> None:
        pass

    def act(self, observation: SkirmishObservation) -> SkirmishAction:
        path = next(path_id for path_id, bit in enumerate(observation["action_mask"]["path"]) if bit)
        return {"path": path, "target": 0}


def test_template_play_loop_completes_a_bounded_episode():
    env = make_env(resolve_parameters(META))
    # The naive builtin hunts, so binding it to every other player keeps the match from stalling
    # into the round cap: verified empirically, seed 4 finishes in well under a second.
    rivals: dict[str, Any] = {f"player_{index}": naive.Agent() for index in range(1, 6)}
    started = time.monotonic()
    try:
        score = play_episode(FirstLegalAgent(), env, seed=SEED, other_agents=rivals)
        assert not env.agents
        assert 0.0 <= score <= 100.0
    finally:
        env.close()
    assert time.monotonic() - started < RUNTIME_LIMIT_S


def test_rival_players_cover_the_opposing_side():
    parameters = resolve_parameters(META)

    assert rival_player_ids(0, parameters) == {"player_3", "player_4", "player_5"}
    assert rival_player_ids(1, parameters) == {"player_0", "player_1", "player_2"}


def test_headless_default_uses_the_bundled_baseline_without_human_fallback(capsys):
    started = time.monotonic()

    score = run_headless(seed=SEED, max_steps=None, seat=0)

    assert 0.0 <= score <= 100.0
    assert "human player" not in capsys.readouterr().err
    assert time.monotonic() - started < RUNTIME_LIMIT_S
