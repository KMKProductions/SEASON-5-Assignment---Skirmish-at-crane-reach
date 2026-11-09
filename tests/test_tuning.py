"""Checks for the tuning tools and the learned tactics critic in agent.py.

The critic may only ever hand the game a legal order, the tuner's overrides must reach the values
they name, and the faster path search used while labeling must agree with the engine exactly.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

import agent
from sandbox.env import make_env
from tuning import tactics
from tuning.arena import AGENT, Contender, load_module, play_match, season_parameters


def test_fast_legal_paths_match_the_engine():
    parameters, _, _ = season_parameters()
    env = make_env(parameters)
    env.reset(seed=3)
    field = env.match.battlefield
    cells = [(q, r) for q in range(25) for r in range(25) if field.tile_at((q, r)).passable]
    rng = random.Random(0)
    for _ in range(200):
        start = rng.choice(cells)
        occupied = set(rng.sample(cells, 40)) - {start}
        points = rng.choice((2, 4))
        tactics._path_cache.clear()
        assert tactics.fast_legal_paths(field, start, points, occupied) == tactics._original_legal_paths(
            field, start, points, occupied
        )


def test_overrides_reach_constants_classes_and_playbook_blocks():
    module = load_module(AGENT)
    skipped = module.apply_overrides(
        {
            "ENGAGE_RADIUS": 4,
            "ArmyPlan.ARCHER_BAND": [2, 4],
            "Agent.CALM_ROUNDS": 3,
            "archer.KiteFromThreat.radius": 3,
            "archer.ShootFromBehindFootmen.nothing": 1,
        }
    )
    assert module.ENGAGE_RADIUS == 4
    assert module.ArmyPlan.ARCHER_BAND == (2, 4)
    assert module.Agent.CALM_ROUNDS == 3
    assert module._playbook_blocks("archer", "KiteFromThreat")[0].radius == 3
    assert skipped == ["archer.ShootFromBehindFootmen.nothing"]
    with pytest.raises(KeyError):
        module.apply_overrides({"ENGAGE_RADIOS": 4}, strict=True)


def _fight_positions(count=40):
    """Real fight activations from one match, each with the candidates the critic would weigh."""
    seen = []

    def grab(ctx, candidates):
        if len(seen) < count:
            seen.append((ctx.obs, candidates))

    tactics.grab_for_test = lambda module: setattr(module.Agent, "tactics_hook", staticmethod(grab))
    ours = Contender.of("me", AGENT, setup="tuning.tactics:grab_for_test")
    play_match(ours, Contender.of("naive"), 7, limits=False)
    return seen


def test_every_critic_alternative_is_a_legal_order():
    positions = _fight_positions()
    assert positions, "the match should contain fights"
    for observation, candidates in positions:
        assert candidates[0].features[0] == 1.0  # the playbook's order comes first
        for candidate in candidates:
            assert len(candidate.features) == len(agent.TACTICS_FEATURES)
            assert observation["action_mask"]["path"][candidate.order["path"]] == 1
            assert observation["action_mask"]["target"][candidate.order["target"]] == 1


def test_a_saved_critic_scores_candidates_and_only_switches_past_the_margin(tmp_path, monkeypatch):
    width = len(agent.TACTICS_FEATURES)
    # A one-layer "network" that scores a landing by how much damage its strike deals.
    weight = np.zeros((1, width))
    weight[0, agent.TACTICS_FEATURES.index("damage")] = 10.0
    data = {
        "version": agent.TACTICS_VERSION,
        "features": list(agent.TACTICS_FEATURES),
        "mean": [0.0] * width,
        "scale": [1.0] * width,
        "layers": [{"weight": weight.tolist(), "bias": [0.0]}],
    }
    path = tmp_path / "tactics-model.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(agent, "_TACTICS_FILE", path)
    monkeypatch.setattr(agent, "_tactics_cache", {})
    model = agent._tactics_model()
    assert model is not None
    rows = [[0.0] * width, [0.0] * width]
    rows[1][agent.TACTICS_FEATURES.index("damage")] = 0.6
    assert model.score(rows) == pytest.approx([0.0, 6.0])

    stale = {**data, "features": list(agent.TACTICS_FEATURES[:-1])}
    path.write_text(json.dumps(stale), encoding="utf-8")
    monkeypatch.setattr(agent, "_tactics_cache", {})
    assert agent._tactics_model() is None  # a model built for other features is ignored


def test_the_agent_finishes_a_timed_match_with_the_critic_on():
    result = play_match(Contender.of("me", AGENT, {"Agent.TACTICS": True}), Contender.of("naive"), 11)
    assert result["error"] is None
    assert result["failed_player"] is None
    assert result["timeouts"] == [0, 0]
