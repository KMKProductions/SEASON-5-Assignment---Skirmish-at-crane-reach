"""Example-specific tests, added on top of the inherited template tests.

Three groups. The model checks build a policy on one hand-made activation and pin the properties
training depends on: padding and ordering never change a score, a checkpoint refuses an encoder it
was not trained with, and a choice that depends on the battlefield can be learned. The encoder and
collector checks pin what a demonstration means: mirrored sides read the same, the teacher's
labels are checked and recorded before anything can modify the observation, and every unit gets its
own teacher. The last two play real matches: the shipped learner finishes one (an illegal order
would raise), and a small collect and train run goes from teacher to saved model.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from agent import choose_action
from encoder import (
    ENTITY_FEATURES,
    FAR,
    MOVE_FEATURES,
    describe_move,
    describe_target,
    encode_entities,
    move_options,
    target_options,
)
from imitation import experiment, experiment_tools, train
from imitation.experiment import demonstration
from imitation.experiment_tools import RecordingPlayer, action_labels
from imitation.train import batch_loss, fit_model, predict
from imitation.train_tools import collate_rows, decision_loss, load_rows, model_inputs
from imitation_model import Policy, load_policy, save_policy
from routes import NO_ROUTE, route_cost
from sandbox.crane import paths, tile
from sandbox.harness.session import REASON_TERMINATED, REASON_TRUNCATED


def sample_observation(kind="footman", side="red"):
    """One activation on a 5-wide grass field, with one wounded enemy archer in sight."""
    enemy_side = "blue" if side == "red" else "red"
    own = {
        "unit_id": f"{side}_{kind}_0",
        "type": kind,
        "position": {"q": 2, "r": 2},
        "hit_points": 6,
        "movement_points": 2,
        "direction": 2 if side == "red" else 5,
    }
    enemies = [
        {"unit_id": f"{enemy_side}_archer_{i}", "side": enemy_side, "type": "archer"} for i in range(2)
    ]
    # Legal paths: stay, east (2), and two routes to (+2, -1): NE then E (8), E then NE (13).
    path_mask = np.zeros(14, dtype=int)
    path_mask[[0, 2, 8, 13]] = 1
    return {
        "observation": {
            "self": own,
            "round": 1,
            "capture": {"red": 3, "blue": 1, "target": 10},
            "rosters": {side: [{"unit_id": own["unit_id"], "side": side, "type": kind}], enemy_side: enemies},
            "visible_units": [
                {**enemies[1], "position": {"q": 4, "r": 2}, "hit_points": 4, "has_acted": True}
            ],
            "battlefield": {
                "side": 5,
                "zones": [],
                "tiles": [[{"terrain": "grass", "feature": "none"} for _ in range(5)] for _ in range(5)],
            },
        },
        "action_mask": {"path": path_mask, "target": np.array([1, 0, 1])},
    }


def training_row(observation=None, choice=0, seed=0):
    """A recorded demonstration of the sample activation, as collection would write it."""
    observation = sample_observation() if observation is None else observation
    row = demonstration(observation, {"path": move_options(observation)[choice][0], "target": 0})
    return {**row, "seed": seed, "seat": 0, "driver": "teacher"}


@pytest.fixture(autouse=True)
def _small_and_repeatable():
    torch.set_num_threads(1)
    torch.manual_seed(7)


# --- The model --------------------------------------------------------------------------------


def test_padding_changes_no_score_loss_or_gradient():
    row = training_row()
    batch = collate_rows([row])
    padded = {key: value.clone() for key, value in batch.items()}
    for values, mask in (
        ("entities", "entity_mask"),
        ("move_options", "move_mask"),
        ("target_options", "target_mask"),
    ):
        filler = torch.full((1, 3, batch[values].shape[-1]), 99.0)
        padded[values] = torch.cat((batch[values], filler), dim=1)
        padded[mask] = torch.cat((batch[mask], torch.zeros(1, 3, dtype=torch.bool)), dim=1)
    pointers = batch["target_entity_indices"]
    padded["target_entity_indices"] = torch.cat((pointers, torch.full((1, 3), -1)), dim=1)

    model = Policy(width=8, layers=1, heads=2)
    original, extra = model(**model_inputs(batch)), model(**model_inputs(padded))
    for head in ("move", "target"):
        count = original[head].shape[1]
        torch.testing.assert_close(original[head], extra[head][:, :count])
        assert torch.isneginf(extra[head][:, count:]).all()

    batch_loss(model, batch).backward()
    gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    batch_loss(model, padded).backward()
    for expected, parameter in zip(gradients, model.parameters(), strict=True):
        if expected is not None:
            torch.testing.assert_close(expected, parameter.grad, atol=2e-6, rtol=2e-5)

    # A padded candidate must also stay out of the loss's softmax denominator.
    scores = torch.tensor([[0.1, 0.9, 1000.0]], requires_grad=True)
    loss = decision_loss(scores, torch.tensor([1]), torch.tensor([[True, True, False]]))
    torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(scores[:, :2], torch.tensor([1])))


def test_reordering_entities_and_options_reorders_the_scores_with_them():
    row = training_row()
    row["entities"].append([0, 0, 0, 1] + [0] * (len(ENTITY_FEATURES) - 4))  # a zone token
    original = collate_rows([row])
    reordered = {key: value.clone() for key, value in original.items()}
    reordered["entities"] = reordered["entities"][:, [0, 2, 1]]
    reordered["target_entity_indices"][reordered["target_entity_indices"] == 1] = 2
    for head in ("move", "target"):
        reordered[f"{head}_options"] = reordered[f"{head}_options"].flip(1)
    reordered["target_entity_indices"] = reordered["target_entity_indices"].flip(1)

    model = Policy(width=8, layers=1, heads=2).eval()
    before, after = model(**model_inputs(original)), model(**model_inputs(reordered))
    for head in before:
        torch.testing.assert_close(before[head], after[head].flip(1))


def test_a_checkpoint_round_trips_and_refuses_another_encoder(tmp_path):
    model = Policy(width=8, layers=1, heads=2).eval()
    path = tmp_path / "policy.pt"
    save_policy(model, path)
    observation = sample_observation()
    assert choose_action(load_policy(path), observation) == choose_action(model, observation)

    saved = torch.load(path, weights_only=True)
    saved["encoder_version"] = "old"
    torch.save(saved, path)
    with pytest.raises(ValueError, match="encoder"):
        load_policy(path)


def test_a_choice_that_depends_on_the_battlefield_can_be_learned():
    # Two activations with identical options: only the acting unit's token tells them apart.
    rows = []
    for sign in (-1, 1):
        row = training_row()
        row["entities"][0][-1] = sign
        row["move_options"] = [[0.0] * len(MOVE_FEATURES) for _ in range(2)]
        row["move_options"][0][-2], row["move_options"][1][-2] = -1, 1
        row["paths"] = [0, 2]
        row["move_choice"] = int(sign > 0)
        row.update(target_options=row["target_options"][:1], targets=[0], target_entity_indices=[-1])
        rows.append(row)
    model, _ = fit_model(rows * 8, rows, width=16, layers=1, heads=2, epochs=100, batch_size=4, threads=1)
    assert [pick["move"] for pick in predict(model, rows)] == [0, 1]


@pytest.mark.parametrize(
    "change",
    [
        {"encoder_version": "options-v1"},
        {"entities": []},
        {"move_options": []},
        {"target_entity_indices": [-1, 0]},
        {"target_entity_indices": [-1, 999]},
        {"move_choice": 99},
    ],
)
def test_malformed_or_stale_records_are_rejected(tmp_path, change):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({**training_row(), **change}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_rows([path], train.HEADS)


# --- The encoder and the collector ------------------------------------------------------------


def test_a_mirrored_observation_reads_the_same_and_gets_the_mirrored_action():
    original = sample_observation()
    original["observation"]["battlefield"]["zones"] = [
        {"center": {"q": 3, "r": 1}, "tiles": [{"q": 3, "r": 1}, {"q": 4, "r": 1}]}
    ]
    original["observation"]["battlefield"]["tiles"][1][4] = {"terrain": "hill", "feature": "forest"}

    # Rebuild the same situation from the blue side: every position reflected through the center.
    mirrored = deepcopy(original)
    state = mirrored["observation"]
    swap = {"red": "blue", "blue": "red"}

    def swap_unit(unit):
        side, rest = unit["unit_id"].split("_", 1)
        unit["unit_id"] = f"{swap[side]}_{rest}"
        if "side" in unit:
            unit["side"] = swap[unit["side"]]
        if "position" in unit:
            unit["position"] = tile.at_mirror(unit["position"], original)

    swap_unit(state["self"])
    state["self"]["direction"] = 5
    for unit in state["visible_units"]:
        swap_unit(unit)
    for roster in state["rosters"].values():
        for unit in roster:
            swap_unit(unit)
    state["rosters"] = {swap[side]: roster for side, roster in state["rosters"].items()}
    state["capture"]["red"], state["capture"]["blue"] = state["capture"]["blue"], state["capture"]["red"]
    state["battlefield"]["tiles"] = [row[::-1] for row in state["battlefield"]["tiles"][::-1]]
    for area in state["battlefield"]["zones"]:
        area["center"] = tile.at_mirror(area["center"], original)
        area["tiles"] = [tile.at_mirror(position, original) for position in area["tiles"]]
    mirrored["action_mask"]["path"] = np.zeros(paths.MAX_ID + 1, dtype=int)
    for path, allowed in enumerate(original["action_mask"]["path"]):
        if allowed:
            mirrored["action_mask"]["path"][paths.encode([(d + 2) % 6 + 1 for d in paths.decode(path)])] = 1

    np.testing.assert_allclose(encode_entities(original)[0], encode_entities(mirrored)[0])
    for _, end in move_options(original):
        opposite = tile.at_mirror(end, original)
        np.testing.assert_allclose(describe_move(original, end), describe_move(mirrored, opposite))
        for target in target_options(original):
            np.testing.assert_allclose(
                describe_target(original, end, target), describe_target(mirrored, opposite, target)
            )

    model = Policy(width=8, layers=1, heads=2).eval()
    first, second = choose_action(model, original), choose_action(model, mirrored)
    first_end = tile.at_path_end(original["observation"]["self"]["position"], first["path"])
    second_end = tile.at_path_end(state["self"]["position"], second["path"])
    assert tile.at_mirror(first_end, original) == second_end
    assert first["target"] == second["target"]


def test_move_features_read_only_visible_enemies_and_face_the_enemy_side():
    east = {"q": 3, "r": 2}
    red = dict(zip(MOVE_FEATURES, describe_move(sample_observation(), east), strict=True))
    assert (red["forward"], red["nearest_enemy"], red["can_strike"], red["threatened_by"]) == (1, 1, 1, 1)
    blue = dict(zip(MOVE_FEATURES, describe_move(sample_observation(side="blue"), east), strict=True))
    assert blue["forward"] == -1

    unseen = sample_observation()
    unseen["observation"]["visible_units"] = []
    hidden = dict(zip(MOVE_FEATURES, describe_move(unseen, east), strict=True))
    assert (hidden["visible_enemies"], hidden["nearest_enemy"], hidden["can_strike"]) == (0, FAR, 0)


def test_route_cost_counts_terrain_and_stops_at_water():
    def cost(changes, goal):
        observation = sample_observation()
        grid = observation["observation"]["battlefield"]["tiles"]
        for (q, r), ground in changes.items():
            grid[r][q] = ground
        return route_cost(observation, {"q": 2, "r": 2}, goal)

    east, two_east = {"q": 3, "r": 2}, {"q": 4, "r": 2}
    assert cost({}, east) == 1
    assert cost({(3, 2): {"terrain": "hill", "feature": "marsh"}}, east) == 4
    assert cost({(3, 2): {"terrain": "grass", "feature": "marsh"}}, two_east) == 3
    assert cost({(3, r): {"terrain": "water", "feature": "none"} for r in range(5)}, two_east) == NO_ROUTE


@pytest.mark.parametrize(
    "action",
    [
        None,
        {"path": 0},
        {"path": True, "target": 0},
        {"path": 2.0, "target": 0},
        {"path": 1, "target": 0},
        {"path": 0, "target": 1},
    ],
)
def test_an_illegal_teacher_action_is_rejected_before_it_is_recorded(action):
    rows = []
    teacher = SimpleNamespace(act=lambda observation: action)
    player = RecordingPlayer(teacher, 4, 0, rows, [], record=demonstration)
    with pytest.raises(ValueError):
        player.act(sample_observation())
    assert rows == []
    assert action_labels(sample_observation(), {"path": np.int64(2), "target": 0}) == [2, 0]


def test_options_are_recorded_before_the_teacher_can_modify_the_observation():
    observation = sample_observation()
    expected = [describe_move(observation, end) for _, end in move_options(observation)]

    def careless_teacher(observation):
        observation["observation"]["self"]["hit_points"] = 99
        observation["action_mask"]["path"][:] = 0
        return {"path": 13, "target": 2}

    rows = []
    player = RecordingPlayer(SimpleNamespace(act=careless_teacher), 4, 0, rows, [], record=demonstration)
    assert player.act(observation) == {"path": 13, "target": 2}
    assert rows[0]["move_options"] == expected
    # Paths 8 and 13 end on the same tile, so the label is that tile, kept under its first path.
    assert (rows[0]["paths"], rows[0]["move_choice"], rows[0]["target_choice"]) == ([0, 2, 8], 2, 1)


def test_the_learner_plays_while_the_teacher_labels_its_own_copy():
    def teacher_act(observation):
        observation["observation"]["self"]["hit_points"] = 99
        return {"path": 13, "target": 2}

    learner = SimpleNamespace(reset=Mock(), act=Mock(return_value={"path": 2, "target": 0}))
    teacher = SimpleNamespace(reset=Mock(), act=Mock(side_effect=teacher_act))
    rows = []
    player = RecordingPlayer(learner, 4, 1, rows, [], labeler=teacher, record=demonstration)
    observation = sample_observation()
    player.reset(4, observation)

    assert player.act(observation) == {"path": 2, "target": 0}
    assert observation["observation"]["self"]["hit_points"] == 6
    assert (rows[0]["driver"], rows[0]["path"], rows[0]["move_choice"]) == ("learner", 13, 2)


def test_every_unit_gets_its_own_teacher_module_and_helpers(tmp_path, monkeypatch):
    (tmp_path / "isolated_helper.py").write_text("calls = []\n", encoding="utf-8")
    (tmp_path / "isolated_teacher.py").write_text(
        "import isolated_helper\n"
        "calls = []\n"
        "class Agent:\n"
        "    calls = []\n"
        "    def reset(self, seed, observation): pass\n"
        "    def act(self, observation):\n"
        "        calls.append(1)\n"
        "        self.calls.append(1)\n"
        "        isolated_helper.calls.append(1)\n"
        "        return len(calls), len(self.calls), len(isolated_helper.calls)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(experiment_tools, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    try:
        factory = experiment_tools.load_teacher("isolated_teacher:Agent")
        first, second = factory(), factory()
        assert first.act(None) == (1, 1, 1)
        assert second.act(None) == (1, 1, 1)
        assert first.act(None) == (2, 2, 2)
    finally:
        experiment_tools._evict_modules_from_roots((tmp_path,))


# --- Whole matches ----------------------------------------------------------------------------


def test_the_shipped_learner_finishes_a_match_against_naive():
    # The match uses season.json. An order outside the mask would raise, so finishing proves
    # every order was legal, and loading proves imitation-policy.pt matches the encoder.
    teacher = experiment_tools.load_teacher("teacher:Agent")
    result = experiment_tools.episode("learner", 100, 0, teacher)
    assert result["reason"] in (REASON_TERMINATED, REASON_TRUNCATED)
    assert result["failed_player"] is None
    assert result["own_timeouts"] == 0


def test_learner_driven_collection_gives_learner_and_teacher_each_a_full_budget(monkeypatch):
    # The teacher labels inside the learner's timed activation, so the limits must double rather
    # than fall back to the environment's defaults.
    limits = []
    real_run_episode = experiment_tools.run_episode

    def watched_run_episode(*args, **kwargs):
        limits.append((kwargs["step_limit_ms"], kwargs["episode_limit_ms"]))
        return real_run_episode(*args, **kwargs)

    monkeypatch.setattr(experiment_tools, "run_episode", watched_run_episode)
    teacher = experiment_tools.load_teacher("teacher:Agent")
    meta = experiment_tools.META
    season = experiment_tools.load_season_settings(experiment_tools.ROOT, meta)
    step = season.decision_limit_ms if season and season.decision_limit_ms else meta.step_limit_ms
    game = season.game_limit_ms if season and season.game_limit_ms else meta.episode_limit_ms

    rows = []
    experiment_tools.episode("learner", 100, 0, teacher, rows, record=demonstration)
    assert limits == [(2 * step, 2 * game)]
    assert rows and {row["driver"] for row in rows} == {"learner"}


def test_collect_then_train_produces_a_model_and_a_report(tmp_path, monkeypatch):
    data, model = tmp_path / "demonstrations.jsonl", tmp_path / "policy.pt"
    monkeypatch.setattr(sys, "argv", ["experiment", "collect", "--episodes", "2", "--output", str(data)])
    experiment.main()
    assert {json.loads(line)["seed"] for line in data.read_text(encoding="utf-8").splitlines()} == {0, 1}

    monkeypatch.setattr(train, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", str(data), "--output", str(model), "--validation-seeds", "1", "--epochs", "2"]
        + ["--width", "8", "--layers", "1", "--heads", "2", "--threads", "1", "--device", "cpu"],
    )
    train.main()
    report = json.loads((tmp_path / "runs" / "policy.training.json").read_text(encoding="utf-8"))
    assert report["validation_seeds"] == [1]
    assert 0 <= report["validation_agreement"]["all"]["whole"] <= 1
    assert load_policy(model).config == {"width": 8, "layers": 1, "heads": 2}
