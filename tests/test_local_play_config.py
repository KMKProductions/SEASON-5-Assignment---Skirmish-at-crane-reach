"""The generic browser launcher always gives the live runner complete player attribution."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from sandbox import evaluate, play
from sandbox.harness.environment import (
    BuiltinAgent,
    EnvParameter,
    EnvPreset,
    PlayerBounds,
    SeatDeclaration,
    SeatPlan,
    SeatPlans,
)
from sandbox.harness.live import parse_config
from sandbox.season import SeasonSettings


def _rival_repo(tmp_path: Path) -> Path:
    rival = tmp_path / "rivals" / "v1"
    rival.mkdir(parents=True)
    (rival / "manifest.json").write_text("{}", encoding="utf-8")
    return rival


def _use_player_bounds(
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    *,
    human_players: tuple[str, ...] | None = None,
) -> None:
    humans = tuple(f"player_{index}" for index in range(count)) if human_players is None else human_players
    monkeypatch.setattr(
        play,
        "META",
        replace(
            play.META,
            layout=PlayerBounds(min=count, max=count),
            human_players=humans,
            presets=(),
        ),
    )


def _use_partnership_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    plans = SeatPlans(
        (
            SeatPlan(
                "partnership",
                "Partnership",
                (SeatDeclaration((0, 2)), SeatDeclaration((1, 3))),
            ),
        )
    )
    monkeypatch.setattr(
        play,
        "META",
        replace(
            play.META,
            layout=plans,
            human_players=("player_0", "player_1", "player_2", "player_3"),
            presets=(),
            builtin_agents=(BuiltinAgent("naive", "Test baseline"),),
        ),
    )


def _use_restricted_layout(monkeypatch: pytest.MonkeyPatch, *, wide: bool = True) -> None:
    plans = SeatPlans(
        (
            SeatPlan(
                "roles",
                "Roles",
                (
                    SeatDeclaration((2,) if wide else (1, 2)),
                    SeatDeclaration((0, 1) if wide else (0,), restricted_builtin="naive"),
                ),
            ),
        )
    )
    monkeypatch.setattr(
        play,
        "META",
        replace(play.META, layout=plans, human_players=("player_0",), parameters=(), presets=()),
    )


def _use_trial_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_player_bounds(monkeypatch, 2)
    monkeypatch.setattr(
        play,
        "META",
        replace(
            play.META,
            parameters=(
                EnvParameter("trial_knob", "Trial", "Test option.", type="int", default=100, min=1, max=1000),
                EnvParameter("other_knob", "Other", "Test option.", type="int", default=10, min=1, max=1000),
            ),
        ),
    )


def test_single_player_local_config_uses_metadata_timeout_when_omitted(monkeypatch, tmp_path: Path):
    _use_player_bounds(monkeypatch, 1)
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")

    config = play.local_config(
        seed=7,
        mode="human",
        seat=0,
        recording_dir=tmp_path / "recordings",
        step_limit=None,
    )

    assert config["player_bindings"] == {"player_0": {"kind": "external"}}
    assert config["players"] == {"player_0": {"kind": "human", "label": "You"}}
    assert "human_timeout_ms" not in config
    assert "max_steps" not in config


def test_four_player_local_config_covers_every_player_and_preserves_null_timeout(monkeypatch, tmp_path: Path):
    _use_player_bounds(monkeypatch, 4)
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")
    config = play.local_config(
        seed=8,
        mode="human",
        seat=2,
        recording_dir=tmp_path / "recordings",
        step_limit=52,
        human_timeout_ms=None,
    )

    agent = {"kind": "builtin-agent", "path": str(tmp_path / "repo")}
    attribution = {"kind": "agent", "submission_id": "local", "label": "Your agent"}
    assert config["player_bindings"] == {
        "player_0": agent,
        "player_1": agent,
        "player_2": {"kind": "external"},
        "player_3": agent,
    }
    assert config["players"] == {
        "player_0": attribution,
        "player_1": attribution,
        "player_2": {"kind": "human", "label": "You"},
        "player_3": attribution,
    }
    assert config["human_timeout_ms"] is None
    assert config["max_steps"] == 52


def test_human_mode_rejects_a_seat_excluded_from_metadata(monkeypatch, capsys):
    _use_player_bounds(monkeypatch, 2, human_players=("player_0",))
    monkeypatch.setattr(play, "launch_browser", lambda *a, **k: pytest.fail("invalid seat launched"))

    with pytest.raises(SystemExit) as error:
        play.main(["human", "--seat", "1"])

    assert error.value.code
    message = capsys.readouterr().err
    assert "seat 1" in message and "not human" in message


@pytest.mark.parametrize("default_seat", [False, True])
def test_headless_selects_an_unrestricted_seat_without_requiring_human_control(monkeypatch, default_seat):
    if default_seat:
        _use_restricted_layout(monkeypatch)
        arguments = []
        expected_seat, expected_parameters = 0, {"seat_plan": "roles"}
    else:
        _use_player_bounds(monkeypatch, 2, human_players=("player_0",))
        monkeypatch.setattr(play, "META", replace(play.META, parameters=()))
        arguments = ["--seat", "1"]
        expected_seat, expected_parameters = 1, {"players": 2}
    captured = {}
    monkeypatch.setattr(play, "run_headless", lambda **kwargs: captured.update(kwargs) or 3.5)

    assert play.main(["human", "--headless", "--seed", "7", *arguments]) == 0
    assert (captured["seat"], captured["seed"]) == (expected_seat, 7)
    assert captured["parameters"] == expected_parameters


@pytest.mark.parametrize("wide", [False, True])
def test_restricted_human_seat_defaults_and_companions(monkeypatch, tmp_path: Path, capsys, wide):
    _use_restricted_layout(monkeypatch, wide=wide)
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(play, "BUILTIN_AGENT_ROOT", tmp_path / "builtins")
    (tmp_path / "builtins" / "naive").mkdir(parents=True)
    captured = {}
    monkeypatch.setattr(play, "launch_browser", lambda config, **kwargs: captured.update(config) or 0)

    assert play.main(["human", "--no-browser"]) == 0
    assert captured["external_chat_player"] == "player_0"
    assert captured["player_bindings"]["player_0"] == {"kind": "external"}
    assert captured["players"]["player_0"] == {"kind": "human", "label": "You"}
    if wide:
        assert captured["player_bindings"]["player_1"] == {
            "kind": "builtin-agent",
            "path": str(tmp_path / "builtins" / "naive"),
            "name": "naive",
        }
        assert captured["players"]["player_1"]["builtin_name"] == "naive"
    else:
        assert captured["player_bindings"]["player_1"] == {
            "kind": "builtin-agent",
            "path": str(tmp_path / "repo"),
        }

    with pytest.raises(SystemExit) as error:
        play.main(["human", "--companion", "self"])
    assert error.value.code
    message = capsys.readouterr().err
    assert "companion" in message and "restricted" in message


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ({}, {"players": 2, "trial_knob": 100, "other_knob": 10}),
        ({"trial_knob": 150}, {"players": 2, "trial_knob": 150, "other_knob": 10}),
    ],
)
def test_preset_replaces_season_parameters_and_preserves_limits(monkeypatch, capsys, preset, expected):
    _use_trial_parameters(monkeypatch)
    monkeypatch.setattr(play, "META", replace(play.META, presets=(EnvPreset("short", "Short", preset),)))
    monkeypatch.setattr(
        play,
        "load_season_settings",
        lambda root, meta: SeasonSettings(
            "Downloaded", {"players": 2, "trial_knob": 900, "other_knob": 90}, 123, 456
        ),
    )
    captured = {}
    monkeypatch.setattr(play, "run_headless", lambda **kwargs: captured.update(kwargs) or 2.0)

    assert play.main(["agent", "--headless", "--preset", "short"]) == 0
    assert captured["parameters"] == expected
    assert (captured["decision_limit_ms"], captured["game_limit_ms"]) == (123, 456)
    output = capsys.readouterr().out
    assert "short" in output and "season.json" in output


def test_preset_parameters_yield_to_explicit_parameter_overrides(monkeypatch):
    _use_trial_parameters(monkeypatch)
    monkeypatch.setattr(
        play,
        "META",
        replace(play.META, presets=(EnvPreset("short", "Short", {"trial_knob": 150, "other_knob": 20}),)),
    )
    captured = {}
    monkeypatch.setattr(play, "run_headless", lambda **kwargs: captured.update(kwargs) or 2.0)

    assert play.main(["agent", "--headless", "--preset", "short", "--parameter", "trial_knob=200"]) == 0
    assert captured["parameters"] == {"players": 2, "trial_knob": 200, "other_knob": 20}


def test_unknown_preset_reports_the_available_names(monkeypatch, capsys):
    monkeypatch.setattr(
        play,
        "META",
        replace(play.META, presets=(EnvPreset("season_1", "Season 1", {}),)),
    )
    monkeypatch.setattr(play, "launch_browser", lambda *a, **k: pytest.fail("unknown preset launched"))

    with pytest.raises(SystemExit) as error:
        play.main(["agent", "--preset", "missing"])

    assert error.value.code
    message = capsys.readouterr().err
    assert "missing" in message and "season_1" in message


@pytest.mark.parametrize(
    ("arguments", "seeds", "mean"),
    [
        ([], [0, 1, 2, 3, 4], "2.00"),
        (["--episodes", "2"], [0, 1], "0.50"),
        (["--seeds", "5", "9"], [5, 9], "7.00"),
    ],
)
def test_evaluate_runs_the_requested_seeds_seat_and_rival(
    monkeypatch, tmp_path: Path, capsys, arguments, seeds, mean
):
    _use_player_bounds(monkeypatch, 2)
    monkeypatch.setattr(play, "META", replace(play.META, parameters=()))
    rival = _rival_repo(tmp_path)
    calls = []

    def run_headless(**kwargs):
        calls.append(kwargs)
        return float(kwargs["seed"])

    monkeypatch.setattr(evaluate, "run_headless", run_headless)

    assert evaluate.main([*arguments, "--seat", "1", "--vs", str(rival)]) == 0
    assert [(call["seed"], call["seat"], call["vs"]) for call in calls] == [
        (seed, 1, rival.resolve()) for seed in seeds
    ]
    assert all(call["parameters"] == {"players": 2} for call in calls)
    assert mean in capsys.readouterr().out.splitlines()[-1]


@pytest.mark.parametrize(
    ("command", "arguments"),
    ((play, ["agent", "--headless"]), (evaluate, ["--episodes", "1"])),
)
def test_headless_commands_propagate_participant_value_errors(monkeypatch, command, arguments):
    _use_player_bounds(monkeypatch, 1)

    def crash(**kwargs: object) -> float:
        raise ValueError("student act crashed")

    monkeypatch.setattr(command, "run_headless", crash)

    with pytest.raises(ValueError, match="student act crashed"):
        command.main(arguments)


def test_resolve_rival_accepts_a_folder_and_its_manifest_path(monkeypatch, tmp_path: Path):
    rival = _rival_repo(tmp_path)

    assert play.resolve_rival(str(rival)) == rival.resolve()
    assert play.resolve_rival(str(rival / "manifest.json")) == rival.resolve()

    monkeypatch.chdir(tmp_path)
    assert play.resolve_rival("rivals/v1") == rival.resolve()


def test_resolve_rival_rejects_missing_and_manifestless_paths(tmp_path: Path):
    with pytest.raises(ValueError, match="could not find"):
        play.resolve_rival(str(tmp_path / "missing"))

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no manifest.json"):
        play.resolve_rival(str(empty))


def test_vs_fills_every_opposing_player_in_a_one_player_per_seat_layout(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(play, "META", replace(play.META, layout=PlayerBounds(min=4, max=4), presets=()))
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")
    rival = _rival_repo(tmp_path)

    config = play.local_config(
        seed=1, mode="agent", seat=0, recording_dir=tmp_path / "recordings", step_limit=None, vs=rival
    )

    bindings = config["player_bindings"]
    assert bindings["player_0"] == {"kind": "builtin-agent", "path": str(tmp_path / "repo")}
    for other in ("player_1", "player_2", "player_3"):
        assert bindings[other] == {"kind": "builtin-agent", "path": str(rival)}
    assert config["players"]["player_0"] == {
        "kind": "agent",
        "submission_id": "local",
        "label": "Your agent",
    }
    assert config["players"]["player_1"] == {
        "kind": "agent",
        "submission_id": "local-rival",
        "label": "Rival (v1)",
    }
    parse_config([json.dumps(config)], entry=play._entry())


def test_vs_keeps_the_selected_seat_on_your_agent_in_a_partnership_layout(monkeypatch, tmp_path: Path):
    _use_partnership_layout(monkeypatch)
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")
    rival = _rival_repo(tmp_path)

    config = play.local_config(
        seed=1, mode="agent", seat=0, recording_dir=tmp_path / "recordings", step_limit=None, vs=rival
    )

    bindings = config["player_bindings"]
    assert bindings["player_0"]["path"] == str(tmp_path / "repo")
    assert bindings["player_2"]["path"] == str(tmp_path / "repo")
    assert bindings["player_1"]["path"] == str(rival)
    assert bindings["player_3"]["path"] == str(rival)


def test_saved_companion_keeps_the_first_human_capable_member_as_the_chat_sender(
    monkeypatch,
    tmp_path: Path,
):
    _use_partnership_layout(monkeypatch)
    monkeypatch.setattr(play, "META", replace(play.META, human_players=("player_2",)))
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")
    companion = _rival_repo(tmp_path)

    config = play.local_config(
        seed=1,
        mode="human",
        seat=0,
        recording_dir=tmp_path / "recordings",
        step_limit=None,
        companion=str(companion),
    )

    assert config["external_chat_player"] == "player_2"
    assert config["player_bindings"]["player_2"] == {"kind": "external"}
    assert config["player_bindings"]["player_0"]["path"] == str(companion.resolve())
    assert config["players"]["player_0"]["submission_id"] == "local-companion"


@pytest.mark.parametrize(("companion", "use_rival"), [(None, False), (None, True), ("self", True)])
def test_whole_seat_control_and_opponents(monkeypatch, tmp_path: Path, companion, use_rival):
    _use_partnership_layout(monkeypatch)
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")
    rival = _rival_repo(tmp_path) if use_rival else None

    config = play.local_config(
        seed=1,
        mode="human",
        seat=0,
        recording_dir=tmp_path / "recordings",
        step_limit=None,
        companion=companion,
        vs=rival,
    )

    assert config["external_chat_player"] == "player_0"
    opponent = {"kind": "builtin-agent", "path": str(rival if use_rival else tmp_path / "repo")}
    assert config["player_bindings"] == {
        "player_0": {"kind": "external"},
        "player_1": opponent,
        "player_2": {"kind": "external"},
        "player_3": opponent,
    }
    assert config["players"]["player_2"] == {"kind": "human", "label": "You"}


def test_watching_without_vs_puts_the_naive_baseline_opposite_your_agent(monkeypatch, tmp_path: Path):
    _use_partnership_layout(monkeypatch)
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(play, "BUILTIN_AGENT_ROOT", tmp_path / "builtins")
    (play.BUILTIN_AGENT_ROOT / "naive").mkdir(parents=True)

    config = play.local_config(
        seed=1, mode="agent", seat=0, recording_dir=tmp_path / "recordings", step_limit=None
    )

    for player_id in ("player_0", "player_2"):
        assert config["player_bindings"][player_id] == {
            "kind": "builtin-agent",
            "path": str(tmp_path / "repo"),
        }
        assert config["players"][player_id] == {
            "kind": "agent",
            "submission_id": "local",
            "label": "Your agent",
        }
    for player_id in ("player_1", "player_3"):
        assert config["player_bindings"][player_id] == {
            "kind": "builtin-agent",
            "path": str(play.BUILTIN_AGENT_ROOT / "naive"),
            "name": "naive",
        }
        assert config["players"][player_id] == {
            "kind": "agent",
            "builtin_name": "naive",
            "label": "Test baseline",
        }


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize(("command", "arguments"), [(play, ["agent"]), (evaluate, [])])
def test_vs_requires_an_opposing_seat(monkeypatch, capsys, tmp_path: Path, command, arguments, wide):
    if wide:
        plans = SeatPlans((SeatPlan("coop", "Cooperative", (SeatDeclaration((0, 1, 2)),)),))
        monkeypatch.setattr(play, "META", replace(play.META, layout=plans, presets=()))
    else:
        _use_player_bounds(monkeypatch, 1)
    monkeypatch.setattr(play, "launch_browser", lambda *a, **k: pytest.fail("invalid rival launched"))
    monkeypatch.setattr(command, "run_headless", lambda **kwargs: pytest.fail("invalid rival ran"))
    rival = _rival_repo(tmp_path)

    with pytest.raises(SystemExit) as error:
        command.main([*arguments, "--vs", str(rival)])

    assert error.value.code
    message = capsys.readouterr().err
    assert "--vs" in message
    assert ("part of your team" if wide else "only one player") in message


@pytest.mark.parametrize(
    ("case", "arguments", "reason"),
    [
        ("singleton", ["human", "--companion", "self"], "more than one player"),
        ("partial-human", ["human", "--companion", "self"], "cannot be controlled"),
        ("missing", ["human"], "pass --companion"),
        ("watch", ["agent", "--companion", "self"], "browser human mode"),
        ("bad-repo", ["human", "--companion", "missing-agent"], "could not find"),
    ],
)
def test_companion_errors_fail_before_launch(monkeypatch, tmp_path: Path, capsys, case, arguments, reason):
    if case == "singleton":
        _use_player_bounds(monkeypatch, 1)
    else:
        _use_partnership_layout(monkeypatch)
        monkeypatch.setattr(play, "META", replace(play.META, human_players=("player_0",)))
    monkeypatch.setattr(play, "launch_browser", lambda *a, **k: pytest.fail("invalid companion launched"))
    if case == "bad-repo":
        arguments = ["human", "--companion", str(tmp_path / "missing-agent")]

    with pytest.raises(SystemExit) as error:
        play.main(arguments)

    assert error.value.code
    message = capsys.readouterr().err
    assert "companion" in message and reason in message


def test_named_builtin_companion_carries_matching_binding_and_attribution(monkeypatch, tmp_path: Path):
    _use_partnership_layout(monkeypatch)
    monkeypatch.setattr(play, "BUILTIN_AGENT_ROOT", tmp_path / "builtins")
    (play.BUILTIN_AGENT_ROOT / "naive").mkdir(parents=True)

    config = play.local_config(
        seed=1,
        mode="human",
        seat=0,
        recording_dir=tmp_path / "recordings",
        step_limit=None,
        companion="naive",
    )

    assert config["player_bindings"]["player_0"] == {"kind": "external"}
    assert config["player_bindings"]["player_2"] == {
        "kind": "builtin-agent",
        "path": str(play.BUILTIN_AGENT_ROOT / "naive"),
        "name": "naive",
    }
    assert config["players"]["player_2"] == {
        "kind": "agent",
        "builtin_name": "naive",
        "label": "Test baseline",
    }


def test_run_headless_vs_loads_a_separate_agent_for_every_other_player(monkeypatch, tmp_path: Path):
    _use_partnership_layout(monkeypatch)
    repo = tmp_path / "repo"
    monkeypatch.setattr(play, "REPO_ROOT", repo)
    monkeypatch.setattr(play, "make_env", lambda parameters: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(play, "load_agent", lambda root: SimpleNamespace(root=Path(root)))
    captured = {}

    def episode(agent, env, **kwargs):
        captured.update(kwargs, agent=agent)
        return 1.5

    monkeypatch.setattr(play, "play_episode", episode)
    rival = _rival_repo(tmp_path)

    assert play.run_headless(seed=3, max_steps=None, seat=0, vs=rival) == 1.5
    others = captured["other_agents"]
    assert {player: agent.root for player, agent in others.items()} == {
        "player_1": rival,
        "player_2": repo,
        "player_3": rival,
    }
    assert captured["agent"].root == repo
    assert captured["score_player_ids"] == ("player_0", "player_2")
    assert len({id(agent) for agent in [captured["agent"], *others.values()]}) == 4


def test_run_headless_without_vs_loads_naive_for_each_opposing_player(monkeypatch, tmp_path: Path):
    _use_player_bounds(monkeypatch, 2)
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")
    baseline = tmp_path / "builtins" / "naive"
    monkeypatch.setattr(play, "builtin_agent_path", lambda name: baseline)
    monkeypatch.setattr(play, "make_env", lambda parameters: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(play, "load_agent", lambda root: SimpleNamespace(root=Path(root)))
    captured = {}

    def episode(agent, env, **kwargs):
        captured.update(kwargs, agent=agent)
        return 0.0

    monkeypatch.setattr(play, "play_episode", episode)

    assert play.run_headless(seed=0, max_steps=None, seat=0) == 0.0
    assert {player: agent.root for player, agent in captured["other_agents"].items()} == {
        "player_1": baseline
    }
    assert captured["agent"].root == tmp_path / "repo"
    assert captured["score_player_ids"] == ("player_0",)


def test_play_episode_binds_other_agents_and_defaults(monkeypatch):
    players = ("player_0", "player_1", "player_2")
    monkeypatch.setattr(play, "possible_players", lambda parameters=None: players)
    captured: dict[str, object] = {}

    def fake_run_episode(entry: object, players: object, **kwargs: object) -> SimpleNamespace:
        captured["players"] = players
        return SimpleNamespace(scores={"player_0": 4.0})

    monkeypatch.setattr(play, "run_episode", fake_run_episode)
    mine, rival_agent = object(), object()

    score = play.play_episode(
        mine, object(), seed=0, player_id="player_0", other_agents={"player_2": rival_agent}
    )

    assert score == 4.0
    players = captured["players"]
    assert isinstance(players, dict)
    assert isinstance(players["player_0"], play.AgentPlayer)
    assert players["player_0"].agent is mine
    assert isinstance(players["player_2"], play.AgentPlayer)
    assert players["player_2"].agent is rival_agent
    assert isinstance(players["player_1"], play.ExternalPlayer)


def test_play_episode_averages_the_selected_seat_scores(monkeypatch):
    players = ("player_0", "player_1", "player_2")
    monkeypatch.setattr(play, "possible_players", lambda parameters=None: players)
    monkeypatch.setattr(
        play,
        "run_episode",
        lambda *args, **kwargs: SimpleNamespace(scores={"player_0": 2.0, "player_1": 99.0, "player_2": 8.0}),
    )

    assert (
        play.play_episode(
            object(),
            object(),
            seed=0,
            player_id="player_0",
            score_player_ids=("player_0", "player_2"),
        )
        == 5.0
    )
