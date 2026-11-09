"""Tests for optional local season settings and their command-line precedence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from sandbox import evaluate, play
from sandbox.harness.environment import EnvParameter, PlayerBounds, resolve_parameters
from sandbox.season import SeasonSettings, load_season_settings

_ENV_ID = "synthetic_env"

_SYNTHETIC_META = replace(
    play.META,
    env_id=_ENV_ID,
    layout=PlayerBounds(min=1, max=1),
    human_players=("player_0",),
    presets=(),
    parameters=(
        EnvParameter("speed", "Speed", "Movement speed.", type="int", default=3, min=1, max=9),
        EnvParameter("goal", "Goal", "Score needed to win.", type="int", default=7, min=1, max=20),
    ),
)


def _write_settings(root: Path, document: object) -> None:
    (root / "season.json").write_text(json.dumps(document), encoding="utf-8")


def test_load_season_settings_keeps_current_behavior_when_the_file_is_absent(tmp_path: Path):
    assert load_season_settings(tmp_path, play.META) is None


def test_load_season_settings_reads_a_complete_handwritten_parameter_map(tmp_path: Path):
    _write_settings(
        tmp_path,
        {
            "env_id": _ENV_ID,
            "season": "Spring practice",
            "parameters": {"speed": 5},
            "decision_limit_ms": 50,
            "game_limit_ms": 3_000,
        },
    )

    settings = load_season_settings(tmp_path, _SYNTHETIC_META)

    assert settings is not None
    assert settings.label == "Spring practice"
    assert settings.parameters == {"players": 1, "speed": 5, "goal": 7}
    assert settings.decision_limit_ms == 50
    assert settings.game_limit_ms == 3_000


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ("not JSON", ("season.json", "JSON")),
        (
            {"env_id": _ENV_ID, "season": "Here", "game_limit_mss": 3_000},
            ("season.json", "game_limit_mss"),
        ),
        ({"env_id": "other", "season": "Elsewhere"}, ("other", _ENV_ID)),
        ({"env_id": _ENV_ID, "season": "  "}, ("season", "nonempty")),
        ({"env_id": _ENV_ID, "season": "Here", "parameters": []}, ("season.json", "parameters")),
        (
            {"env_id": _ENV_ID, "season": "Here", "parameters": {"unknown": 1}},
            ("parameters", "unknown"),
        ),
        (
            {"env_id": _ENV_ID, "season": "Here", "parameters": {"speed": "fast"}},
            ("parameters", "speed"),
        ),
        (
            {"env_id": _ENV_ID, "season": "Here", "parameters": {"speed": 99}},
            ("parameters", "speed"),
        ),
        (
            {"env_id": _ENV_ID, "season": "Here", "decision_limit_ms": 0},
            ("season.json", "decision_limit_ms"),
        ),
        (
            {"env_id": _ENV_ID, "season": "Here", "game_limit_ms": -5},
            ("season.json", "game_limit_ms"),
        ),
    ],
)
def test_load_season_settings_rejects_invalid_files(
    tmp_path: Path, document: object, expected: tuple[str, ...]
):
    path = tmp_path / "season.json"
    path.write_text(document if isinstance(document, str) else json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError) as error:
        load_season_settings(tmp_path, _SYNTHETIC_META)

    message = str(error.value)
    assert all(fragment in message for fragment in expected)


@pytest.mark.parametrize(
    ("main", "extra_args"),
    [(play.main, ["--headless"]), (evaluate.main, ["--episodes", "1"])],
    ids=["play", "evaluate"],
)
def test_cli_overrides_season_parameters_and_limits(monkeypatch, capsys, main, extra_args):
    settings = SeasonSettings("Spring practice", {"players": 1, "speed": 2, "goal": 4}, 50, 3_000)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(play, "META", _SYNTHETIC_META)
    monkeypatch.setattr(play, "load_season_settings", lambda *_args: settings)
    monkeypatch.setattr(evaluate, "load_season_settings", lambda *_args: settings)
    monkeypatch.setattr(play, "run_headless", lambda **kwargs: calls.append(kwargs) or 1.0)
    monkeypatch.setattr(evaluate, "run_headless", lambda **kwargs: calls.append(kwargs) or 1.0)

    assert (
        main(
            [
                *extra_args,
                "--parameter",
                "speed=5",
                "--parameter",
                "goal=9",
                "--decision-limit-ms",
                "70",
                "--game-limit-ms",
                "4000",
            ]
        )
        == 0
    )

    assert len(calls) == 1
    assert calls[0]["parameters"] == {"players": 1, "speed": 5, "goal": 9}
    assert calls[0]["decision_limit_ms"] == 70
    assert calls[0]["game_limit_ms"] == 4_000
    assert "Spring practice" in capsys.readouterr().out


def test_play_browser_passes_season_limits_to_the_local_runner(monkeypatch):
    settings = SeasonSettings("Spring practice", {"players": 1, "speed": 6, "goal": 7}, 50, 3_000)
    monkeypatch.setattr(play, "META", _SYNTHETIC_META)
    monkeypatch.setattr(play, "load_season_settings", lambda *_args: settings)
    configs: list[dict[str, object]] = []
    monkeypatch.setattr(play, "launch_browser", lambda config, **_kwargs: configs.append(config) or 0)

    assert play.main(["agent", "--no-browser"]) == 0

    config = configs[0]
    assert config["step_timeout_ms"] == 50
    assert config["episode_timeout_ms"] == 3_000
    assert config["parameters"] == {"players": 1, "speed": 6, "goal": 7}


def test_eval_applies_season_settings_to_every_requested_seed(monkeypatch, capsys):
    settings = SeasonSettings("Spring practice", {"players": 1, "speed": 6, "goal": 7}, 50, 3_000)
    monkeypatch.setattr(play, "META", _SYNTHETIC_META)
    monkeypatch.setattr(evaluate, "load_season_settings", lambda *_args: settings)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(evaluate, "run_headless", lambda **kwargs: calls.append(kwargs) or 1.0)

    assert evaluate.main(["--seeds", "3", "8"]) == 0

    assert [call["seed"] for call in calls] == [3, 8]
    for call in calls:
        assert call["parameters"] == {"players": 1, "speed": 6, "goal": 7}
        assert call["decision_limit_ms"] == 50
        assert call["game_limit_ms"] == 3_000
    output = capsys.readouterr().out
    assert "Spring practice" in output
    assert "season.json" in output


def test_eval_rejects_a_seat_outside_the_resolved_season_layout(monkeypatch, capsys):
    meta = replace(play.META, layout=PlayerBounds(min=1, max=1), presets=())
    monkeypatch.setattr(play, "META", meta)
    settings = SeasonSettings("Solo season", resolve_parameters(meta), 50, 3_000)
    monkeypatch.setattr(evaluate, "load_season_settings", lambda root, meta: settings)

    with pytest.raises(SystemExit) as error:
        evaluate.main(["--seat", "1"])

    assert error.value.code == 2
    assert "--seat must name one of 0..0" in capsys.readouterr().err
