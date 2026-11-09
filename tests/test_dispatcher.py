"""Tests for the dependency-aware sandbox command."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sandbox import __main__ as dispatcher


def test_help_lists_commands(capsys):
    assert dispatcher.main(["--help"]) == 0
    output = capsys.readouterr().out
    commands = {line.split()[0] for line in output.splitlines() if line.split()}
    assert {"play", "watch", "eval", "test", "llm", "setup"} <= commands
    assert "season.json" in output


def test_setup_help_prints_usage_instead_of_installing(capsys, monkeypatch):
    def unexpected_setup() -> str:
        raise AssertionError("setup called")

    monkeypatch.setattr(dispatcher, "setup", unexpected_setup)

    assert dispatcher.main(["setup", "--help"]) == 0
    assert "usage: python -m sandbox setup" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("arguments", "command"),
    [
        ([], ["-m", "sandbox.play", "human"]),
        (["--seed", "6"], ["-m", "sandbox.play", "human", "--seed", "6"]),
        (["play", "--seed", "7"], ["-m", "sandbox.play", "human", "--seed", "7"]),
        (["watch", "--seed", "8"], ["-m", "sandbox.play", "agent", "--seed", "8"]),
        (["eval", "--episodes", "2"], ["-m", "sandbox.evaluate", "--episodes", "2"]),
        (["test", "-k", "mine"], ["-m", "pytest", "-k", "mine"]),
        (["llm", "medium"], ["-m", "sandbox.llm_example", "medium"]),
    ],
)
def test_commands_launch_with_arguments_and_return_the_child_status(
    monkeypatch, tmp_path, arguments, command
):
    monkeypatch.setattr(dispatcher, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(dispatcher, "_venv_python", lambda: tmp_path / "missing-python")
    monkeypatch.setattr(dispatcher, "_has_runtime", lambda python, probe: True)
    launches = []

    def run(args, **kwargs):
        launches.append((args, kwargs["cwd"]))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(dispatcher.subprocess, "run", run)

    assert dispatcher.main(arguments) == 7
    assert launches == [([sys.executable, *command], str(tmp_path))]


def test_unknown_command_fails_without_launching(monkeypatch, capsys):
    monkeypatch.setattr(dispatcher.subprocess, "run", lambda *a, **k: pytest.fail("command launched"))
    monkeypatch.setattr(dispatcher, "setup", lambda: pytest.fail("dependencies installed"))

    assert dispatcher.main(["unknown-command"])
    assert "unknown-command" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("venv_exists", "missing", "expected"),
    [
        (False, "openai", "repaired"),
        (False, "dotenv", "repaired"),
        (True, "openai", "repaired"),
        (True, "dotenv", "repaired"),
        (False, None, "current"),
        (True, None, "project"),
    ],
)
def test_llm_launch_uses_a_compatible_runtime(monkeypatch, tmp_path: Path, venv_exists, missing, expected):
    project_python = tmp_path / "project-python"
    if venv_exists:
        project_python.touch()
    monkeypatch.setattr(dispatcher, "_venv_python", lambda: project_python)
    monkeypatch.setattr(dispatcher, "REPO_ROOT", tmp_path)
    launches = []

    def run(args, **kwargs):
        if args[1] == "-c":
            missing_llm = missing is not None and missing in args[2]
            return SimpleNamespace(returncode=1 if missing_llm else 0)
        launches.append(args)
        return SimpleNamespace(returncode=0)

    def setup():
        if missing is None:
            pytest.fail("a compatible runtime was reinstalled")
        return "repaired-python"

    monkeypatch.setattr(dispatcher.subprocess, "run", run)
    monkeypatch.setattr(dispatcher, "setup", setup)

    assert dispatcher.main(["llm", "large"]) == 0
    python = {"repaired": "repaired-python", "current": sys.executable, "project": str(project_python)}[
        expected
    ]
    assert launches == [[python, "-m", "sandbox.llm_example", "large"]]
