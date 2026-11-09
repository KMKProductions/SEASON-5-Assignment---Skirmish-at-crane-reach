"""The background LLM helper keeps one non-blocking plain-text request slot."""

from __future__ import annotations

import threading
from collections.abc import Callable
from types import SimpleNamespace

import pytest
from sandbox import llm


class _Completions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.reply: object = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="later reply"))]
        )
        self.failure: Exception | None = None

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        self.started.set()
        assert self.release.wait(timeout=1)
        if self.failure is not None:
            raise self.failure
        return self.reply


def _install_client(monkeypatch: pytest.MonkeyPatch) -> tuple[_Completions, list[dict[str, object]]]:
    completions = _Completions()
    clients: list[dict[str, object]] = []

    def openai(**kwargs: object) -> object:
        clients.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    monkeypatch.setattr(llm, "OpenAI", openai)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://course.example/api/llm/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "player-key")
    return completions, clients


def test_request_is_non_blocking_and_preserves_an_unread_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    completions, clients = _install_client(monkeypatch)
    helper = llm.BackgroundLLM()

    assert helper.request(
        model="small",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0,
        extra_headers={"X-Trace": "yes", "x-game-sandbox-background": "0"},
    )
    assert completions.started.wait(timeout=1)
    assert helper.requesting
    assert helper.response() is None
    assert helper.request(model="medium", messages=[]) is False

    completions.release.set()
    assert helper.request(model="medium", messages=[]) is False
    assert _eventually(lambda: len(completions.calls) == 1 and helper.response() == "later reply")
    assert helper.response() is None
    assert not helper.requesting
    assert helper.error is None
    assert clients == [
        {
            "base_url": "https://course.example/api/llm/v1",
            "api_key": "player-key",
            "max_retries": 0,
        }
    ]
    assert completions.calls == [
        {
            "model": "small",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
            "extra_headers": {"X-Trace": "yes", "X-Game-Sandbox-Background": "1"},
            "stream": False,
        }
    ]


def test_finished_unread_reply_is_not_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    completions, _clients = _install_client(monkeypatch)
    completions.release.set()

    class ImmediateThread:
        def __init__(
            self, *, target: Callable[..., None], args: tuple[object, ...], **_kwargs: object
        ) -> None:
            self._target = target
            self._args = args

        def start(self) -> None:
            self._target(*self._args)

    monkeypatch.setattr(llm.threading, "Thread", ImmediateThread)
    helper = llm.BackgroundLLM()

    assert helper.request(model="small", messages=[])
    assert helper.requesting
    assert helper.request(model="medium", messages=[], response_format={"type": "json_object"}) is False
    assert len(completions.calls) == 1
    assert helper.response() == "later reply"
    assert not helper.requesting


@pytest.mark.parametrize("argument", ["tools", "tool_choice", "functions", "response_format"])
def test_advanced_completion_shapes_raise_in_the_calling_hook(
    monkeypatch: pytest.MonkeyPatch, argument: str
) -> None:
    _completions, clients = _install_client(monkeypatch)
    helper = llm.BackgroundLLM()

    with pytest.raises(ValueError, match=argument):
        helper.request(model="small", messages=[], **{argument: None})

    assert clients == []
    assert not helper.requesting


@pytest.mark.parametrize(
    "reply",
    [
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))]),
    ],
)
def test_text_free_success_is_a_background_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    reply: object,
) -> None:
    completions, _clients = _install_client(monkeypatch)
    completions.reply = reply
    helper = llm.BackgroundLLM()

    assert helper.request(model="small", messages=[])
    completions.release.set()
    assert _eventually(lambda: not helper.requesting)

    assert helper.response() is None
    assert helper.error is not None
    assert "BackgroundLLM request failed:" in capsys.readouterr().err


def test_request_failure_sets_error_and_frees_the_slot(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    completions, _clients = _install_client(monkeypatch)
    completions.failure = RuntimeError("provider unavailable")
    helper = llm.BackgroundLLM()

    assert helper.request(model="small", messages=[])
    completions.release.set()
    assert _eventually(lambda: not helper.requesting)

    assert isinstance(helper.error, RuntimeError)
    assert helper.response() is None
    assert "provider unavailable" in capsys.readouterr().err

    completions.failure = None
    assert helper.request(model="small", messages=[])
    assert _eventually(lambda: helper.response() == "later reply")
    assert helper.error is None


@pytest.mark.parametrize("missing", ["OPENAI_BASE_URL", "OPENAI_API_KEY"])
def test_credentials_are_required_on_first_request(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    _completions, clients = _install_client(monkeypatch)
    monkeypatch.delenv(missing)
    helper = llm.BackgroundLLM()

    with pytest.raises(RuntimeError, match=missing):
        helper.request(model="small", messages=[])

    assert not helper.requesting
    assert clients == []


def test_client_captures_credentials_on_first_request_and_reuses_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completions, clients = _install_client(monkeypatch)
    monkeypatch.delenv("OPENAI_BASE_URL")
    monkeypatch.delenv("OPENAI_API_KEY")
    helper = llm.BackgroundLLM()
    assert clients == []

    monkeypatch.setenv("OPENAI_BASE_URL", "https://first.example/api/llm/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "first-player-key")
    assert helper.request(model="small", messages=[])
    completions.release.set()
    assert _eventually(lambda: helper.response() == "later reply")

    monkeypatch.setenv("OPENAI_BASE_URL", "https://other.example/api/llm/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "other-player-key")
    assert helper.request(model="medium", messages=[])
    assert _eventually(lambda: helper.response() == "later reply")

    assert clients == [
        {
            "base_url": "https://first.example/api/llm/v1",
            "api_key": "first-player-key",
            "max_retries": 0,
        }
    ]
    assert [call["model"] for call in completions.calls] == ["small", "medium"]


def _eventually(predicate: Callable[[], bool]) -> bool:
    for _ in range(1000):
        if predicate():
            return True
        threading.Event().wait(0.001)
    return False
