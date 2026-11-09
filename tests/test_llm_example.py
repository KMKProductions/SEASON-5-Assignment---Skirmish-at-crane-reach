"""The LLM smoke command uses a requested tier and never exposes the key."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sandbox import llm_example

_BASE_URL = "https://course.example/api/llm/v1"
_API_KEY = "test-only-api-key"
_REPLY = "A reply from the test upstream"


def _install_smoke_client(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []
    completions = SimpleNamespace(
        create=lambda **kwargs: (
            calls.append(kwargs)
            or SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=_REPLY))],
                usage=SimpleNamespace(total_tokens=4_321),
            )
        )
    )
    clients: list[tuple[str, str, int]] = []

    def openai(*, base_url: str, api_key: str, max_retries: int):
        clients.append((base_url, api_key, max_retries))
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    monkeypatch.setattr(llm_example, "load_dotenv", lambda: None)
    monkeypatch.setattr(llm_example, "OpenAI", openai)
    monkeypatch.setenv("OPENAI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("OPENAI_API_KEY", _API_KEY)
    return calls, clients


@pytest.mark.parametrize(
    ("argv", "model_tier"),
    [
        ([], "small"),
        (["small"], "small"),
        (["medium"], "medium"),
        (["large"], "large"),
    ],
)
def test_smoke_call_uses_requested_tier_and_reports_usage(monkeypatch, capsys, argv, model_tier):
    calls, clients = _install_smoke_client(monkeypatch)

    assert llm_example.main(argv) == 0
    captured = capsys.readouterr()
    assert _REPLY in captured.out
    assert model_tier in captured.out
    assert "4321" in captured.out
    assert clients == [(_BASE_URL, _API_KEY, 0)]
    assert len(calls) == 1
    assert calls[0]["model"] == model_tier
    assert calls[0]["stream"] is False
    assert _API_KEY not in captured.out
    assert _API_KEY not in captured.err


@pytest.mark.parametrize("missing", ["OPENAI_BASE_URL", "OPENAI_API_KEY"])
def test_smoke_call_requires_credentials(monkeypatch, capsys, missing):
    monkeypatch.setattr(llm_example, "load_dotenv", lambda: None)
    monkeypatch.setattr(llm_example, "OpenAI", lambda **_kwargs: pytest.fail("client was created"))
    monkeypatch.setenv("OPENAI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("OPENAI_API_KEY", _API_KEY)
    monkeypatch.delenv(missing)

    assert llm_example.main([])
    captured = capsys.readouterr()
    assert missing in captured.err
    assert _API_KEY not in captured.out
    assert _API_KEY not in captured.err


def test_smoke_call_rejects_unknown_tier_before_creating_client(monkeypatch, capsys):
    monkeypatch.setattr(llm_example, "OpenAI", lambda **_kwargs: pytest.fail("client was created"))

    with pytest.raises(SystemExit, match="2"):
        llm_example.main(["unknown"])
    error = capsys.readouterr().err
    assert "usage: python -m sandbox llm" in error
    assert "llm_example.py" not in error
