"""Non-blocking access to one plain-text LLM completion.

``BackgroundLLM`` owns the thread used for the request. Agent hooks stay non-blocking and can
collect the reply during a later hook without creating or managing threads themselves.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Mapping
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

_BACKGROUND_HEADER = "X-Game-Sandbox-Background"
_UNSUPPORTED_ARGUMENTS = frozenset({"tools", "tool_choice", "functions", "response_format"})


class BackgroundLLM:
    """Run at most one plain-text chat completion in the background."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._client: Any = None
        self._requesting = False
        self._response_text: str | None = None
        self.error: Exception | None = None

    @property
    def requesting(self) -> bool:
        """Whether a request is running or has a successful reply waiting to be read."""
        with self._lock:
            return self._requesting

    def request(self, *, model: str, messages: Any, **kwargs: Any) -> bool:
        """Start one plain-text completion, or return ``False`` while the slot is occupied."""
        with self._lock:
            if self._requesting:
                return False
            unsupported = sorted(_UNSUPPORTED_ARGUMENTS.intersection(kwargs))
            if unsupported:
                names = ", ".join(unsupported)
                raise ValueError(f"BackgroundLLM does not support these arguments: {names}")
            client = self._client
            if client is None:
                load_dotenv()
                base_url = os.environ.get("OPENAI_BASE_URL")
                api_key = os.environ.get("OPENAI_API_KEY")
                if not base_url or not api_key:
                    raise RuntimeError("OPENAI_BASE_URL and OPENAI_API_KEY must be set")
                client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
                self._client = client

            extra_headers_value = kwargs.pop("extra_headers", None)
            extra_headers: dict[str, str] = {}
            if extra_headers_value is not None:
                if not isinstance(extra_headers_value, Mapping):
                    raise TypeError("extra_headers must be a mapping")
                for name, value in extra_headers_value.items():
                    if not isinstance(name, str):
                        raise TypeError("extra_headers names must be strings")
                    if name.lower() != _BACKGROUND_HEADER.lower():
                        extra_headers[name] = value
            extra_headers[_BACKGROUND_HEADER] = "1"
            kwargs["extra_headers"] = extra_headers
            kwargs["stream"] = False

            self.error = None
            self._response_text = None
            self._requesting = True

        thread = threading.Thread(
            target=self._run_request,
            args=(client, model, messages, kwargs),
            daemon=True,
        )
        try:
            thread.start()
        except Exception as error:
            self._fail(error)
            return False
        return True

    def response(self) -> str | None:
        """Return a finished reply once, or ``None`` while waiting or while the slot is idle."""
        with self._lock:
            if self._response_text is None:
                return None
            text = self._response_text
            self._response_text = None
            self._requesting = False
            return text

    def _run_request(self, client: Any, model: str, messages: Any, kwargs: dict[str, Any]) -> None:
        try:
            reply = client.chat.completions.create(model=model, messages=messages, **kwargs)
            text = reply.choices[0].message.content
            if not isinstance(text, str) or not text:
                raise ValueError("completion did not contain message text")
        except Exception as error:
            self._fail(error)
            return

        with self._lock:
            self._response_text = text

    def _fail(self, error: Exception) -> None:
        print(f"BackgroundLLM request failed: {error}", file=sys.stderr, flush=True)
        with self._lock:
            self.error = error
            self._response_text = None
            self._requesting = False
