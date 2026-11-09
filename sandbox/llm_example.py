"""Minimal example of calling the provided OpenAI-compatible LLM API.

Your agent may call an LLM through a standard OpenAI-compatible endpoint. The endpoint and key
come from environment variables ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY``:

- Locally, put the key your instructor provides in a ``.env`` file (copy ``.env.example``).
- Server-side, the runner injects the same two variables per player with a one-off key scoped
  to your session. Choose the ``small``, ``medium``, or ``large`` model tier in your code.

    python -m sandbox llm [small|medium|large]
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

_MODEL_TIERS = ("small", "medium", "large")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m sandbox llm",
        description="Make one LLM API smoke-test call.",
    )
    parser.add_argument(
        "model_tier",
        nargs="?",
        choices=_MODEL_TIERS,
        default="small",
        help="model tier to request (default: small)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    load_dotenv()  # read the local development settings from .env, if present

    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not base_url or not api_key:
        print(
            "Set OPENAI_BASE_URL and OPENAI_API_KEY (copy .env.example to .env and fill them in).",
            file=sys.stderr,
        )
        return 1

    # The class proxy owns retries. Disabling SDK retries keeps this smoke command to one logical
    # request and exposes the backend's terminal response directly.
    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
    response = client.chat.completions.create(
        model=args.model_tier,
        messages=[{"role": "user", "content": "Reply with exactly: Game Sandbox LLM ready"}],
        stream=False,
    )
    content = response.choices[0].message.content or "(empty response)"
    print(content)
    print(f"model tier: {args.model_tier}")
    print(f"tokens used: {response.usage.total_tokens if response.usage is not None else 'unavailable'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
