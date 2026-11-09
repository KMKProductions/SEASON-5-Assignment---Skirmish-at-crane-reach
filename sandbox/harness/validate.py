"""The submission load check: ``python -m game_sandbox_harness.validate``.

The dynamic half of submission validation (Stage 5.4). It runs the very same
:func:`~game_sandbox_harness.manifest.load_agent` a live session uses — put the repo root on
``sys.path``, import the entry-point module, instantiate the named class, and confirm the instance
exposes callable ``reset``/``act`` — **exactly once**, and it never constructs or steps the
environment. That is the whole command: import-and-construct, classify, report.

It emits one structured result envelope on stdout (the protocol stream, classified the same way the
session runner's output is — stdout is protocol, stderr is diagnostics) and exits ``0`` on success
or ``1`` on a load failure, so the backend turns the typed ``code`` into the owner-visible message.
The closed failure set is ``import_error`` (the module failed to import, or resolved outside the
repo root), ``class_not_found`` (the manifest names a class the module does not define),
``constructor_error`` (instantiation raised), and ``missing_hook`` (no callable ``reset``/``act``),
each carrying the underlying Python error as ``detail`` for the owner's debug view.

The static manifest checks (present, valid JSON, required fields, known/ matching template version)
are **not** the gate here: that is the backend's step-3 mirror, which already ran before the build.
``load_agent`` does re-parse the manifest, so a manifest problem would still surface, but in the
pipeline it never reaches this command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import IO, Any

from .manifest import ManifestError, describe_agent_hooks, load_agent

#: The single outbound event-envelope kind this command defines (cf. ``result`` for the runner).
RESULT_KIND = "validate-result"

#: Where the overlay build stages a submission's repo root for the canonical validation seat.
DEFAULT_REPO_ROOT = "/opt/agents/submissions/seat_0"


def _claim_stdout() -> IO[str]:
    """Reserve the protocol stdout and redirect later participant prints to diagnostics."""
    sys.stdout.flush()
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    return os.fdopen(protocol_fd, "w", encoding="utf-8", newline="\n")


def _emit(protocol: IO[str], envelope: dict[str, Any]) -> None:
    """Write one result envelope as a single compact JSON line on protocol stdout, then flush."""
    protocol.write(json.dumps(envelope, separators=(",", ":"), sort_keys=True) + "\n")
    protocol.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m game_sandbox_harness.validate",
        description="Load-check a submission's agent without constructing or stepping the game.",
    )
    parser.add_argument(
        "repo_root",
        nargs="?",
        default=DEFAULT_REPO_ROOT,
        help=f"repo root with a manifest.json (default {DEFAULT_REPO_ROOT})",
    )
    args = parser.parse_args(argv)
    protocol = _claim_stdout()

    try:
        agent = load_agent(args.repo_root)
    except ManifestError as error:
        _emit(
            protocol,
            {"kind": RESULT_KIND, "ok": False, "code": error.code, "detail": str(error)},
        )
        return 1

    _emit(protocol, {"kind": RESULT_KIND, "ok": True, "hooks": describe_agent_hooks(agent)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
