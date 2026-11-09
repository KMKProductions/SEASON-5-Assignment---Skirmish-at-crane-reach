"""One command to set up, run, and human-test your agent: ``python -m sandbox``.

    python -m sandbox            # set up if needed, then play it yourself
    python -m sandbox play       # control a player in your browser
    python -m sandbox watch      # watch YOUR agent in your browser
    python -m sandbox eval       # run several seeded episodes, print the mean
    python -m sandbox test       # run the checks
    python -m sandbox llm [tier] # smoke-test small, medium, or large (default: small)
    python -m sandbox setup      # just install dependencies into .venv

The first time you run any of these from a fresh clone, it creates a local ``.venv`` and installs
the pinned dependencies for you, then runs your command inside it, so there is no separate
install step. Extra arguments are passed straight through, e.g. ``python -m sandbox watch
--headless --seed 7`` or ``python -m sandbox test -k manifest``.

This module imports only the standard library, so it works before any dependencies exist.
"""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

_USAGE = """\
usage: python -m sandbox [command] [args...]

commands:
  (none)   control a player in your browser (same as `play`)
  play     control a player in your browser
  watch    watch YOUR agent in your browser
  eval     run several seeded episodes and print the mean
  test     run the checks (pytest)
  llm      smoke-test small, medium, or large (default: small)
  setup    install dependencies into .venv

Extra args pass straight through, e.g. `python -m sandbox watch --seed 7` or
`python -m sandbox watch --decision-limit-ms 500`. Local episode commands also use an optional
`season.json` beside manifest.json.\
"""

#: Repo root is the parent of this ``sandbox`` package, where requirements + manifest live.
REPO_ROOT = Path(__file__).resolve().parent.parent
#: Import lines that prove an interpreter has what a command needs. The runtime deps cover
#: play/eval/human; ``test`` additionally needs pytest, so each command probes for its own deps —
#: otherwise ``test`` could run under an interpreter that lacks pytest instead of bootstrapping.
_RUNTIME_PROBE = "import pettingzoo, gymnasium, jsonschema, websockets, wcwidth"
_TEST_PROBE = "import pettingzoo, gymnasium, pytest"
_LLM_PROBE = "import openai, dotenv"


def _venv_python() -> Path:
    """Path to the project ``.venv`` interpreter for the current OS (may not exist yet)."""
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    return REPO_ROOT / ".venv" / bin_dir / exe


def _has_runtime(python: str, probe: str) -> bool:
    """True if ``python`` satisfies ``probe`` — the import line a command's deps must cover."""
    result = subprocess.run(
        [python, "-c", probe],
        capture_output=True,
    )
    return result.returncode == 0


def setup() -> str:
    """Create ``.venv`` (if missing) and install the pinned requirements into it; return its python."""
    venv_dir = REPO_ROOT / ".venv"
    if not _venv_python().exists():
        print(f"creating virtual environment in {venv_dir} ...", flush=True)
        venv.EnvBuilder(with_pip=True).create(venv_dir)
    python = str(_venv_python())
    reqs = ["-r", str(REPO_ROOT / "requirements.txt")]
    dev = REPO_ROOT / "requirements-dev.txt"
    if dev.exists():
        reqs += ["-r", str(dev)]
    print("installing dependencies (first run only) ...", flush=True)
    subprocess.run([python, "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=True)
    subprocess.run([python, "-m", "pip", "install", "--quiet", *reqs], check=True)
    return python


def _runtime_python(probe: str) -> str:
    """Return an interpreter that satisfies ``probe``, bootstrapping ``.venv`` on first use.

    Prefers an existing ``.venv`` when it passes the selected command's probe. A stale environment
    is repaired from the pinned requirements before it is used. With no ``.venv``, a current
    interpreter that already passes the probe is used; otherwise setup creates the environment.
    """
    if _venv_python().exists():
        python = str(_venv_python())
        if _has_runtime(python, probe):
            return python
        return setup()
    if _has_runtime(sys.executable, probe):
        return sys.executable
    return setup()


def _run(module_args: list[str], probe: str) -> int:
    """Run ``python <module_args>`` from the repo root, under a runtime that satisfies ``probe``."""
    python = _runtime_python(probe)
    return subprocess.run([python, *module_args], cwd=str(REPO_ROOT)).returncode


#: command -> (argv run under the resolved runtime, the probe that runtime must satisfy). ``setup``
#: is special-cased in ``main`` (it builds the runtime rather than running under it), and a bare
#: ``python -m sandbox``, or a leading flag, maps to ``play``. The launcher's own positional mode
#: names the controller (``human`` or ``agent``); the command names what you do (play or watch).
_DISPATCH = {
    "play": (["-m", "sandbox.play", "human"], _RUNTIME_PROBE),
    "watch": (["-m", "sandbox.play", "agent"], _RUNTIME_PROBE),
    "eval": (["-m", "sandbox.evaluate"], _RUNTIME_PROBE),
    "test": (["-m", "pytest"], _TEST_PROBE),
    "llm": (["-m", "sandbox.llm_example"], _LLM_PROBE),
}
_COMMANDS = {"setup", *_DISPATCH}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in ("-h", "--help"):
        print(_USAGE)
        return 0

    # A bare `python -m sandbox` plays it yourself; a leading known command selects the action,
    # and everything after it passes straight through to the underlying tool.
    if not argv:
        command, rest = "play", []
    elif argv[0] in _COMMANDS:
        command, rest = argv[0], argv[1:]
    elif argv[0].startswith("-"):
        command, rest = "play", argv  # `python -m sandbox --seed 7` -> play with flags
    else:
        print(f"unknown command {argv[0]!r}\n", file=sys.stderr)
        print(_USAGE, file=sys.stderr)
        return 2

    if command == "setup":
        if any(arg in ("-h", "--help") for arg in rest):
            print("usage: python -m sandbox setup\n\ninstall dependencies into .venv without starting a game")
            return 0
        setup()
        print(f"done. dependencies installed in {REPO_ROOT / '.venv'}", flush=True)
        return 0
    module_args, probe = _DISPATCH[command]
    return _run([*module_args, *rest], probe)


if __name__ == "__main__":
    sys.exit(main())
