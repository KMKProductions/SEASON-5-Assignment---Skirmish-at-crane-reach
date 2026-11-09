"""The submission manifest and the agent loader.

``manifest.json`` at a repository root names the entry-point module, the agent class, and
the integer template-dependency-set version the repo targets. :func:`load_agent` parses and
validates the manifest, puts the repo root on ``sys.path``, imports the module, resolves the
class, and instantiates it with no arguments — all episode state is established in
``reset(seed)``. This is exactly the mechanism the Stage 3 session container uses per player,
one repo root each, loaded by this same function. Stage 2 does not sandbox the import:
participant code runs in-process with the harness by design; isolation is the container's
job.

Every failure raises :class:`ManifestError` naming the repo, the field, and the failure,
because in Stage 5 these messages are surfaced to the participant whose build failed.

Known limitation: per-player code isolation holds at *load* time, not at *act* time. When one
container hosts several players, :func:`load_agent` evicts a prior root's modules from
``sys.modules`` before importing the next entry point, so two repos that import a same-named
helper *at module top* each get their own (see the loader tests). But every loaded root stays on
``sys.path`` (most-recent first) for the life of the process, and ``sys.modules`` is shared. So a
helper a repo imports *lazily inside* ``act`` (``import helper`` in the method body rather than at
the top of the module) resolves against the last-loaded player's directory and is then cached under
that bare name for every player. Two players that each lazily import their own ``helper`` therefore
share whichever one imported first. Fixing this properly needs per-player module namespacing (or a
separate interpreter per player), which is a larger change than the container's current in-process
model; until then, keep the container to one player per process, or have submissions import their
helpers at module top, where the eviction already isolates them.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .agent import has_chat, has_learn

_MANIFEST_FILENAME = "manifest.json"
_REQUIRED_FIELDS = ("entry_point", "class_name", "template_version")
# Submission roots loaded so far this process, so the next load can evict a prior repo's
# modules. Pruned to live directories on each load (see load_agent) so it stays bounded to
# the submissions still on disk rather than growing for the life of the process.
_LOADED_REPO_ROOTS: set[Path] = set()


class ManifestError(Exception):
    """Raised when a manifest is missing, malformed, or names an unloadable agent.

    Carries a ``code`` classifying the failure for the Stage 5.4 ``validate`` command, which turns
    it into the owner-visible load-check reason. The dynamic-load codes (``import_error``,
    ``class_not_found``, ``constructor_error``, ``missing_hook``) are the closed set that command
    reports; static manifest problems keep the default ``manifest_invalid`` because the backend's
    step-3 static mirror is the gate for those and they never reach the load check in practice.
    """

    def __init__(self, message: str, *, code: str = "manifest_invalid") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Manifest:
    """A parsed, validated ``manifest.json``."""

    entry_point: str
    class_name: str
    template_version: int


def load_manifest(repo_root: Path | str) -> Manifest:
    """Parse and validate ``manifest.json`` under ``repo_root``."""
    root = Path(repo_root)
    path = root / _MANIFEST_FILENAME
    if not path.is_file():
        raise ManifestError(f"no {_MANIFEST_FILENAME} at repo root {root}")

    try:
        parsed: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ManifestError(f"{path} is not valid JSON: {error}") from error

    if not isinstance(parsed, dict):
        raise ManifestError(f"{path} must be a JSON object, got {type(parsed).__name__}")
    raw = cast("dict[str, Any]", parsed)

    missing = [field for field in _REQUIRED_FIELDS if field not in raw]
    if missing:
        raise ManifestError(f"{path} is missing required field(s): {', '.join(missing)}")

    unknown = sorted(set(raw) - set(_REQUIRED_FIELDS))
    if unknown:
        raise ManifestError(
            f"{path} has unknown key(s): {', '.join(unknown)}. Allowed keys are {', '.join(_REQUIRED_FIELDS)}"
        )

    entry_point = raw["entry_point"]
    class_name = raw["class_name"]
    template_version = raw["template_version"]

    if not isinstance(entry_point, str) or not entry_point:
        raise ManifestError(f"{path} field 'entry_point' must be a non-empty string")
    if not isinstance(class_name, str) or not class_name:
        raise ManifestError(f"{path} field 'class_name' must be a non-empty string")
    # bool is an int subclass; a JSON true/false must not pass as a version.
    if not isinstance(template_version, int) or isinstance(template_version, bool):
        raise ManifestError(f"{path} field 'template_version' must be an integer")

    return Manifest(
        entry_point=entry_point,
        class_name=class_name,
        template_version=template_version,
    )


def load_agent(repo_root: Path | str) -> Any:
    """Load and instantiate the agent named by ``repo_root``'s manifest.

    Prepends the repo root to ``sys.path``, imports the entry-point module, resolves the
    class, and constructs it with no arguments. Raises :class:`ManifestError` on any failure
    (import error, missing class, or a class missing a callable ``reset``/``act``).
    """
    root = Path(repo_root).resolve()
    manifest = load_manifest(root)

    root_str = str(root)
    sys.path[:] = [path for path in sys.path if path != root_str]
    sys.path.insert(0, root_str)

    # Record the root for eviction now, before importing, not only on full success. Importing
    # the entry point runs its body, which may cache helper modules under this root; if a
    # later step (missing class, bad constructor) then fails, those helpers must still be
    # evicted on the next load, so the root has to be remembered regardless of the outcome.
    _LOADED_REPO_ROOTS.add(root)

    # Drop roots whose directory is gone (e.g. a finished submission's temp dir) before using
    # the set, so it never grows without bound across a long-lived loader process.
    _LOADED_REPO_ROOTS.difference_update({r for r in _LOADED_REPO_ROOTS if not r.exists()})
    _evict_modules_from_roots(tuple(_LOADED_REPO_ROOTS))
    _evict_entry_modules(manifest.entry_point)
    importlib.invalidate_caches()
    try:
        module = importlib.import_module(manifest.entry_point)
    except Exception as error:  # noqa: BLE001 - a raising import is an import failure too
        # ImportError covers a missing module; a module whose body raises (a bad top-level
        # statement) is a failed import too, so both classify as import_error rather than escaping.
        raise ManifestError(
            f"repo {root}: could not import entry-point module {manifest.entry_point!r}: {error}",
            code="import_error",
        ) from error
    _ensure_module_loaded_from_repo(root, manifest.entry_point, module)

    try:
        agent_cls = getattr(module, manifest.class_name)
    except AttributeError as error:
        raise ManifestError(
            f"repo {root}: module {manifest.entry_point!r} has no class {manifest.class_name!r}",
            code="class_not_found",
        ) from error

    try:
        agent = agent_cls()
    except Exception as error:  # noqa: BLE001 - surfaced verbatim to the participant
        raise ManifestError(
            f"repo {root}: constructing {manifest.class_name!r} failed: {error}",
            code="constructor_error",
        ) from error

    if not callable(getattr(agent, "reset", None)):
        raise ManifestError(
            f"repo {root}: {manifest.class_name!r} has no callable 'reset' method",
            code="missing_hook",
        )
    if not callable(getattr(agent, "act", None)):
        raise ManifestError(
            f"repo {root}: {manifest.class_name!r} has no callable 'act' method",
            code="missing_hook",
        )

    return agent


def _evict_modules_from_roots(roots: tuple[Path, ...]) -> None:
    """Remove cached modules loaded from submission roots.

    This avoids reusing a previous repo's local helper module when the next repo imports a
    same-named helper during agent module import.
    """
    resolved_roots = tuple(root.resolve() for root in roots)
    interpreter_root = Path(sys.prefix).resolve()
    for name, module in list(sys.modules.items()):
        raw_path = getattr(module, "__file__", None)
        if raw_path is None:
            continue
        module_path = Path(raw_path).resolve()
        if module_path.is_relative_to(interpreter_root):
            continue
        if any(module_path.is_relative_to(root) for root in resolved_roots):
            del sys.modules[name]


def _evict_entry_modules(entry_point: str) -> None:
    """Remove a previous load of this entry point from Python's module cache.

    Student repos usually use the template's default ``entry_point`` of ``agent``. Without
    evicting the cached module, loading two different repo roots in one harness process can
    silently return the first repo's ``agent`` module for the second repo.
    """
    prefix = f"{entry_point}."
    for name in [name for name in sys.modules if name == entry_point or name.startswith(prefix)]:
        del sys.modules[name]


def _ensure_module_loaded_from_repo(root: Path, entry_point: str, module: Any) -> None:
    """Reject a manifest entry point that resolved outside the submitted repo root."""
    raw_path = getattr(module, "__file__", None)
    if raw_path is None:
        raise ManifestError(
            f"repo {root}: entry-point module {entry_point!r} has no file on disk; "
            "the entry point must resolve inside the repo root",
            code="import_error",
        )
    module_path = Path(raw_path).resolve()
    if not module_path.is_relative_to(root):
        raise ManifestError(
            f"repo {root}: entry-point module {entry_point!r} resolved to {module_path}, "
            "which is outside the repo root",
            code="import_error",
        )


def describe_agent_hooks(agent: object) -> dict[str, bool]:
    """Return which optional hooks an agent provides, for logging and CLI summaries."""
    return {"learn": has_learn(agent), "chat": has_chat(agent)}
