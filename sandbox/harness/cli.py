"""Development runner over :func:`run_episode`: ``python -m game_sandbox_harness.cli``.

A thin argument-parsing shell — anything the CLI does, Stage 3 does programmatically through
the same ``run_episode`` API, which is the whole point. It resolves an environment through
the entry-point registry, binds the single player to either a manifest-loaded agent or an
external action source, plays one seeded episode, optionally records it, and prints the
result summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

from .clock import SystemClock
from .environment import EnvironmentEntry, load_environment, resolve_parameters
from .manifest import load_agent
from .recording.local import FolderRecordingStore
from .session import (
    AgentPlayer,
    ExternalPlayer,
    NoopSource,
    Player,
    ScriptedSource,
    run_episode,
)
from .state import PlayerAttribution


def _build_player(entry: EnvironmentEntry, agent_root: str | None, source: str | None) -> Player:
    """Build the binding for the environment's single human-capable player."""
    if agent_root is not None:
        return AgentPlayer(load_agent(agent_root))
    if source == "noop":
        return ExternalPlayer(NoopSource())
    if source is not None and source.startswith("scripted:"):
        path = Path(source.split(":", 1)[1])
        parsed: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, list):
            raise ValueError(f"scripted source {path} must contain a JSON list of actions")
        return ExternalPlayer(ScriptedSource(cast("list[Any]", parsed)))
    raise ValueError(f"unrecognised --source {source!r}; expected 'noop' or 'scripted:<file>'")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m game_sandbox_harness.cli",
        description="Run one seeded episode of an environment through the session harness.",
    )
    parser.add_argument("--env", required=True, help="environment id (for example flappy_bird)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--agent", metavar="REPO_ROOT", help="repo root with a manifest.json")
    group.add_argument("--source", help="external action source: 'noop' or 'scripted:<file.json>'")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--record", metavar="DIR", help="record the episode under this FolderRecordingStore root"
    )
    parser.add_argument("--steps", type=int, help="cap the episode at this many steps")
    args = parser.parse_args(argv)

    entry = load_environment(args.env)

    # The single player is the environment's first human-capable player when one is declared.
    player_id = entry.meta.human_players[0] if entry.meta.human_players else "player_0"
    player = _build_player(entry, args.agent, args.source)
    player_attribution: dict[str, PlayerAttribution] | None = (
        {player_id: {"kind": "agent", "submission_id": "local", "label": "Local agent"}}
        if args.agent is not None
        else None
    )

    store = FolderRecordingStore(args.record) if args.record else None
    result = run_episode(
        entry,
        {player_id: player},
        seed=args.seed,
        parameters=resolve_parameters(entry.meta),
        store=store,
        clock=SystemClock(),
        max_steps=args.steps,
        player_attribution=player_attribution,
    )

    print(f"environment : {args.env}")
    print(f"seed        : {args.seed}")
    print(f"ticks       : {result.ticks}")
    print(f"reason      : {result.reason}")
    print(f"scores      : {result.scores}")
    print(f"step_timeouts: {result.step_timeouts}")
    if result.recording_id is not None:
        print(f"recording_id: {result.recording_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
