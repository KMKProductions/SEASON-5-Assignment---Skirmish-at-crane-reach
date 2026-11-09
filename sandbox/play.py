"""Run one local browser session through the same live runner used by production.

``python -m sandbox play`` hands you a seat to play; ``python -m sandbox watch`` runs your agent in
one instead. Commands select a resolved seat rather than one PettingZoo player. Watching and
headless runs face the declared Naive baseline in every unrestricted opposing seat, while playing
by hand faces your own agent there. ``--vs`` replaces those opponents with a saved rival agent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import webbrowser
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

from sandbox.env import META, PLAYER_ID, default_action, extract_overlay, extract_overlay_static, make_env
from sandbox.harness.environment import (
    EnvironmentEntry,
    ParameterValue,
    ResolvedLayout,
    ResolvedSeat,
    preset_values,
    resolve_layout,
    resolve_parameters,
)
from sandbox.harness.live import UNSET_TIMEOUT, UnsetTimeout
from sandbox.harness.local_server import LocalServer
from sandbox.harness.manifest import load_agent as _load_agent
from sandbox.harness.session import AgentPlayer, ExternalPlayer, run_episode
from sandbox.season import announce, load_season_settings, parse_parameter_overrides

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = Path(__file__).resolve().parent / "web"
BUILTIN_AGENT_ROOT = Path(__file__).resolve().parent / "builtins"


class _DefaultSource:
    """An action source that lets ``ExternalPlayer`` select the environment's legal default."""

    def get_action(self, player_id: str, observation: object, window_ms: int | None) -> None:
        return None


def load_agent(repo_root: Path) -> Any:
    """Load the manifest-selected agent through the same harness loader used by live sessions."""
    return _load_agent(str(repo_root))


def _entry(make: Any = make_env) -> EnvironmentEntry:
    return EnvironmentEntry(
        meta=META,
        make=make,
        default_action=default_action,
        overlay=extract_overlay,
        overlay_static=extract_overlay_static,
    )


def possible_players(parameters: Mapping[str, ParameterValue] | None = None) -> tuple[str, ...]:
    """Return the canonical players declared by the resolved assignment layout."""
    return resolved_layout(parameters).players


def resolved_layout(parameters: Mapping[str, ParameterValue] | None = None) -> ResolvedLayout:
    """Resolve the environment's complete seat layout from one parameter map."""
    values = resolve_parameters(META) if parameters is None else parameters
    return resolve_layout(META, values)


def _selected_seat(layout: ResolvedLayout, seat: int) -> ResolvedSeat:
    """Return one seat from an already resolved layout."""
    if not 0 <= seat < layout.seat_count:
        raise ValueError(f"--seat must name one of 0..{layout.seat_count - 1}")
    return layout.seats[seat]


def selected_seat(seat: int, parameters: Mapping[str, ParameterValue] | None = None) -> ResolvedSeat:
    """Return one resolved seat, rejecting an index outside the current layout."""
    return _selected_seat(resolved_layout(parameters), seat)


def default_human_seat_index(layout: ResolvedLayout) -> int:
    """Choose the preferred human-playable seat."""
    human_players = frozenset(META.human_players)
    capable = [
        (index, seat)
        for index, seat in enumerate(layout.seats)
        if any(player in human_players for player in seat.players)
    ]
    restricted = next(
        (index for index, seat in capable if seat.restricted_builtin is not None),
        None,
    )
    if restricted is not None:
        return restricted
    if capable:
        return capable[0][0]
    raise ValueError(f"{META.env_id!r} has no human-playable seat")


def default_agent_seat_index(layout: ResolvedLayout) -> int:
    """Choose the first seat available to the student's agent."""
    for index, seat in enumerate(layout.seats):
        if seat.restricted_builtin is None:
            return index
    raise ValueError(f"{META.env_id!r} has no unrestricted seat for your agent")


def _rival_player_ids(layout: ResolvedLayout, seat: int) -> frozenset[str]:
    """Return replaceable opponents from an already resolved layout."""
    _selected_seat(layout, seat)
    return frozenset(
        player
        for index, candidate in enumerate(layout.seats)
        if index != seat and candidate.restricted_builtin is None
        for player in candidate.players
    )


def rival_player_ids(seat: int, parameters: Mapping[str, ParameterValue]) -> frozenset[str]:
    """Return players in unrestricted seats outside the selected seat."""
    return _rival_player_ids(resolved_layout(parameters), seat)


def resolve_agent_repo(raw: str, *, option: str) -> Path:
    """Resolve an agent folder or its manifest path for one CLI option."""
    supplied = Path(raw)
    folder = (supplied.parent if supplied.name == "manifest.json" else supplied).resolve()
    if not folder.is_dir():
        raise ValueError(
            f"{option} could not find {folder}. Pass a folder that contains a manifest.json, "
            "or the manifest.json file itself."
        )
    if not (folder / "manifest.json").is_file():
        raise ValueError(
            f"{option} found no manifest.json in {folder}. Copy agent.py and manifest.json from "
            "the version you want to play against into that folder."
        )
    return folder


def resolve_rival(raw: str) -> Path:
    """Resolve a ``--vs`` folder or manifest path."""
    return resolve_agent_repo(raw, option="--vs")


def builtin_agent_path(name: str) -> Path:
    """Return one bundled builtin repository after checking its metadata declaration."""
    if name not in {agent.name for agent in META.builtin_agents}:
        raise ValueError(f"unknown builtin agent {name!r} for {META.env_id!r}")
    path = BUILTIN_AGENT_ROOT / name
    if not path.is_dir():
        raise ValueError(f"builtin agent {name!r} is missing from this template at {path}")
    return path


def builtin_agent_label(name: str) -> str:
    """Return one declared builtin's display label."""
    return next(agent.label for agent in META.builtin_agents if agent.name == name)


def parse_rival(
    parser: argparse.ArgumentParser,
    raw: str | None,
    seat: int,
    parameters: Mapping[str, ParameterValue] | None = None,
) -> Path | None:
    """Turn a ``--vs`` value into the rival's folder, reporting problems through the parser.

    Restricted seats retain their designated builtin, so a rival must have at least one unrestricted
    opposing seat to replace.
    """
    if raw is None:
        return None
    layout = resolved_layout(parameters)
    if not _rival_player_ids(layout, seat):
        reason = (
            "it has only one player"
            if layout.player_count == 1
            else "every opposing seat is restricted or part of your team"
        )
        parser.error(f"--vs is not available in this game: {reason}, so there are no opponents to replace")
    try:
        return resolve_rival(raw)
    except ValueError as error:
        parser.error(str(error))


def _rival_label(rival: Path) -> str:
    """Name the rival after its folder so two saved versions stay distinguishable in the viewer."""
    return f"Rival ({rival.name})" if rival.name else "Rival"


def _repo_assignment(path: Path, *, submission_id: str, label: str) -> tuple[dict[str, str], dict[str, str]]:
    """Build one local or saved-repository binding and its recording attribution."""
    return (
        {"kind": "builtin-agent", "path": str(path)},
        {"kind": "agent", "submission_id": submission_id, "label": label},
    )


def _builtin_assignment(name: str) -> tuple[dict[str, str], dict[str, str]]:
    """Build one named-builtin binding and matching recording attribution."""
    return (
        {"kind": "builtin-agent", "path": str(builtin_agent_path(name)), "name": name},
        {"kind": "agent", "builtin_name": name, "label": builtin_agent_label(name)},
    )


def _companion_assignment(raw: str) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve a declared builtin name or saved agent path for a wide human seat."""
    if raw in {agent.name for agent in META.builtin_agents}:
        return _builtin_assignment(raw)
    path = resolve_agent_repo(raw, option="--companion")
    label = f"Companion ({path.name})" if path.name else "Companion"
    return _repo_assignment(path, submission_id="local-companion", label=label)


def play_episode(
    agent: Any,
    env: Any,
    *,
    seed: int,
    max_steps: int | None = None,
    player_id: str = PLAYER_ID,
    parameters: Mapping[str, ParameterValue] | None = None,
    decision_limit_ms: int | None = None,
    game_limit_ms: int | None = None,
    other_agents: Mapping[str, Any] | None = None,
    score_player_ids: tuple[str, ...] | None = None,
) -> float:
    """Play one headless episode with one selected agent and legal defaults for every other player.

    ``env`` is already built, so the factory below returns it as-is and ignores the map the harness
    hands it. Pass the same ``parameters`` the environment was built from, otherwise the recording
    would describe settings the game did not actually run with. Omitting them means plain defaults,
    which is what ``make_env(resolve_parameters(META))`` produces. ``other_agents`` names agent
    instances for specific other players; players it leaves out keep the legal default. By default
    the result is ``player_id``'s score. ``score_player_ids`` reduces a wide seat by averaging its
    member scores, matching official evaluation.
    """
    others: Mapping[str, Any] = {} if other_agents is None else other_agents

    def _player(candidate: str) -> AgentPlayer | ExternalPlayer:
        if candidate == player_id:
            return AgentPlayer(agent)
        if candidate in others:
            return AgentPlayer(others[candidate])
        return ExternalPlayer(_DefaultSource())

    resolved_parameters = resolve_parameters(META) if parameters is None else parameters
    players = {candidate: _player(candidate) for candidate in possible_players(resolved_parameters)}
    result = run_episode(
        _entry(lambda _parameters: env),
        players,
        seed=seed,
        parameters=resolved_parameters,
        step_limit_ms=decision_limit_ms,
        episode_limit_ms=game_limit_ms,
        max_steps=max_steps,
    )
    scored = (player_id,) if score_player_ids is None else score_player_ids
    return sum(result.scores[candidate] for candidate in scored) / len(scored)


def _headless_agent_path(
    layout: ResolvedLayout,
    *,
    selected_index: int,
    player_id: str,
    vs: Path | None,
) -> Path:
    """Resolve the real agent repository assigned to one headless player."""
    for index, seat in enumerate(layout.seats):
        if player_id not in seat.players:
            continue
        if index == selected_index:
            return REPO_ROOT
        if seat.restricted_builtin is not None:
            return builtin_agent_path(seat.restricted_builtin)
        return builtin_agent_path("naive") if vs is None else vs
    raise ValueError(f"layout carries no seat for {player_id!r}")


def run_headless(
    *,
    seed: int,
    max_steps: int | None,
    seat: int,
    vs: Path | None = None,
    parameters: Mapping[str, ParameterValue] | None = None,
    decision_limit_ms: int | None = None,
    game_limit_ms: int | None = None,
) -> float:
    """Run the selected seat through the harness without local networking or browser rendering.

    Every player receives a separately constructed agent. The selected seat runs this repository,
    opposing unrestricted seats run ``vs`` or Naive, and restricted seats keep their designated
    builtin. The returned score is the mean across the selected seat's players.
    """
    resolved_parameters = resolve_parameters(META) if parameters is None else parameters
    layout = resolved_layout(resolved_parameters)
    chosen = _selected_seat(layout, seat)
    if chosen.restricted_builtin is not None:
        raise ValueError(f"seat {seat} is restricted to builtin {chosen.restricted_builtin!r}")
    if vs is not None and not _rival_player_ids(layout, seat):
        raise ValueError("--vs has no unrestricted opposing seat to replace")
    player_id = chosen.players[0]
    # One resolution feeds both the environment and the episode, so the recorded parameters always
    # describe the environment that actually ran.
    other_agents = {
        candidate: load_agent(
            _headless_agent_path(
                layout,
                selected_index=seat,
                player_id=candidate,
                vs=vs,
            )
        )
        for candidate in layout.players
        if candidate != player_id
    }
    env = make_env(resolved_parameters)
    try:
        return play_episode(
            load_agent(REPO_ROOT),
            env,
            seed=seed,
            max_steps=max_steps,
            player_id=player_id,
            parameters=resolved_parameters,
            decision_limit_ms=decision_limit_ms,
            game_limit_ms=game_limit_ms,
            other_agents=other_agents,
            score_player_ids=chosen.players,
        )
    finally:
        env.close()


def local_config(
    *,
    seed: int,
    mode: Literal["human", "agent"],
    seat: int,
    recording_dir: Path,
    step_limit: int | None,
    human_timeout_ms: int | None | UnsetTimeout = UNSET_TIMEOUT,
    vs: Path | None = None,
    companion: str | None = None,
    parameters: Mapping[str, ParameterValue] | None = None,
    decision_limit_ms: int | None = None,
    game_limit_ms: int | None = None,
) -> dict[str, object]:
    """Build the complete runner config and header attribution for one local launch.

    Watching your agent puts the Naive baseline in every unrestricted opposing seat, while playing
    by hand leaves your own agent there. ``vs`` replaces those opponents with the saved rival.
    Restricted seats always retain their designated builtin. A wide human seat defaults to
    whole-seat control; an explicit builtin name or repository path gives every other member a
    companion instance.
    """
    resolved_parameters = resolve_parameters(META) if parameters is None else parameters
    layout = resolved_layout(resolved_parameters)
    chosen = _selected_seat(layout, seat)
    if mode == "agent" and chosen.restricted_builtin is not None:
        raise ValueError(f"seat {seat} is restricted to builtin {chosen.restricted_builtin!r}")
    if mode != "human" and companion is not None:
        raise ValueError("--companion is only available in human mode")
    if vs is not None and not _rival_player_ids(layout, seat):
        raise ValueError("--vs has no unrestricted opposing seat to replace")

    human_player: str | None = None
    externally_controlled: frozenset[str] = frozenset()
    selected_companion: tuple[dict[str, str], dict[str, str]] | None = None
    if mode == "human":
        human_player = next(
            (player for player in chosen.players if player in META.human_players),
            None,
        )
        if human_player is None:
            raise ValueError(f"seat {seat} is not human-playable in {META.env_id!r}")
        if chosen.restricted_builtin is not None:
            if companion is not None:
                raise ValueError("--companion is not available for a restricted human seat")
            externally_controlled = frozenset((human_player,))
        elif len(chosen.players) == 1:
            if companion is not None:
                raise ValueError("--companion needs a seat with more than one player")
            externally_controlled = frozenset((human_player,))
        elif companion in (None, "self"):
            if any(player not in META.human_players for player in chosen.players):
                raise ValueError(
                    f"seat {seat} includes players that cannot be controlled by a human; "
                    "pass --companion NAME_OR_PATH"
                )
            externally_controlled = frozenset(chosen.players)
        else:
            externally_controlled = frozenset((human_player,))
            selected_companion = _companion_assignment(companion)

    local_assignment = _repo_assignment(REPO_ROOT, submission_id="local", label="Your agent")
    opposing_assignment: tuple[dict[str, str], dict[str, str]] | None = None
    if vs is not None:
        opposing_assignment = _repo_assignment(vs, submission_id="local-rival", label=_rival_label(vs))
    elif mode == "agent" and _rival_player_ids(layout, seat):
        # Watching your agent means watching it against the same Naive baseline ``eval`` reports on.
        # Playing by hand keeps your agent opposite you, so a session can test it.
        opposing_assignment = _builtin_assignment("naive")
    bindings: dict[str, dict[str, str]] = {}
    players: dict[str, dict[str, str]] = {}
    for seat_index, candidate_seat in enumerate(layout.seats):
        for player_id in candidate_seat.players:
            assignment: tuple[dict[str, str], dict[str, str]]
            if player_id in externally_controlled:
                assignment = ({"kind": "external"}, {"kind": "human", "label": "You"})
            elif seat_index == seat and mode == "human" and selected_companion is not None:
                assignment = selected_companion
            elif candidate_seat.restricted_builtin is not None:
                assignment = _builtin_assignment(candidate_seat.restricted_builtin)
            elif seat_index != seat and opposing_assignment is not None:
                assignment = opposing_assignment
            else:
                assignment = local_assignment
            bindings[player_id], players[player_id] = assignment
    config: dict[str, object] = {
        "env_id": META.env_id,
        "parameters": resolved_parameters,
        "seed": seed,
        "player_bindings": bindings,
        "players": players,
        "external_chat_player": human_player,
        "recording_dir": str(recording_dir),
        "recording_id": "local",
        "human_timeout_ms": None,
        "llm": None,
        "start_paused": True,
    }
    # ``max_steps`` is the local runner's explicit step cap. Omit it for normal unlimited sessions.
    if step_limit is not None:
        config["max_steps"] = step_limit
    if decision_limit_ms is not None:
        config["step_timeout_ms"] = decision_limit_ms
    if game_limit_ms is not None:
        config["episode_timeout_ms"] = game_limit_ms
    # Omission means the metadata default. JSON null is reserved for an explicit disabled timeout.
    if human_timeout_ms is not UNSET_TIMEOUT:
        config["human_timeout_ms"] = human_timeout_ms
    else:
        config.pop("human_timeout_ms")
    return config


def launch_browser(config: dict[str, object], *, port: int, open_browser: bool) -> int:
    """Serve the local bundle and runner until the player closes the command with Ctrl+C."""

    command = [sys.executable, "-m", "sandbox.live_local", json.dumps(config, separators=(",", ":"))]
    # The externally bound players are the local viewer's controlled players (empty in watch-style
    # modes), so the relay applies the production controller view to bounded broadcasts.
    bindings = cast("Mapping[str, Mapping[str, str]]", config.get("player_bindings") or {})
    controller_players = [player for player, binding in bindings.items() if binding["kind"] == "external"]

    async def serve() -> None:
        async with LocalServer(
            _entry(),
            command=command,
            static_root=WEB_ROOT,
            start_paused=True,
            port=port,
            controller_players=controller_players,
        ) as server:
            print(f"local play: {server.url}", flush=True)
            if open_browser:
                webbrowser.open(server.url)
            await server.wait()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        return 0
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play your environment in a local browser session.")
    parser.add_argument("mode", nargs="?", choices=("human", "agent"), default="human")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, help="cap headless steps")
    parser.add_argument("--seat", type=int, help="seat index; defaults to the first valid seat")
    parser.add_argument(
        "--vs",
        metavar="PATH",
        help="play against the agent saved in PATH, a folder holding that agent's manifest.json",
    )
    parser.add_argument(
        "--companion",
        metavar="NAME_OR_PATH",
        help=(
            "wide-seat companion: self, a declared builtin name, a manifest.json path, "
            "or its repository directory"
        ),
    )
    parser.add_argument("--port", type=int, default=0, help="loopback port, or 0 for an available port")
    parser.add_argument("--no-browser", action="store_true", help="serve without opening a browser")
    parser.add_argument("--headless", action="store_true", help="run one episode without a browser")
    parser.add_argument(
        "--parameter",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="typed environment parameter override; repeat for several values",
    )
    parser.add_argument("--preset", help="named environment preset")
    parser.add_argument("--decision-limit-ms", type=int, help="override the agent decision limit")
    parser.add_argument("--game-limit-ms", type=int, help="override the game time limit")
    timeouts = parser.add_mutually_exclusive_group()
    timeouts.add_argument("--human-timeout-ms", type=int, help="override the human turn timeout")
    timeouts.add_argument(
        "--no-human-timeout",
        action="store_true",
        help="disable the turn timeout for turn-based local play",
    )
    args = parser.parse_args(argv)

    try:
        season = load_season_settings(REPO_ROOT, META)
        parameter_overrides = parse_parameter_overrides(META, args.parameter)
        if args.preset is not None:
            base = preset_values(META, args.preset)
        else:
            base = {} if season is None else season.parameters
        parameters = resolve_parameters(META, base, parameter_overrides)
    except ValueError as error:
        parser.error(str(error))
    if args.decision_limit_ms is not None and args.decision_limit_ms <= 0:
        parser.error("--decision-limit-ms must be positive")
    if args.game_limit_ms is not None and args.game_limit_ms <= 0:
        parser.error("--game-limit-ms must be positive")
    decision_limit_ms = (
        args.decision_limit_ms
        if args.decision_limit_ms is not None
        else None
        if season is None
        else season.decision_limit_ms
    )
    game_limit_ms = (
        args.game_limit_ms
        if args.game_limit_ms is not None
        else None
        if season is None
        else season.game_limit_ms
    )
    if args.preset is None:
        announce(season)
    elif season is None:
        print(f"Using the {args.preset} preset.")
    else:
        print(f"Using the {args.preset} preset with the time limits from season.json.")
    layout = resolved_layout(parameters)
    human_selection = args.mode == "human" and not args.headless
    try:
        if args.seat is not None:
            seat = args.seat
        elif human_selection:
            seat = default_human_seat_index(layout)
        else:
            seat = default_agent_seat_index(layout)
        chosen = _selected_seat(layout, seat)
    except ValueError as error:
        parser.error(str(error))
    if human_selection and not any(player in META.human_players for player in chosen.players):
        parser.error(f"seat {seat} is not human-playable in {META.env_id!r}")
    if not human_selection and chosen.restricted_builtin is not None:
        parser.error(f"seat {seat} is restricted to builtin {chosen.restricted_builtin!r}")
    rival = parse_rival(parser, args.vs, seat, parameters)
    if args.companion is not None and (args.mode != "human" or args.headless):
        parser.error("--companion is only available in browser human mode")
    if args.headless:
        score = run_headless(
            seed=args.seed,
            max_steps=args.steps,
            seat=seat,
            vs=rival,
            parameters=parameters,
            decision_limit_ms=decision_limit_ms,
            game_limit_ms=game_limit_ms,
        )
        print(f"seed {args.seed}: score {score:.2f}")
        return 0
    with TemporaryDirectory(prefix="game-sandbox-local-") as recording_dir:
        try:
            config = local_config(
                seed=args.seed,
                mode=args.mode,
                seat=seat,
                recording_dir=Path(recording_dir),
                step_limit=args.steps,
                vs=rival,
                companion=args.companion,
                parameters=parameters,
                decision_limit_ms=decision_limit_ms,
                game_limit_ms=game_limit_ms,
                human_timeout_ms=(
                    None
                    if args.no_human_timeout
                    else args.human_timeout_ms
                    if args.human_timeout_ms is not None
                    else UNSET_TIMEOUT
                ),
            )
        except ValueError as error:
            parser.error(str(error))
        return launch_browser(config, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
