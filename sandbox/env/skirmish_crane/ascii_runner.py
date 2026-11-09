"""A tiny command-line replay surface for the pure tactical rules engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import Match, MatchConfig, OrderSource, scripted_order
from .overlay import TILE_CODES, decode_overlay

# One ASCII glyph per compact overlay tile code. Uppercase marks a feature sitting on a hill.
_TILE_MARKS = {
    "g": ".",
    "h": "^",
    "w": "~",
    "v": " ",
    "f": "f",
    "m": "m",
    "s": "s",
    "F": "F",
    "M": "M",
    "S": "S",
}


def render_overlay(overlay: dict[str, object], static: dict[str, object]) -> str:
    """Render one compact Stage 2 overlay without reconstructing the live rules engine."""
    decoded = decode_overlay(overlay, static)
    battlefield = decoded["battlefield"]
    capture = decoded["capture"]
    units = decoded["units"]
    marks = {(unit["position"]["q"], unit["position"]["r"]): unit["side"][0].upper() for unit in units}
    rows = [f"round {decoded['round']}  red={capture['red']} blue={capture['blue']}"]
    for r, row in enumerate(battlefield["tiles"]):
        rows.append(
            "".join(
                marks.get((q, r), _TILE_MARKS[TILE_CODES[cell["terrain"], cell["feature"]]])
                for q, cell in enumerate(row)
            ).rstrip()
        )
    return "\n".join(rows)


def replay_jsonl(path: Path | str) -> str:
    """Replay the recorded Skirmish overlays in a JSONL recording as ASCII frames."""
    frames: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError("recording is missing its header")
        header = json.loads(header_line)
        static = header.get("overlay_static")
        if not isinstance(static, dict):
            raise ValueError("recording header is missing Crane Reach overlay static data")
        for line in handle:
            state = json.loads(line)
            overlay = state.get("overlay")
            if overlay is not None:
                if not isinstance(overlay, dict):
                    raise ValueError("recording state has a malformed Crane Reach overlay")
                frames.append(render_overlay(overlay, static))
    return "\n\n".join(frames)


def render(match: Match, *, round_number: int | None = None) -> str:
    shown_round = match.round if round_number is None else round_number
    marks = {unit.position: unit.side[0].upper() for unit in match.units.values()}
    rows: list[str] = [
        f"round {shown_round}  red={match.capture_scores['red']} blue={match.capture_scores['blue']}"
    ]
    for r, row in enumerate(match.battlefield.tiles):
        rows.append(
            "".join(
                marks.get((q, r), _TILE_MARKS[TILE_CODES[tile.terrain, tile.feature]])
                for q, tile in enumerate(row)
            ).rstrip()
        )
    return "\n".join(rows)


def run_scripted_match(config: MatchConfig, orders: OrderSource | None = None) -> tuple[Match, str]:
    """Run to completion and render exactly once after each completed round."""
    match = Match(config)
    frames: list[str] = []
    source = orders if orders is not None else {}
    while match.result is None:
        completed_round = match.round
        while match.result is None and match.round == completed_round:
            unit_id = match.current_unit_id
            if unit_id is None:
                break
            match.apply_order(scripted_order(source, match, unit_id))
        frames.append(render(match, round_number=completed_round))
    return match, "\n\n".join(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic tactical replay.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seat-plan", choices=("skirmish", "army"), default="skirmish")
    parser.add_argument("--round-cap", type=int, default=100)
    arguments = parser.parse_args()
    config = MatchConfig(
        seed=arguments.seed,
        seat_plan=arguments.seat_plan,
        terrain=True,
        wasteland=True,
        unit_abilities=True,
        messages=True,
        capture_zones=3,
        round_cap=arguments.round_cap,
    )
    match, transcript = run_scripted_match(config)
    print(transcript)
    print(match.result)


if __name__ == "__main__":
    main()
