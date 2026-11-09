"""Capture accounting and terminal team-score formulas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .battlefield import Battlefield

if TYPE_CHECKING:
    from .engine import Unit


@dataclass(frozen=True)
class Result:
    red: float
    blue: float
    winner: str | None
    reason: str


def score_capture(battlefield: Battlefield, units: dict[str, Unit], scores: dict[str, int]) -> dict[str, int]:
    """Award one point per solely occupied zone and return the changed scores."""
    changed = dict(scores)
    for zone in battlefield.zones:
        occupants = {unit.side for unit in units.values() if unit.position in zone.tiles}
        if len(occupants) == 1:
            changed[occupants.pop()] += 1
    return changed


def _result(winner: str | None, winner_score: float, loser_score: float, reason: str) -> Result:
    if winner is None:
        return Result(50.0, 50.0, None, reason)
    return (
        Result(winner_score, loser_score, winner, reason)
        if winner == "red"
        else Result(loser_score, winner_score, winner, reason)
    )


def elimination_result(remaining: dict[str, int], starting: dict[str, int], *, round_cap: bool) -> Result:
    """Score an elimination game after elimination or the round-cap comparison."""
    red, blue = remaining["red"], remaining["blue"]
    if red == blue:
        return _result(None, 50, 50, "round_cap_draw" if round_cap else "elimination_draw")
    winner = "red" if red > blue else "blue"
    winner_hp, loser_hp = (red, blue) if winner == "red" else (blue, red)
    if round_cap:
        margin = (winner_hp - loser_hp) / starting[winner]
        return _result(winner, 70 + 30 * margin, 30 * (1 - margin), "round_cap")
    fraction = winner_hp / starting[winner]
    return _result(winner, 70 + 30 * fraction, 0, "elimination")


def capture_result(
    remaining: dict[str, int],
    capture: dict[str, int],
    target: int,
    *,
    capture_won: bool,
    capped: bool,
) -> Result:
    """Score a capture match after its terminating condition, naming that condition."""
    if remaining["red"] == 0 or remaining["blue"] == 0:
        if remaining["red"] == remaining["blue"]:
            return _result(None, 50, 50, "elimination_draw")
        return _result("red" if remaining["red"] else "blue", 100, 0, "elimination")
    reason = "capture" if capture_won else "round_cap" if capped else "elimination"
    if capture["red"] == capture["blue"]:
        if remaining["red"] == remaining["blue"]:
            return _result(None, 50, 50, reason)
        winner = "red" if remaining["red"] > remaining["blue"] else "blue"
    else:
        winner = "red" if capture["red"] > capture["blue"] else "blue"
    margin = min(1, abs(capture["red"] - capture["blue"]) / target)
    return _result(winner, 70 + 30 * margin, 30 * (1 - margin), reason)
