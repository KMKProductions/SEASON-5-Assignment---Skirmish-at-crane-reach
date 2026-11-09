"""The Season 4 starter tactical blocks: a working action space for a strategic layer.

A block is one job a unit can be given, written as a plain function::

    decide(observation, memory, goal) -> order or None

``goal`` is the position the job is about, or None. ``memory`` is this unit's own dictionary,
private to it, where a block keeps anything it needs to remember under a key of its own name.
Returning None means the situation is not this block's: the unit is holding a job that has
nothing to say right now, and whoever called it should fall back to something else. A block
that needs a destination and is handed None uses the middle of the field.

Blocks read the action mask and never plan a route. Each one asks ``action.legal_paths`` for the
paths that are legal this activation, looks at the tile each one lands on, and picks the landing
it likes best, so every order is legal by construction. Only the landing tile is scored, because
a path's middle tiles change nothing about what happens when the walking stops. The exception is
Season 6's wasteland, where entering a wasted tile costs hit points and two routes onto the same
tile can charge very different prices, so scoring routes rather than destinations becomes work
worth doing. There is no pathfinder here either: getting a unit across the field to somewhere it
cannot reach this activation is still your own.

Every score in this file is a tuple that is minimized, so lower is better throughout, and a
condition worth preferring is written as ``0 if wanted else 1``.

:func:`assign` is the placeholder. It hands each unit one block and one goal at the start of the
match and never revisits the choice, which is exactly the piece Season 4 asks you to replace:
deciding which units should do what and where, and when those assignments should change as the
battle turns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sandbox.crane import action, me, tile, units, visible, zone

if TYPE_CHECKING:
    from collections.abc import Callable

    from sandbox.observation_types import (
        AxialPosition,
        SkirmishAction,
        SkirmishObservation,
        VisibleUnit,
    )

    Block = Callable[[SkirmishObservation, dict, AxialPosition | None], SkirmishAction | None]
    Score = Callable[[AxialPosition], tuple[int, ...]]

__all__ = [
    "BLOCKS",
    "advance",
    "assign",
    "capture",
    "charge",
    "fall_back",
    "flank",
    "harass",
    "hold_ground",
    "kite",
    "screen",
    "shield_wall",
]

CHARGE_DISTANCE = 3  # start-to-end displacement that earns the cavalry its charge bonus
TETHER = 2  # how far from its goal a holding block is content to stand


class _View:
    """One activation, read once: who we are, what we see, and where we could finish."""

    def __init__(self, observation: SkirmishObservation) -> None:
        self.observation = observation
        self.side = me.side(observation)
        self.position = me.position(observation)
        self.stats = units.STATS[me.unit_type(observation)]
        self.abilities = bool(observation["observation"]["parameters"]["unit_abilities"])
        self.enemies = visible.enemies(observation)
        self.allies = visible.allies(observation)
        self.nameable = set(action.possible_targets(observation))
        self.landings = _landings(observation)


def _landings(observation: SkirmishObservation) -> list[tuple[int, AxialPosition]]:
    """Pair every tile this unit can finish on with the fewest-steps path that gets there."""
    start = me.position(observation)
    found: dict[tuple[int, int], tuple[int, AxialPosition]] = {}
    for path_id in action.legal_paths(observation):
        spot = tile.at_path_end(start, path_id)
        found.setdefault((spot["q"], spot["r"]), (path_id, spot))
    return list(found.values())


def _victims(view: _View, spot: AxialPosition) -> list[VisibleUnit]:
    """Return the nameable enemies in range from ``spot``."""
    reach = view.stats.attack_range
    return [
        enemy
        for enemy in view.enemies
        if enemy["unit_id"] in view.nameable and tile.distance(spot, enemy["position"]) <= reach
    ]


def _victim(view: _View, spot: AxialPosition) -> VisibleUnit | None:
    """Return the enemy worth naming from ``spot``: the weakest one in range, then the nearest."""
    options = _victims(view, spot)
    if not options:
        return None
    return min(options, key=lambda enemy: (enemy["hit_points"], tile.distance(spot, enemy["position"])))


def _threats(view: _View, spot: AxialPosition) -> list[VisibleUnit]:
    """Return the visible enemies that could strike ``spot`` on their next activation.

    This estimate errs in both directions. It adds movement points to attack range while ignoring
    the ground between, so an enemy that a hill or a marsh would slow down can count as arriving
    when it cannot. It also sees only what your unit sees, so an archer outside that sight can
    still cover a tile it calls quiet. Finally, the activation order is shuffled every round, so
    an enemy that acts last in one round and first in the next gets two moves before you choose
    again. Read a quiet tile as safer than the alternatives, not as safe.
    """
    found = []
    for enemy in view.enemies:
        stats = units.STATS[enemy["type"]]
        if tile.distance(spot, enemy["position"]) <= stats.movement_points + stats.attack_range:
            found.append(enemy)
    return found


def _in_shield_wall(spot: AxialPosition, friends: list[VisibleUnit]) -> bool:
    """Whether a footman on ``spot`` would stand beside one of ``friends``, its own side's footmen."""
    around = {(near["q"], near["r"]) for near in tile.neighbors(spot).values()}
    return any(
        unit["type"] == "footman" and (unit["position"]["q"], unit["position"]["r"]) in around
        for unit in friends
    )


def _denies_charge(view: _View, victim: VisibleUnit) -> bool:
    """Whether the charge bonus would be lost against ``victim`` where it stands.

    Forest denies it, and so does a shield wall, which is the defender's own side's footmen
    standing beside it. Those are our enemies, which is why this reads the enemy list.
    """
    ground = tile.terrain_at(view.observation, victim["position"])
    walled = victim["type"] == "footman" and _in_shield_wall(victim["position"], view.enemies)
    return ground["feature"] == "forest" or walled


def _order(view: _View, path_id: int, victim: VisibleUnit | None) -> SkirmishAction:
    target_id = victim["unit_id"] if victim is not None else None
    if path_id == 0:
        return action.stay(target_id, view.observation)
    return action.move(path_id, target_id, view.observation)


def _best(view: _View, score: Score) -> SkirmishAction:
    """Return the order for the landing that scores lowest, shortest path breaking a tie."""
    path_id, spot = min(view.landings, key=lambda landing: (score(landing[1]), landing[0]))
    return _order(view, path_id, _victim(view, spot))


def _destination(view: _View, goal: AxialPosition | None) -> AxialPosition:
    return goal if goal is not None else tile.at_center(view.observation)


def _turn(digit: int, sixths: int) -> int:
    """Return the direction digit ``sixths`` of a turn clockwise from ``digit``."""
    return (digit - 1 + sixths) % 6 + 1


def _number(observation: SkirmishObservation) -> int:
    """Return this unit's number within its own type, read off its ``side_type_number`` id."""
    return int(me.unit_id(observation).rsplit("_", 1)[1])


def advance(observation: SkirmishObservation, memory: dict, goal: AxialPosition | None) -> SkirmishAction:
    """Close on the goal, striking from wherever the walk ends.

    The plainest block there is, and the one worth falling back to: it always has an answer,
    because standing still is itself a landing.
    """
    view = _View(observation)
    target = _destination(view, goal)
    return _best(view, lambda spot: (tile.distance(spot, target),))


def hold_ground(
    observation: SkirmishObservation, memory: dict, goal: AxialPosition | None
) -> SkirmishAction | None:
    """Stay within arm's length of the goal and strike whatever walks into range.

    Guarding a place rather than taking one: it prefers a landing that gives it a target, but
    never at the price of wandering away from what it is holding.
    """
    view = _View(observation)
    target = _destination(view, goal)

    def score(spot: AxialPosition) -> tuple[int, ...]:
        room = tile.distance(spot, target)
        return (max(0, room - TETHER), 0 if _victim(view, spot) is not None else 1, room)

    return _best(view, score)


def kite(observation: SkirmishObservation, memory: dict, goal: AxialPosition | None) -> SkirmishAction | None:
    """Back away from the nearest enemy without losing it, the archer's shape.

    A path and a strike are one order, so opening the distance and shooting happen together.
    It gives up as much ground as its bow still covers and no more, which means that at full
    range it simply holds. When no tile it can reach keeps the enemy in range it closes instead,
    since a shot it cannot take is worth less than a shot next activation. That makes this the
    archer's block: a melee unit's range is one tile, so kiting walks it into contact. With
    nothing in sight there is nobody to back away from, so this returns None.
    """
    view = _View(observation)
    if not view.enemies:
        return None
    reach = view.stats.attack_range
    nearest = min(view.enemies, key=lambda enemy: tile.distance(view.position, enemy["position"]))
    covering = [
        landing for landing in view.landings if tile.distance(landing[1], nearest["position"]) <= reach
    ]
    if covering:
        path_id, spot = max(
            covering, key=lambda landing: (tile.distance(landing[1], nearest["position"]), -landing[0])
        )
    else:
        path_id, spot = min(
            view.landings, key=lambda landing: (tile.distance(landing[1], nearest["position"]), landing[0])
        )
    return _order(view, path_id, _victim(view, spot))


def charge(
    observation: SkirmishObservation, memory: dict, goal: AxialPosition | None
) -> SkirmishAction | None:
    """Ride the length of the field into a strike, the cavalry's shape.

    The bonus wants a run of at least three tiles from where the unit starts to where it stops,
    and it is denied by a defender standing in forest or shoulder to shoulder with another
    footman, so a defender the bonus actually lands on is worth more than a weaker one it does
    not. Only cavalry earns it, and only while the abilities variant is on, so everyone else
    gets None and so does a cavalry with no strike at the end of such a run. Riding three tiles
    for a bonus that is switched off just walks past nearer enemies.
    """
    if me.unit_type(observation) != "cavalry":
        return None
    view = _View(observation)
    if not view.abilities:
        return None
    runs = []
    for path_id, spot in view.landings:
        if tile.distance(view.position, spot) < CHARGE_DISTANCE:
            continue
        for victim in _victims(view, spot):
            score = (1 if _denies_charge(view, victim) else 0, victim["hit_points"], path_id)
            runs.append((score, path_id, victim))
    if not runs:
        return None
    _score, path_id, victim = min(runs, key=lambda run: run[0])
    return _order(view, path_id, victim)


def capture(
    observation: SkirmishObservation, memory: dict, goal: AxialPosition | None
) -> SkirmishAction | None:
    """Stand in the capture zone the goal belongs to and stay there.

    A zone pays only the side that holds it alone, which makes standing in one the whole job:
    holding an empty zone scores, and stepping into one the enemy holds stops their scoring
    even while nobody is paid. Since a unit already inside prefers to stay inside, it never
    wanders off the ground it is being paid for. A goal that is not in a zone means this unit
    was given the wrong job, so this returns None.
    """
    if goal is None:
        return None
    held = zone.at(observation, goal)
    if held is None:
        return None
    view = _View(observation)
    center = held["center"]
    inside = {(spot["q"], spot["r"]) for spot in held["tiles"]}

    def score(spot: AxialPosition) -> tuple[int, ...]:
        return (0 if (spot["q"], spot["r"]) in inside else 1, tile.distance(spot, center))

    return _best(view, score)


def fall_back(
    observation: SkirmishObservation, memory: dict, goal: AxialPosition | None
) -> SkirmishAction | None:
    """Put as much ground as possible between this unit and everything that can see it.

    Leaving every threat behind is usually impossible on a crowded field, so this maximizes the
    distance to the closest one rather than demanding safety it cannot have, and drifts toward
    the goal among the tiles that are equally clear. With nothing in sight there is nothing to
    leave, so this returns None.
    """
    view = _View(observation)
    if not view.enemies:
        return None
    target = _destination(view, goal)

    def score(spot: AxialPosition) -> tuple[int, ...]:
        room = min(tile.distance(spot, enemy["position"]) for enemy in view.enemies)
        return (-room, tile.distance(spot, target))

    return _best(view, score)


def screen(
    observation: SkirmishObservation, memory: dict, goal: AxialPosition | None
) -> SkirmishAction | None:
    """Put this unit's body between the goal and whatever is closest to it.

    The escort's job: stand on the line the threat would walk, as far up that line as the goal
    can still be called guarded, so that reaching what is behind means going through this unit
    first. With nothing in sight there is nothing to stand in front of, so this returns None.
    """
    view = _View(observation)
    if not view.enemies:
        return None
    target = _destination(view, goal)
    threat = min(view.enemies, key=lambda enemy: tile.distance(target, enemy["position"]))
    span = tile.distance(target, threat["position"])

    def score(spot: AxialPosition) -> tuple[int, ...]:
        detour = tile.distance(target, spot) + tile.distance(spot, threat["position"]) - span
        return (max(0, tile.distance(spot, target) - TETHER), detour, tile.distance(spot, threat["position"]))

    return _best(view, score)


def flank(
    observation: SkirmishObservation, memory: dict, goal: AxialPosition | None
) -> SkirmishAction | None:
    """Close on the goal along ground nothing can reach, the careful approach.

    It walks only onto tiles no visible enemy could strike on its next activation, which is what
    makes it slower and safer than :func:`advance`. Every candidate has to close the distance as
    well as be clear: a block with no safe way forward should say so and let its caller decide,
    not drift sideways or backwards while reporting progress. So when nothing in sight can be
    avoided on the way in, and when there is nothing in sight at all, this returns None and
    plain advancing becomes the honest answer.
    """
    view = _View(observation)
    if not view.enemies:
        return None
    target = _destination(view, goal)
    room = tile.distance(view.position, target)
    clear = [
        landing
        for landing in view.landings
        if tile.distance(landing[1], target) < room and not _threats(view, landing[1])
    ]
    if not clear:
        return None
    path_id, spot = min(clear, key=lambda landing: (tile.distance(landing[1], target), landing[0]))
    return _order(view, path_id, _victim(view, spot))


def harass(
    observation: SkirmishObservation, memory: dict, goal: AxialPosition | None
) -> SkirmishAction | None:
    """Strike where only the victim can strike back, then leave before anyone else arrives.

    Hit and run, alternating: on a striking activation it takes a shot only from a tile that
    nobody except its victim could answer, and on the activation after a hit it withdraws to the
    quietest tile near the goal no matter what is on offer. It remembers which half of that
    cycle it is in under ``harass.struck``. With nothing in sight there is nobody to harass, so
    this returns None, and losing contact ends the cycle: a withdrawal owed to a hit landed
    several rounds ago is not worth paying once the enemy has gone.
    """
    view = _View(observation)
    if not view.enemies:
        memory["harass.struck"] = False
        return None
    target = _destination(view, goal)
    if not memory.get("harass.struck", False):
        shots = []
        for path_id, spot in view.landings:
            victim = _victim(view, spot)
            if victim is None:
                continue
            others = [enemy for enemy in _threats(view, spot) if enemy["unit_id"] != victim["unit_id"]]
            if others:
                continue
            score = (victim["hit_points"], -tile.distance(view.position, spot), path_id)
            shots.append((score, path_id, victim))
        if shots:
            _score, path_id, victim = min(shots, key=lambda shot: shot[0])
            memory["harass.struck"] = True
            return _order(view, path_id, victim)
    memory["harass.struck"] = False
    return _best(view, lambda spot: (len(_threats(view, spot)), tile.distance(spot, target)))


def shield_wall(
    observation: SkirmishObservation, memory: dict, goal: AxialPosition | None
) -> SkirmishAction | None:
    """Hold near the goal shoulder to shoulder with another footman.

    A footman standing beside a friendly footman takes one less damage and cannot be run down by
    a charge, so where it stands is worth as much as whether it swings. Only a footman has that
    bonus to earn, so anyone else gets None.
    """
    if me.unit_type(observation) != "footman":
        return None
    view = _View(observation)
    target = _destination(view, goal)

    def score(spot: AxialPosition) -> tuple[int, ...]:
        room = tile.distance(spot, target)
        return (max(0, room - TETHER), 0 if _in_shield_wall(spot, view.allies) else 1, room)

    return _best(view, score)


BLOCKS: tuple[Block, ...] = (
    advance,
    capture,
    charge,
    fall_back,
    flank,
    harass,
    hold_ground,
    kite,
    screen,
    shield_wall,
)


def assign(observation: SkirmishObservation, memory: dict) -> tuple[Block, AxialPosition | None]:
    """Return the block and goal this unit plays, chosen once and never reconsidered.

    Everything it reads is standing knowledge, fixed for the match: the unit's type, its number
    within that type, and where the capture zone is. The side spreads out because those numbers
    differ, not because any unit is watching the battle. Footmen split between taking the zone
    and bracing inside it, cavalry split between charging and harassing from either side of the
    zone center, and the archers work off the zone itself.

    Every goal sits on or beside the zone, which matters more than it looks. A goal is where a
    unit goes when its block has nothing to say, so a goal placed off the contested ground is an
    order to walk away from the battle and wait there: point archers at a tile behind the line
    and the survivors of a fight will stand on it, out of sight of each other, until the round
    cap. The zone is where the fight is, so that is where a unit with nothing better to do
    belongs.

    This is the placeholder Season 4 replaces. A commander worth the name re-reads the battle
    every few rounds and moves units between jobs as it turns; this one commits everybody in
    round one and lets them live with it. It is narrow in another way too: Seasons 5 and 6 field
    three zones, and this reads the first and ignores the others.
    """
    found = zone.zones(observation)
    heart = found[0]["center"] if found else tile.at_center(observation)
    number = _number(observation)
    kind = me.unit_type(observation)
    if kind == "footman":
        return (capture if number % 2 == 0 else shield_wall), heart
    if kind == "archer":
        return kite, heart
    aside = _turn(me.direction(observation), 5 if number % 2 == 0 else 1)
    return (charge if number % 2 == 0 else harass), tile.neighbors(heart)[aside]
