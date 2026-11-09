"""Skirmish at Crane Reach agent built from reusable behavior blocks.

Read ``environment.md`` beside this file for the rules and helpers. Every unit runs its own
``Agent`` instance, so the army never shares memory. Each unit instead derives the same army plan
from facts every ally agrees on: the roster, the battlefield, and the current round.

The army plays for all three capture zones: our home zone at our back corner, the middle zone on
the river seam at the field center, and the enemy's home zone at their back corner.

* Home: the footman nearest our home zone walks onto it and never leaves. When it sees enemies
  near the zone it raises an alarm, and the archers nearest home come back to defend it, two of
  them for a small raid and more archers and footmen as the raid grows.
* Middle: one footman runs straight to the middle zone and holds it. The other middle footmen
  and the archers regroup just short of the zone, then move up together. There the footmen hold
  the zone and never leave it, while the archers stand back, climbing a nearby hill when an enemy
  is in range.
* Flanks: two footmen, one per side, each go with half the cavalry, but only once the middle is
  won. Each wing first falls back to its support post behind the middle infantry and forms up
  there, ignoring the enemy, so the wing that spawns beside the middle passage does not charge
  through the middle. From there the riders and the flank footman help win the middle fight.
  When a footman holding the middle zone has seen it quiet for a few rounds, it tells the flanks
  to set out. Each flank footman then walks through its outer passage to the enemy's home zone
  and holds it, while its riders ride around it and strike whoever comes for it, archers first.

Every move toward a goal is routed over the whole field, so units walk around the river through
a passage instead of stalling on its bank.

An ``Agent`` is only a playbook: an ordered list of behavior blocks per unit type. The first block
that proposes an order wins. Blocks are small, stateless, and reusable across unit types.
"""

from __future__ import annotations

import heapq
import json
from pathlib import Path

from sandbox.crane import action, me, paths, roster, tile, units, visible
from sandbox.observation_types import AxialPosition, SkirmishAction, SkirmishObservation

_BAD_TERRAIN = {"marsh", "waste"}
_WATER = {"water"}
_IMPASSABLE = {"water", "void"}

# Movement point cost to enter a tile, from the rules table in environment.md.
_TERRAIN_COST = {"grass": 1, "hill": 2}
_FEATURE_COST = {"none": 0, "forest": 1, "marsh": 2, "waste": 0}
# Extra routing cost for ground that hurts on entry, so routes avoid it when they can.
_ROUTE_PENALTY = {"waste": 2}

ENGAGE_RADIUS = 6  # an enemy this close ends the march and starts the fight
HOME_RADIUS = 5  # an enemy this close to the home zone's center raises the alarm
ESCORT_ARCHER_REACH = 6  # a rider dives on an enemy archer this close to its flank footman
ESCORT_THREAT_REACH = 4  # and on any other enemy this close to it
FLANK_GATHER = 0  # rounds a flank footman waits after the middle is won, so its riders gather
SUPPORT_RIDES = 2  # how many rides away a held-back rider still goes to strike in the middle
SORTIE_DEPTH = 24  # columns past the river a held-back rider may ride to strike in the middle
_UNREACHABLE = 10**9

Hex = tuple[int, int]

# Offsets that fill a hex outward from its center, ring by ring. The first seven are a center and
# its six neighbors: the shape of a capture zone.
_CLUSTER: list[Hex] = sorted(
    ((dq, dr) for dq in range(-3, 4) for dr in range(-3, 4) if (abs(dq) + abs(dr) + abs(dq + dr)) // 2 <= 3),
    key=lambda offset: ((abs(offset[0]) + abs(offset[1]) + abs(offset[0] + offset[1])) // 2, offset),
)


def _hex(position: AxialPosition) -> Hex:
    return position["q"], position["r"]


def _hex_neighbors(position: Hex) -> list[Hex]:
    return [(position[0] + dq, position[1] + dr) for dq, dr in tile.DIRECTIONS.values()]


def _hex_distance(first: Hex, second: Hex) -> int:
    dq = first[0] - second[0]
    dr = first[1] - second[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def _cube_round(fq: float, fr: float) -> Hex:
    fs = -fq - fr
    q, r, s = round(fq), round(fr), round(fs)
    dq, dr, ds = abs(q - fq), abs(r - fr), abs(s - fs)
    if dq > dr and dq > ds:
        q = -r - s
    elif dr > ds:
        r = -q - s
    return q, r


def _hex_line(start: Hex, end: Hex) -> list[Hex]:
    """Return every tile on the straight hex line from start to end, both included."""
    steps = _hex_distance(start, end)
    if steps == 0:
        return [start]
    # The tiny nudge keeps rounding consistent when the line runs exactly along a tile edge.
    return [
        _cube_round(
            start[0] + (end[0] - start[0]) * i / steps + 1e-6,
            start[1] + (end[1] - start[1]) * i / steps + 1e-6,
        )
        for i in range(steps + 1)
    ]


# --------------------------------------------------------------------------------------------
# Whole-field routing
# --------------------------------------------------------------------------------------------


class TerrainMap:
    """Travel costs over the whole field for one match, so routes go around the river.

    The action mask only covers paths of up to four steps, which cannot see a passage further
    down the bank. ``field`` runs one cheapest-route search from a goal over every passable tile,
    and a unit then picks the legal path whose landing has the cheapest remaining route.
    """

    CACHE_LIMIT = 256  # goals whose cost fields are kept for reuse within the match
    OCCUPIED_PENALTY = 5  # extra cost for a tile a unit stands on: it may move, but likely not

    def __init__(self, observation: SkirmishObservation) -> None:
        battlefield = observation["observation"]["battlefield"]
        view = {"observation": {"battlefield": battlefield}}
        side = battlefield["side"]
        self.entry_cost: dict[Hex, int] = {}
        self._route_cost: dict[Hex, int] = {}
        self.hills: frozenset[Hex] = frozenset(
            (q, r)
            for q in range(side)
            for r in range(side)
            if tile.terrain_at(view, {"q": q, "r": r})["terrain"] == "hill"
        )
        self.forests: frozenset[Hex] = frozenset(
            (q, r)
            for q in range(side)
            for r in range(side)
            if tile.terrain_at(view, {"q": q, "r": r})["feature"] == "forest"
        )
        for q in range(side):
            for r in range(side):
                ground = tile.terrain_at(view, {"q": q, "r": r})
                if ground["terrain"] in _IMPASSABLE:
                    continue
                cost = _TERRAIN_COST.get(ground["terrain"], 1) + _FEATURE_COST.get(ground["feature"], 0)
                self.entry_cost[(q, r)] = cost
                self._route_cost[(q, r)] = cost + _ROUTE_PENALTY.get(ground["feature"], 0)
        self._fields: dict[Hex, dict[Hex, int]] = {}

    def passable(self, position: Hex) -> bool:
        return position in self.entry_cost

    def field(self, goal: Hex, occupied: frozenset[Hex] = frozenset()) -> dict[Hex, int]:
        """Return the cheapest route cost from every reachable tile to ``goal``.

        Tiles in ``occupied`` cost extra, so routes flow around units standing in a narrow
        passage. Those fields depend on the moment, so only unoccupied ones are cached.
        """
        if not occupied:
            cached = self._fields.get(goal)
            if cached is not None:
                return cached
            if len(self._fields) >= self.CACHE_LIMIT:
                self._fields.clear()

        # An impassable goal, such as water, is approached from its passable neighbors.
        seeds = [goal] if self.passable(goal) else [n for n in _hex_neighbors(goal) if self.passable(n)]
        costs = {seed: 0 for seed in seeds}
        frontier = [(0, seed) for seed in seeds]
        while frontier:
            cost, position = heapq.heappop(frontier)
            if cost > costs[position]:
                continue
            # Searching backward from the goal: a neighbor reaches here by entering this tile.
            through = cost + self._route_cost[position]
            if position in occupied:
                through += self.OCCUPIED_PENALTY
            for neighbor in _hex_neighbors(position):
                if self.passable(neighbor) and through < costs.get(neighbor, _UNREACHABLE):
                    costs[neighbor] = through
                    heapq.heappush(frontier, (through, neighbor))

        if not occupied:
            self._fields[goal] = costs
        return costs

    def path_cost(self, start: Hex, path_id: int) -> int:
        """Return the movement points a path spends, entering each tile along it."""
        q, r = start
        total = 0
        for digit in paths.decode(path_id):
            dq, dr = tile.DIRECTIONS[digit]
            q, r = q + dq, r + dr
            total += self.entry_cost.get((q, r), _UNREACHABLE)
        return total


# --------------------------------------------------------------------------------------------
# Movement primitives
# --------------------------------------------------------------------------------------------


class Navigator:
    """Path choices for one activation. Every path it returns comes from the legal mask."""

    def __init__(self, observation: SkirmishObservation, terrain: TerrainMap) -> None:
        self.observation = observation
        self.terrain = terrain
        self.here = me.position(observation)
        self._occupied = frozenset(_hex(unit["position"]) for unit in observation["observation"]["visible_units"])

    def field(self, goal: Hex) -> dict[Hex, int]:
        """Return route costs to goal that steer around the units this unit can see."""
        return self.terrain.field(goal, self._occupied - {goal})

    def path_has_bad_terrain(self, path_id: int) -> bool:
        """True when the path enters water, marsh, or waste terrain."""
        position = self.here
        for digit in paths.decode(path_id):
            dq, dr = tile.DIRECTIONS[digit]
            position = {"q": position["q"] + dq, "r": position["r"] + dr}
            terrain = tile.terrain_at(self.observation, position)
            if terrain["terrain"] in _BAD_TERRAIN | _WATER or terrain["feature"] in _BAD_TERRAIN | _WATER:
                return True
        return False

    def best_path(self, key, max_cost: int | None = None) -> int:
        """Return the legal path whose landing tile has the lowest ``key``, or 0 to stay.

        Standing still wins ties, and shorter paths come first in the mask, so a unit never walks
        further than the gain needs. ``max_cost`` limits the movement points the path may spend.
        """
        here = _hex(self.here)
        best_path, best_key = 0, key(here)
        for path_id in action.legal_paths(self.observation):
            if path_id == 0:
                continue
            if max_cost is not None and self.terrain.path_cost(here, path_id) > max_cost:
                continue
            candidate = key(_hex(tile.at_path_end(self.here, path_id)))
            if candidate < best_key:
                best_path, best_key = path_id, candidate
        return best_path

    def step_toward(self, goal: AxialPosition, max_cost: int | None = None) -> int:
        """Return the legal path that gets furthest along the real route to goal, or 0.

        The route comes from the whole field, so a goal across the river leads through the
        nearest passage instead of to the bank opposite it.
        """
        target = _hex(goal)
        field = self.field(target)
        return self.best_path(
            lambda landing: (field.get(landing, _UNREACHABLE), _hex_distance(landing, target)), max_cost
        )

    def travel_cost(self, start: AxialPosition, goal: AxialPosition) -> int:
        """Return the movement points the cheapest route from start to goal spends."""
        return self.terrain.field(_hex(goal)).get(_hex(start), _UNREACHABLE)

    def path_away(self, threat: AxialPosition) -> int:
        """Return the legal path that maximizes distance from a threat, avoiding bad terrain."""
        best_path = 0
        best_distance = tile.distance(self.here, threat)

        for path_id in action.legal_paths(self.observation):
            if path_id == 0 or self.path_has_bad_terrain(path_id):
                continue
            landing_distance = tile.distance(tile.at_path_end(self.here, path_id), threat)
            if landing_distance > best_distance:
                best_path, best_distance = path_id, landing_distance

        return best_path

    def flank_goal(self, enemy: AxialPosition) -> AxialPosition:
        """Choose the nearer of the two side tiles next to an enemy."""
        enemy_sides = tile.neighbors(enemy)
        side_tiles = (enemy_sides[1], enemy_sides[3])
        return min(side_tiles, key=lambda side: tile.distance(self.here, side))


# --------------------------------------------------------------------------------------------
# Army plan: the zone runner, the infantry's regroup and march, and the cavalry wings
# --------------------------------------------------------------------------------------------


class ArmyPlan:
    """Gives every unit a role and a tile for each round, built around the three capture zones.

    Every unit builds this from shared match knowledge only, so all allies agree on it without
    messaging. The plan lives in red's frame, where the enemy lies toward +q, and is mirrored
    through the field center for blue. ``zone_center`` and ``zone_tiles`` are the middle zone.

    * The home guard, the footman with the cheapest route to our home zone, holds it.
    * One flank footman per wing, the footman left with the cheapest route to that wing's
      passage, goes with the wing to the enemy's home zone.
    * The runner, the footman with the shortest route to the middle zone, heads for its center.
    * The other footmen and every archer form up at a rally point just short of the zone, the
      footmen packed into a zone-shaped hex and the archers in a line behind them, then take their
      posts together. The footmen take the zone's other
      tiles, with one spare footman in reserve behind it, and the archers take posts further back
      that stay within bow range of the zone.
    * The cavalry splits into a north and a south wing. Each regroups at a support post behind
      the middle zone on its side and helps win the middle, then escorts its flank footman
      through the outer river passage on its side to the enemy's home zone.

    Without a home zone on each side of the river, there are no home guard or flank footmen and
    every footman plays for the middle, while the wings swing in beside the middle zone instead.
    """

    FILE_SPACING = 2  # rows between neighboring archers: 2 leaves one open tile between them
    RANK_SPACING = 2  # columns from the footman block's back edge to the archers: 2 leaves one open
    RALLY_SEARCH = 6  # how far a rally point may shift from its spawn centroid to fit the group
    REGROUP_SLACK = 2  # extra rounds allowed for forming up, for terrain and traffic
    MARCH_STOP = 3  # columns short of the zone center where the infantry forms up
    RESERVE_DEPTH = 2  # columns behind the zone center where the spare footman waits
    ARCHER_BAND = (3, 5)  # nearest and furthest an archer post may be from the zone center
    ARCHER_GAP = 2  # least distance between two archer posts
    STAGE_DEPTH = 2  # columns short of the river where a flank footman enters its passage
    SUPPORT_DEPTH = 3  # columns behind the middle zone's center where a wing's support post lies
    SUPPORT_SPREAD = 6  # rows out from the middle zone's center toward the south wing's side
    FOOTMAN_DEPTH = 2  # the same two numbers for the flank footman's waiting tile, closer in
    FOOTMAN_SPREAD = 3
    STRIKE_REACH = 3  # tiles beyond the zone's far corners where the wings swing in
    SCOUT_AFTER = 10  # rounds past arrival, without contact, before archers and cavalry scout

    def __init__(self, observation: SkirmishObservation, terrain: TerrainMap) -> None:
        self.terrain = terrain
        self.side = observation["observation"]["battlefield"]["side"]
        self.extent = (self.side - 1) // 2
        self.is_red = me.side(observation) == "red"
        # The march moves at the pace of the slowest line troops, spending their full points.
        self.pace = min(units.STATS[kind].movement_points for kind in ("footman", "archer"))

        # Zones in red's frame: the middle one sits at the field center, our home zone lies furthest
        # back and the enemy's home zone furthest forward.
        red_zones = [
            (self._to_red(area["center"]), frozenset(self._to_red(spot) for spot in area["tiles"]))
            for area in observation["observation"]["battlefield"]["zones"]
        ]
        field_center = (self.extent, self.extent)
        middle = min(red_zones, key=lambda area: (_hex_distance(area[0], field_center), area[0]), default=None)
        if middle is not None:
            self.zone_center, zone_tiles = middle
        else:
            self.zone_center = field_center
            zone_tiles = {self.zone_center, *(n for n in _hex_neighbors(self.zone_center) if terrain.passable(n))}
        self.zone_tiles = frozenset(zone_tiles)
        self.zone_board = frozenset(_hex(self._to_board(*position)) for position in zone_tiles)
        others = [area for area in red_zones if area is not middle]
        home = min(others, key=lambda area: (area[0][0], area[0]), default=None)
        enemy = max(others, key=lambda area: (area[0][0], area[0]), default=None)
        # Home and away play needs one zone on each side of the river.
        self.three_zones = (
            home is not None
            and enemy is not None
            and home[0][0] < self.zone_center[0] < enemy[0][0]
        )
        self.centers: dict[str, Hex] = {"middle": self.zone_center}
        self.zones: dict[str, frozenset[Hex]] = {"middle": self.zone_board}
        if self.three_zones:
            for name, (center, tiles) in (("home", home), ("enemy", enemy)):
                self.centers[name] = center
                self.zones[name] = frozenset(_hex(self._to_board(*position)) for position in tiles)

        roster_order = [(entry["type"], int(entry["unit_id"].rsplit("_", 1)[1])) for entry in roster.allies(observation)]
        spawns = dict(zip(roster_order, self._spawns(len(roster_order))))
        own_key = (me.unit_type(observation), int(me.unit_id(observation).rsplit("_", 1)[1]))
        if spawns.get(own_key) != self._to_red(me.position(observation)):
            spawns = None  # the spawn rule changed, so plan without knowing where allies start
        self._spawn_tiles = spawns

        by_kind: dict[str, list[int]] = {}
        for kind, index in roster_order:
            by_kind.setdefault(kind, []).append(index)
        footmen = list(by_kind.get("footman", []))
        passages = self._passage_rows()
        self._passage = {0: passages[0], 1: passages[-1]}
        self._stage_anchor = {
            wing: self._onto_field(self.zone_center[0] - self.STAGE_DEPTH, row) for wing, row in self._passage.items()
        }

        def cheapest(goal: Hex) -> int:
            """The free footman with the cheapest route to goal, by roster order without spawns."""
            if spawns is None:
                return footmen[0]
            costs = terrain.field(goal)
            return min(footmen, key=lambda index: (costs.get(spawns[("footman", index)], _UNREACHABLE), index))

        # One footman holds the home zone and one walks each flank, while at least two stay for
        # the middle, so the plan only splits a roster large enough for it.
        self.home_guard: int | None = None
        self.flankers: dict[int, int] = {}  # wing -> its flank footman
        if self.three_zones and len(footmen) >= 5:
            self.home_guard = cheapest(self.centers["home"])
            footmen.remove(self.home_guard)
            for wing in (0, 1):
                self.flankers[wing] = cheapest(self._stage_anchor[wing])
                footmen.remove(self.flankers[wing])
        self.flank_wing = {index: wing for wing, index in self.flankers.items()}

        self.runner = cheapest(self.zone_center) if footmen else None
        self.infantry = {
            "footman": [index for index in footmen if index != self.runner],
            "archer": by_kind.get("archer", []),
        }

        self._plan_infantry(spawns)
        self._plan_posts()
        self._plan_wings(by_kind.get("cavalry", []), spawns)
        self._plan_home_order()

    # -- Lookups used by Context ----------------------------------------------------------------

    def is_runner(self, kind: str, index: int) -> bool:
        return kind == "footman" and index == self.runner

    def role(self, kind: str, index: int) -> str:
        """``home`` and ``flank`` for the footmen with those jobs, otherwise the unit type."""
        if kind == "footman":
            if index == self.home_guard:
                return "home"
            if index in self.flank_wing:
                return "flank"
        return kind

    def wing(self, index: int) -> int:
        return self._wings[index][0]

    def rider_slot(self, index: int) -> int:
        """This rider's place among its wing's riders, used to spread the escort out."""
        return self._wings[index][1]

    def zone_for(self, role: str) -> frozenset[Hex]:
        """The zone a role plays for, in board coordinates."""
        if role == "home" and self.three_zones:
            return self.zones["home"]
        if role in ("flank", "cavalry") and self.three_zones:
            return self.zones["enemy"]
        return self.zone_board

    def center(self, name: str) -> AxialPosition:
        return self._to_board(*self.centers[name])

    def post(self, kind: str, index: int) -> AxialPosition:
        """The support post tile where a flank footman or rider regroups and waits out the
        middle fight."""
        spot = self._flank_post[index] if kind == "footman" else self._wings[index][2]
        return self._to_board(*spot)

    def stage(self, wing: int) -> AxialPosition:
        """Our mouth of this wing's passage, where the flank footman's march goes through."""
        return self._to_board(*self._stage_anchor[wing])

    def flank_exit(self, wing: int) -> AxialPosition:
        """The tile just across the river at the far mouth of this wing's passage."""
        return self._to_board(*self._flank_exit[wing])

    def flank_target(self, wing: int) -> AxialPosition:
        """Where the flank ends: the enemy's home zone, or beside the middle zone without one."""
        return self._to_board(*self._flank_target[wing])

    def flank_anchor(self, wing: int, marched: int) -> AxialPosition:
        """Where this wing's flank footman should stand ``marched`` rounds after leaving its
        stage, had it kept the plan's pace. Before that, it stands at its support post."""
        if marched < 0:
            footman = self.flankers.get(wing)
            if footman is not None:
                return self.post("footman", footman)
        route, schedule = self._flank_route[wing], self._flank_schedule[wing]
        return self._to_board(*route[schedule[min(len(schedule) - 1, max(0, marched))]])

    def crossed(self, position: Hex) -> bool:
        """True for a tile beyond the river seam, on the enemy's half."""
        return self.depth(position) > self.zone_center[0]

    def home_rank(self, kind: str, index: int) -> int | None:
        """This unit's place in line to defend the home zone, nearest post first, or None."""
        order = self._home_order.get(kind, [])
        return order.index(index) if index in order else None

    def slot(
        self, kind: str, index: int, round_number: int, released: bool = False, advanced: bool = False
    ) -> AxialPosition:
        """Return where this unit belongs this round, in board coordinates.

        A flank footman or rider belongs at its support post; its march out is driven by its own
        blocks. An ``advanced`` footman or archer skips the rest of the regroup and heads for its
        post.
        """
        role = self.role(kind, index)
        if role in ("flank", "cavalry"):
            return self.post(kind, index)
        if role == "home":
            return self.center("home")
        if self.is_runner(kind, index):
            return self._to_board(*self.zone_center)
        slot_index = self.infantry[kind].index(index)
        if advanced:
            position = self._posts[(kind, slot_index)]
        elif round_number <= self.ready_round:
            position = self._formation(self.infantry, self.rally)[(kind, slot_index)]
        elif round_number <= self.arrive_round:
            position = self._formation(self.infantry, self._anchor(round_number))[(kind, slot_index)]
        else:
            position = self._posts[(kind, slot_index)]
        return self._to_board(*position)

    def back_tiles(self) -> list[Hex]:
        """The zone's tiles nearest our own side, in board coordinates."""
        nearest = min(self.depth(cell) for cell in self.zone_board)
        return sorted(cell for cell in self.zone_board if self.depth(cell) == nearest)

    def rally_tiles(self, kind: str) -> frozenset[Hex]:
        """The board tiles one unit type's slots cover at the rally."""
        slots = self._formation(self.infantry, self.rally)
        return frozenset(
            _hex(self._to_board(*position)) for (slot_kind, _), position in slots.items() if slot_kind == kind
        )

    def depth(self, position: Hex) -> int:
        """How far a board tile lies toward the enemy: its column in red's frame."""
        return self._to_red({"q": position[0], "r": position[1]})[0]

    def on_our_side(self, position: Hex) -> bool:
        """True for a tile on our half of the river or on the seam itself."""
        return self.depth(position) <= self.zone_center[0]

    def scouting(self, round_number: int, last_contact: int | None) -> bool:
        """True once the infantry has held the zone for a while and this unit has seen nobody."""
        if round_number <= self.arrive_round + self.SCOUT_AFTER:
            return False
        return last_contact is None or round_number - last_contact > self.SCOUT_AFTER

    # -- Planning ---------------------------------------------------------------------------------

    def _plan_infantry(self, spawns: dict | None) -> None:
        """Choose the infantry's rally point, when it has formed, and its march to the zone."""
        members = self.infantry
        own = [spawns[(kind, index)] for kind, indices in members.items() for index in indices] if spawns else []
        # Form up just short of the zone, so the infantry can take its posts as soon as it is done.
        stop = self._onto_field(self.zone_center[0] - self.MARCH_STOP, self.zone_center[1])
        self.rally = self._choose_rally(members, stop)

        if own:
            slowest = 0
            rally_slots = self._formation(members, self.rally)
            for kind, indices in members.items():
                speed = units.STATS[kind].movement_points
                for slot_index, index in enumerate(indices):
                    walk = _hex_distance(spawns[(kind, index)], rally_slots[(kind, slot_index)])
                    slowest = max(slowest, -(-walk // speed))
            self.ready_round = slowest + self.REGROUP_SLACK
        else:
            self.ready_round = 8

        self.route = self._route([self.rally, stop])
        self.schedule = self._march_schedule(self.route)
        self.arrive_round = self.ready_round + len(self.schedule) - 1

    def _plan_posts(self) -> None:
        """Give every infantry slot the post it takes once the march is over."""
        march_end = self._formation(self.infantry, self.route[-1])
        center_q, center_r = self.zone_center
        # The zone's far edge fills first, since that is where the enemy arrives.
        ring = sorted(
            self.zone_tiles - {self.zone_center},
            key=lambda position: (-position[0], abs(position[1] - center_r), position[1]),
        )
        reserve = self._settle(center_q - self.RESERVE_DEPTH, center_r, center_r, set(self.zone_tiles))
        archer_posts = self._archer_posts(set(self.zone_tiles) | {reserve}, len(self.infantry["archer"]))
        self._posts = {
            **self._assign("footman", [*ring, reserve], march_end),
            **self._assign("archer", archer_posts, march_end),
        }

    def _archer_posts(self, taken: set[Hex], count: int) -> list[Hex]:
        """Pick spaced-out tiles behind the zone and within bow range of it, hills first."""
        nearest, furthest = self.ARCHER_BAND
        candidates = sorted(
            (
                position
                for position in self.terrain.entry_cost
                if position[0] < self.zone_center[0] - 1
                and nearest <= _hex_distance(position, self.zone_center) <= furthest
                and position not in taken
            ),
            key=lambda position: (
                position not in self.terrain.hills,
                abs(_hex_distance(position, self.zone_center) - nearest - 1),
                abs(position[1] - self.zone_center[1]),
                position,
            ),
        )
        chosen: list[Hex] = []
        for position in candidates:
            if len(chosen) == count:
                break
            if all(_hex_distance(position, other) >= self.ARCHER_GAP for other in chosen):
                chosen.append(position)
        return chosen

    def _assign(self, kind: str, posts: list[Hex], march_end: dict[tuple[str, int], Hex]) -> dict[tuple[str, int], Hex]:
        """Hand each post, in order, to the nearest slot still free at the end of the march.

        A slot left without a post holds where the march ended.
        """
        free = list(range(len(self.infantry[kind])))
        assigned: dict[tuple[str, int], Hex] = {}
        for post in posts:
            if not free:
                break
            slot_index = min(free, key=lambda index: (_hex_distance(march_end[(kind, index)], post), index))
            free.remove(slot_index)
            assigned[(kind, slot_index)] = post
        for slot_index in free:
            assigned[(kind, slot_index)] = march_end[(kind, slot_index)]
        return assigned

    def _plan_wings(self, cavalry: list[int], spawns: dict | None) -> None:
        """Split the cavalry into two wings, each escorting one flank footman through its passage.

        Each wing first regroups at its support post, behind the middle infantry on its own side,
        the footman on the post's anchor tile and the riders packed beside it. The wing that
        spawns beside the middle passage therefore falls back from the river first, instead of
        charging through the middle. From the post the riders help win the middle fight. Once it
        is won, the footman walks to our mouth of its passage, through it, and on to the enemy's
        home zone, and that walk at its own pace is the plan's guess of where the footman is when
        no rider can see it.
        """
        if spawns is not None:
            ordered = sorted(cavalry, key=lambda index: (spawns[("cavalry", index)][1], index))
        else:
            ordered = sorted(cavalry, key=lambda index: (index % 2, index))
        half = (len(ordered) + 1) // 2
        center_q, center_r = self.zone_center
        reach = self.STRIKE_REACH
        fallback = {
            # Without an enemy home zone, swing in beside the middle zone's far corners instead.
            0: (center_q + 1 + reach, center_r - 1 - reach),
            1: (center_q + 1, center_r + reach),
        }
        self._wings: dict[int, tuple[int, int, Hex]] = {}  # rider -> (wing, slot, support post tile)
        self._flank_post: dict[int, Hex] = {}  # flank footman -> support post tile
        self._flank_exit: dict[int, Hex] = {}
        self._flank_target: dict[int, Hex] = {}
        self._flank_route: dict[int, list[Hex]] = {}
        self._flank_schedule: dict[int, list[int]] = {}
        self.flank_ready: dict[int, int] = {}
        self.stage_walk: dict[int, int] = {}  # rounds a flank footman needs from its post to its stage
        for wing, riders in enumerate((ordered[:half], ordered[half:])):
            footman = self.flankers.get(wing)

            # Both posts lie behind the middle zone, mirror images across the line from it straight
            # back to our side. The river runs along r, so mirroring an offset (dq, dr) north to
            # south gives (dq, -dq - dr), and the two posts sit equally far from the zone.
            def mirrored(back: int, out: int) -> Hex:
                offset = (-back, back - out) if wing == 0 else (-back, out)
                return self._onto_field(center_q + offset[0], center_r + offset[1])

            post = mirrored(self.SUPPORT_DEPTH, self.SUPPORT_SPREAD)
            post_tiles = self._cluster(post, max(1, len(riders)))
            if footman is not None:
                # The flank footman waits closer in, beside the middle zone, where it can see the
                # zone and help hold it.
                self._flank_post[footman] = mirrored(self.FOOTMAN_DEPTH, self.FOOTMAN_SPREAD)
            for slot_index, index in enumerate(riders):
                self._wings[index] = (wing, slot_index, post_tiles[slot_index % len(post_tiles)])

            self._flank_exit[wing] = self._onto_field(center_q + self.STAGE_DEPTH, self._passage[wing])
            self._flank_target[wing] = self.centers["enemy"] if self.three_zones else self._onto_field(*fallback[wing])

            # The opening regroup ends once the wing's slowest member could have reached the post.
            rounds = 0
            if spawns is not None:
                members = [("cavalry", index, self._wings[index][2]) for index in riders]
                if footman is not None:
                    members.append(("footman", footman, self._flank_post[footman]))
                for kind, index, spot in members:
                    cost = self.terrain.field(spot).get(spawns[(kind, index)], _UNREACHABLE)
                    rounds = max(rounds, -(-cost // units.STATS[kind].movement_points))
            else:
                rounds = 6
            self.flank_ready[wing] = rounds + self.REGROUP_SLACK

            start = self._flank_post[footman] if footman is not None else post
            stage = self._stage_anchor[wing]
            walk = self.terrain.field(stage).get(start, _UNREACHABLE)
            self.stage_walk[wing] = -(-walk // self.pace) if walk < _UNREACHABLE else 0
            route = self._route([stage, self._flank_exit[wing], self._flank_target[wing]])
            self._flank_route[wing] = route
            self._flank_schedule[wing] = self._march_schedule(route)

    def _plan_home_order(self) -> None:
        """Rank the archers and the middle footmen by how near their posts lie to the home zone.

        The runner never leaves the middle zone's center. Every unit ranks from the same plan, so
        all of them agree on who answers the home zone's alarm first.
        """
        self._home_order: dict[str, list[int]] = {}
        if not self.three_zones:
            return
        home = self.centers["home"]
        for kind in ("archer", "footman"):
            members = self.infantry.get(kind, [])
            self._home_order[kind] = sorted(
                members,
                key=lambda index: (_hex_distance(self._posts[(kind, members.index(index))], home), index),
            )

    def _cluster(self, anchor: Hex, count: int) -> list[Hex]:
        """Return the ``count`` passable tiles nearest ``anchor``, the anchor itself first."""
        return sorted(self.terrain.entry_cost, key=lambda position: (_hex_distance(position, anchor), position))[:count]

    def _route(self, waypoints: list[Hex]) -> list[Hex]:
        """Join straight hex lines between waypoints into one route, skipping impassable tiles."""
        route = [waypoints[0]]
        for start, end in zip(waypoints, waypoints[1:]):
            for position in _hex_line(start, end)[1:]:
                if position != route[-1] and self.terrain.passable(position):
                    route.append(position)
        return route

    def _choose_rally(self, members: dict[str, list[int]], centroid: Hex) -> Hex:
        """Return the tile nearest the centroid where the whole spaced-out formation fits."""
        candidates = sorted(self.terrain.entry_cost, key=lambda position: (_hex_distance(position, centroid), position))
        for anchor in candidates:
            if _hex_distance(anchor, centroid) > self.RALLY_SEARCH:
                break
            footprint = [
                self._ideal_slot(members, kind, slot_index, anchor)
                for kind, indices in members.items()
                for slot_index in range(len(indices))
            ]
            if len(set(footprint)) == len(footprint) and all(self.terrain.passable(position) for position in footprint):
                return anchor
        return centroid

    def _march_schedule(self, route: list[Hex]) -> list[int]:
        """Walk the route the way a line troop would, spending ``pace`` movement points a round.

        As in the rules, the first step of a round is always allowed and later steps need enough
        points left. Entry ``k`` is the route index the anchor reaches after ``k`` rounds.
        """
        schedule = [0]
        index = 0
        while index < len(route) - 1:
            points = self.pace
            moved = False
            while index < len(route) - 1:
                cost = self.terrain.entry_cost.get(route[index + 1], _UNREACHABLE)
                if moved and cost > points:
                    break
                points -= cost
                index += 1
                moved = True
                if points <= 0:
                    break
            schedule.append(index)
        return schedule

    def _anchor(self, round_number: int) -> Hex:
        marched = max(0, round_number - self.ready_round)
        return self.route[self.schedule[min(len(self.schedule) - 1, marched)]]

    def _formation(self, members: dict[str, list[int]], anchor: Hex) -> dict[tuple[str, int], Hex]:
        """Return every slot of the infantry formation around ``anchor``, keyed by (type, slot index).

        Slots are placed footmen first, then archers, and a slot never takes a tile an earlier slot
        already holds.
        """
        taken: set[Hex] = set()
        slots: dict[tuple[str, int], Hex] = {}
        for kind in ("footman", "archer"):
            for slot_index in range(len(members.get(kind, []))):
                q, r = self._ideal_slot(members, kind, slot_index, anchor)
                position = self._settle(q, r, anchor[1], taken)
                taken.add(position)
                slots[(kind, slot_index)] = position
        return slots

    def _ideal_slot(self, members: dict[str, list[int]], kind: str, slot_index: int, anchor: Hex) -> Hex:
        """Footmen pack into a tight hex around ``anchor``, the same shape as the zone, so they
        stand shoulder to shoulder and slide into it together. Archers line up behind them.
        """
        front, middle = anchor
        if kind == "footman":
            dq, dr = _CLUSTER[slot_index % len(_CLUSTER)]
            return front + dq, middle + dr
        footmen = len(members.get("footman", []))
        depth = max((_hex_distance((0, 0), offset) for offset in _CLUSTER[:footmen]), default=0)
        count = len(members.get(kind, []))
        row = middle + slot_index * self.FILE_SPACING - ((count - 1) * self.FILE_SPACING) // 2
        return front - depth - self.RANK_SPACING, row

    def _settle(self, q: int, r: int, toward_r: int, taken: set[Hex] = frozenset()) -> Hex:
        """Slide a slot toward the anchor row, then back toward home, until it is on free ground.

        That keeps slots on the field near an edge, and squeezes a formation through tight ground,
        queueing the rest single file behind it.
        """
        start = (q, r)
        for _ in range(4 * self.side):
            if self.terrain.passable((q, r)) and (q, r) not in taken:
                return q, r
            if r != toward_r:
                r += 1 if r < toward_r else -1
            elif q > 0:
                q -= 1
            else:
                break
        return start if self.terrain.passable(start) else (q, r)

    def _onto_field(self, q: int, r: int) -> Hex:
        """Slide a waypoint toward the middle row until it is on passable ground."""
        return self._settle(q, r, self.extent)

    def _spawns(self, count: int) -> list[Hex]:
        """Rebuild red's spawn tiles exactly as the battlefield generator lays them out."""
        left = [(q, r) for q in range(self.extent) for r in range(self.side) if self.terrain.passable((q, r))]
        if len(left) < count:
            return []
        return [left[index * len(left) // count] for index in range(count)]

    def _passage_rows(self) -> list[int]:
        """Return the middle row of each gap in the river seam, from north to south."""
        runs: list[list[int]] = []
        for r in range(self.side):
            if self.terrain.passable((self.extent, r)):
                if runs and runs[-1][-1] == r - 1:
                    runs[-1].append(r)
                else:
                    runs.append([r])
        if not runs:
            return [self.extent]
        return [run[len(run) // 2] for run in runs]

    def _to_red(self, position: AxialPosition) -> Hex:
        if self.is_red:
            return position["q"], position["r"]
        return self.side - 1 - position["q"], self.side - 1 - position["r"]

    def _to_board(self, q: int, r: int) -> AxialPosition:
        if self.is_red:
            return {"q": q, "r": r}
        return {"q": self.side - 1 - q, "r": self.side - 1 - r}


# --------------------------------------------------------------------------------------------
# Per-activation context
# --------------------------------------------------------------------------------------------


class Context:
    """Everything a behavior block needs, computed once per activation."""

    def __init__(
        self,
        observation: SkirmishObservation,
        plan: ArmyPlan,
        terrain: TerrainMap,
        last_contact: int | None,
        released: bool = False,
        reports: list[tuple[str, Hex]] | None = None,
        advanced: bool = False,
        guard: tuple[Hex, list[tuple[str, Hex]]] | None = None,
        senders: list[Hex] | None = None,
        responding: bool = False,
        flank_footman: Hex | None = None,
        launched: int | None = None,
        staged: bool = False,
    ) -> None:
        self.obs = observation
        self.here = me.position(observation)
        self.kind = me.unit_type(observation)
        self.index = int(me.unit_id(observation).rsplit("_", 1)[1])
        self.role = plan.role(self.kind, self.index)
        self.round = observation["observation"]["round"]
        self.enemies = visible.enemies(observation)
        self.allies = visible.allies(observation)
        self.nearest = (
            min(self.enemies, key=lambda enemy: tile.distance(self.here, enemy["position"])) if self.enemies else None
        )
        # The nearest enemy close enough to fight. Enemies seen farther off do not stop the march.
        self.threat = self.nearest if self.nearest is not None and self.nearest_distance <= ENGAGE_RADIUS else None
        self.nav = Navigator(observation, terrain)
        self.terrain = terrain
        self.plan = plan
        # The zone this unit plays for: home for the home guard, the enemy's home zone for the
        # flank footmen and the riders escorting them, the middle zone for everyone else.
        self.zone = plan.zone_for(self.role)
        self.occupied = frozenset(_hex(unit["position"]) for unit in observation["observation"]["visible_units"])
        self.runner = plan.is_runner(self.kind, self.index)
        if self.kind == "cavalry":
            self.wing = plan.wing(self.index)
        elif self.role == "flank":
            self.wing = plan.flank_wing[self.index]
        else:
            self.wing = 0
        # Whether this unit is answering the home zone's alarm, and the enemies it sees there.
        self.responding = responding
        self.home_enemies = (
            [
                enemy
                for enemy in self.enemies
                if tile.distance(enemy["position"], plan.center("home")) <= HOME_RADIUS
            ]
            if plan.three_zones
            else []
        )
        # Where a rider last knew its flank footman to be: in sight, or from the footman's message.
        self._flank_footman = flank_footman
        # The round this unit's flank set out, once the middle fight was won, or None before.
        self.launched = launched
        # Whether a launched flank footman has reached our mouth of its passage.
        self.staged = staged
        self.last_contact = last_contact
        # Whether this unit has seen an enemy, or been told of one, so far this match.
        self.released = released
        # Enemies allies of the same type reported recently, as (type, board tile). They may have
        # moved since.
        self.reports = reports or []
        # A footman or archer that has seen the regroup finish heads for its post at once.
        self.advanced = advanced
        # For the one footman sent to defend an archer: that archer's tile and its attackers.
        self.guard = guard
        # Where the allies who sent recent reports were standing when they sent them.
        self.senders = senders or []

    @property
    def nearest_distance(self) -> int | None:
        return tile.distance(self.here, self.nearest["position"]) if self.nearest is not None else None

    def arrived(self) -> bool:
        """True once this unit's group holds its zone.

        The middle group has arrived once its regroup is over. A rider has arrived once the flank
        footman it escorts stands at the enemy's home zone, and the home guard always has.
        """
        if self.role == "home":
            return True
        if self.role in ("flank", "cavalry"):
            if self.launched is None:
                return False
            return _hex_distance(_hex(self.anchor()), _hex(self.plan.flank_target(self.wing))) <= 2
        return self.advanced or self.round > self.plan.arrive_round

    def marched(self) -> int:
        """Rounds this wing's flank footman has spent walking from its stage, negative before."""
        if self.launched is None:
            return -1
        return self.round - self.launched - FLANK_GATHER - self.plan.stage_walk[self.wing]

    def anchor(self) -> AxialPosition:
        """Where this wing's flank footman is: in sight, reported, or else where the plan has it."""
        if self.role == "flank":
            return self.here
        footman = self.plan.flankers.get(self.wing)
        for ally in self.allies:
            if ally["type"] == "footman" and int(ally["unit_id"].rsplit("_", 1)[1]) == footman:
                return ally["position"]
        if self._flank_footman is not None:
            return _board(self._flank_footman)
        return self.plan.flank_anchor(self.wing, self.marched())

    def scouting(self) -> bool:
        return self.plan.scouting(self.round, self.last_contact)

    def slot(self) -> AxialPosition:
        return self.plan.slot(self.kind, self.index, self.round, self.released, self.advanced)

    def allied(self, kind: str) -> list[AxialPosition]:
        """Positions of the visible allies of one unit type."""
        return [ally["position"] for ally in self.allies if ally["type"] == kind]

    def near_zone(self, position: AxialPosition) -> bool:
        """True for a tile inside the zone or touching it."""
        spot = _hex(position)
        return any(_hex_distance(spot, cell) <= 1 for cell in self.zone)

    def priority_targets(self) -> list[dict]:
        """Visible enemies, most urgent first: at the zone, then the weakest, then the nearest."""
        return sorted(
            self.enemies,
            key=lambda enemy: (
                not self.near_zone(enemy["position"]),
                enemy["hit_points"],
                tile.distance(self.here, enemy["position"]),
            ),
        )

    def target_from(self, landing: AxialPosition) -> tuple[int, dict | None]:
        """Return the rank and the most urgent enemy this unit can strike from ``landing``."""
        attack_range = units.STATS[self.kind].attack_range
        for rank, enemy in enumerate(self.priority_targets()):
            if tile.distance(landing, enemy["position"]) <= attack_range:
                return rank, enemy
        return len(self.enemies), None

    def order(self, path_id: int, target: dict | None = None) -> SkirmishAction:
        """Build an order along ``path_id`` naming ``target``, or the nearest visible enemy."""
        target = target or self.nearest
        if target is None:
            return action.move(path_id) if path_id else action.stay()
        if path_id:
            return action.move(path_id, target["unit_id"], self.obs)
        return action.stay(target["unit_id"], self.obs)


def _board(position: Hex) -> AxialPosition:
    return {"q": position[0], "r": position[1]}


# --------------------------------------------------------------------------------------------
# Behavior blocks
# --------------------------------------------------------------------------------------------


class Behavior:
    """A reusable decision rule. ``propose`` returns an order, or None when it does not apply."""

    def propose(self, ctx: Context) -> SkirmishAction | None:
        raise NotImplementedError


class WhenThreatened(Behavior):
    """Gate a combat block: it acts only once an enemy comes within ENGAGE_RADIUS."""

    def __init__(self, block: Behavior) -> None:
        self.block = block

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.threat is None:
            return None
        return self.block.propose(ctx)


class FightBack(Behavior):
    """Footman off the zone: close on an enemy within ``radius`` and strike it.

    The short leash keeps a marching footman from being picked apart without answering, while
    never drawing it far from its formation or its post.
    """

    def __init__(self, radius: int = 2) -> None:
        self.radius = radius

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.nearest is None or ctx.nearest_distance > self.radius:
            return None
        target = min(
            (enemy for enemy in ctx.enemies if tile.distance(ctx.here, enemy["position"]) <= self.radius),
            key=lambda enemy: (enemy["hit_points"], tile.distance(ctx.here, enemy["position"])),
        )
        goal = _hex(target["position"])
        footmen = [_hex(ally) for ally in ctx.allied("footman")]

        def key(landing: Hex) -> tuple[int, int]:
            # Next to the target first, then more adjacent footmen for the shield wall.
            buddies = sum(1 for ally in footmen if _hex_distance(ally, landing) == 1)
            return _hex_distance(landing, goal), -buddies

        return ctx.order(ctx.nav.best_path(key), target)


class KiteFromThreat(Behavior):
    """Back away from an enemy that is too close, unless the river keeps it from reaching us."""

    def __init__(self, radius: int = 2) -> None:
        self.radius = radius

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.nearest is None or ctx.nearest_distance > self.radius:
            return None
        # An enemy that needs more than two activations to walk around to us is no danger yet.
        reach = 2 * units.STATS[ctx.nearest["type"]].movement_points
        if ctx.nav.travel_cost(ctx.nearest["position"], ctx.here) > reach:
            return None
        return ctx.order(ctx.nav.path_away(ctx.nearest["position"]))


class HoldZone(Behavior):
    """Footman: once standing in the zone, never step off it. Strike from where it stands.

    The runner holds the zone center from the start. Everyone else holds whichever zone tile they
    reach once the march is over. The river seam walls the zone in above and below, so its far
    tiles can only be reached through its near ones: a holder steps forward onto an empty deeper
    zone tile beside it, so the zone fills from the front and those behind flow in after.
    With no enemy beside it, a holder moves to an empty zone tile that has one, so it can strike
    without ever leaving the zone. An enemy inside the zone is struck first, then the weakest one.
    """

    def propose(self, ctx: Context) -> SkirmishAction | None:
        here = _hex(ctx.here)
        if here not in ctx.zone:
            return None
        if not ctx.arrived():
            if not ctx.runner:
                return None
            # Before the others arrive, the runner keeps walking until it reaches a free center.
            center = _hex(ctx.slot())
            if here != center and center not in ctx.occupied:
                return None
        else:
            forward = [
                step
                for step in action.legal_steps(ctx.obs)
                if (landing := _hex(tile.at_path_end(ctx.here, step))) in ctx.zone
                and ctx.plan.depth(landing) > ctx.plan.depth(here)
            ]
            if forward:
                step = max(forward, key=lambda step: (ctx.plan.depth(_hex(tile.at_path_end(ctx.here, step))), -step))
                return ctx.order(step, ctx.target_from(tile.at_path_end(ctx.here, step))[1])
        adjacent = [enemy for enemy in ctx.enemies if tile.distance(ctx.here, enemy["position"]) == 1]
        if adjacent:
            return ctx.order(0, self._pick(ctx, adjacent))
        shift = self._shift(ctx)
        if shift is not None:
            return shift
        return ctx.order(0)

    @staticmethod
    def _pick(ctx: Context, adjacent: list[dict]) -> dict:
        return min(adjacent, key=lambda enemy: (_hex(enemy["position"]) not in ctx.zone, enemy["hit_points"]))

    def _shift(self, ctx: Context) -> SkirmishAction | None:
        """Move to an empty zone tile beside an enemy, or return None when there is none."""
        footmen = [_hex(ally) for ally in ctx.allied("footman")]
        best_path, best_key, best_target = None, None, None
        for path_id in action.legal_paths(ctx.obs):
            if path_id == 0:
                continue
            landing = tile.at_path_end(ctx.here, path_id)
            if _hex(landing) not in ctx.zone:
                continue
            adjacent = [enemy for enemy in ctx.enemies if tile.distance(landing, enemy["position"]) == 1]
            if not adjacent:
                continue
            target = self._pick(ctx, adjacent)
            buddies = sum(1 for ally in footmen if _hex_distance(ally, _hex(landing)) == 1)
            # An enemy inside the zone first, then the weakest, then the shield wall, then the
            # shortest walk.
            candidate = (
                _hex(target["position"]) in ctx.zone,
                -target["hit_points"],
                buddies,
                -len(paths.decode(path_id)),
            )
            if best_key is None or candidate > best_key:
                best_path, best_key, best_target = path_id, candidate, target
        if best_path is None:
            return None
        return ctx.order(best_path, best_target)


class DefendArcher(Behavior):
    """Footman sent from the zone: attack the enemies an archer reported attacking it.

    Enemies in sight near that archer come first. Otherwise the footman walks toward where the
    attackers were reported, or toward the archer itself.
    """

    def __init__(self, radius: int = 4) -> None:
        self.radius = radius

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.guard is None:
            return None
        archer, attackers = ctx.guard
        near = [enemy for enemy in ctx.enemies if _hex_distance(_hex(enemy["position"]), archer) <= self.radius]
        if near:
            target = min(near, key=lambda enemy: (tile.distance(ctx.here, enemy["position"]), enemy["hit_points"]))
            goal = _hex(target["position"])
            field = ctx.nav.field(goal)

            def key(landing: Hex) -> tuple[int, int]:
                return (0 if _hex_distance(landing, goal) == 1 else 1), field.get(landing, _UNREACHABLE)

            return ctx.order(ctx.nav.best_path(key), target)
        here = _hex(ctx.here)
        goal = min(
            (position for _, position in attackers),
            key=lambda position: _hex_distance(here, position),
            default=archer,
        )
        return ctx.order(ctx.nav.step_toward(_board(goal)))


class FillZoneGap(Behavior):
    """Footman off the zone after arrival: step onto an empty zone tile, or strike at its edge.

    This is how the reserve replaces a fallen zone footman, and how anyone who loses a race for
    a tile finds another one.
    """

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if not ctx.arrived():
            return None
        here = _hex(ctx.here)
        sight = units.STATS[ctx.kind].vision
        # Only tiles in sight are known to be empty.
        gaps = [cell for cell in ctx.zone if cell not in ctx.occupied and _hex_distance(cell, here) <= sight]
        if gaps:
            gap = min(gaps, key=lambda cell: (ctx.nav.travel_cost(ctx.here, _board(cell)), cell))
            path = ctx.nav.step_toward(_board(gap))
            if path:
                return ctx.order(path)

        intruders = [enemy for enemy in ctx.enemies if ctx.near_zone(enemy["position"])]
        if not intruders:
            return None
        target = min(intruders, key=lambda enemy: (enemy["hit_points"], tile.distance(ctx.here, enemy["position"])))
        goal = _hex(target["position"])

        def key(landing: Hex) -> tuple[int, int]:
            # Only fight from the zone's edge, so nobody gets drawn away from it.
            if not ctx.near_zone(_board(landing)):
                return 1, _UNREACHABLE
            return (0 if _hex_distance(landing, goal) == 1 else 1), _hex_distance(landing, goal)

        return ctx.order(ctx.nav.best_path(key), target)


class ClaimEmptyZone(Behavior):
    """Archer or cavalry: once no ally stands in the zone, step in so it keeps scoring.

    The first unit in becomes the holder and the rest see it and stay at their posts. A holder
    leaves again once a footman is back in the zone beside it.
    """

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if not ctx.arrived():
            return None
        here = _hex(ctx.here)
        holders = [ally for ally in ctx.allies if _hex(ally["position"]) in ctx.zone]
        if here in ctx.zone:
            if any(holder["type"] == "footman" for holder in holders):
                return None
            return ctx.order(0, ctx.target_from(ctx.here)[1])
        if holders:
            return None
        sight = units.STATS[ctx.kind].vision
        if any(_hex_distance(cell, here) > sight for cell in ctx.zone):
            return None  # the zone is not fully in sight, so it may not be empty
        cells = [cell for cell in ctx.zone if cell not in ctx.occupied]
        if not cells:
            return None
        cell = min(cells, key=lambda cell: (ctx.nav.travel_cost(ctx.here, _board(cell)), cell))
        path = ctx.nav.step_toward(_board(cell))
        if not path:
            return None
        return ctx.order(path, ctx.target_from(tile.at_path_end(ctx.here, path))[1])


class TakeHighGround(Behavior):
    """Archer: with an enemy in bow range of a free hill nearby, climb it and shoot from there.

    Only hills on our side of the river count, so the footmen still screen the archer, and none
    within ``safe`` tiles of an enemy. An archer already on such a hill stays on it.
    ``KiteFromThreat`` runs first, so an enemy that gets too close still drives it off.
    """

    def __init__(self, search: int = 4, safe: int = 2) -> None:
        self.search = search
        self.safe = safe

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if not ctx.enemies:
            return None
        here = _hex(ctx.here)
        attack_range = units.STATS[ctx.kind].attack_range
        enemies = [_hex(enemy["position"]) for enemy in ctx.enemies]

        def usable(hill: Hex) -> bool:
            return (
                ctx.plan.on_our_side(hill)
                and (hill == here or hill not in ctx.occupied)
                and any(_hex_distance(hill, enemy) <= attack_range for enemy in enemies)
                and all(_hex_distance(hill, enemy) > self.safe for enemy in enemies)
            )

        if here in ctx.terrain.hills and usable(here):
            return ctx.order(0, ctx.target_from(ctx.here)[1])

        # A hill counts as close when the archer can reach it within two activations.
        reach = 2 * units.STATS[ctx.kind].movement_points
        options = []
        for hill in ctx.terrain.hills:
            if _hex_distance(hill, here) > self.search or not usable(hill):
                continue
            cost = ctx.terrain.field(hill).get(here, _UNREACHABLE)
            if cost <= reach:
                options.append((cost, hill))
        if not options:
            return None
        hill = _board(min(options)[1])
        path = ctx.nav.step_toward(hill)
        if not path:
            return None
        return ctx.order(path, ctx.target_from(tile.at_path_end(ctx.here, path))[1])


class ShootFromBehindFootmen(Behavior):
    """Archer: shoot the most urgent enemy in reach from a tile screened by a footman."""

    def propose(self, ctx: Context) -> SkirmishAction | None:
        enemies = [enemy["position"] for enemy in ctx.enemies]
        footmen = ctx.allied("footman")

        best_path, best_key, best_target = None, None, None
        for path_id in action.legal_paths(ctx.obs):
            if path_id and ctx.nav.path_has_bad_terrain(path_id):
                continue
            landing = tile.at_path_end(ctx.here, path_id)
            if path_id and not ctx.plan.on_our_side(_hex(landing)):
                continue  # never cross the river ahead of the footmen
            rank, target = ctx.target_from(landing)
            if target is None:
                continue
            shot = tile.distance(landing, target["position"])
            closest_enemy = min(tile.distance(landing, enemy) for enemy in enemies)
            screened = any(
                tile.distance(landing, ally) <= 2 and tile.distance(ally, target["position"]) < shot for ally in footmen
            )
            # Screened and out of reach first, then the most urgent target, then as far from
            # enemies as the shot allows, then the shortest walk.
            candidate = (screened, closest_enemy > 2, -rank, closest_enemy, -len(paths.decode(path_id)))
            if best_key is None or candidate > best_key:
                best_path, best_key, best_target = path_id, candidate, target

        if best_path is None:
            return None  # nothing is in reach this turn, so keep moving with the group
        return ctx.order(best_path, best_target)


def _approach(ctx: Context, goal: AxialPosition, slack: int = 2) -> tuple[int, dict | None]:
    """Close on ``goal`` along the real route, striking on the way when that costs little.

    Among the landings at most ``slack`` route cost worse than the best approach, one beside an
    enemy wins, so the move still ends in a strike instead of wasting the turn. Returns the path
    and the enemy to name, or the plain best approach and None when no such landing exists.
    """
    field = ctx.nav.field(_hex(goal))
    approach = ctx.nav.step_toward(goal)
    best = field.get(_hex(tile.at_path_end(ctx.here, approach)), _UNREACHABLE)
    pick, pick_key, victim = None, None, None
    for path_id in action.legal_paths(ctx.obs):
        if path_id and ctx.nav.path_has_bad_terrain(path_id):
            continue
        landing = tile.at_path_end(ctx.here, path_id)
        cost = field.get(_hex(landing), _UNREACHABLE)
        if cost > best + slack:
            continue
        adjacent = [enemy for enemy in ctx.enemies if tile.distance(landing, enemy["position"]) == 1]
        if not adjacent:
            continue
        enemy = min(adjacent, key=lambda enemy: (enemy["type"] != "archer", enemy["hit_points"]))
        # Closest to the goal first, then a charge, then the weakest victim.
        candidate = (cost, tile.distance(ctx.here, landing) < 3, enemy["hit_points"])
        if pick_key is None or candidate < pick_key:
            pick, pick_key, victim = path_id, candidate, enemy
    if pick is None:
        return approach, None
    return pick, victim


def _strike(ctx: Context, target: dict) -> SkirmishAction:
    """Cavalry: land beside ``target`` and strike it, charging when the ride allows.

    Among landings beside the target, a charge comes first, then the fewest other enemies around.
    Out of reach this turn, the rider follows the real route toward the target's side, striking
    whoever stands beside the way so the turn is not wasted.
    """
    best_path = _landing_beside(ctx, target)
    if best_path is None:
        best_path, victim = _approach(ctx, ctx.nav.flank_goal(target["position"]))
        return ctx.order(best_path, victim or target)
    return ctx.order(best_path, target)


def _landing_beside(ctx: Context, target: dict) -> int | None:
    """The legal path that lands beside ``target``, a charge and fewer enemies around first."""
    goal = target["position"]
    best_path, best_key = None, None
    for path_id in action.legal_paths(ctx.obs):
        if path_id and ctx.nav.path_has_bad_terrain(path_id):
            continue
        landing = tile.at_path_end(ctx.here, path_id)
        if tile.distance(landing, goal) != 1:
            continue
        # A walk ending 3 or more tiles from the start earns the charge bonus.
        charging = tile.distance(ctx.here, landing) >= 3
        crowding = sum(
            1 for enemy in ctx.enemies if enemy is not target and tile.distance(landing, enemy["position"]) <= 1
        )
        candidate = (charging, -crowding)
        if best_key is None or candidate > best_key:
            best_path, best_key = path_id, candidate
    return best_path


def _guarded(ctx: Context, enemy: dict, guards: int = 2) -> bool:
    """True when ``guards`` or more enemy footmen or cavalry stand beside this enemy."""
    return (
        sum(
            1
            for other in ctx.enemies
            if other["type"] != "archer" and other is not enemy and tile.distance(other["position"], enemy["position"]) <= 1
        )
        >= guards
    )


class Regroup(Behavior):
    """Cavalry, at the start of the match: ride to its support post behind the middle infantry.

    On the way the rider never chases and never crosses the river, so the wing that spawns beside
    the middle passage falls back and forms up instead of charging through the middle. It still
    punishes an enemy that has crossed onto our half of the river, when it can land beside that
    enemy this very turn, so a raider diving on our archers does not go unanswered. Once at its
    post it holds there, fighting only enemies that come within ``guard`` tiles.
    """

    def __init__(self, guard: int = 2) -> None:
        self.guard = guard

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.round > ctx.plan.flank_ready[ctx.wing] or ctx.launched is not None:
            return None
        seam = ctx.plan.zone_center[0]
        raiders = sorted(
            (enemy for enemy in ctx.enemies if ctx.plan.depth(_hex(enemy["position"])) < seam),
            key=lambda enemy: (enemy["type"] != "cavalry", enemy["hit_points"], enemy["unit_id"]),
        )
        for raider in raiders:
            path = _landing_beside(ctx, raider)
            # Strike from our own half only, so the counter never carries the rider across.
            if path is not None and ctx.plan.depth(_hex(tile.at_path_end(ctx.here, path))) < seam:
                return ctx.order(path, raider)
        post = ctx.plan.post(ctx.kind, ctx.index)
        if ctx.here == post:
            near = [enemy for enemy in ctx.enemies if tile.distance(ctx.here, enemy["position"]) <= self.guard]
            if near:
                return _strike(ctx, min(near, key=lambda enemy: (enemy["hit_points"], enemy["unit_id"])))
            return ctx.order(0)
        path = ctx.nav.step_toward(post)
        return ctx.order(path, ctx.target_from(tile.at_path_end(ctx.here, path))[1])


class SupportMiddle(Behavior):
    """Cavalry held back from its flank: help win the middle fight from the support post.

    Until the middle is won, a rider strikes any enemy within SUPPORT_RIDES rides that stands on
    our half of the river, at the middle zone, or no more than SORTIE_DEPTH columns past the
    river: enemy archers first, unless two fighters guard them, then the weakest. It never rides
    deeper into the enemy's half than that. With nothing to strike, it rides back to its post.
    """

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.launched is not None:
            return None
        reach = SUPPORT_RIDES * units.STATS[ctx.kind].movement_points
        limit = ctx.plan.zone_center[0] + SORTIE_DEPTH
        options = []
        for enemy in ctx.enemies:
            spot = _hex(enemy["position"])
            if ctx.plan.depth(spot) > limit:
                continue
            if enemy["type"] == "archer" and _guarded(ctx, enemy):
                continue
            cost = ctx.nav.travel_cost(ctx.here, enemy["position"])
            if cost <= reach:
                options.append((enemy["type"] != "archer", enemy["hit_points"], cost, enemy["unit_id"], enemy))
        if options:
            return _strike(ctx, min(options)[-1])
        post = ctx.plan.post(ctx.kind, ctx.index)
        if ctx.here == post:
            return ctx.order(0)
        path = ctx.nav.step_toward(post)
        return ctx.order(path, ctx.target_from(tile.at_path_end(ctx.here, path))[1])


class Escort(Behavior):
    """Cavalry: guard this wing's flank footman on its walk to the enemy's home zone, and there.

    While the footman is still walking on our side of the river, the riders do not wait for it:
    they ride ahead and take the far mouth of its passage, so the way is clear when it gets there.
    Once it is across, they stay with it. An enemy archer within ESCORT_ARCHER_REACH of the footman
    or of the passage the riders hold comes first, unless two fighters guard it, then the weakest
    enemy within ESCORT_THREAT_REACH of either or right beside the rider. With nobody to strike,
    the riders spread out two tiles from the footman on the side it is heading, or two tiles around
    the passage mouth, so they meet whoever comes first. At the zone they face the field center,
    where the rest of the enemy army comes from.
    """

    def __init__(self, spread: int = 2) -> None:
        self.spread = spread

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.launched is None:
            return None
        anchor = ctx.anchor()
        ahead = not ctx.arrived() and not ctx.plan.crossed(_hex(anchor))
        # What the riders guard: the footman, and while it is on our side the passage ahead too.
        guarded = [anchor, ctx.plan.flank_exit(ctx.wing)] if ahead else [anchor]

        def near(enemy: dict, reach: int) -> bool:
            return any(tile.distance(enemy["position"], point) <= reach for point in guarded)

        archers = [
            enemy
            for enemy in ctx.enemies
            if enemy["type"] == "archer" and near(enemy, ESCORT_ARCHER_REACH) and not _guarded(ctx, enemy)
        ]
        if archers:
            target = min(
                archers, key=lambda enemy: (tile.distance(ctx.here, enemy["position"]), enemy["hit_points"])
            )
            return _strike(ctx, target)
        threats = [
            enemy
            for enemy in ctx.enemies
            if near(enemy, ESCORT_THREAT_REACH) or tile.distance(enemy["position"], ctx.here) <= 1
        ]
        if threats:
            target = min(
                threats,
                key=lambda enemy: (enemy["hit_points"], tile.distance(enemy["position"], anchor), enemy["unit_id"]),
            )
            return _strike(ctx, target)

        if ahead:
            exit_ = _hex(ctx.plan.flank_exit(ctx.wing))
            spots = self._escort_tiles(ctx, exit_, _hex(ctx.plan.flank_target(ctx.wing)))
        else:
            spots = self._escort_tiles(ctx, _hex(anchor), self._heading(ctx, anchor))
        if not spots:
            return ctx.order(ctx.nav.step_toward(anchor))
        spot = _board(spots[ctx.plan.rider_slot(ctx.index) % len(spots)])
        if ctx.here == spot:
            return ctx.order(0)
        return ctx.order(ctx.nav.step_toward(spot))

    @staticmethod
    def _heading(ctx: Context, anchor: AxialPosition) -> Hex:
        """Where the footman goes next: the far mouth of its passage, the zone, or, there, the
        field center the enemy comes from."""
        if ctx.arrived():
            return _hex(tile.at_center(ctx.obs))
        if not ctx.plan.crossed(_hex(anchor)):
            return _hex(ctx.plan.flank_exit(ctx.wing))
        return _hex(ctx.plan.flank_target(ctx.wing))

    def _escort_tiles(self, ctx: Context, anchor: Hex, heading: Hex) -> list[Hex]:
        """Passable tiles ``spread`` from the footman, nearest its heading first, kept apart."""
        ring = sorted(
            (
                position
                for position in ctx.terrain.entry_cost
                if _hex_distance(position, anchor) == self.spread
            ),
            key=lambda position: (_hex_distance(position, heading), position),
        )
        chosen: list[Hex] = []
        for position in ring:
            if all(_hex_distance(position, other) >= 2 for other in chosen):
                chosen.append(position)
        return chosen


class HoldLoneZone(HoldZone):
    """Footman holding a zone alone: once standing in its zone, never step off it.

    It strikes an enemy beside it, an enemy inside the zone first. Otherwise it moves to an empty
    zone tile beside an enemy, so it can strike without leaving. Otherwise it stands still.
    """

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if _hex(ctx.here) not in ctx.zone:
            return None
        adjacent = [enemy for enemy in ctx.enemies if tile.distance(ctx.here, enemy["position"]) == 1]
        if adjacent:
            return ctx.order(0, self._pick(ctx, adjacent))
        shift = self._shift(ctx)
        if shift is not None:
            return shift
        return ctx.order(0, ctx.target_from(ctx.here)[1])


def _walk_to_zone(ctx: Context, zone: frozenset[Hex], center: AxialPosition) -> SkirmishAction:
    """Walk toward the cheapest free tile of ``zone`` in sight, or its center, striking on the way."""
    here = _hex(ctx.here)
    sight = units.STATS[ctx.kind].vision
    free = [cell for cell in zone if cell not in ctx.occupied or cell == here]
    known = [cell for cell in free if _hex_distance(cell, here) <= sight]
    goal = (
        _board(min(known, key=lambda cell: (ctx.nav.travel_cost(ctx.here, _board(cell)), cell)))
        if known
        else center
    )
    path = ctx.nav.step_toward(goal)
    return ctx.order(path, ctx.target_from(tile.at_path_end(ctx.here, path))[1])


class MarchToZone(Behavior):
    """Footman: walk to this unit's own zone, onto a free tile of it."""

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if _hex(ctx.here) in ctx.zone:
            return None
        return _walk_to_zone(ctx, ctx.zone, _board(min(ctx.zone, key=lambda cell: _hex_distance(cell, _hex(ctx.here)))))


class FlankMarch(Behavior):
    """Flank footman: wait out the middle fight at the support post, then march on the enemy zone.

    Until the middle is won it walks to its post and waits there. Once it is won, the footman waits
    FLANK_GATHER more rounds so its riders gather, then walks to our mouth of its passage, on to
    the far mouth, so the route goes around the river through that passage and not the middle
    one, and once across the river it heads for a free tile of the enemy's home zone.
    """

    def propose(self, ctx: Context) -> SkirmishAction | None:
        plan = ctx.plan
        if ctx.launched is None or ctx.round < ctx.launched + FLANK_GATHER:
            goal = plan.post(ctx.kind, ctx.index)
            if ctx.here == goal:
                return ctx.order(0, ctx.target_from(ctx.here)[1])
        elif not ctx.staged and not plan.crossed(_hex(ctx.here)):
            goal = plan.stage(ctx.wing)
        elif not plan.crossed(_hex(ctx.here)):
            goal = plan.flank_exit(ctx.wing)
        else:
            return _walk_to_zone(ctx, ctx.zone, plan.flank_target(ctx.wing))
        path = ctx.nav.step_toward(goal)
        return ctx.order(path, ctx.target_from(tile.at_path_end(ctx.here, path))[1])


class DefendHome(Behavior):
    """Archer or middle footman answering the home zone's alarm: go back and drive the enemy off.

    An archer shoots the weakest attacker it can reach from a tile three or more tiles from every
    enemy. A footman closes on the weakest attacker and strikes it, and steps into the home zone
    when it finds the zone empty. With no attacker in sight, both head for the home zone.
    """

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if not ctx.responding:
            return None
        home = ctx.plan.zones["home"]
        center = ctx.plan.center("home")
        attackers = ctx.home_enemies
        if ctx.kind == "archer":
            order = self._shoot(ctx, attackers)
            if order is not None:
                return order
        elif attackers:
            target = min(
                attackers, key=lambda enemy: (enemy["hit_points"], tile.distance(ctx.here, enemy["position"]))
            )
            path, victim = _approach(ctx, target["position"])
            return ctx.order(path, victim or target)
        else:
            here = _hex(ctx.here)
            if here in home:
                return ctx.order(0, ctx.target_from(ctx.here)[1])
            sight = units.STATS[ctx.kind].vision
            in_sight = all(_hex_distance(cell, here) <= sight for cell in home)
            if in_sight and not any(_hex(ally["position"]) in home for ally in ctx.allies):
                return _walk_to_zone(ctx, home, center)  # the home guard is gone: hold it ourselves
        # With nothing to fight, wait beside the home zone until the alarm is over.
        if tile.distance(ctx.here, center) <= 2:
            return ctx.order(0, ctx.target_from(ctx.here)[1])
        path = ctx.nav.step_toward(center)
        return ctx.order(path, ctx.target_from(tile.at_path_end(ctx.here, path))[1])

    @staticmethod
    def _shoot(ctx: Context, attackers: list[dict]) -> SkirmishAction | None:
        if not attackers:
            return None
        attack_range = units.STATS[ctx.kind].attack_range
        enemies = [enemy["position"] for enemy in ctx.enemies]
        best_path, best_key, best_target = None, None, None
        for path_id in action.legal_paths(ctx.obs):
            if path_id and ctx.nav.path_has_bad_terrain(path_id):
                continue
            landing = tile.at_path_end(ctx.here, path_id)
            reachable = [enemy for enemy in attackers if tile.distance(landing, enemy["position"]) <= attack_range]
            if not reachable:
                continue
            target = min(reachable, key=lambda enemy: (enemy["hit_points"], enemy["unit_id"]))
            closest = min(tile.distance(landing, enemy) for enemy in enemies)
            # Out of reach of every enemy first, then the weakest target, then the shortest walk.
            candidate = (closest >= 3, -target["hit_points"], min(closest, 4), -len(paths.decode(path_id)))
            if best_key is None or candidate > best_key:
                best_path, best_key, best_target = path_id, candidate, target
        if best_path is None:
            return None
        return ctx.order(best_path, best_target)


class AnswerSignal(Behavior):
    """Go to a fight an ally of the same type reported, though the enemy is out of our own sight.

    Only reports within ``rides`` activations of walking count, so units far from the fight keep
    their own post. A reported archer comes first. A report of a tile this unit can see with no
    enemy on it is stale and ignored. With ``zone_only`` set, only enemies at the zone count, so a
    footman defends the zone instead of chasing.
    """

    def __init__(self, rides: int = 3, archers_only: bool = False, zone_only: bool = False) -> None:
        self.rides = rides
        self.archers_only = archers_only
        self.zone_only = zone_only

    def propose(self, ctx: Context) -> SkirmishAction | None:
        reach = self.rides * units.STATS[ctx.kind].movement_points
        sight = units.STATS[ctx.kind].vision
        seen = {_hex(enemy["position"]) for enemy in ctx.enemies}
        options = []
        for kind, position in ctx.reports:
            if self.archers_only and kind != "archer":
                continue
            if self.zone_only and not ctx.near_zone(_board(position)):
                continue
            if position not in seen and _hex_distance(_hex(ctx.here), position) <= sight:
                continue
            cost = ctx.nav.travel_cost(ctx.here, _board(position))
            if cost <= reach:
                options.append((kind != "archer", cost, position))
        if not options:
            return None
        goal = min(options)[2]
        path, victim = _approach(ctx, ctx.nav.flank_goal(_board(goal)))
        return ctx.order(path, victim) if path or victim else None


class JoinFight(Behavior):
    """Archer: walk toward a fight another archer reported, however far, until it is in bow range.

    The archer never crosses the river or steps into the zone. It heads for the tile on our side
    that has the reported enemy in bow range, or, when the enemy stands too far across the river
    for any such tile, the tile on our side that comes closest. A reported tile the archer can see
    with nobody on it is stale, so the archer heads for the archers who sent the reports instead,
    since the fight is around them.
    """

    def __init__(self, beside: int = 2) -> None:
        self.beside = beside  # how close to a reporting archer is close enough

    def propose(self, ctx: Context) -> SkirmishAction | None:
        here = _hex(ctx.here)
        sight = units.STATS[ctx.kind].vision
        seen = {_hex(enemy["position"]) for enemy in ctx.enemies}
        live = [
            position
            for _, position in ctx.reports
            if position in seen or _hex_distance(here, position) > sight
        ]
        if live:
            reach, targets = units.STATS[ctx.kind].attack_range, live
        elif ctx.senders:
            reach, targets = self.beside, ctx.senders
        else:
            return None

        # Route cost of walking from a tile to here, which is close to the cost of walking there.
        walk = ctx.terrain.field(here)

        def key(position: Hex) -> tuple[int, int]:
            short = min(max(0, _hex_distance(position, target) - reach) for target in targets)
            return short, walk.get(position, _UNREACHABLE)

        spots = [
            position
            for position in ctx.terrain.entry_cost
            if ctx.plan.on_our_side(position)
            and position not in ctx.zone
            and (position == here or position not in ctx.occupied)
        ]
        best = min(spots, key=key, default=here)
        if key(best) >= key(here):
            return None  # already as close as our side of the river allows
        path = ctx.nav.step_toward(_board(best))
        return ctx.order(path) if path else None


class ChaseNearest(Behavior):
    """Close on the nearest visible enemy along the real route, around the river if needed."""

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.nearest is None:
            return None
        return ctx.order(ctx.nav.step_toward(ctx.nearest["position"]))


class KeepFormation(Behavior):
    """Walk to this unit's tile in the plan for this round, or hold it once there.

    With ``scout`` set, the block steps aside once the plan says to scout, so a unit can go
    looking for enemies that never came to the zone.
    """

    def __init__(self, scout: bool = False) -> None:
        self.scout = scout

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if self.scout and ctx.scouting():
            return None
        slot = ctx.slot()
        if ctx.here == slot:
            return ctx.order(0)
        return ctx.order(ctx.nav.step_toward(slot))


class SweepField(Behavior):
    """With no enemy in sight, tour the field's six outer regions in turn to flush out stragglers.

    Each wing starts on the opposite side of the tour, and every unit moves on to the next
    region after ``dwell`` rounds, so a unit hiding in a corner cannot stall the match.
    """

    def __init__(self, dwell: int = 12, inset: int = 3) -> None:
        self.dwell = dwell
        self.inset = inset

    def propose(self, ctx: Context) -> SkirmishAction | None:
        if ctx.enemies:
            return None
        center = tile.at_center(ctx.obs)
        reach = center["q"] - self.inset
        regions = [{"q": center["q"] + dq * reach, "r": center["r"] + dr * reach} for dq, dr in tile.DIRECTIONS.values()]
        goal = regions[(ctx.round // self.dwell + 3 * ctx.wing) % len(regions)]
        path = ctx.nav.step_toward(goal)
        return action.move(path) if path else None


class HoldPosition(Behavior):
    """Last resort: stay put and strike whatever is in range."""

    def propose(self, ctx: Context) -> SkirmishAction | None:
        return ctx.order(0)


# --------------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------------


class Agent:
    """Runs the first behavior block in this unit type's playbook that proposes an order."""

    # Blocks are stateless, so every unit can share these instances. KeepFormation still names
    # the nearest enemy, so units strike anything in range on the way to their tile.
    PLAYBOOK: dict[str, list[Behavior]] = {
        # The home guard walks to our home zone and never leaves it. When enemies come near it
        # calls for help from outside the playbook, in act.
        "home": [
            HoldLoneZone(),
            FightBack(1),
            MarchToZone(),
            HoldPosition(),
        ],
        # A flank footman regroups with its riders, walks through its passage to the enemy's home
        # zone, and holds it. It only fights what stands right beside it: the riders do the rest.
        "flank": [
            HoldLoneZone(),
            FightBack(2),
            FlankMarch(),
            HoldPosition(),
        ],
        # Footmen never chase: they hold the zone, fill its gaps, answer enemies right beside them
        # or reported at the zone, and otherwise keep the plan. The exceptions are the footman
        # sent from the back of the zone to defend an archer under attack, and the footmen sent
        # home when the home zone's alarm reports a large attack.
        "footman": [
            DefendHome(),
            DefendArcher(),
            HoldZone(),
            FillZoneGap(),
            FightBack(2),
            AnswerSignal(rides=2, zone_only=True),
            KeepFormation(),
            HoldPosition(),
        ],
        # The archers nearest home answer the home zone's alarm first, more of them as it grows.
        "archer": [
            DefendHome(),
            KiteFromThreat(2),
            ClaimEmptyZone(),
            TakeHighGround(),
            WhenThreatened(ShootFromBehindFootmen()),
            # An archer out of the fight walks toward one another archer reported, however far,
            # until it has the enemy in bow range, so no archer is left out of a battle.
            JoinFight(),
            # An archer that has seen nobody for a while goes looking, so a lone enemy picking
            # off the zone's footmen from out of sight cannot do it forever.
            KeepFormation(scout=True),
            ChaseNearest(),
            SweepField(),
            HoldPosition(),
        ],
        # Each cavalry wing falls back to its support post behind the middle infantry and forms up
        # there, ignoring the enemy, then helps win the middle fight from that post. Once the
        # middle is won it escorts its flank footman to the enemy's home zone and guards it there.
        # A rider steps into that zone itself whenever no ally holds it, so the zone keeps scoring
        # after the footman falls.
        "cavalry": [
            Regroup(),
            SupportMiddle(),
            ClaimEmptyZone(),
            Escort(),
            HoldPosition(),
        ],
    }

    def __init__(self) -> None:
        self.terrain: TerrainMap | None = None
        self.plan: ArmyPlan | None = None
        self.last_contact: int | None = None
        self.released = False
        self.reports: list[tuple[int, str, Hex]] = []  # (round seen, type, board tile)
        self.senders: list[tuple[int, Hex]] = []  # (round sent, where the reporting ally stood)
        self.round = 0
        self.outbox: dict[str, str] = {}  # player id -> text to send after this activation
        self.peers: list[str] = []
        self.footmen: list[str] = []
        self.advanced = False
        self.distress: list[tuple[int, Hex, list[tuple[str, Hex]]]] = []  # (round, archer, attackers)
        self.guard: tuple[int, Hex, list[tuple[str, Hex]]] | None = None  # the call this footman answers
        self.guard_heard: int | None = None  # last round another footman said it went to defend
        self.alarms: list[tuple[int, int]] = []  # (round, enemies the home guard saw near home)
        self.responding = False
        self.responding_since = 0  # last round this unit had reason to keep defending home
        self.flank_report: tuple[int, Hex] | None = None  # (round, tile) the flank footman reported
        self.escorts: list[str] = []  # flank footman: player ids of its wing's riders
        self.alarm_list: list[str] = []  # home guard: player ids that may answer its alarm
        self.flank_group: list[str] = []  # middle footman: player ids to tell the middle is won
        self.calm = 0  # middle footman: activations in a row holding a quiet middle zone
        self.won_heard: int | None = None  # flank footman or rider: round it heard the middle won
        self.launched: int | None = None  # flank footman or rider: round its flank set out
        self.staged = False  # flank footman: reached our mouth of its passage since setting out

    def reset(self, seed, observation) -> None:
        # The routing map and the plan depend only on constant match knowledge and the opening
        # position, so both are built once here, outside the decision clock.
        self.terrain = TerrainMap(observation)
        self.plan = ArmyPlan(observation, self.terrain)
        if self.TACTICS:
            _tactics_model()  # load numpy and the critic here, outside the decision clock
        self.last_contact = None
        self.released = False
        self.reports = []
        self.senders = []
        self.round = 0
        self.outbox = {}
        self.peers = self._peers(observation)
        self.footmen = self._of_type(observation, "footman")
        self.advanced = False
        self.distress = []
        self.guard = None
        self.guard_heard = None
        self._reset_roles(observation)

    def _reset_roles(self, observation: SkirmishObservation) -> None:
        """Forget the home alarm and the flank, and look up who this unit messages for them."""
        self.alarms = []
        self.responding = False
        self.responding_since = 0
        self.flank_report = None
        self.calm = 0
        self.won_heard = None
        self.launched = None
        self.staged = False
        plan = self.plan
        kind = me.unit_type(observation)
        index = int(me.unit_id(observation).rsplit("_", 1)[1])
        self._me = (kind, index)
        role = plan.role(kind, index)
        allies = [
            (entry["player"], entry["type"], int(entry["unit_id"].rsplit("_", 1)[1]))
            for entry in roster.allies(observation)
            if entry["unit_id"] != me.unit_id(observation)
        ]
        self.escorts = (
            [player for player, other, other_index in allies if other == "cavalry" and plan.wing(other_index) == plan.flank_wing[index]]
            if role == "flank"
            else []
        )
        self.alarm_list = (
            [
                player
                for player, other, other_index in allies
                if plan.home_rank(other, other_index) is not None
            ]
            if role == "home"
            else []
        )
        # Every middle footman, the runner included, and every flank unit judges whether the middle
        # fight is won, and tells all the flank units when it is.
        self.flank_group = (
            [player for player, other, other_index in allies if plan.role(other, other_index) in ("flank", "cavalry")]
            if role in ("footman", "flank", "cavalry")
            else []
        )

    def act(self, observation: SkirmishObservation) -> SkirmishAction:
        if self.plan is None:
            self.terrain = TerrainMap(observation)
            self.plan = ArmyPlan(observation, self.terrain)
            self._reset_roles(observation)

        self.round = observation["observation"]["round"]
        self.reports = [report for report in self.reports if self.round - report[0] <= self.REPORT_AGE]
        self.senders = [sender for sender in self.senders if self.round - sender[0] <= self.REPORT_AGE]
        enemies = visible.enemies(observation)
        kind = me.unit_type(observation)
        if not self.released and (enemies or self.reports):
            self.released = True
        if not self.advanced and kind in ("footman", "archer"):
            self.advanced = self._regrouped(observation, kind)
        self.outbox = {}
        if kind in self.SIGNALLERS and enemies:
            report = self._report(enemies, _hex(me.position(observation)))
            for player in self.peers:
                self.outbox[player] = report
        if kind == "archer":
            self._call_for_help(observation, enemies)
        if kind == "footman":
            self._answer_call(observation, enemies)
        index = int(me.unit_id(observation).rsplit("_", 1)[1])
        role = self.plan.role(kind, index)
        here = _hex(me.position(observation))
        if role == "home":
            self._raise_alarm(enemies)
        elif role == "flank":
            # Tell this wing's riders where we are, for when they lose sight of us.
            self._send(self.escorts, f"F{self.round}@{here[0]},{here[1]}")
        elif role == "footman":
            self._judge_middle(observation, enemies)
        if role in ("flank", "cavalry"):
            self._check_launch(observation, enemies, here)
        self._update_responding(kind, index, enemies)
        report = self.flank_report
        ctx = Context(
            observation,
            self.plan,
            self.terrain,
            self.last_contact,
            self.released,
            [(kind, position) for _, kind, position in self.reports],
            self.advanced,
            (self.guard[1], self.guard[2]) if self.guard is not None else None,
            [position for _, position in self.senders],
            self.responding,
            report[1] if report is not None and self.round - report[0] <= self.REPORT_AGE else None,
            self.launched,
            self.staged,
        )
        if ctx.threat is not None:
            self.last_contact = ctx.round
            ctx.last_contact = ctx.round

        for block in self.PLAYBOOK.get(role, [HoldPosition()]):
            order = block.propose(ctx)
            if order is not None:
                return self._refine(ctx, order)
        return action.stay()

    # -- Learned tactics ---------------------------------------------------------------------------

    TACTICS = True  # let the learned critic adjust fight orders, when tactics-model.json exists
    TACTICS_MARGIN = 0.5  # predicted gain, in hit points, an alternative needs to replace the order
    TACTICS_RADIUS = 2  # alternatives land at most this far from the playbook's landing tile
    # tuning/tactics.py sets this while collecting training data. Play never does.
    tactics_hook = None

    def _refine(self, ctx: Context, order: SkirmishAction) -> SkirmishAction:
        """Let the learned critic swap a fight order for a nearby one it expects to trade better.

        The playbook still decides where the unit is going. The critic only sees the landing tiles
        within TACTICS_RADIUS of the playbook's, inside the same zone and on the same side of the
        river, and keeps the playbook's order unless one of them is predicted to come out at least
        TACTICS_MARGIN hit points better before this unit acts again.
        """
        hook = Agent.tactics_hook
        model = _tactics_model() if self.TACTICS else None
        if ctx.threat is None or (model is None and hook is None):
            return order
        if hook is not None:
            # Collecting: let errors surface, since a broken label is worse than a lost one.
            candidates = tactical_candidates(ctx, order, self.TACTICS_RADIUS)
            if len(candidates) >= 2:
                hook(ctx, candidates)
            if model is None:
                return order
        try:
            if hook is None:
                candidates = tactical_candidates(ctx, order, self.TACTICS_RADIUS)
            if len(candidates) < 2:
                return order
            scores = model.score([candidate.features for candidate in candidates])
            best = max(range(len(candidates)), key=lambda index: scores[index])
            if scores[best] - scores[0] > self.TACTICS_MARGIN:
                return candidates[best].order
        except Exception:  # noqa: BLE001 - the critic is optional, the playbook's order never is
            pass
        return order

    def _regrouped(self, observation: SkirmishObservation, kind: str) -> bool:
        """True once this unit can tell the regroup is over and it should head for its post.

        A footman sees the whole tight footman block from inside it: once every block tile holds a
        footman, the regroup is done. Every unit also follows a footman already standing in the zone,
        since that one has advanced, and a footman answers a report of an enemy at the zone.
        """
        plan = self.plan
        footmen = [
            ally
            for ally in visible.allies(observation)
            if ally["type"] == "footman" and int(ally["unit_id"].rsplit("_", 1)[1]) != plan.runner
        ]
        if any(_hex(ally["position"]) in plan.zone_board for ally in footmen):
            return True
        if kind != "footman" or plan.is_runner(kind, int(me.unit_id(observation).rsplit("_", 1)[1])):
            return False
        if any(
            any(_hex_distance(position, cell) <= 1 for cell in plan.zone_board) for _, _, position in self.reports
        ):
            return True
        block = plan.rally_tiles("footman")
        filled = {_hex(ally["position"]) for ally in footmen} | {_hex(me.position(observation))}
        return block <= filled

    # -- Defending the archers ---------------------------------------------------------------------

    DISTRESS_RADIUS = 4  # an enemy this close to an archer, on our half of the river, is a breakthrough
    GUARD_HOLD = 3  # rounds a defender stays out after the last call, while no attacker is in sight
    MIN_HOLDERS = 3  # footmen that must stand in the zone before one of them may leave it

    def _call_for_help(self, observation: SkirmishObservation, enemies: list[dict]) -> None:
        """Archer: tell the footmen where it is and who is on it, when an enemy has broken through."""
        here = _hex(me.position(observation))
        center_depth = self.plan.zone_center[0]
        attackers = [
            enemy
            for enemy in enemies
            if _hex_distance(here, _hex(enemy["position"])) <= self.DISTRESS_RADIUS
            and self.plan.depth(_hex(enemy["position"])) < center_depth
        ]
        if not attackers:
            return
        listed = ";".join(
            f"{enemy['type'][0]}{enemy['position']['q']},{enemy['position']['r']}" for enemy in attackers[:8]
        )
        call = f"H{self.round}@{here[0]},{here[1]}|{listed}"
        for player in self.footmen:
            self.outbox[player] = call

    def _answer_call(self, observation: SkirmishObservation, enemies: list[dict]) -> None:
        """Footman: decide whether to go defend an archer that called, or come back from doing so."""
        self.distress = [call for call in self.distress if self.round - call[0] <= self.REPORT_AGE]
        if self.guard is not None:
            if self.distress:
                self.guard = max(self.distress)
            archer = self.guard[1]
            attacker_near = any(
                _hex_distance(_hex(enemy["position"]), archer) <= self.DISTRESS_RADIUS for enemy in enemies
            )
            if not attacker_near and self.round - self.guard[0] > self.GUARD_HOLD:
                self.guard = None  # the fight is over: back to the zone
            return
        if not self.distress:
            return
        if self.guard_heard is not None and self.round - self.guard_heard <= self.GUARD_HOLD:
            return  # another footman already went
        here = _hex(me.position(observation))
        zone = self.plan.zone_board
        holders = sum(
            1 for ally in visible.allies(observation) if ally["type"] == "footman" and _hex(ally["position"]) in zone
        )
        if here not in zone or holders + 1 < self.MIN_HOLDERS:
            return
        call = max(self.distress)
        # Only the footman on the back tile nearest the archer goes, so only one ever leaves.
        chosen = min(self.plan.back_tiles(), key=lambda cell: (_hex_distance(cell, call[1]), cell))
        if here != chosen:
            return
        self.guard = call
        notice = f"G{self.round}"
        for player in self.peers:
            self.outbox[player] = f"{self.outbox[player]} {notice}" if player in self.outbox else notice

    # -- Defending the home zone ---------------------------------------------------------------------

    HOME_HOLD = 3  # rounds a defender stays after the last alarm, while no attacker is in sight
    # How many of each type answer the alarm, by the most enemies the home guard reported: two
    # archers by default, then more archers and footmen as the attack grows.
    HOME_QUOTA = (
        (5, {"archer": 6, "footman": 2}),
        (3, {"archer": 4, "footman": 1}),
        (1, {"archer": 2, "footman": 0}),
    )

    def _send(self, players: list[str], part: str) -> None:
        """Queue one message part for each player, after anything already queued for them."""
        for player in players:
            self.outbox[player] = f"{self.outbox[player]} {part}" if player in self.outbox else part

    def _raise_alarm(self, enemies: list[dict]) -> None:
        """Home guard: tell the defenders how many enemies it sees near the home zone."""
        center = self.plan.center("home")
        near = [enemy for enemy in enemies if tile.distance(enemy["position"], center) <= HOME_RADIUS]
        if near:
            self._send(self.alarm_list, f"A{self.round}|{len(near)}")

    def _update_responding(self, kind: str, index: int, enemies: list[dict]) -> None:
        """Archer or middle footman: decide whether to go home to defend it, or come back."""
        self.alarms = [alarm for alarm in self.alarms if self.round - alarm[0] <= self.HOME_HOLD]
        rank = self.plan.home_rank(kind, index) if self.plan.three_zones else None
        if rank is None:
            self.responding = False
            return
        count = max((alarm[1] for alarm in self.alarms), default=0)
        quota = next((needed[kind] for least, needed in self.HOME_QUOTA if count >= least), 0)
        center = self.plan.center("home")
        attacker_seen = any(tile.distance(enemy["position"], center) <= HOME_RADIUS for enemy in enemies)
        if rank < quota or (self.responding and attacker_seen):
            self.responding = True
            self.responding_since = self.round
        elif self.responding and self.round - self.responding_since > self.HOME_HOLD:
            self.responding = False  # the attack is over: back to the post

    # -- Holding the flanks back until the middle is won -------------------------------------------

    CALM_RADIUS = 6  # no enemy this close to the middle zone's center means the fight there is over
    CALM_ROUNDS = 1  # activations in a row the middle must stay quiet before the flanks set out
    CALM_HOLDERS = 2  # our footmen that must stand in the middle zone for it to count as won
    LATEST_LAUNCH = 60  # the round the flanks set out anyway, if the middle never goes quiet

    def _judge_middle(self, observation: SkirmishObservation, enemies: list[dict]) -> None:
        """Middle footman: once the middle zone has been ours and quiet for a while, tell the
        flank groups to set out. It keeps saying so, since a footman may die before it is heard."""
        here = _hex(me.position(observation))
        zone = self.plan.zone_board
        if here not in zone:
            self.calm = 0
            return
        holders = 1 + sum(
            1 for ally in visible.allies(observation) if ally["type"] == "footman" and _hex(ally["position"]) in zone
        )
        center = self.plan.center("middle")
        quiet = not any(tile.distance(enemy["position"], center) <= self.CALM_RADIUS for enemy in enemies)
        self.calm = self.calm + 1 if holders >= self.CALM_HOLDERS and quiet else 0
        if self.calm >= self.CALM_ROUNDS:
            self._send(self.flank_group, f"W{self.round}")

    def _middle_clear(self, observation: SkirmishObservation, enemies: list[dict]) -> bool:
        """True when this unit can see our footmen holding the middle zone and no enemy near it.

        Enemies on our half of the river count too, so a raid behind the lines keeps the flanks
        home. This only covers what this unit can see, so it is a quick judgment, not proof.
        """
        zone = self.plan.zone_board
        here = _hex(me.position(observation))
        holding = here in zone or any(
            ally["type"] == "footman" and _hex(ally["position"]) in zone for ally in visible.allies(observation)
        )
        if not holding:
            return False
        center = self.plan.center("middle")
        seam = self.plan.zone_center[0]
        return not any(
            tile.distance(enemy["position"], center) <= self.CALM_RADIUS
            or self.plan.depth(_hex(enemy["position"])) < seam
            for enemy in enemies
        )

    def _check_launch(self, observation: SkirmishObservation, enemies: list[dict], here: Hex) -> None:
        """Flank footman or rider: set out as soon as the middle is won.

        A flank footman waits right beside the middle zone, so it sets out the moment it sees the
        middle clear itself, once the opening regroup is over, and tells both flank groups so
        nobody waits on a message from the zone. Riders wait further back, where a lull in what
        they can see does not mean the fight is over, so they set out on hearing the middle is won
        from a footman, or at the latest round.
        """
        if self.launched is None:
            kind, index = self._me
            wing = self.plan.wing(index) if kind == "cavalry" else self.plan.flank_wing[index]
            seen = (
                kind == "footman"
                and self.round > self.plan.flank_ready[wing]
                and self._middle_clear(observation, enemies)
            )
            if seen or self.won_heard is not None or self.round >= self.LATEST_LAUNCH:
                self.launched = self.round
                if seen:
                    self._send(self.flank_group, f"W{self.round}")
        if self.launched is not None and not self.staged and self.plan.role(*self._me) == "flank":
            wing = self.plan.flank_wing[self._me[1]]
            self.staged = _hex_distance(here, _hex(self.plan.stage(wing))) <= 1

    # Optional: a reinforcement-learning hook called after every step with that step's
    # transition. Its time counts against the timing and episode budget. The order argument is
    # what act returned. It is named order so it does not shadow the action helpers.
    #
    # def learn(self, observation, order: SkirmishAction, reward: float, terminated: bool) -> None:
    #     ...

    # -- Messaging: every unit reports the enemies it sees to allies of its own type -------------
    #
    # A report reads "14@9,12|a15,8;c16,9": in round 14 the sender, standing at (9,12), saw an
    # archer at (15,8) and a cavalry at (16,9), all in board coordinates. It reaches an ally at its next activation, after that ally has already
    # chosen its order, so it acts on the report one activation later still. A unit writes reports
    # only to allies of its own type, and never as a broadcast, since that reaches the enemy too.
    #
    # Two more messages break that rule on purpose. An archer under attack calls the footmen with
    # "H14@8,12|c9,13": in round 14 the archer at (8,12) has a cavalry on it at (9,13). The footman
    # that goes to help tells the other footmen "G14", so nobody else leaves the zone. One message
    # may carry several parts separated by spaces, since each ally gets one message per activation.

    SIGNALLERS = ("cavalry", "footman", "archer")  # unit types that send reports to their own kind

    REPORT_AGE = 2  # rounds a report stays worth acting on, since the enemy keeps moving
    REPORT_LIMIT = 12  # enemies per report, archers first, to stay inside the text limit

    def chat(self, inbox: list[dict]) -> list[dict] | None:
        for message in inbox:
            for part in message.get("text", "").split(" "):
                if part.startswith("H"):
                    call = self._parse_call(part)
                    if call is not None:
                        self.distress.append(call)
                elif part.startswith("G"):
                    try:
                        self.guard_heard = max(self.guard_heard or 0, int(part[1:]))
                    except ValueError:
                        pass
                elif part.startswith("A"):
                    try:
                        round_text, count = part[1:].split("|")
                        self.alarms.append((int(round_text), int(count)))
                    except ValueError:
                        pass
                elif part.startswith("W"):
                    try:
                        self.won_heard = max(self.won_heard or 0, int(part[1:]))
                    except ValueError:
                        pass
                elif part.startswith("F"):
                    sender = self._parse_sender(part[1:])
                    if sender is not None:
                        self.flank_report = sender
                else:
                    self.reports.extend(self._parse(part))
                    sender = self._parse_sender(part)
                    if sender is not None:
                        self.senders.append(sender)
        if not self.outbox:
            return None
        outbox, self.outbox = self.outbox, {}
        return [{"to": player, "text": text[:200]} for player, text in outbox.items()]

    @staticmethod
    def _parse_sender(text: str) -> tuple[int, Hex] | None:
        try:
            round_text, where = text.split("|", 1)[0].split("@")
            q, r = where.split(",")
            return int(round_text), (int(q), int(r))
        except ValueError:
            return None

    @classmethod
    def _parse_call(cls, part: str) -> tuple[int, Hex, list[tuple[str, Hex]]] | None:
        try:
            head, listed = part[1:].split("|", 1)
            round_text, where = head.split("@")
            q, r = where.split(",")
            attackers = [(kind, position) for _, kind, position in cls._parse(f"{round_text}|{listed}")]
            return int(round_text), (int(q), int(r)), attackers
        except ValueError:
            return None

    def _report(self, enemies: list[dict], here: Hex) -> str:
        ordered = sorted(enemies, key=lambda enemy: (enemy["type"] != "archer", enemy["hit_points"]))
        spotted = ";".join(
            f"{enemy['type'][0]}{enemy['position']['q']},{enemy['position']['r']}"
            for enemy in ordered[: self.REPORT_LIMIT]
        )
        return f"{self.round}@{here[0]},{here[1]}|{spotted}"

    @staticmethod
    def _parse(text: str) -> list[tuple[int, str, Hex]]:
        kinds = {"f": "footman", "a": "archer", "c": "cavalry"}
        try:
            round_text, spotted = text.split("|", 1)
            seen = int(round_text.split("@")[0])
            reports = []
            for entry in spotted.split(";"):
                q, r = entry[1:].split(",")
                reports.append((seen, kinds[entry[0]], (int(q), int(r))))
            return reports
        except (ValueError, KeyError, IndexError):
            return []  # not one of our reports

    @classmethod
    def _peers(cls, observation: SkirmishObservation) -> list[str]:
        """Player ids of every other ally of this unit's own type. Messages to the fallen are dropped."""
        return cls._of_type(observation, me.unit_type(observation))

    @staticmethod
    def _of_type(observation: SkirmishObservation, kind: str) -> list[str]:
        """Player ids of every other ally of one unit type."""
        own = me.unit_id(observation)
        return [
            entry["player"] for entry in roster.allies(observation) if entry["type"] == kind and entry["unit_id"] != own
        ]


# --------------------------------------------------------------------------------------------
# Learned tactics
# --------------------------------------------------------------------------------------------
#
# In a fight the playbook picks one order by fixed rules. The critic lists the landing tiles near
# that order's landing, describes each one with the features below, and a small neural network
# predicts how many hit points the local trade comes out ahead before this unit acts again. The
# network learned that from tuning/tactics.py, which labels real fight positions by playing each
# alternative forward in the forecast simulator. That search is far too slow for every activation,
# so the network stands in for it at play time, in about a millisecond.

TACTICS_VERSION = "tactics-v1"
TACTICS_FEATURES = (
    "is_playbook",  # 1 for the playbook's own order
    "off_playbook",  # tiles between this landing and the playbook's
    "moved",  # tiles between the start and this landing
    "charge",  # cavalry ending 3 or more tiles from its start
    "strikes",  # some enemy is in range from this landing
    "damage",  # damage the strike deals, by the rules
    "kills",  # the strike finishes its target
    "target_hp_after",
    "target_footman",
    "target_archer",
    "target_cavalry",
    "target_waiting",  # the target has not acted yet this round, so it can still answer
    "threat_waiting",  # damage enemies yet to act this round could deal here
    "threat_acted",  # damage enemies that already acted could deal here next round
    "melee_threats",  # footmen and cavalry that can reach this tile
    "archer_threats",  # archers that can shoot this tile
    "lethal",  # the threats add up to at least this unit's hit points
    "own_hp",
    "own_hp_share",
    "is_footman",
    "is_archer",
    "is_cavalry",
    "adjacent_enemies",
    "adjacent_allies",
    "shield_wall",  # a footman landing beside an allied footman
    "on_hill",
    "in_forest",
    "high_ground",  # striking from a hill onto lower ground, minus the reverse
    "allies_near",  # allies within 2 tiles
    "enemies_near",
    "ally_hp_near",  # allied hit points within 3 tiles
    "enemy_hp_near",
    "nearest_enemy",
    "nearest_melee",
    "closer",  # how much nearer the nearest enemy this landing is than the start
    "in_zone",
)


class Candidate:
    """One alternative order in a fight: where it lands, whom it strikes, and its features."""

    __slots__ = ("path", "landing", "target", "order", "features")

    def __init__(self, path: int, landing: Hex, target: dict | None, order: SkirmishAction, features: list[float]):
        self.path, self.landing, self.target, self.order, self.features = path, landing, target, order, features


class _FightView:
    """What every candidate in one activation shares: who is where, and the rules switches."""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.terrain = ctx.terrain
        self.kind = ctx.kind
        self.stats = units.STATS[ctx.kind]
        self.start = _hex(ctx.here)
        self.hp = me.hit_points(ctx.obs)
        self.abilities = bool(ctx.obs["observation"].get("parameters", {}).get("unit_abilities", 1))
        self.enemies = [(enemy, _hex(enemy["position"])) for enemy in ctx.enemies]
        self.allies = [(ally, _hex(ally["position"])) for ally in ctx.allies]
        self.enemy_footmen = [spot for enemy, spot in self.enemies if enemy["type"] == "footman"]
        self.ally_footmen = [spot for ally, spot in self.allies if ally["type"] == "footman"]
        self.zone_tiles = frozenset().union(*ctx.plan.zones.values()) if ctx.plan.zones else frozenset()
        self.start_nearest = min((_hex_distance(self.start, spot) for _, spot in self.enemies), default=10)

    def strike_damage(self, landing: Hex, enemy: dict, spot: Hex) -> int:
        """Damage this unit's strike from ``landing`` deals, by the rules in combat.py."""
        hills, forests = self.terrain.hills, self.terrain.forests
        total = self.stats.damage
        shielded = (
            self.abilities
            and enemy["type"] == "footman"
            and any(_hex_distance(guard, spot) == 1 for guard in self.enemy_footmen if guard != spot)
        )
        if (
            self.abilities
            and self.kind == "cavalry"
            and _hex_distance(self.start, landing) >= 3
            and spot not in forests
            and not shielded
        ):
            total += 2
        if landing in hills and spot not in hills:
            total += 1
        if landing not in hills and spot in hills:
            total -= 1
        if spot in forests and _hex_distance(landing, spot) > 1:
            total -= 1
        if shielded:
            total -= 1
        return max(1, total)

    def incoming(self, enemy: dict, spot: Hex, landing: Hex) -> int:
        """Damage one enemy could deal to this unit standing on ``landing``.

        An archer shoots from about where it stands. A melee attacker strikes from beside the
        tile, usually from lower ground when the tile is a hill, and a rider with room to ride
        charges unless forest or a shield wall stops it.
        """
        hills, forests = self.terrain.hills, self.terrain.forests
        stats = units.STATS[enemy["type"]]
        total = stats.damage
        shielded = self.abilities and self.kind == "footman" and any(
            _hex_distance(guard, landing) == 1 for guard in self.ally_footmen
        )
        if stats.attack_range > 1:
            if spot in hills and landing not in hills:
                total += 1
            if spot not in hills and landing in hills:
                total -= 1
            if landing in forests:
                total -= 1
        else:
            gap = _hex_distance(spot, landing)
            if self.abilities and enemy["type"] == "cavalry" and gap >= 2 and landing not in forests and not shielded:
                total += 2
            if landing in hills:
                total -= 1
        if shielded:
            total -= 1
        return max(1, total)

    def in_range(self, landing: Hex) -> list[tuple[dict, Hex]]:
        reach = self.stats.attack_range
        return [(enemy, spot) for enemy, spot in self.enemies if _hex_distance(landing, spot) <= reach]

    def effective_target(self, landing: Hex, named: dict | None) -> tuple[dict, Hex] | None:
        """Whom the strike hits: the named enemy when in range, else the nearest one in range."""
        reachable = self.in_range(landing)
        if not reachable:
            return None
        if named is not None:
            for enemy, spot in reachable:
                if enemy["unit_id"] == named["unit_id"]:
                    return enemy, spot
        return min(reachable, key=lambda pair: _hex_distance(landing, pair[1]))

    def best_target(self, landing: Hex) -> dict | None:
        """The enemy in range worth naming: a kill first, then the most damage, archers first."""
        best_key, best = None, None
        for enemy, spot in self.in_range(landing):
            damage = self.strike_damage(landing, enemy, spot)
            key = (damage >= enemy["hit_points"], damage, enemy["type"] == "archer", -enemy["hit_points"], enemy["unit_id"])
            if best_key is None or key > best_key:
                best_key, best = key, enemy
        return best

    def features(self, landing: Hex, named: dict | None, is_playbook: bool, playbook_end: Hex) -> list[float]:
        hills, forests = self.terrain.hills, self.terrain.forests
        struck = self.effective_target(landing, named)
        damage = kills = waiting = high_ground = 0
        target_hp_after = 0
        target_type = ""
        if struck is not None:
            enemy, spot = struck
            damage = self.strike_damage(landing, enemy, spot)
            kills = int(damage >= enemy["hit_points"])
            target_hp_after = max(0, enemy["hit_points"] - damage)
            target_type = enemy["type"]
            waiting = int(not enemy.get("has_acted", False))
            high_ground = int(landing in hills and spot not in hills) - int(landing not in hills and spot in hills)

        # What could hit this tile before this unit acts again. A target the strike kills cannot.
        threat_waiting = threat_acted = melee = archers = 0
        for enemy, spot in self.enemies:
            if kills and enemy is struck[0]:
                continue
            stats = units.STATS[enemy["type"]]
            if _hex_distance(spot, landing) > stats.movement_points + stats.attack_range:
                continue
            if stats.attack_range > 1:
                archers += 1
            else:
                melee += 1
            hurt = self.incoming(enemy, spot, landing)
            if enemy.get("has_acted", False):
                threat_acted += hurt
            else:
                threat_waiting += hurt

        nearest = min((_hex_distance(landing, spot) for _, spot in self.enemies), default=10)
        nearest_melee = min(
            (_hex_distance(landing, spot) for enemy, spot in self.enemies if enemy["type"] != "archer"), default=10
        )
        shield = self.kind == "footman" and any(_hex_distance(spot, landing) == 1 for spot in self.ally_footmen)
        return [
            float(is_playbook),
            _hex_distance(landing, playbook_end) / 2,
            _hex_distance(landing, self.start) / 4,
            float(self.abilities and self.kind == "cavalry" and _hex_distance(self.start, landing) >= 3),
            float(struck is not None),
            damage / 5,
            float(kills),
            target_hp_after / 12,
            float(target_type == "footman"),
            float(target_type == "archer"),
            float(target_type == "cavalry"),
            float(waiting),
            threat_waiting / 10,
            threat_acted / 10,
            melee / 4,
            archers / 4,
            float(threat_waiting + threat_acted >= self.hp),
            self.hp / 12,
            self.hp / self.stats.hit_points,
            float(self.kind == "footman"),
            float(self.kind == "archer"),
            float(self.kind == "cavalry"),
            sum(1 for _, spot in self.enemies if _hex_distance(spot, landing) == 1) / 3,
            sum(1 for _, spot in self.allies if _hex_distance(spot, landing) == 1) / 3,
            float(shield),
            float(landing in hills),
            float(landing in forests),
            float(high_ground),
            sum(1 for _, spot in self.allies if _hex_distance(spot, landing) <= 2) / 5,
            sum(1 for _, spot in self.enemies if _hex_distance(spot, landing) <= 2) / 5,
            sum(ally["hit_points"] for ally, spot in self.allies if _hex_distance(spot, landing) <= 3) / 30,
            sum(enemy["hit_points"] for enemy, spot in self.enemies if _hex_distance(spot, landing) <= 3) / 30,
            min(nearest, 10) / 6,
            min(nearest_melee, 10) / 6,
            (self.start_nearest - nearest) / 4,
            float(landing in self.zone_tiles),
        ]


def _named_enemy(ctx: Context, order: SkirmishAction) -> dict | None:
    """The visible enemy an order names, or None for no target."""
    slot = order.get("target", 0)
    if not slot:
        return None
    entries = roster.enemies(ctx.obs)
    if not 0 < slot <= len(entries):
        return None
    unit_id = entries[slot - 1]["unit_id"]
    return next((enemy for enemy in ctx.enemies if enemy["unit_id"] == unit_id), None)


def tactical_candidates(ctx: Context, order: SkirmishAction, radius: int) -> list[Candidate]:
    """The playbook's order first, then one order per nearby landing tile the critic may pick.

    Alternatives stay within ``radius`` of the playbook's landing, inside its zone when it lands in
    one, and on our side of the river when it stays there, so the critic changes how a unit fights
    but never where the army is going.
    """
    view = _FightView(ctx)
    playbook_end = _hex(tile.at_path_end(ctx.here, order["path"]))
    named = _named_enemy(ctx, order)
    candidates = [
        Candidate(order["path"], playbook_end, named, order, view.features(playbook_end, named, True, playbook_end))
    ]
    zone_lock = next((cells for cells in ctx.plan.zones.values() if playbook_end in cells), None)
    side_lock = ctx.plan.on_our_side(playbook_end)
    seen = {playbook_end}
    for path_id in action.legal_paths(ctx.obs):
        landing = _hex(tile.at_path_end(ctx.here, path_id))
        if landing in seen or _hex_distance(landing, playbook_end) > radius:
            continue
        if zone_lock is not None and landing not in zone_lock:
            continue
        if side_lock and not ctx.plan.on_our_side(landing):
            continue
        if path_id and ctx.nav.path_has_bad_terrain(path_id):
            continue
        seen.add(landing)  # the mask lists shorter paths first, so this is the shortest way there
        target = view.best_target(landing)
        candidates.append(
            Candidate(
                path_id, landing, target, ctx.order(path_id, target), view.features(landing, target, False, playbook_end)
            )
        )
    return candidates


class _TacticsModel:
    """A small fully connected network, run with numpy: features in, predicted gain out."""

    def __init__(self, data: dict) -> None:
        import numpy as np

        self._np = np
        self.mean = np.asarray(data["mean"], dtype=np.float64)
        self.scale = np.asarray(data["scale"], dtype=np.float64)
        self.layers = [
            (np.asarray(layer["weight"], dtype=np.float64), np.asarray(layer["bias"], dtype=np.float64))
            for layer in data["layers"]
        ]

    def score(self, rows: list[list[float]]) -> list[float]:
        np = self._np
        values = (np.asarray(rows, dtype=np.float64) - self.mean) / self.scale
        for index, (weight, bias) in enumerate(self.layers):
            values = values @ weight.T + bias
            if index < len(self.layers) - 1:
                values = np.maximum(values, 0.0)
        return values[:, 0].tolist()


_TACTICS_FILE = Path(__file__).with_name("tactics-model.json")
_tactics_cache: dict[str, _TacticsModel | None] = {}


def _tactics_model() -> _TacticsModel | None:
    """The trained critic, or None when tactics-model.json is missing or was built for other features."""
    if "model" not in _tactics_cache:
        model = None
        try:
            data = json.loads(_TACTICS_FILE.read_text(encoding="utf-8"))
            if data.get("version") == TACTICS_VERSION and tuple(data.get("features", ())) == TACTICS_FEATURES:
                model = _TacticsModel(data)
        except (OSError, ValueError, KeyError, TypeError, ImportError):
            model = None
        _tactics_cache["model"] = model
    return _tactics_cache["model"]


# --------------------------------------------------------------------------------------------
# Tuned parameters
# --------------------------------------------------------------------------------------------


def _playbook_blocks(role: str, name: str) -> list[Behavior]:
    """The blocks of one class in one role's playbook, looking inside WhenThreatened wrappers."""
    found = []
    for block in Agent.PLAYBOOK[role]:
        for candidate in (block, getattr(block, "block", None)):
            if candidate is not None and type(candidate).__name__ == name:
                found.append(candidate)
    return found


def apply_overrides(overrides: dict, strict: bool = False) -> list[str]:
    """Set tuned values on this module, and return the keys that could not be applied.

    A key names what to change:

    * ``ENGAGE_RADIUS``: a module constant.
    * ``ArmyPlan.MARCH_STOP`` or ``Agent.CALM_RADIUS``: a class attribute.
    * ``archer.KiteFromThreat.radius``: one argument of a block in one role's playbook.

    Lists become tuples, so ``ArmyPlan.ARCHER_BAND`` stays a pair. With ``strict`` set, an unknown
    key raises instead of being skipped, which catches typos while tuning.
    """
    skipped = []
    namespace = globals()
    for key, value in overrides.items():
        if isinstance(value, list):
            value = tuple(value)
        parts = key.split(".")
        try:
            if len(parts) == 1 and parts[0] in namespace and parts[0].isupper():
                namespace[parts[0]] = value
            elif len(parts) == 2 and isinstance(namespace.get(parts[0]), type) and hasattr(namespace[parts[0]], parts[1]):
                setattr(namespace[parts[0]], parts[1], value)
            elif len(parts) == 3 and parts[0] in Agent.PLAYBOOK:
                blocks = _playbook_blocks(parts[0], parts[1])
                if not blocks or not all(hasattr(block, parts[2]) for block in blocks):
                    raise KeyError(key)
                for block in blocks:
                    setattr(block, parts[2], value)
            else:
                raise KeyError(key)
        except KeyError:
            if strict:
                raise KeyError(f"apply_overrides: nothing in agent.py is named {key!r}") from None
            skipped.append(key)
    return skipped


# tuning/tune.py writes the values that won its matches here. A missing or broken file leaves the
# hand-picked values in place, so the agent always loads.
_TUNED = Path(__file__).with_name("tuned_params.json")
try:
    apply_overrides(json.loads(_TUNED.read_text(encoding="utf-8")).get("params", {}))
except (OSError, ValueError, AttributeError):
    pass
