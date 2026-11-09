"""Tests for the provided ``sandbox.crane`` helpers.

These pin the two guarantees the helpers rely on: that the path encoding matches the synced
rules engine exactly and never drifts (so the helpers stay a stable compatibility promise), and
that the readers agree with the raw observation and mask the environment produces while driving
real match states. A final check confirms that importing the helpers does not drag in the heavy
environment stack, which is why an agent may import them at module top without slowing down
loading.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest
from sandbox.crane import action, forecast, me, paths, roster, tile, units, visible, zone
from sandbox.env import META, make_env
from sandbox.env.skirmish_crane import engine, hexes
from sandbox.env.skirmish_crane import paths as engine_paths
from sandbox.harness.environment import resolve_parameters

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 5
STEP_BUDGET = 60
FIELD_SIDE = 15
# The smallest field extent the battlefield generator accepts (see FIELD_EXTENT_BOUNDS in
# environments/skirmish_crane/battlefield.py), used to keep the dedicated zone pass fast. With a
# single zone, capture_zones=1 always centers it at (extent, extent).
ZONE_FIELD_EXTENT = 5
ZONE_STEP_BUDGET = 40


def _position(q: int, r: int) -> dict:
    return {"q": q, "r": r}


def _unit(unit_id: str, q: int, r: int) -> dict:
    side, kind, _ = unit_id.split("_")
    return {
        "unit_id": unit_id,
        "side": side,
        "type": kind,
        "position": _position(q, r),
        "hit_points": 7,
    }


def _zone(center: dict, tiles: tuple[dict, ...]) -> dict:
    return {"center": center, "tiles": tiles}


def _engine_path_end(start: tuple[int, int], path_id: int) -> tuple[int, int]:
    """Walk ``path_id`` from ``start`` with the engine's own decoder and hex step.

    Used to steer the dedicated zone test below without going through the sandbox.crane helpers
    it exists to exercise, keeping the steering independent of the code under test.
    """
    position = start
    for digit in engine_paths.decode_path(path_id):
        position = hexes.neighbor(position, digit)
    return position


def _observation(
    own_id: str = "red_footman_0",
    enemy_ids: tuple[str, ...] = ("blue_footman_0", "blue_archer_0"),
    seen: tuple[dict, ...] = (),
    tiles: tuple[tuple[dict, ...], ...] = (),
    zones: tuple[dict, ...] = (),
) -> dict:
    own_side = own_id.split("_", 1)[0]
    enemy_side = "blue" if own_side == "red" else "red"
    return {
        "observation": {
            "self": {
                "unit_id": own_id,
                "type": own_id.split("_")[1],
                "position": _position(3, 4),
                "hit_points": 9,
                "movement_points": 2,
                "direction": 2 if own_side == "red" else 5,
            },
            "visible_units": seen,
            "battlefield": {"side": FIELD_SIDE, "tiles": tiles, "zones": zones},
            "rosters": {
                own_side: ({"unit_id": own_id, "side": own_side},),
                enemy_side: tuple({"unit_id": enemy_id, "side": enemy_side} for enemy_id in enemy_ids),
            },
        },
        "action_mask": {"path": [1, 0], "target": [1, 1, 1]},
    }


# -- the path encoding, frozen forever -----------------------------------------------------------


def test_encode_and_decode_pin_the_stable_vectors():
    assert paths.encode([]) == 0
    assert paths.encode([1]) == 1
    assert paths.encode([6]) == 6
    assert paths.encode([1, 1]) == 7
    assert paths.encode([6, 6, 6, 6]) == 1554

    assert paths.decode(0) == ()
    assert paths.decode(1) == (1,)
    assert paths.decode(6) == (6,)
    assert paths.decode(7) == (1, 1)
    assert paths.decode(1554) == (6, 6, 6, 6)


def test_full_range_round_trips_against_the_step_one_decoder():
    assert tile.DIRECTIONS == hexes.DIRECTIONS
    assert paths.MAX_ID == engine_paths.MAX_PATH_ID
    for path_id in range(paths.MAX_ID + 1):
        digits = paths.decode(path_id)
        assert digits == engine_paths.decode_path(path_id)
        assert paths.encode(digits) == path_id
        assert engine_paths.encode_path(digits) == path_id


@pytest.mark.parametrize(
    "directions",
    [(0,), (7,), (1, 1, 1, 1, 1), (True,), (1.0,)],
    ids=["digit-zero", "digit-seven", "five-steps", "bool-digit", "non-int-digit"],
)
def test_encode_rejects_invalid_digit_sequences(directions):
    with pytest.raises(ValueError):
        paths.encode(directions)


@pytest.mark.parametrize("path_id", [-1, 1555, True, "3"])
def test_decode_rejects_invalid_ids(path_id):
    with pytest.raises(ValueError):
        paths.decode(path_id)


def test_importing_the_helpers_stays_light():
    # An agent imports sandbox.crane at module top, so it must not pull in the environment engine.
    # Check in a fresh interpreter, since this test process has already loaded it.
    code = (
        "import sys; "
        "from sandbox.crane import action, forecast, me, paths, roster, tile, units, visible, zone; "
        "assert 'pettingzoo' not in sys.modules; "
        "assert 'gymnasium' not in sys.modules; "
        "assert 'numpy' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)


# -- hex geometry and the ground ------------------------------------------------------------------


def test_at_path_end_agrees_with_walking_the_neighbors_one_digit_at_a_time():
    start = _position(6, 5)
    assert tile.at_path_end(start, 0) == start
    for path_id in range(paths.MAX_ID + 1):
        walked = start
        for digit in paths.decode(path_id):
            walked = tile.neighbors(walked)[digit]
        assert tile.at_path_end(start, path_id) == walked


@pytest.mark.parametrize("path_id", [-1, 1555])
def test_at_path_end_rejects_invalid_ids(path_id):
    with pytest.raises(ValueError):
        tile.at_path_end(_position(0, 0), path_id)


def test_at_center_is_the_middle_of_the_field():
    assert tile.at_center(_observation()) == _position(7, 7)


def test_at_mirror_reflects_through_the_center_and_undoes_itself():
    observation = _observation()
    center = tile.at_center(observation)

    assert tile.at_mirror(center, observation) == center
    spawn = _position(2, 6)
    opposite = tile.at_mirror(spawn, observation)
    assert opposite == _position(12, 8)
    assert tile.at_mirror(opposite, observation) == spawn
    # The reflection is a rotation about the middle, so it never changes how far a tile sits.
    assert tile.distance(opposite, center) == tile.distance(spawn, center)


def test_terrain_at_reads_the_grid_row_first_and_calls_everything_outside_it_void():
    grass = {"terrain": "grass", "feature": "none"}
    marsh = {"terrain": "marsh", "feature": "none"}
    observation = _observation(tiles=((grass, grass), (grass, marsh)))

    assert tile.terrain_at(observation, _position(1, 1)) == marsh
    assert tile.terrain_at(observation, _position(0, 1)) == grass
    assert tile.terrain_at(observation, _position(9, 9)) == {"terrain": "void", "feature": "none"}
    assert tile.terrain_at(observation, _position(-1, 0)) == {"terrain": "void", "feature": "none"}


# -- reading your own unit, the rosters, and what is in sight --------------------------------------


def test_me_reads_every_field_of_your_own_unit():
    observation = _observation("blue_archer_1")

    assert me.unit_id(observation) == "blue_archer_1"
    assert me.side(observation) == "blue"
    assert me.unit_type(observation) == "archer"
    assert me.position(observation) == _position(3, 4)
    assert me.direction(observation) == 5
    assert me.hit_points(observation) == 9
    assert me.movement_points(observation) == 2


def test_rosters_are_picked_by_your_own_side():
    observation = _observation()

    assert [entry["unit_id"] for entry in roster.allies(observation)] == ["red_footman_0"]
    assert [entry["unit_id"] for entry in roster.enemies(observation)] == [
        "blue_footman_0",
        "blue_archer_0",
    ]


def test_visible_splits_what_the_unit_sees_and_keeps_the_observation_order():
    seen = (
        _unit("blue_footman_0", 4, 4),
        _unit("red_archer_0", 2, 4),
        _unit("blue_archer_0", 5, 4),
    )
    observation = _observation(seen=seen)

    assert [unit["unit_id"] for unit in visible.enemies(observation)] == [
        "blue_footman_0",
        "blue_archer_0",
    ]
    assert [unit["unit_id"] for unit in visible.allies(observation)] == ["red_archer_0"]


# -- unit stats and capture zones -------------------------------------------------------------------


def test_units_stats_is_the_table_the_engine_plays_by():
    # units.STATS and engine.UNIT_STATS come from separate composed copies of the same source
    # file (sandbox/unit_stats.py and sandbox/env/skirmish_crane/unit_stats.py), so they are
    # equal-valued UnitStats instances of two distinct classes, not the same object; compare
    # field values rather than relying on dataclass ``==``, which only matches within one class.
    assert set(units.STATS) == {"footman", "archer", "cavalry"}
    for kind, live in engine.UNIT_STATS.items():
        assert dataclasses.astuple(units.STATS[kind]) == dataclasses.astuple(live)


def test_zones_returns_the_battlefields_zones_and_empty_when_capture_is_off():
    center = _position(5, 5)
    one_zone = _zone(center, (center, *tile.neighbors(center).values()))

    assert zone.zones(_observation(zones=(one_zone,))) == (one_zone,)
    assert zone.zones(_observation()) == ()


def test_at_finds_the_zone_covering_a_tile_including_its_center():
    center = _position(5, 5)
    ring = tile.neighbors(center)
    one_zone = _zone(center, (center, *ring.values()))
    observation = _observation(zones=(one_zone,))

    assert zone.at(observation, center) == one_zone
    assert zone.at(observation, ring[1]) == one_zone
    assert zone.at(observation, _position(0, 0)) is None


def test_occupants_lists_your_own_unit_first_then_the_visible_units_standing_in_the_zone():
    own_position = _position(3, 4)  # matches _observation's own unit position
    ring = tile.neighbors(own_position)
    inside = ring[1]
    outside = _position(9, 9)
    one_zone = _zone(own_position, (own_position, inside))
    inside_enemy = _unit("blue_footman_0", inside["q"], inside["r"])
    outside_enemy = _unit("blue_archer_0", outside["q"], outside["r"])
    observation = _observation(seen=(inside_enemy, outside_enemy), zones=(one_zone,))

    occupants = zone.occupants(observation, one_zone)

    assert [unit["unit_id"] for unit in occupants] == ["red_footman_0", "blue_footman_0"]
    assert occupants[0] == {
        "unit_id": "red_footman_0",
        "side": "red",
        "type": "footman",
        "position": own_position,
        "hit_points": 9,
    }


def test_occupants_omits_your_own_unit_when_it_stands_outside_the_zone():
    center = _position(8, 8)
    one_zone = _zone(center, (center,))
    inside_enemy = _unit("blue_footman_0", 8, 8)
    observation = _observation(seen=(inside_enemy,), zones=(one_zone,))

    assert [unit["unit_id"] for unit in zone.occupants(observation, one_zone)] == ["blue_footman_0"]


# -- the mask and the order it builds ---------------------------------------------------------------


def test_forecast_helper_builds_an_independent_environment_ready_for_an_order():
    live = make_env(resolve_parameters(META))
    live.reset(seed=SEED)
    source = live.observe(live.agent_selection)
    predicted = forecast.from_observation(source)
    assert predicted.agent_selection == live.agent_selection
    assert me.unit_id(predicted.last()[0]) == me.unit_id(source)
    assert set(predicted.match.units) == {
        me.unit_id(source),
        *(unit["unit_id"] for unit in source["observation"]["visible_units"]),
    }
    predicted.step(action.stay())
    assert len(predicted.match.history) == 1
    assert live.match.history == []
    live.close()
    predicted.close()


def test_forecast_guide_agent_chooses_a_move_that_survives_the_archer_reply():
    guide = (REPO_ROOT / "environment.md").read_text(encoding="utf-8")
    section = guide.split("### Forecast a local future", 1)[1].split("\n## ", 1)[0]
    example = section.split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {}
    exec(compile(example, "environment.md forecast example", "exec"), namespace)
    agent = namespace["Agent"]()

    live = make_env(resolve_parameters(META))
    live.reset(seed=SEED)
    own = live.match.units["red_cavalry_0"]
    enemy = live.match.units["blue_archer_0"]
    own.position, own.hit_points = (7, 7), 2
    enemy.position = (13, 7)
    live.match.units = {own.unit_id: own, enemy.unit_id: enemy}
    live.match.activation_order = [own.unit_id, enemy.unit_id]
    live.match.activation_index = 0
    live.agent_selection = live.agent_by_unit[own.unit_id]
    observation = live.observe(live.agent_selection)
    agent.reset(SEED, observation)
    order = agent.act(observation)

    assert order["path"] in action.legal_steps(observation)
    destination = tile.at_path_end(me.position(observation), order["path"])
    assert tile.distance(destination, _position(*enemy.position)) > enemy.stats.attack_range
    assert live.match.history == []
    assert own.position == (7, 7)
    live.close()


def test_helper_accessors_agree_with_live_environment_states():
    parameters = resolve_parameters(META)
    extent = parameters["field_extent"]
    env = make_env(parameters)
    try:
        env.reset(seed=SEED)
        acted = 0
        while env.agents and acted < STEP_BUDGET:
            observation, _reward, termination, truncation, _info = env.last()
            if termination or truncation:
                env.step(None)
                continue
            agent = env.agent_selection
            state = observation["observation"]
            mask = observation["action_mask"]

            expected_paths = [path_id for path_id, bit in enumerate(mask["path"]) if bit]
            assert action.legal_paths(observation) == expected_paths
            assert action.legal_steps(observation) == [
                path_id for path_id in expected_paths if 1 <= path_id <= 6
            ]

            own_side = state["self"]["unit_id"].split("_", 1)[0]
            enemy_side = "blue" if own_side == "red" else "red"
            enemy_roster = state["rosters"][enemy_side]
            expected_targets = [
                entry["unit_id"] for index, entry in enumerate(enemy_roster) if mask["target"][index + 1]
            ]
            assert action.possible_targets(observation) == expected_targets

            # Independent oracle: the mask-based checks above restate legal_paths and
            # possible_targets almost verbatim, so they cannot catch a wrong assumption the helper
            # and the test happen to share. Ask the engine directly instead, bypassing the mask.
            unit_id = state["self"]["unit_id"]
            walkable_paths, nameable_unit_ids = env.unwrapped.match.legal_orders(unit_id)
            oracle_paths = {0, *(engine_paths.encode_path(walked) for walked in walkable_paths)}
            assert set(action.legal_paths(observation)) == oracle_paths
            assert set(action.possible_targets(observation)) == set(nameable_unit_ids)

            # The unit's own readings agree with the raw observation, and the two visible-unit
            # readers partition it. Every enemy in sight is one this unit may name, which is what
            # lets an agent name a target straight off visible.enemies.
            assert me.unit_id(observation) == unit_id
            assert me.side(observation) == own_side
            assert me.position(observation) == state["self"]["position"]
            assert me.direction(observation) == (2 if own_side == "red" else 5)
            assert roster.enemies(observation) == enemy_roster
            assert roster.allies(observation) == state["rosters"][own_side]
            partition = visible.enemies(observation) + visible.allies(observation)
            assert sorted(unit["unit_id"] for unit in partition) == sorted(
                unit["unit_id"] for unit in state["visible_units"]
            )
            assert {unit["unit_id"] for unit in visible.enemies(observation)} == set(expected_targets)

            assert tile.at_center(observation) == {"q": extent, "r": extent}
            here = me.position(observation)
            assert tile.terrain_at(observation, here) == state["battlefield"]["tiles"][here["r"]][here["q"]]

            # This pass always resolves capture_zones to 0 (see resolve_parameters above), so the
            # live battlefield never publishes a zone here; both readers must agree it is empty.
            # test_zone_helpers_agree_with_the_live_battlefield_when_capture_is_on, below, is the
            # dedicated pass that turns capture on and drives real zone occupancy.
            assert zone.zones(observation) == ()
            assert zone.at(observation, here) is None

            sampled_path = max(expected_paths)
            space = env.action_space(agent)
            assert space.contains(action.stay())
            assert space.contains(action.move(sampled_path))
            assert mask["path"][sampled_path] == 1
            assert mask["target"][0] == 1

            if expected_targets:
                name = expected_targets[0]
                slot = next(i for i, entry in enumerate(enemy_roster) if entry["unit_id"] == name)
                order = action.move(sampled_path, target_id=name, observation=observation)
                assert order["target"] == slot + 1

            env.step(action.move(sampled_path))
            acted += 1
        assert acted > 0
    finally:
        env.close()


def test_zone_helpers_agree_with_the_live_battlefield_when_capture_is_on():
    """A short, dedicated live pass that gives the zone helpers real coverage.

    The mask-parity test above always plays with capture off, so its zone assertions can only ever
    see the empty case, and a branch that reads a unit standing inside a zone would never run.
    Resolving with capture_zones=1 and the smallest allowed field_extent turns exactly one zone on,
    centered on the field (see the odd-count case in environments/skirmish_crane/battlefield.py's
    ``_zones``), and keeps the episode short. Every activation, whichever side it belongs to, walks
    the legal path that lands closest to the zone's center, so several units end up standing inside
    it well before the match resolves.
    """
    parameters = resolve_parameters(META, {"capture_zones": 1, "field_extent": ZONE_FIELD_EXTENT})
    extent = parameters["field_extent"]
    env = make_env(parameters)
    try:
        env.reset(seed=SEED)
        acted = 0
        occupant_hits = 0
        while env.agents and acted < ZONE_STEP_BUDGET:
            observation, _reward, termination, truncation, _info = env.last()
            if termination or truncation:
                env.step(None)
                continue
            state = observation["observation"]
            mask = observation["action_mask"]
            unit_id = state["self"]["unit_id"]
            here = state["self"]["position"]

            # Independent oracle: derive the covering zone straight from the raw battlefield,
            # rather than through the helper under test, the same approach the test above uses.
            zones_state = state["battlefield"]["zones"]
            assert zone.zones(observation) == zones_state
            assert len(zones_state) == 1  # capture_zones=1 always publishes exactly one zone
            one_zone = zones_state[0]
            inside = any(t["q"] == here["q"] and t["r"] == here["r"] for t in one_zone["tiles"])
            own_zone = one_zone if inside else None
            assert zone.at(observation, here) == own_zone

            # A tile far outside the single centered zone stays outside no matter which unit reads
            # it, checked against the raw tile list rather than by calling the helper twice.
            far_corner = {"q": 0, "r": extent}
            assert far_corner not in one_zone["tiles"]
            assert zone.at(observation, far_corner) is None

            if own_zone is not None:
                occupant_hits += 1
                assert unit_id in {unit["unit_id"] for unit in zone.occupants(observation, own_zone)}

            start = (here["q"], here["r"])
            target = (one_zone["center"]["q"], one_zone["center"]["r"])
            legal_paths = [path_id for path_id, bit in enumerate(mask["path"]) if bit]
            sampled_path = min(
                legal_paths, key=lambda path_id: hexes.distance(_engine_path_end(start, path_id), target)
            )
            env.step(action.move(sampled_path))
            acted += 1
        # The guard the rest of this test exists for: prove the occupants branch actually ran,
        # so this test can never again read as live zone coverage while asserting nothing.
        assert occupant_hits > 0
    finally:
        env.close()


def test_naming_a_target_without_an_observation_raises():
    with pytest.raises(ValueError):
        action.stay(target_id="blue_footman_0")
    with pytest.raises(ValueError):
        action.move(0, target_id="blue_footman_0")


def test_naming_an_unknown_unit_id_raises():
    observation = _observation()
    with pytest.raises(ValueError):
        action.move(0, target_id="blue_cavalry_9", observation=observation)


def test_naming_an_allied_unit_id_raises():
    observation = _observation()
    with pytest.raises(ValueError):
        action.move(0, target_id="red_footman_0", observation=observation)


def test_move_rejects_out_of_range_path_ids():
    with pytest.raises(ValueError):
        action.move(-1)
    with pytest.raises(ValueError):
        action.move(paths.MAX_ID + 1)
