"""PettingZoo interface for Skirmish at Crane Reach.

The rules engine owns battlefield generation, movement, combat, and scoring.  This module maps
that state into the stable object-shaped observation and composite action contract used by the
platform.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv

from .battlefield import CAPTURE_ZONES_BOUNDS, FIELD_EXTENT_BOUNDS
from .engine import COMPOSITIONS, Match, MatchConfig, Order
from .hexes import DIRECTIONS
from .paths import MAX_PATH_ID, decode_path, encode_path

if TYPE_CHECKING:
    from game_sandbox_harness.environment import ParameterValue

    from .observation_types import (
        ActionMask,
        AxialPosition,
        Battlefield,
        Capture,
        MatchParameters,
        RosterEntry,
        Rosters,
        SelfUnit,
        SkirmishAction,
        SkirmishObservation,
        SkirmishObservationData,
        Tile,
        VisibleUnit,
        Zone,
    )


_TEXT_CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789_"
_SIDES = ("red", "blue")
_KINDS = ("footman", "archer", "cavalry")
# The direction digit a side walks to head toward the enemy, published in every unit's
# observation. Red spawns on the low-q half of the field and blue on its point reflection, so
# red walks east and blue west, and neither has to work out which way it is facing.
FORWARD_DIRECTION = {"red": 2, "blue": 5}
# The battlefield generator owns the field bounds; these two have no engine-side check, so the
# factory enforces them. The metadata declarations and parameter spaces reuse all four pairs.
CAPTURE_TARGET_BOUNDS = (10, 10_000)
ROUND_CAP_BOUNDS = (100, 10_000)
SEAT_PLAN_SPECS = (
    ("skirmish", "Skirmish", ((0, 1, 2), (3, 4, 5))),
    ("army", "Army", (tuple(range(20)), tuple(range(20, 40)))),
)


class IllegalMoveError(ValueError):
    """Raised when a submitted action is outside the current published mask."""


def _parameter(parameters: Mapping[str, ParameterValue], name: str, expected: type) -> Any:
    if name not in parameters:
        raise ValueError(f"missing environment parameter {name!r}")
    value = parameters[name]
    if type(value) is not expected:
        raise ValueError(f"{name} must be a {expected.__name__}")
    return value


def _check_bounds(name: str, value: int, bounds: tuple[int, int]) -> None:
    low, high = bounds
    if not low <= value <= high:
        raise ValueError(f"{name} must be from {low} through {high}")


def make_env(parameters: Mapping[str, ParameterValue]) -> SkirmishCraneEnv:
    """Construct an environment from one complete, already resolved parameter map."""
    seat_plan = _parameter(parameters, "seat_plan", str)
    if seat_plan not in COMPOSITIONS:
        raise ValueError("seat_plan must be skirmish or army")
    config = MatchConfig(
        seat_plan=seat_plan,
        field_extent=_parameter(parameters, "field_extent", int),
        terrain=_parameter(parameters, "terrain", bool),
        wasteland=_parameter(parameters, "wasteland", bool),
        unit_abilities=_parameter(parameters, "unit_abilities", bool),
        capture_zones=_parameter(parameters, "capture_zones", int),
        capture_target=_parameter(parameters, "capture_target", int),
        round_cap=_parameter(parameters, "round_cap", int),
    )
    _check_bounds("capture_target", config.capture_target, CAPTURE_TARGET_BOUNDS)
    _check_bounds("round_cap", config.round_cap, ROUND_CAP_BOUNDS)
    return SkirmishCraneEnv(config)


def default_action(env: SkirmishCraneEnv, player_id: str) -> SkirmishAction:
    """Return the always-legal stand-still order used by a timed-out participant."""
    if player_id not in env.possible_agents:
        raise ValueError(f"unknown player {player_id!r}")
    return {"path": 0, "target": 0}


def _position_space(side: int) -> spaces.Dict:
    return spaces.Dict({"q": spaces.Discrete(side), "r": spaces.Discrete(side)})


def _bounded(bounds: tuple[int, int]) -> spaces.Discrete:
    low, high = bounds
    return spaces.Discrete(high - low + 1, start=low)


def _text(max_length: int) -> spaces.Text:
    return spaces.Text(max_length=max_length, min_length=1, charset=_TEXT_CHARSET)


class SkirmishCraneEnv(AECEnv):
    """Sequential tactical combat over the pure :class:`~.engine.Match` state."""

    metadata = {"name": "skirmish_crane_v0", "is_parallelizable": False}

    def __init__(self, config: MatchConfig) -> None:
        super().__init__()
        self.config = config
        count_per_side = sum(COMPOSITIONS[config.seat_plan].values())
        self.possible_agents = [f"player_{index}" for index in range(count_per_side * 2)]
        self.agents: list[str] = []
        self.match: Match
        self.last_activation = None
        self.last_capture_changes = {"red": 0, "blue": 0}
        self._last_observations: dict[str, SkirmishObservation] = {}
        self._battlefield_snapshot: Battlefield
        self._rosters_snapshot: Rosters
        self._parameters_snapshot: MatchParameters
        # Player order is part of the public contract. These maps keep the initial roster slots
        # stable even after units die and leave the engine's living-unit mapping.
        self.agent_by_unit = self._agent_mapping()
        self.unit_by_agent = {agent: unit_id for unit_id, agent in self.agent_by_unit.items()}
        self._enemy_roster = {
            agent: self._roster_ids("blue" if self._side_for_agent(agent) == "red" else "red")
            for agent in self.possible_agents
        }

        # Every space depends only on resolved construction parameters. Sharing the finished space
        # objects across players also prevents an episode from changing shape as units disappear.
        side = 2 * config.field_extent + 1
        position = _position_space(side)
        unit = spaces.Dict(
            {
                "unit_id": _text(16),
                "side": _text(4),
                "type": _text(7),
                "position": position,
                "hit_points": spaces.Discrete(13),
                "has_acted": spaces.Discrete(2),
            }
        )
        self_unit = spaces.Dict(
            {
                "unit_id": _text(16),
                "type": _text(7),
                "position": position,
                "hit_points": spaces.Discrete(13),
                "movement_points": spaces.Discrete(5),
                "direction": spaces.Discrete(len(DIRECTIONS) + 1),
            }
        )
        tile = spaces.Dict({"terrain": _text(5), "feature": _text(6)})
        zone = spaces.Dict({"center": position, "tiles": spaces.Tuple([position] * 7)})
        roster_entry = spaces.Dict(
            {"player": _text(9), "unit_id": _text(16), "side": _text(4), "type": _text(7)}
        )
        capture_high = config.capture_target + max(1, config.capture_zones)
        parameters = spaces.Dict(
            {
                "seat_plan": _text(8),
                "field_extent": _bounded(FIELD_EXTENT_BOUNDS),
                "terrain": spaces.Discrete(2),
                "wasteland": spaces.Discrete(2),
                "unit_abilities": spaces.Discrete(2),
                "capture_zones": _bounded(CAPTURE_ZONES_BOUNDS),
                "capture_target": _bounded(CAPTURE_TARGET_BOUNDS),
                "round_cap": _bounded(ROUND_CAP_BOUNDS),
            }
        )
        observation = spaces.Dict(
            {
                "self": self_unit,
                "visible_units": spaces.Sequence(unit),
                "round": spaces.Discrete(config.round_cap, start=1),
                "capture": spaces.Dict(
                    {
                        "red": spaces.Discrete(capture_high),
                        "blue": spaces.Discrete(capture_high),
                        "target": spaces.Discrete(config.capture_target + 1),
                    }
                ),
                "battlefield": spaces.Dict(
                    {
                        "side": spaces.Discrete(side + 1),
                        "tiles": spaces.Tuple([spaces.Tuple([tile] * side)] * side),
                        "zones": spaces.Tuple([zone] * config.capture_zones),
                    }
                ),
                "rosters": spaces.Dict(
                    {name: spaces.Tuple([roster_entry] * count_per_side) for name in _SIDES}
                ),
                "parameters": parameters,
            }
        )
        action = spaces.Dict(
            {"path": spaces.Discrete(MAX_PATH_ID + 1), "target": spaces.Discrete(count_per_side + 1)}
        )
        wrapped = spaces.Dict(
            {
                "observation": observation,
                "action_mask": spaces.Dict(
                    {
                        "path": spaces.Box(0, 1, shape=(MAX_PATH_ID + 1,), dtype=np.int8),
                        "target": spaces.Box(0, 1, shape=(count_per_side + 1,), dtype=np.int8),
                    }
                ),
            }
        )
        self.observation_spaces = {agent: wrapped for agent in self.possible_agents}
        self.action_spaces = {agent: action for agent in self.possible_agents}

    def _roster_ids(self, side: str) -> tuple[str, ...]:
        return tuple(
            f"{side}_{kind}_{index}"
            for kind, count in COMPOSITIONS[self.config.seat_plan].items()
            for index in range(count)
        )

    def _agent_mapping(self) -> dict[str, str]:
        unit_ids = (*self._roster_ids("red"), *self._roster_ids("blue"))
        return {unit_id: f"player_{index}" for index, unit_id in enumerate(unit_ids)}

    def _side_for_agent(self, agent: str) -> str:
        return "red" if self.possible_agents.index(agent) < len(self.possible_agents) // 2 else "blue"

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        """Start a fresh match. The harness always seeds; an omitted seed replays seed 0."""
        self.match = Match(MatchConfig(**{**self.config.__dict__, "seed": 0 if seed is None else seed}))
        self.last_activation = None
        self.last_capture_changes = {"red": 0, "blue": 0}
        self.agents = list(self.possible_agents)
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        current = self.match.current_unit_id
        assert current is not None
        self.agent_selection = self.agent_by_unit[current]
        self._battlefield_snapshot = self._battlefield()
        self._rosters_snapshot = self._rosters()
        self._parameters_snapshot = self._parameters()
        # PettingZoo can request a final observation while consuming a dead step. Preserve each
        # player's last living observation because the engine removes killed units immediately.
        self._last_observations = {agent: self._observe_living(agent) for agent in self.possible_agents}

    @classmethod
    def from_observation(cls, observation: SkirmishObservation) -> SkirmishCraneEnv:
        """Build a visible-only forecast environment from one current observation.

        The source observation has no state for hidden units, so this environment contains only
        the observing unit and the units it can see. It uses the real rules for that reduced
        state, while capture and elimination results remain estimates when hidden units exist.
        """
        match = Match.from_observation(observation["observation"])
        env = cls(match.config)
        state: SkirmishObservationData = observation["observation"]
        rosters = state["rosters"]
        entries = (*rosters["red"], *rosters["blue"])
        env.possible_agents = [entry["player"] for entry in entries]
        env.agent_by_unit = {entry["unit_id"]: entry["player"] for entry in entries}
        env.unit_by_agent = {agent: unit_id for unit_id, agent in env.agent_by_unit.items()}
        env._enemy_roster = {
            entry["player"]: tuple(
                opponent["unit_id"] for opponent in rosters["blue" if entry["side"] == "red" else "red"]
            )
            for entry in entries
        }
        env.match = match
        env.last_activation = None
        env.last_capture_changes = {"red": 0, "blue": 0}
        env.agents = [agent for agent in env.possible_agents if env.unit_by_agent[agent] in match.units]
        env.rewards = {agent: 0.0 for agent in env.agents}
        env._cumulative_rewards = {agent: 0.0 for agent in env.agents}
        env.terminations = {agent: False for agent in env.agents}
        env.truncations = {agent: False for agent in env.agents}
        env.infos = {agent: {} for agent in env.agents}
        current = match.current_unit_id
        if current is None:
            raise RuntimeError("a forecast needs its observing unit as the current activation")
        env.agent_selection = env.agent_by_unit[current]
        env._battlefield_snapshot = env._battlefield()
        env._rosters_snapshot = env._rosters()
        env._parameters_snapshot = env._parameters()
        env._last_observations = {agent: env._observe_living(agent) for agent in env.agents}
        return env

    def _position(self, value: tuple[int, int]) -> AxialPosition:
        return {"q": value[0], "r": value[1]}

    def _parameters(self) -> MatchParameters:
        return {
            "seat_plan": self.config.seat_plan,
            "field_extent": self.config.field_extent,
            "terrain": int(self.config.terrain),
            "wasteland": int(self.config.wasteland),
            "unit_abilities": int(self.config.unit_abilities),
            "capture_zones": self.config.capture_zones,
            "capture_target": self.config.capture_target,
            "round_cap": self.config.round_cap,
        }

    def _roster_entries(self, side: str) -> tuple[RosterEntry, ...]:
        return tuple(
            {
                "player": self.agent_by_unit[entry.unit_id],
                "unit_id": entry.unit_id,
                "side": entry.side,
                "type": entry.kind,
            }
            for entry in self.match.initial_rosters[side]
        )

    def _rosters(self) -> Rosters:
        return {"red": self._roster_entries("red"), "blue": self._roster_entries("blue")}

    def _capture(self) -> Capture:
        return {
            "red": self.match.capture_scores["red"],
            "blue": self.match.capture_scores["blue"],
            "target": self.config.capture_target if self.config.capture else 0,
        }

    def _battlefield(self) -> Battlefield:
        field = self.match.battlefield
        tiles: tuple[tuple[Tile, ...], ...] = tuple(
            tuple({"terrain": tile.terrain, "feature": tile.feature} for tile in row) for row in field.tiles
        )
        zones: tuple[Zone, ...] = tuple(
            {
                "center": self._position(zone.center),
                "tiles": tuple(self._position(tile) for tile in zone.tiles),
            }
            for zone in field.zones
        )
        return {"side": field.side, "tiles": tiles, "zones": zones}

    def observe(self, agent: str) -> SkirmishObservation:
        # A killed unit has no engine perception to reshape, so its required dead-step observation
        # is the last one produced while that unit was alive.
        if self.unit_by_agent[agent] not in self.match.units and agent in self._last_observations:
            return self._last_observations[agent]
        observation = self._observe_living(agent)
        self._last_observations[agent] = observation
        return observation

    def _observe_living(self, agent: str) -> SkirmishObservation:
        unit_id = self.unit_by_agent[agent]
        unit = self.match.units.get(unit_id)
        if unit is None:
            raise RuntimeError(f"terminated player {agent!r} has no observation")
        perception = self.match.perception(unit_id)
        path_mask = np.zeros(MAX_PATH_ID + 1, dtype=np.int8)
        target_mask = np.zeros(len(self._enemy_roster[agent]) + 1, dtype=np.int8)
        path_mask[0] = 1
        target_mask[0] = 1
        # Nonacting players still need space-valid observations. Only the current activation
        # publishes movement and target choices beyond the always-legal stay and none values.
        if self.match.current_unit_id == unit_id:
            walkable_paths = cast("tuple[tuple[int, ...], ...]", perception["walkable_paths"])
            nameable_targets = cast("tuple[str, ...]", perception["nameable_targets"])
            for path in walkable_paths:
                path_mask[encode_path(path)] = 1
            for target in nameable_targets:
                target_mask[self._enemy_roster[agent].index(target) + 1] = 1
        visible_units = cast("tuple[dict[str, Any], ...]", perception["visible_units"])
        round_number = cast("int", perception["round"])
        self_unit: SelfUnit = {
            "unit_id": unit.unit_id,
            "type": unit.kind,
            "position": self._position(unit.position),
            "hit_points": unit.hit_points,
            "movement_points": unit.stats.movement_points,
            "direction": FORWARD_DIRECTION[unit.side],
        }
        visible: tuple[VisibleUnit, ...] = tuple(
            {
                "unit_id": other["unit_id"],
                "side": other["side"],
                "type": other["type"],
                "position": self._position(other["position"]),
                "hit_points": other["hit_points"],
                "has_acted": other["has_acted"],
            }
            for other in visible_units
        )
        action_mask: ActionMask = {"path": path_mask, "target": target_mask}
        return {
            "observation": {
                "self": self_unit,
                "visible_units": visible,
                "round": round_number,
                "capture": self._capture(),
                "battlefield": self._battlefield_snapshot,
                "rosters": self._rosters_snapshot,
                "parameters": self._parameters_snapshot,
            },
            "action_mask": action_mask,
        }

    def _order(self, action: Any, agent: str) -> Order:
        if not self.action_space(agent).contains(action):
            raise IllegalMoveError(f"{agent} supplied an action outside its Dict space")
        path_id = int(action["path"])
        target_slot = int(action["target"])
        path = decode_path(path_id)
        unit_id = self.unit_by_agent[agent]
        legal_paths, legal_targets = self.match.legal_orders(unit_id)
        if path not in legal_paths:
            raise IllegalMoveError(f"{agent} supplied an unwalkable path {path_id}")
        # Target values index the opponent's initial roster. A dead target keeps its slot but is no
        # longer nameable, so later deaths never renumber an agent's action space.
        target = None if target_slot == 0 else self._enemy_roster[agent][target_slot - 1]
        if target is not None and target not in legal_targets:
            raise IllegalMoveError(f"{agent} supplied an unnameable target {target_slot}")
        return Order(path=path, target=target)

    def step(self, action: Any) -> None:
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        before_capture = dict(self.match.capture_scores)
        activation = self.match.apply_order(self._order(action, agent))
        self.last_activation = activation
        self.last_capture_changes = {
            side: self.match.capture_scores[side] - before_capture[side] for side in _SIDES
        }
        self.rewards = {player: 0.0 for player in self.agents}
        if activation.killed_id is not None:
            # Keep the killed player in env.agents until PettingZoo receives its mandatory None step.
            self.terminations[self.agent_by_unit[activation.killed_id]] = True
        if self.match.result is not None:
            scores = {"red": self.match.result.red, "blue": self.match.result.blue}
            final = self.match.result.reason.startswith("round_cap")
            for player in self.agents:
                if not self.terminations[player]:
                    self.rewards[player] = float(scores[self._side_for_agent(player)])
                    if final:
                        self.truncations[player] = True
                    else:
                        self.terminations[player] = True
        self._cumulative_rewards[agent] = 0
        current = self.match.current_unit_id
        # The engine chooses the next living activation. PettingZoo then moves any newly dead player
        # ahead of it temporarily so the harness can perform the required cleanup step.
        self.agent_selection = self.agent_by_unit[current] if current is not None else agent
        self._accumulate_rewards()
        self._deads_step_first()

    def result_scores(self) -> dict[str, float] | None:
        """Return complete final team scores, or ``None`` before natural match completion."""
        if self.match.result is None:
            return None
        # Terminal rewards reach only players still active on the final transition. The harness uses
        # this complete mapping so previously killed units receive the same official team outcome.
        return {
            player: float(
                self.match.result.red if self._side_for_agent(player) == "red" else self.match.result.blue
            )
            for player in self.possible_agents
        }

    def chat_policy(self, sender: str) -> dict[str, object]:
        """Allow direct messages only to the sender's living allies, in player order."""
        side = self._side_for_agent(sender)
        recipients = tuple(
            player
            for player in self.possible_agents
            if player != sender
            and self._side_for_agent(player) == side
            and player in self.agents
            and not self.terminations.get(player, False)
            and not self.truncations.get(player, False)
        )
        return {"target_recipients": recipients, "default_recipient": None}

    def render(self) -> None:
        """Browser renderers consume the semantic overlay, not pixels from this environment."""
        return None

    def close(self) -> None:
        """Release no external resources. Required by PettingZoo's API surface."""
