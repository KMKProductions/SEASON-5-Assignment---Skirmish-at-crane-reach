"""The encoder: what the learner sees.

For each legal choice, the encoder produces one row of numbers called features. To make sure
the model works as expected, each feature must mean the same thing in training and during play
(inference). Collection and play share this one encoder, which follows three design rules:

- Describe each option relative to the unit. For example, "nearest_enemy" measures the distance
  from a candidate tile to the nearest visible enemy. The model can use that distance wherever
  the tile lies on the battlefield.
- Measure only what this unit can see: visible units, the battlefield, and public unit stats.
- Give each category its own 0-or-1 entry (one-hot encoding). For example, an archer's unit type
  becomes footman=0, archer=1, cavalry=0. Use FAR for a distance when there is nothing to measure.

To change what the learner sees, edit encode_entities, describe_move, or describe_target and
update the matching feature list below. Give VERSION a new value, collect new demonstrations,
and retrain with the updated encoder.
For a terrain-aware feature, such as a route cost to each enemy, use route_cost from routes.py.

encode_entities describes the acting unit, visible units, and capture zones as tokens,
one row per entity. Target pointers let the model read the named enemy's token directly.
"""

from sandbox.crane import me, roster, tile, units, visible, zone

VERSION = "entity-attention-v2"
KINDS = ("footman", "archer", "cavalry")
TERRAINS = ("grass", "hill", "water", "void")
GROUND_FEATURES = ("none", "forest", "marsh", "waste")
FAR = 20  # The distance we report when no enemy is visible or the episode has no zones.

# The feature names in a tile row and in a target row, in the order the model receives them.
MOVE_FEATURES = (
    # The acting unit
    *KINDS,
    "hit_points",
    "visible_enemies",
    # Movement
    "moved",
    "forward",
    "center_distance",
    "zone_distance",
    "in_zone",
    # Support
    "adjacent_allies",
    # Danger
    "nearest_enemy",
    "weakest_enemy",
    "threatened_by",
    "reachable_by",
    # Opportunity
    "can_strike",
    "weakest_strikeable_hit_points",
    # Ground
    *TERRAINS,
    *GROUND_FEATURES,
    # The oriented destination
    "end_q",
    "end_r",
)
TARGET_FEATURES = (
    "no_target",
    *KINDS,
    "hit_points",
    "has_acted",
    "distance",
    "in_range",
    "weakest_visible",
    "weakest_in_range",
    "enemy_from_end_q",
    "enemy_from_end_r",
    "end_q",
    "end_r",
)
ENTITY_FEATURES = (
    "self",
    "ally",
    "enemy",
    "zone",
    *KINDS,
    "relative_q",
    "relative_r",
    "health_fraction",
    "has_acted",
    "movement",
    "attack_range",
    "zone_size",
    "visible_allies",
    "visible_enemies",
    "own_score",
    "opponent_score",
    "score_difference",
)


def oriented_offset(observation, start, end):
    """Use the square grid width and rotate the other side by 180 degrees.

    battlefield.side is 25 when field_extent is 12. A unit facing direction 5 (the blue side)
    gets both axial offsets negated, a half turn; reflecting just one coordinate would change
    the hex geometry.
    """
    scale = (1 if me.direction(observation) == 2 else -1) / observation["observation"]["battlefield"]["side"]
    return ((end["q"] - start["q"]) * scale, (end["r"] - start["r"]) * scale)


def encode_entities(observation):
    """Return entity rows and a map from visible enemy IDs to their row indices."""
    own = observation["observation"]["self"]
    side = me.side(observation)
    rows, enemies = [], {}

    def token(kind, position):
        values: dict[str, float] = dict.fromkeys(ENTITY_FEATURES, 0)
        values[kind] = 1
        values["relative_q"], values["relative_r"] = oriented_offset(observation, own["position"], position)
        return values

    # Token 0 always describes the acting unit. The target scorer reads it directly.
    for unit in [own, *observation["observation"]["visible_units"]]:
        kind = "self" if unit is own else "ally" if unit["side"] == side else "enemy"
        values = token(kind, unit["position"])
        stats = units.STATS[unit["type"]]
        values.update(one_hot(unit["type"], KINDS))
        values.update(
            health_fraction=unit["hit_points"] / stats.hit_points,
            has_acted=0 if unit is own else int(unit["has_acted"]),
            movement=(own["movement_points"] if unit is own else stats.movement_points) / 4,
            attack_range=stats.attack_range / 3,
        )
        if unit is own:
            # Capture scores are public. Divide by the winning score to keep their scale steady.
            capture = observation["observation"]["capture"]
            if capture["target"]:
                opponent = "blue" if side == "red" else "red"
                values["own_score"] = capture[side] / capture["target"]
                values["opponent_score"] = capture[opponent] / capture["target"]
                values["score_difference"] = (capture[side] - capture[opponent]) / capture["target"]
        if kind == "enemy":
            # In the practice, you can add a route cost to this token with route_cost from routes.py.
            # IDs connect targets to rows, but never become numeric model features.
            enemies[unit["unit_id"]] = len(rows)
        rows.append(as_row(values, ENTITY_FEATURES))
    grid_width = observation["observation"]["battlefield"]["side"]
    for area in zone.zones(observation):
        # Zones have their own rows, including only the occupants this unit can see.
        values = token("zone", area["center"])
        occupants = zone.occupants(observation, area)
        values["zone_size"] = len(area["tiles"]) / grid_width**2
        values["visible_allies"] = sum(unit["side"] == side for unit in occupants)
        values["visible_enemies"] = sum(unit["side"] != side for unit in occupants)
        rows.append(as_row(values, ENTITY_FEATURES))
    return rows, enemies


def target_entity_indices(observation, targets, enemy_indices):
    """Point each target at its visible enemy token; -1 means no target."""
    enemy_roster = roster.enemies(observation)
    indices = []
    for target in targets:
        if target == 0:
            indices.append(-1)
        elif 0 < target <= len(enemy_roster) and enemy_roster[target - 1]["unit_id"] in enemy_indices:
            # A target ID is a roster slot, not a token index. Look up the unit ID in between.
            indices.append(enemy_indices[enemy_roster[target - 1]["unit_id"]])
        else:
            raise ValueError(f"Target {target} does not identify a visible enemy")
    return indices


# --- The options: what the model chooses between ----------------------------------------------


def move_options(observation):
    """Return (path ID, end tile) for each reachable tile, with staying in place first.

    To compare destinations, we keep one legal path per tile. Path IDs are ordered by length,
    making the first legal path to each tile its shortest route, with the lowest ID breaking
    ties. The learner scores the tiles and follows the kept path to the one it chooses.
    """
    here = me.position(observation)
    options = {}  # Maps the (q, r) of each end tile to its (path ID, end tile).
    for path_id, allowed in enumerate(observation["action_mask"]["path"]):
        if allowed:
            end = tile.at_path_end(here, path_id)
            # setdefault keeps the first path that reaches this tile and ignores the later ones.
            options.setdefault((end["q"], end["r"]), (path_id, end))
    return list(options.values())


def target_options(observation):
    """Return the legal target IDs, with no target (0) first. Target i is enemy roster slot i - 1."""
    return [index for index, allowed in enumerate(observation["action_mask"]["target"]) if allowed]


# --- The rows: how one option becomes numbers -------------------------------------------------


def describe_move(observation, end):
    """Describe ending this activation on the tile end, with one value per name in MOVE_FEATURES."""
    # We use the current observation to describe the candidate tile end.
    own = observation["observation"]["self"]
    here = own["position"]
    enemies = visible.enemies(observation)  # Only visible enemies; hidden ones never enter a feature.
    # The weakest visible enemy. When two are equally weak, we take the closer one.
    weakest = min(
        enemies, key=lambda enemy: (enemy["hit_points"], tile.distance(here, enemy["position"])), default=None
    )
    # The enemies this unit could strike after ending on the tile.
    in_reach = [enemy for enemy in enemies if strikes(own["type"], end, enemy["position"])]
    # To make "forward" mean the same thing for both sides, we measure it along one step
    # toward the enemy side.
    step_q, step_r = tile.DIRECTIONS[me.direction(observation)]
    zone_centers = [area["center"] for area in zone.zones(observation)]
    ground = tile.terrain_at(observation, end)
    # The destination itself, as a step offset turned to face the enemy side.
    end_q, end_r = oriented_offset(observation, here, end)

    features = {
        # Acting unit: these values are the same for every tile in one decision.
        **one_hot(own["type"], KINDS),
        "hit_points": own["hit_points"],
        "visible_enemies": len(enemies),
        # Movement: how far the tile is from here, how far it is toward the enemy side, and how
        # far it is from the objectives. All distances are straight-line steps.
        "moved": tile.distance(here, end),
        # Progress along the unit's forward direction is positive, and backward is negative.
        "forward": (end["q"] - here["q"]) * step_q + (end["r"] - here["r"]) * step_r,
        "center_distance": tile.distance(end, tile.at_center(observation)),
        "zone_distance": min((tile.distance(end, center) for center in zone_centers), default=FAR),
        "in_zone": int(zone.at(observation, end) is not None),
        # Support: how many allies stand on the six neighboring tiles.
        "adjacent_allies": sum(
            tile.distance(end, ally["position"]) == 1 for ally in visible.allies(observation)
        ),
        # Danger: how close the visible enemies are to the tile, and how many could strike it.
        "nearest_enemy": min((tile.distance(end, enemy["position"]) for enemy in enemies), default=FAR),
        "weakest_enemy": FAR if weakest is None else tile.distance(end, weakest["position"]),
        # The enemies that could strike the tile from where they stand now.
        "threatened_by": sum(strikes(enemy["type"], enemy["position"], end) for enemy in enemies),
        # The enemies that could move next turn and then strike the tile, in straight-line steps.
        "reachable_by": sum(
            tile.distance(end, enemy["position"])
            <= units.STATS[enemy["type"]].movement_points + units.STATS[enemy["type"]].attack_range
            for enemy in enemies
        ),
        # Opportunity: how many enemies this unit could strike from the tile, and how hurt the
        # weakest of them is.
        "can_strike": len(in_reach),
        "weakest_strikeable_hit_points": min((enemy["hit_points"] for enemy in in_reach), default=0),
        # Ground: the terrain and feature of the tile itself.
        **one_hot(ground["terrain"], TERRAINS),
        **one_hot(ground["feature"], GROUND_FEATURES),
        "end_q": end_q,
        "end_r": end_r,
    }
    return as_row(features, MOVE_FEATURES)


def describe_target(observation, end, target_id):
    """Describe naming target_id after moving to end, with one value per name in TARGET_FEATURES."""
    # No target keeps the destination coordinates and has zero enemy features.
    features: dict[str, float] = dict.fromkeys(TARGET_FEATURES, 0)
    features["end_q"], features["end_r"] = oriented_offset(observation, me.position(observation), end)
    if target_id == 0:
        features["no_target"] = 1
        return as_row(features, TARGET_FEATURES)
    own_type = observation["observation"]["self"]["type"]
    enemies = visible.enemies(observation)
    # Target i names enemy roster slot i - 1. We find that unit among the visible enemies.
    enemy_id = roster.enemies(observation)[target_id - 1]["unit_id"]
    enemy = next(unit for unit in enemies if unit["unit_id"] == enemy_id)
    features["enemy_from_end_q"], features["enemy_from_end_r"] = oriented_offset(
        observation, end, enemy["position"]
    )
    in_reach = [unit for unit in enemies if strikes(own_type, end, unit["position"])]
    features.update(
        {
            # Who the target is, how hurt it is, and whether it has acted this round.
            **one_hot(enemy["type"], KINDS),
            "hit_points": enemy["hit_points"],
            "has_acted": int(enemy["has_acted"]),
            # The distance to the target and whether it is in strike range from the chosen tile.
            "distance": tile.distance(end, enemy["position"]),
            "in_range": int(enemy in in_reach),
            # Whether the target has the fewest hit points in sight or among enemies in strike range.
            "weakest_visible": int(enemy["hit_points"] == min(unit["hit_points"] for unit in enemies)),
            "weakest_in_range": int(
                bool(in_reach) and enemy["hit_points"] == min(unit["hit_points"] for unit in in_reach)
            ),
        }
    )
    return as_row(features, TARGET_FEATURES)


# --- Small helpers ----------------------------------------------------------------------------


def one_hot(value, choices):
    """Give each choice its own entry: 1 for value and 0 for the rest, e.g., archer -> 0, 1, 0."""
    return {choice: int(value == choice) for choice in choices}


def as_row(features, names):
    """Put the named values in the model's order, and report any name that is missing or extra."""
    if features.keys() != set(names):
        raise ValueError(
            f"Update the feature names to match the encoder: "
            f"missing {sorted(set(names) - features.keys())}, "
            f"extra {sorted(features.keys() - set(names))}"
        )
    return [features[name] for name in names]


def strikes(attacker_type, start, goal):
    """Return whether a unit of attacker_type standing on start could strike goal."""
    return tile.distance(start, goal) <= units.STATS[attacker_type].attack_range
