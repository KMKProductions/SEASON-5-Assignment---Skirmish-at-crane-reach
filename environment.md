# Skirmish at Crane Reach

Skirmish at Crane Reach is a turn-based tactics game on a hex field where Red and Blue fight for ground. Your side wins by defeating the other side, reaching the capture target when capture zones are enabled, or leading at the round limit. The shared [agent interface](https://vox-deorum.github.io/game-sandbox/students/agent-interface/) explains the methods every environment uses.

Most importantly: every unit runs a separate `Agent` instance with its own state. Your code does not command an army, and your agents do not share observations or memory. The [agent interface](https://vox-deorum.github.io/game-sandbox/students/agent-interface/#agent-instances-and-state) installs one instance per player, each with its own memory.

## Start with the template

Complete the [Getting Started guide](https://vox-deorum.github.io/game-sandbox/students/getting-started/) first, then open `agent.py` in the template. An **episode** is one complete match. As you edit, run these commands from the template folder:

```console
python -m sandbox play   # command a side yourself in your browser
python -m sandbox watch  # watch your agent take on Naive
python -m sandbox test   # run the provided checks
python -m sandbox eval   # compare your agent with Naive
```

**Naive** is a simple built-in opponent. It holds the other side in `watch` and `eval`, so both commands measure the same matchup. `eval` runs repeatable matches and reports your selected side's average team score. Use the same seeds before and after a change. Local results cannot predict a leaderboard result.

On the board, muted units with a checkmark have already acted this round. Their normal colors return next round. Hover over a unit to inspect it; the same checkmark appears beside its name. You can still select an enemy that has acted as your attack target.

## Make an order

On each activation, `act` returns one **order**: a path of at most four steps and an optional named enemy. The unit walks its whole path, then strikes once from its final tile. A minimal order is a dictionary:

```python
return {"path": 0, "target": 0}
```

`path: 0` stays in place and `target: 0` names nobody. This order is always legal, and it is played if your code answers late. Prefer the helpers in `sandbox.crane`:

```python
from sandbox.crane import action

return action.stay()
return action.move(path_id)
```

Pass a visible enemy id and the observation when you name a target: `action.stay(enemy_id, observation)` or `action.move(path_id, enemy_id, observation)`. The helpers resolve that id to the target slot the game needs.

The observation has an authoritative `action_mask` with a `path` array and a `target` array. A `1` allows a choice and a `0` rejects it. Use `action.legal_paths(observation)`, `action.legal_steps(observation)`, and `action.possible_targets(observation)` rather than reimplementing legality. Values returned by those readers are legal, but `action.move` and `action.stay` only build dictionaries: an arbitrary path id or unit id can still make an invalid order. A path and target that are individually allowed may be combined.

## How the starter agent works

The helper names used below, such as `visible.enemies`, come from `sandbox.crane` and are each listed in [Helpers](#helpers). The template is weak but complete, so its behavior is easy to recognize in a recording. It keeps no episode state, so `reset` has nothing to prepare. Each `act` first gets the enemies this unit can see:

```python
enemies = visible.enemies(observation)
```

When that list is empty, it tries one legal step in `me.direction(observation)`, the digit toward the enemy side. In the default `skirmish` plan, the spawns begin farther apart than any unit can see, so this forward behavior starts the match. If that step is blocked, it stays. The template checks `action.legal_steps(observation)` before moving, so a wall, ally, or enemy cannot make that order illegal.

Once an enemy is visible, the template reads its own position with `me.position`, chooses the visible enemy with the smallest `tile.distance`, and calls `_step_toward`. That method tries only legal single steps, uses `tile.at_path_end` to find each landing tile, and returns the one that most reduces the distance, or `0` when none does. The final order names the nearest enemy:

```python
if step == 0:
    return action.stay(nearest["unit_id"], observation)
return action.move(step, nearest["unit_id"], observation)
```

Naming a target makes the strike prefer it when that enemy is alive and in range after the move. The `TODO(you)` comments mark the useful starting changes: pick a response when forward is blocked, replace the all-units-chase strategy, and use longer routes in `_step_toward`. Both sides run this same starter, so watch what it actually does before changing it.

## Your first improvement

Make one small edit first. After the template finds `nearest`, but before it calls `_step_toward`, make an archer stay and name that target:

```python
if me.unit_type(observation) == "archer":
    return action.stay(nearest["unit_id"], observation)
```

This change is narrow: when an archer sees the nearest enemy, it stops walking closer and tries to shoot it. Run `watch`, then compare several seeded `eval` episodes before and after. The first improvement needs no new imports. Later, `from sandbox.crane import paths, roster, units, zone` gives you longer routes, full-roster coordination, fixed unit stats, and capture-zone geometry.

Try one experiment at a time:

1. Let cavalry spend more than one movement point. Build a legal multi-step path toward a goal and watch whether its speed creates a useful flank or an isolated target.
2. Give footmen and archers a shared local purpose: protect an archer, or rendezvous at `tile.at_center(observation)` before advancing. An archer sees and shoots six tiles but has only 6 hit points, while a footman can hold adjacent ground.
3. Starting in Season 3, send a delayed message that tells an ally where you expect to be next. The recipient cannot use it until a later activation, so design the message around that delay.

Change one behavior at a time and watch a complete recording, not only the final score. Look for the first contact, the tiles where units become targets, and whether a unit arrives with allies nearby. A change that wins one short fight can still lose a match by leaving a capture zone empty or exposing the archer.

`python -m sandbox eval --vs rivals/v1` puts the saved rival on the enemy side in place of Naive, while your current agent controls the selected side. Add `--seat 1` to switch sides. Before developing these ideas further, read the rules below. When your agent is ready, follow the [submitting guide](https://vox-deorum.github.io/game-sandbox/students/submitting/).

## Match flow

A match proceeds in rounds. Every living unit activates once per round, in an order shuffled again at the start of that round. Results apply immediately, so later units see earlier movement and combat. A unit killed before its activation does not activate that round.

Each activation resolves one complete order. A unit cannot pause after a step to see the new state, and its `Agent` instance never knows which unit will activate next. Plan from the observation you have, return one legal order promptly, then save only information that the same unit can use on a later activation. The recording is the complete match record; your observation is not.

After walking, a unit strikes as follows:

- It strikes its named target if that target is still alive and in range from the final tile.
- Otherwise it strikes the nearest enemy in range. Ties for nearest are random, and a named target that ends out of range uses this same fallback.
- It does nothing when no enemy is in range.

An attack always hits. Staying still can still strike, so avoiding a fight requires ending out of every enemy's range. A unit with 0 hit points leaves the field immediately.

## Seats and units

A **player** is one unit on the field. A **seat** is the group of players controlled by one submission, which is always a whole side here. Red has the first half of players and Blue the second. Within a side, player and roster order is footmen, then archers, then cavalry, each in index order. Every round shuffles the living units for activation. Stable ids use `side_type_index`, such as `red_archer_0`; the rosters in every observation already provide the mapping.

| Plan                 | What one submission controls                          |
| -------------------- | ----------------------------------------------------- |
| `skirmish` (default) | One side of 3 units: 1 footman, 1 archer, 1 cavalry   |
| `army`               | One side of 20 units: 8 footmen, 6 archers, 6 cavalry |

In `skirmish`, `player_0`, `player_1`, and `player_2` are Red's footman, archer, and cavalry. `player_3` through `player_5` repeat that order for Blue. A fixed seed still gives the sides different spawn halves and activation draws, so switching sides changes the match.

> _New to hex maps?_ A tile has six neighbors. Positions are axial coordinate dictionaries such as `{"q": 8, "r": 5}`. The [Wikipedia introduction to hex maps](https://en.wikipedia.org/wiki/Hex_map) gives useful background.

## Battlefield and unit stats

| Stat            | Footman | Archer | Cavalry |
| --------------- | ------- | ------ | ------- |
| Hit points      | 12      | 6      | 10      |
| Movement points | 2       | 2      | 4       |
| Attack range    | 1       | 6      | 1       |
| Damage          | 3       | 2      | 3       |
| Vision          | 4       | 6      | 6       |

Damage is the attacker's damage after the adjustments below and never drops below 1. The point-symmetric field mirrors the spawn positions so neither side gets better ground. Every passable tile is reachable from every other, and each tile holds at most one unit.

Each tile has one terrain and at most one feature. Their effects stack, so a hill with a forest costs 3 to enter and provides both effects.

| Terrain or feature | Move cost | Effect |
| --- | --- | --- |
| Grass (terrain) | 1 | None. It covers the field when terrain is off. |
| Hill (terrain) | 2 | High ground: attacks from a hill onto lower ground deal 1 extra damage, and attacks from lower ground onto a hill deal 1 less. A unit on a hill sees 1 tile farther. |
| Water (terrain) | impassable | Shapes the two or three passages between the halves of the field. |
| Forest (feature) | +1 | Cover: a unit in forest takes 1 less damage from attacks made from more than 1 tile away, and no charge bonus applies against it. |
| Marsh (feature) | +2 | Slow ground. |
| Wasteland (feature, `waste` in the observation) | +0 | Entering costs 2 hit points, never below 1. Standing there is free, so only entry hurts. |

Vision and attacks ignore terrain. Terrain changes movement cost and damage, never sight or arrow paths. A unit starts each activation with full movement points and pays on entry. Its first step may enter any empty passable tile even if that cost exceeds its points. Later steps need enough points, and a tile that would take the balance below zero must end the path. Re-entering wasteland pays its damage again.

When a season enables unit abilities, these rules apply:

- **Charge:** cavalry that ends its walk at least 3 tiles from its start gains 2 damage for that activation.
- **Shield wall:** a footman next to an allied footman takes 1 less damage and cannot receive charge bonus damage. A lone footman gets neither.

## Scoring and rewards

Every player on one side receives the same team score from 0 to 100. A win is 70 to 100, a draw is 50, and a loss is 0 to 30.

- **Elimination**, when capture zones are off: the match ends when a side has no living units or at the round cap. Eliminating the enemy earns 70 plus up to 30 for the fraction of the winner's hit points still standing, while the eliminated side gets 0. At the cap, higher total remaining hit points wins; the margin is the hit point difference divided by the winner's starting total. Equal totals draw.
- **Capture**, when zones are on: all zones score at the end of each round. A zone with living units from exactly one side gives that side 1 point, while an empty or contested zone gives nobody a point. If one or both sides reach the target, the higher resulting capture score wins. Equal capture scores use total remaining hit points, then draw if those totals are equal. Eliminating the enemy earns 100 against 0. The capture margin is the capture-score difference divided by the target, capped at 1.

Rewards are not the official score. Every non-final step gives `0.0`; on the final step, every living player receives its side's team score. A player killed earlier stops at `0.0`, though the recording's match result still assigns it the full team score. A seat that forfeits by crashing, returning an illegal action, or using its game limit scores 0.

## Helpers

Import helpers at the top of `agent.py`:

```python
from sandbox.crane import action, me, tile, visible
```

`roster`, `paths`, `units`, and `zone` are available when you need them, but the first improvement does not need them. The helpers do not choose strategy or include a pathfinder, so you plan routes yourself.

`act` receives one dictionary with `observation` and `action_mask` keys. The current match state is under `observation["observation"]`. Its `self` field describes your unit, `visible_units` lists other units in vision, `round` starts at 1, and `capture` holds both scores and the target. A visible unit's `has_acted` field says whether that unit has completed its activation in the current round. It is false at round start and resets for every unit at the next round. It does not reveal the remaining activation order. `battlefield`, both `rosters`, and `parameters` are shared match knowledge and stay constant for the match, so you may cache them from `reset`. Treat them as read-only: every player receives the same objects, so mutating them corrupts what other players observe.

Units outside vision are absent, with no count of what is missing. The observation has no history and does not say who attacked you. Store information you need on that unit's own `Agent` instance. Information from beyond vision must arrive through a message. In `sandbox.observation_types`, `SkirmishObservation` and `SkirmishAction` provide the exact TypedDict shapes for your editor and type checker.

The rosters list every starting unit, including units no longer alive. Use a roster when a message needs the player name of an ally outside sight. Do not use a roster entry as proof that the unit is still alive or a legal target. `visible.enemies(observation)` and `action.possible_targets(observation)` give the current, legal view.

`units.STATS` looks up a fixed table by type, such as `units.STATS["archer"].attack_range` for how far an archer can strike.

| Group | Callable | Result |
| --- | --- | --- |
| Orders | `action.legal_paths(observation)` | Every legal path id from the mask, including `0`. |
| Orders | `action.legal_steps(observation)` | Legal one-step path ids, `1` through `6`. |
| Orders | `action.possible_targets(observation)` | Enemy unit ids you may name, in roster order. |
| Orders | `action.move(path_id, target_id=None, observation=None)` | Builds an order and resolves a target id with the observation. |
| Orders | `action.stay(target_id=None, observation=None)` | Builds a stay order, optionally with a target. |
| Self | `me.unit_id(observation)`, `me.side(observation)`, `me.unit_type(observation)` | Your id, side, and type. |
| Self | `me.position(observation)`, `me.direction(observation)` | Your position and the digit toward the enemy side, `2` for Red and `5` for Blue. |
| Self | `me.hit_points(observation)`, `me.movement_points(observation)` | Your current hit points and full points for this activation. |
| Visible | `visible.enemies(observation)`, `visible.allies(observation)` | Other units in sight, divided by side. |
| Roster | `roster.enemies(observation)`, `roster.allies(observation)` | Complete starting rosters, alive or not. |
| Unit | `units.STATS[unit_type]` | Fixed `hit_points`, `movement_points`, `attack_range`, `damage`, and `vision` for that type. |
| Tile | `tile.distance(first, second)` | Hex distance in steps. |
| Tile | `tile.neighbors(position)` | Six neighbors keyed by direction digit. |
| Tile | `tile.at_path_end(position, path_id)` | Where a path ends. |
| Tile | `tile.at_center(observation)` | The field center. |
| Tile | `tile.at_mirror(position, observation)` | The opposite position, in mirrored enemy ground for your starting position. |
| Tile | `tile.terrain_at(observation, position)` | The `{"terrain", "feature"}` pair, or void off the field. |
| Tile | `tile.DIRECTIONS` | Direction digit to axial offset. |
| Zone | `zone.zones(observation)` | The battlefield's capture zones, empty when capture play is off. |
| Zone | `zone.at(observation, position)` | The zone covering that position, or `None`. |
| Zone | `zone.occupants(observation, area)` | The units you can see standing in that zone, your own unit first if it stands there. An enemy outside your vision is missing from the list, so an empty result does not prove the zone is free. |
| Paths | `paths.encode(directions)`, `paths.decode(path_id)` | Convert direction sequences and path ids. |
| Paths | `paths.MAX_ID`, `paths.MAX_STEPS` | `1554` and `4`. |
| Forecast | `forecast.from_observation(observation)` | Build an independent local `SkirmishCraneEnv` from one observation. |

Invalid path ids or direction digits passed to `paths` raise `ValueError`. `tile.neighbors` is geometry, not a legality check. The mask remains the source of truth.

### Forecast a local future

`forecast.from_observation(observation)` gives you a separate game in which to try orders before choosing your real action. Replace your starter's `agent.py` with this example: it compares staying with every legal one-step move and chooses the order that preserves the most health until its next activation.

```python
from sandbox.crane import action, forecast, me


class Agent:
    def reset(self, seed, observation):
        # This strategy plans from scratch each time, so it needs no saved state.
        pass

    def act(self, observation):
        # Unit IDs identify pieces on the battlefield, such as "red_cavalry_0".
        own_id = me.unit_id(observation)
        best_order = action.stay()
        # Health cannot be negative here, so the first candidate always beats -1.
        best_health = -1

        # Path 0 means stay. The * adds each legal one-step path to this list.
        # Trying stay first makes it our preferred choice when scores tie.
        for path_id in [0, *action.legal_steps(observation)]:
            # Start every candidate from the same observed position. Changes to
            # this copy affect neither the real game nor the other candidates.
            # It keeps the map, scores, round, and visible has_acted flags.
            # Only your unit and visible units exist in this simulated battle.
            trial = forecast.from_observation(observation)

            # The copy is ready to play, with our unit next. Do not call reset():
            # that would generate a new battle instead of using this position.
            # Player IDs, such as "player_2", identify whose activation is next.
            own_player = trial.agent_selection
            order = action.move(path_id)
            # Resolve our candidate's movement and automatic attack.
            trial.step(order)

            # Play forward until it is our turn again or the simulation ends.
            # Units with has_acted=True wait for the next round. The remaining
            # order and automatic targets use local randomness seeded at 0,
            # so they can differ from what happens in the real game.
            while trial.agents and trial.agent_selection != own_player:
                # last() describes the next player. We only need the end flags;
                # _ discards the observation, reward, and extra information.
                _, _, terminated, truncated, _ = trial.last()
                if terminated or truncated:
                    # A dead player or one whose match ended needs cleanup,
                    # not a new order. None removes it from the turn sequence.
                    trial.step(None)
                else:
                    # Our simple prediction: everyone else stays in place.
                    # They still attack enemies in range. You can replace this
                    # with a better prediction of the other units' choices.
                    trial.step(action.stay())

            # The simulated battlefield stores living units by unit ID.
            # get() returns None if our unit was killed, which scores zero.
            survivor = trial.match.units.get(own_id)
            health = survivor.hit_points if survivor is not None else 0
            # Keep the healthiest outcome. To pursue a different goal, replace
            # this score with one based on damage, capture progress, or position.
            if health > best_health:
                best_order = order
                best_health = health
            # Finished with this candidate's private simulation.
            trial.close()

        # Only this returned order is played in the real game.
        return best_order
```

## Season settings

Your submission uses the settings of the season you submit to. On your computer, `season_N` is a preset that reproduces that season's gameplay parameters, not a separate season. Your observation exposes the resolved values under `parameters`.

| Season | Preset | Seat plan | Field extent | Terrain | Abilities | Capture zones | Wasteland |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1, The Skirmish | `season_1` | `skirmish` | 7 (15 across) | off | off | 0 | off |
| 2, The March | `season_2` | `skirmish` | 7 (15 across) | on | off | 0 | off |
| 3, The Army | `season_3` | `army` | 12 (25 across) | on | on | 0 | off |
| 4, The Commander | `season_4` | `army` | 12 (25 across) | on | on | 1 | off |
| 5, The General | `season_5` | `army` | 12 (25 across) | on | on | 3 | off |
| 6, The Rivals | `season_6` | `army` | 12 (25 across) | on | on | 3 | on |

Field extent is the hex distance from center to edge. All six included presets use a capture target of 200 and a round cap of 1000. Season settings and local overrides can change either. Messaging is controlled separately by the season, so it is always available in local runs. Season 6 adds wasteland, whose entry cost cannot kill a unit by itself.

The local start dialog lets you choose any of the six seasons. For example, run `python -m sandbox watch --preset season_4`.

For `season.json`, `--preset`, `--parameter`, and the time-limit flags, see [Getting started](https://vox-deorum.github.io/game-sandbox/students/getting-started/#more-local-run-options).

## Time limits

The game is turn-based. By default, each agent's `act` must return within 1 second, and an agent may use 600 seconds of measured computation per match. A season may override either limit. A late `act` becomes `{"path": 0, "target": 0}`, so it can still strike but cannot move. See [Time limits](https://vox-deorum.github.io/game-sandbox/students/agent-interface/#time-limits) for measurement and enforcement.

## Messaging

Messaging is enabled starting in Season 3. Add the optional `chat` method for those seasons. `act` chooses an order first, then `chat` receives messages that arrived since this unit's previous activation. An incoming message cannot change the order just chosen, so the earliest it can affect play is the next activation. Store it on that unit's instance if you need it later.

Direct recipients must be living allied players other than the sender, addressed by player string, not unit id. Broadcasts use `None` and reach both sides. Text is limited to 200 characters unless a season lowers it. A message to a dead unit is dropped with a note in the local console. Every message is recorded and visible in replays, so it is never secret.

```python
def chat(self, inbox: list[dict]) -> list[dict]:
    self.last_messages = [message["text"] for message in inbox]
    return [{"to": None, "text": "hold the center"}]
```

See the shared [agent interface](https://vox-deorum.github.io/game-sandbox/students/agent-interface/#chatinbox) for generic message policy, replay details, and chat time accounting.

## Optional advanced raw reference

The helpers are enough for the starter. This section gives the exact raw shapes for agents that need them.

### Path and target encoding

Path `0` stays. Paths `1` through `1554` encode all direction sequences of one to four digits, ordered by length and then lexicographically with the last digit changing fastest. Use `paths.encode` and `paths.decode`; `paths.encode(())` is `0`, `paths.decode(0)` is `()`, and a single-step id is its direction digit. No order can contain a fifth step because every step costs at least 1 and the fastest unit has 4 movement points.

| Digit | Direction | `dq, dr` |
| ----- | --------- | -------- |
| `1`   | northeast | `+1, -1` |
| `2`   | east      | `+1, 0`  |
| `3`   | southeast | `0, +1`  |
| `4`   | southwest | `-1, +1` |
| `5`   | west      | `-1, 0`  |
| `6`   | northwest | `0, -1`  |

Target `0` names nobody. Target `i` is slot `i - 1` in the enemy roster, in player order. The same numeric target can therefore name a different unit for Red and Blue. `action.move` and `action.stay` resolve a unit id for you when given the observation.

### Observation fields and mask

| Field | Content |
| --- | --- |
| `self` | Your `unit_id`, `type`, `position`, `hit_points`, `movement_points`, and `direction`. `movement_points` is always the type's full stat at a new activation. |
| `visible_units` | Every other unit inside vision, in player order: `unit_id`, `side`, `type`, `position`, `hit_points`, and `has_acted: bool`. `has_acted` is true after that unit completes its activation in this round, including a stay order, and false before then. It resets at the next round and does not expose future activation order. |
| `round` | Current round, from `1` through the cap. |
| `capture` | `red`, `blue`, and `target`, all `0` when capture is off. |
| `battlefield` | `side`, square-array width, `tiles[r][q]` terrain-feature pairs, and seven-tile `zones` as `{"center", "tiles"}` pairs. |
| `rosters` | `red` and `blue` tuples of `{"player", "unit_id", "side", "type"}` for every starting unit. |
| `parameters` | `seat_plan`, `field_extent`, `terrain`, `wasteland`, `unit_abilities`, `capture_zones`, `capture_target`, `round_cap`. The three switches are `0` or `1`. |

Terrain values are `grass`, `hill`, `water`, and `void`; features are `none`, `forest`, `marsh`, and `waste`. The square tile array has `void` cells outside the hex field. The path mask has exactly 1555 entries, one for each id. The target mask has one entry for each enemy roster slot plus the no-target entry. A `1` permits a value. A `1` in the target mask means the enemy is alive and visible now, not that it will be in range after movement. `env.step` rejects a zero-mask choice, and an illegal action in an official game forfeits the seat. Stay and no-target are always allowed.
