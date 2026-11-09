"""Terrain-aware route costs for the practice.

Two enemies may stand the same number of steps from a unit, yet the route to one crosses a marsh
while the route to the other stays on grass. To tell them apart, route_cost finds the cheapest
route and returns its movement cost.

For example, to add a route cost to each enemy's token, import route_cost and NO_ROUTE in
encoder.py and add "route_cost" to ENTITY_FEATURES. Then add this line to the enemy branch of
encode_entities, change VERSION, and recollect and retrain. We divide by NO_ROUTE to keep the
number small, like the token's other entries:

    values["route_cost"] = route_cost(observation, own["position"], unit["position"]) / NO_ROUTE
"""

from heapq import heappop, heappush

from sandbox.crane import tile

NO_ROUTE = 100  # The cost we report when water or void cuts start off from goal.
_routes = {"observation": None, "maps": {}}


def route_cost(observation, start, goal):
    """Return the least movement cost from start to goal, counting terrain.

    Entering a tile costs 1, or 2 on a hill, plus 1 more in a forest or 2 more in a marsh. Water
    and void block movement, and units are ignored. To keep act fast, we reuse one search for
    every start tile with the same goal and observation.
    """
    if _routes["observation"] is not observation:
        _routes["observation"], _routes["maps"] = observation, {}
    key = (goal["q"], goal["r"])
    if key not in _routes["maps"]:
        _routes["maps"][key] = _route_map(observation, goal)
    return _routes["maps"][key].get((start["q"], start["r"]), NO_ROUTE)


def _route_map(observation, goal):
    """Find the cost from every tile to goal by searching backward from goal.

    To get the cost for every possible start in one search, we work backward from the goal.
    A forward step pays for the tile it enters. When searching backward, we add the cost of
    the tile we are leaving.
    """
    costs = {(goal["q"], goal["r"]): 0}
    frontier = [(0, goal["q"], goal["r"])]
    while frontier:
        cost, q, r = heappop(frontier)
        if cost != costs[(q, r)]:
            continue
        entry = _entry_cost(tile.terrain_at(observation, {"q": q, "r": r}))
        for neighbor in tile.neighbors({"q": q, "r": r}).values():
            key = (neighbor["q"], neighbor["r"])
            if tile.terrain_at(observation, neighbor)["terrain"] in {"water", "void"}:
                continue
            if cost + entry < costs.get(key, float("inf")):
                costs[key] = cost + entry
                heappush(frontier, (cost + entry, *key))
    return costs


def _entry_cost(ground):
    cost = 2 if ground["terrain"] == "hill" else 1
    return cost + {"forest": 1, "marsh": 2}.get(ground["feature"], 0)
