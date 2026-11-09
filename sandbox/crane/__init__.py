"""Provided helpers for Skirmish at Crane Reach agents, grouped into small namespaces.

Import the namespaces you need at the top of ``agent.py``, not inside a method::

    from sandbox.crane import action, me, tile, visible

Besides ``sandbox.observation_types``, this package is the only piece of ``sandbox`` you are
meant to use. Importing it does not load the environment engine. Building a forecast loads
the engine when you need it.

- ``action``: read what is legal this turn and build the order your ``act`` returns.
- ``me``: your own unit's fields, including ``direction``, the digit toward the enemy side.
- ``visible``: the units your unit can see, split into enemies and allies.
- ``roster``: both sides' starting rosters, standing knowledge for units you cannot see.
- ``units``: each unit type's fixed hit points, movement, range, damage, and vision.
- ``tile``: hex geometry and the ground, including where a path ends and what terrain is where.
- ``zone``: the capture zones, their tiles, and who is standing in one.
- ``paths``: the stable path encoding, for when you plan a route longer than one step.
- ``forecast``: an independent environment built from your unit's current observation.

The readers use the observation and the authoritative action mask. The forecast applies the
game's rules to orders you supply. No helper picks a strategy, a route, or a tile for you.

There is deliberately no pathfinder. Turning a route toward a distant tile into a legal,
mask-checked order, and re-planning as the battlefield changes, is your own work. The full
observation and action reference lives in ``environment.md``, shipped alongside this template.
"""

from __future__ import annotations

from . import action, forecast, me, paths, roster, tile, units, visible, zone

__all__ = ["action", "forecast", "me", "paths", "roster", "tile", "units", "visible", "zone"]
