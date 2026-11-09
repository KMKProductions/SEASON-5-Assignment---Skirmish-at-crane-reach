"""Template tests. Every composed example inherits these and must pass them in CI.

They encode what every submittable repo should satisfy: the manifest parses and names a
loadable, instantiable agent class with the required interface, and that agent can actually
drive a few steps of the synced environment headlessly. They pass on the bare template because
it ships a small working starting agent, so a fresh clone is green out of the box; they keep
gating composed examples in CI and a student's own edits locally.
"""

from __future__ import annotations

import json
from pathlib import Path

from sandbox.env import META, make_env
from sandbox.harness.environment import resolve_parameters
from sandbox.play import load_agent, play_episode

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_manifest_parses_and_names_a_loadable_class():
    manifest = json.loads((REPO_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"entry_point", "class_name", "template_version"}
    assert isinstance(manifest["template_version"], int)
    agent = load_agent(REPO_ROOT)
    assert agent is not None


def test_three_step_headless_episode_runs():
    agent = load_agent(REPO_ROOT)
    env = make_env(resolve_parameters(META))
    try:
        score = play_episode(agent, env, seed=0, max_steps=3)
    finally:
        env.close()
    assert isinstance(score, float)
