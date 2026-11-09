"""Run one local browser-play episode through the injected, relocated harness.

The loopback relay launches this module instead of asking the harness to discover installed Python
entry points. Student repositories ship their own environment copy, so this shim builds the one
``EnvironmentEntry`` explicitly and then hands every transport, timeout, recording, and state concern
to the production ``sandbox.harness.live.run`` function.
"""

from __future__ import annotations

import sys

from sandbox.env import META, default_action, extract_overlay, extract_overlay_static, make_env
from sandbox.harness.clock import SystemClock
from sandbox.harness.environment import EnvironmentEntry
from sandbox.harness.live import LiveConfigError, _claim_stdout, parse_config, run
from sandbox.harness.live_io import (
    PausableClock,
    ProtocolStream,
    RealSleeper,
    SessionControl,
    build_tee_store,
)

ENTRY = EnvironmentEntry(
    meta=META,
    make=make_env,
    default_action=default_action,
    overlay=extract_overlay,
    overlay_static=extract_overlay_static,
)


def main(argv: list[str] | None = None) -> int:
    """Parse the normal live config and execute it against this template's injected environment."""
    protocol = ProtocolStream(_claim_stdout())
    try:
        config = parse_config(list(sys.argv[1:] if argv is None else argv), entry=ENTRY)
    except LiveConfigError as error:
        print(f"live_local: invalid config: {error}", file=sys.stderr, flush=True)
        return 2

    clock = PausableClock(SystemClock())
    control = SessionControl(clock)
    return run(
        ENTRY,
        config,
        protocol=protocol,
        control=control,
        clock=clock,
        sleeper=RealSleeper(),
        store=build_tee_store(config.recording_dir, protocol),
        command_lines=sys.stdin,
    )


if __name__ == "__main__":
    raise SystemExit(main())
