"""The recording store: the save and load interface for JSONL recordings.

A recording is a header line followed by one per-step state per line, the same
line-delimited JSON the harness streams over its transport during a live session, so
the wire form and the stored form are one format.

The protocol deliberately names only ids and streams, never filesystem types, so an
S3-compatible store can be added behind it later as a purely additive change.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from ..schema import (
    SCHEMA_VERSION,
    SchemaValidationError,
    validate_header,
    validate_step,
)
from ..state import RecordingHeader, StepState


class RecordingError(Exception):
    """Raised when a recording cannot be read coherently (for example a version mismatch)."""


class RecordingWriter(AbstractContextManager["RecordingWriter"], Protocol):
    """A live recording being written. Used as a context manager.

    ``write_step`` validates the state, appends exactly one JSONL line, and flushes on
    every write, so a crashed session leaves a readable prefix. ``live_state``, when given,
    is a live-presentation variant of the same step handed to the store's mirror seam in
    place of the persisted line; it is never written to the recording.
    """

    def write_step(self, state: StepState, live_state: StepState | None = None) -> None: ...


class Recording(Protocol):
    """An opened recording: a validated header plus a lazy iterator of validated states."""

    @property
    def header(self) -> RecordingHeader: ...

    def steps(self) -> Iterator[StepState]: ...


@runtime_checkable
class RecordingStore(Protocol):
    """Save and load recordings by id."""

    def create(
        self, recording_id: str, header: RecordingHeader
    ) -> AbstractContextManager[RecordingWriter]: ...

    def open(self, recording_id: str) -> Recording: ...

    def list_ids(self) -> list[str]: ...


def check_header(header: RecordingHeader) -> None:
    """Validate a header and enforce that its version matches this reader."""
    validate_header(header)
    if header["schema_version"] != SCHEMA_VERSION:
        raise RecordingError(
            f"recording header schema_version {header['schema_version']} "
            f"does not match reader version {SCHEMA_VERSION}"
        )


def check_step(state: StepState, header_version: int) -> None:
    """Validate a state line and enforce that its version matches the header's."""
    validate_step(state)
    if state["schema_version"] != header_version:
        raise RecordingError(
            f"state line schema_version {state['schema_version']} "
            f"does not match header version {header_version}"
        )


__all__ = [
    "Recording",
    "RecordingError",
    "RecordingStore",
    "RecordingWriter",
    "SchemaValidationError",
]
