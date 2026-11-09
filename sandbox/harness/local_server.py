"""Loopback-only HTTP and WebSocket relay for local browser play."""

from __future__ import annotations

import asyncio
import json
import mimetypes
from collections.abc import Sequence
from contextlib import suppress
from http import HTTPStatus
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit

from websockets.datastructures import Headers
from websockets.exceptions import InvalidState
from websockets.frames import OP_TEXT, Frame
from websockets.http11 import Request
from websockets.server import ServerProtocol

from .environment import EnvironmentEntry
from .live_io import parse_commands, session_envelope

WS_PATH = "/api/sessions/local/ws"
ENVIRONMENTS_PATH = "/api/environments"
_MAX_REQUEST_BYTES = 16 * 1024


class _Socket:
    """One upgraded connection, with framing delegated to websockets Sans-I/O."""

    def __init__(self, writer: asyncio.StreamWriter, protocol: ServerProtocol) -> None:
        self._writer = writer
        self._protocol = protocol
        self._closed = False

    async def send(self, line: str) -> None:
        if self._closed:
            raise ConnectionError("socket is closed")
        self._protocol.send_text(line.encode())
        await self.flush()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._protocol.state.name == "OPEN":
            self._protocol.send_close()
            with suppress(ConnectionError, OSError):
                await self.flush()
        self._writer.close()
        with suppress(ConnectionError, OSError):
            await self._writer.wait_closed()

    async def receive(self, data: bytes) -> list[str]:
        self._protocol.receive_data(data)
        received: list[str] = []
        for event in self._protocol.events_received():
            if isinstance(event, Frame) and event.opcode == OP_TEXT and event.fin:
                received.append(bytes(event.data).decode("utf-8"))
        await self.flush()
        return received

    async def eof(self) -> None:
        if not self._closed:
            self._protocol.receive_eof()
            with suppress(ConnectionError, OSError):
                await self.flush()
        self._closed = True

    async def flush(self) -> None:
        for data in self._protocol.data_to_send():
            self._writer.write(data)
        await self._writer.drain()


class LocalServer:
    """Serve one supplied runner command and relay its protocol over a loopback socket.

    ``controller_players`` names the players the local viewer controls (a human-mode session's
    external bindings). Every local socket is that one viewer, so with the set non-empty the relay
    filters each state like the production controller view: a broadcast the environment bounded
    (annotated with its delivered ``recipients``) is shown only when a controlled player sent it or
    received it. An empty set is a watch-style session whose viewer sees every delivered line. The
    live-only ``recipients`` annotation is stripped in both cases; no browser receives it.
    """

    def __init__(
        self,
        entry: EnvironmentEntry,
        *,
        command: Sequence[str],
        static_root: Path | str,
        start_paused: bool = False,
        port: int = 0,
        controller_players: Sequence[str] | None = None,
    ) -> None:
        self._entry = entry
        self._command = list(command)
        self._root = Path(static_root).resolve()
        self._start_paused = start_paused
        self._port = port
        self._controller_players = frozenset(controller_players or ())
        self._server: asyncio.Server | None = None
        self._child: asyncio.subprocess.Process | None = None
        self._output_task: asyncio.Task[None] | None = None
        self._exit_task: asyncio.Task[None] | None = None
        self._sockets: set[_Socket] = set()
        self._header: str | None = None
        self._latest_state: str | None = None
        self._status = "starting"
        self._paused = start_paused
        self._awaiting_start = start_paused
        self._ended = False
        self._end_reason: str | None = None
        self._ended_event = asyncio.Event()

    @property
    def port(self) -> int:
        if self._server is None or not self._server.sockets:
            raise RuntimeError("local server has not started")
        address = self._server.sockets[0].getsockname()
        if not isinstance(address, tuple):
            raise RuntimeError("local server did not bind to a TCP port")
        address_parts = cast(tuple[object, ...], address)
        if len(address_parts) < 2 or not isinstance(address_parts[1], int):
            raise RuntimeError("local server did not bind to a TCP port")
        return address_parts[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/local.html"

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await asyncio.start_server(self._handle_connection, "127.0.0.1", self._port)
        self._child = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._output_task = asyncio.create_task(self._pump_output())
        self._exit_task = asyncio.create_task(self._wait_for_exit())

    async def close(self) -> None:
        if self._child is not None and self._child.returncode is None:
            self._child.terminate()
            await self._child.wait()
        if self._output_task is not None:
            await self._output_task
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        for socket in list(self._sockets):
            await socket.close()
        self._server = None

    async def wait(self) -> None:
        """Wait until the runner produces a terminal result or exits."""
        await self._ended_event.wait()

    async def __aenter__(self) -> LocalServer:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await self._read_request(reader)
            if request is None:
                await self._send_http(writer, HTTPStatus.BAD_REQUEST, b"")
                return
            method, target, _headers, raw = request
            path = urlsplit(target).path
            if path == WS_PATH:
                if method != "GET":
                    await self._send_http(writer, HTTPStatus.METHOD_NOT_ALLOWED, b"", allow="GET")
                    return
                await self._upgrade(reader, writer, raw)
                return
            await self._serve_http(writer, method, path)
        except (ConnectionError, OSError, asyncio.IncompleteReadError):
            return
        finally:
            if not writer.is_closing():
                writer.close()
                with suppress(ConnectionError, OSError):
                    await writer.wait_closed()

    async def _read_request(self, reader: asyncio.StreamReader) -> tuple[str, str, Headers, bytes] | None:
        try:
            raw = await reader.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            return None
        if len(raw) > _MAX_REQUEST_BYTES:
            return None
        lines = raw.decode("iso-8859-1").split("\r\n")
        try:
            request_line, raw_headers = lines[0], lines[1:-2]
            method, target, version = request_line.split(" ")
        except ValueError:
            return None
        if version != "HTTP/1.1":
            return None
        parsed_headers = Headers()
        for raw_header in raw_headers:
            if not raw_header or ":" not in raw_header:
                return None
            name, value = raw_header.split(":", 1)
            if not name or name.strip() != name:
                return None
            parsed_headers[name] = value.strip()
        return method, target, parsed_headers, raw

    async def _serve_http(self, writer: asyncio.StreamWriter, method: str, path: str) -> None:
        if method not in ("GET", "HEAD"):
            await self._send_http(writer, HTTPStatus.METHOD_NOT_ALLOWED, b"", allow="GET, HEAD")
            return
        if path == ENVIRONMENTS_PATH:
            body = json.dumps([self._entry.meta.to_json()], separators=(",", ":")).encode()
            await self._send_http(writer, HTTPStatus.OK, body, "application/json", head=method == "HEAD")
            return
        asset = self._asset(path)
        if asset is None:
            await self._send_http(writer, HTTPStatus.NOT_FOUND, b"", head=method == "HEAD")
            return
        body = asset.read_bytes()
        content_type = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
        await self._send_http(writer, HTTPStatus.OK, body, content_type, head=method == "HEAD")

    async def _upgrade(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, raw: bytes) -> None:
        protocol = ServerProtocol()
        protocol.receive_data(raw)
        events = protocol.events_received()
        if len(events) != 1 or not isinstance(events[0], Request):
            await self._send_http(writer, HTTPStatus.BAD_REQUEST, b"")
            return
        response = protocol.accept(events[0])
        protocol.send_response(response)
        socket = _Socket(writer, protocol)
        try:
            await socket.flush()
            if response.status_code != HTTPStatus.SWITCHING_PROTOCOLS:
                return
            protocol.receive_data(b"")
            protocol.events_received()
            self._sockets.add(socket)
            await self._attach(socket)
            while data := await reader.read(4096):
                for command_raw in await socket.receive(data):
                    await self._forward(command_raw)
        finally:
            self._sockets.discard(socket)
            await socket.eof()

    def _asset(self, raw_path: str) -> Path | None:
        decoded = unquote(raw_path)
        if not decoded.startswith("/") or "\\" in decoded:
            return None
        relative = decoded.removeprefix("/")
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            return None
        candidate = (self._root / relative).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    @staticmethod
    async def _send_http(
        writer: asyncio.StreamWriter,
        status: HTTPStatus,
        body: bytes,
        content_type: str = "text/plain",
        *,
        head: bool = False,
        allow: str | None = None,
    ) -> None:
        headers = [
            f"HTTP/1.1 {status.value} {status.phrase}",
            f"Content-Type: {content_type}",
            f"Content-Length: {len(body)}",
            "Cache-Control: no-store",
            "Connection: close",
        ]
        if allow is not None:
            headers.append(f"Allow: {allow}")
        writer.write(("\r\n".join(headers) + "\r\n\r\n").encode())
        if not head:
            writer.write(body)
        await writer.drain()

    async def _attach(self, socket: _Socket) -> None:
        if self._header is not None:
            await socket.send(self._header)
        if self._latest_state is not None:
            await socket.send(self._latest_state)
        if self._status == "running":
            await socket.send(self._json(session_envelope("running", awaiting_start=self._awaiting_start)))
            if self._paused:
                await socket.send('{"kind":"pause"}')
        elif self._status == "ended":
            await socket.send(self._json(session_envelope("ended", self._end_reason)))

    async def _forward(self, raw: str) -> None:
        child = self._child
        if child is None or child.stdin is None or self._ended:
            return
        for command in parse_commands(raw):
            # Mirrors blockedBeforeStart in schema/ts/src/protocol.ts; keep the two lists identical.
            if self._awaiting_start and command["kind"] in ("input", "chat", "clock"):
                continue
            line = self._json(command)
            child.stdin.write((line + "\n").encode())
            await child.stdin.drain()
            if command["kind"] in ("pause", "resume"):
                self._paused = command["kind"] == "pause"
                if command["kind"] == "resume":
                    self._awaiting_start = False
                await self._broadcast(line)

    async def _pump_output(self) -> None:
        assert self._child is not None and self._child.stdout is not None
        while raw := await self._child.stdout.readline():
            line = raw.decode("utf-8").rstrip("\r\n")
            await self._on_runner_line(line)

    async def _on_runner_line(self, line: str) -> None:
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(value, dict):
            return
        untyped_message = cast(dict[object, Any], value)
        if not all(isinstance(key, str) for key in untyped_message):
            return
        message = cast(dict[str, Any], untyped_message)
        if self._header is None and "kind" not in message:
            self._header = line
            self._status = "running"
            await self._broadcast(line)
            await self._broadcast(
                self._json(session_envelope("running", awaiting_start=self._awaiting_start))
            )
            if self._paused:
                await self._broadcast('{"kind":"pause"}')
            return
        if message.get("kind") == "result":
            await self._broadcast(line)
            await self._finish(str(message.get("reason", "terminated")))
            return
        if "kind" not in message:
            line = self._present_state(line, message)
            self._latest_state = line
            await self._broadcast(line)

    def _present_state(self, line: str, state: dict[str, Any]) -> str:
        """Return the viewer's copy of one state line, annotation-free and controller-filtered.

        A state without messages passes through byte-identical, and so does a watch-style viewer's
        (an empty ``_controller_players``): every message reaches it unchanged, annotation included
        for the recipients case below. For the controller, a targeted message (``to`` a non-null
        string) is kept only when ``to`` or ``from`` names a controlled player. A broadcast (``to``
        is ``None``) carrying no ``recipients`` annotation passes through untouched; one the
        environment bounded is kept only when a controlled player sent it or is in its delivered
        audience, and the annotation itself never reaches the browser either way. The stashed
        catch-up state is the returned line, so a late socket gets the same view.
        """
        messages = state.get("messages")
        if not isinstance(messages, list):
            return line
        kept: list[Any] = []
        changed = False
        for message in cast(list[Any], messages):
            if not isinstance(message, dict):
                kept.append(message)
                continue
            entry = cast(dict[str, Any], message)
            to = entry.get("to")
            if to is not None:
                heard = to in self._controller_players or entry.get("from") in self._controller_players
                if self._controller_players and not heard:
                    changed = True
                    continue
                kept.append(entry)
                continue
            recipients = entry.get("recipients")
            if not isinstance(recipients, list):
                kept.append(entry)
                continue
            changed = True
            heard = entry.get("from") in self._controller_players or any(
                player in self._controller_players for player in cast(list[Any], recipients)
            )
            if self._controller_players and not heard:
                continue
            kept.append({key: value for key, value in entry.items() if key != "recipients"})
        if not changed:
            return line
        presented = dict(state)
        if kept:
            presented["messages"] = kept
        else:
            del presented["messages"]
        return self._json(presented)

    async def _wait_for_exit(self) -> None:
        assert self._child is not None
        code = await self._child.wait()
        if self._output_task is not None:
            await self._output_task
        await self._finish("terminated" if code == 0 else "error")

    async def _finish(self, reason: str) -> None:
        if self._ended:
            return
        self._ended = True
        self._end_reason = reason
        self._status = "ended"
        await self._broadcast(self._json(session_envelope("ended", reason)))
        for socket in list(self._sockets):
            await socket.close()
        self._ended_event.set()

    async def _broadcast(self, line: str) -> None:
        for socket in list(self._sockets):
            try:
                await socket.send(line)
            except (ConnectionError, OSError, InvalidState):
                self._sockets.discard(socket)

    @staticmethod
    def _json(value: dict[str, Any]) -> str:
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
