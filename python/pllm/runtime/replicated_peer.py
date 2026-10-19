"""Bounded, one-use directed peer RPC streams (no detached send queue).

Each direction owns one socket, one in-flight request/response and at most one
waiting control/data caller (no queued serialization). WebSocket
message buffers and serialization copies are covered by STREAM_MEMORY per worker,
in addition to the ordinary native/pending-frame admission. Deployments must also
set the server's WebSocket max_size and max_queue to FRAME_LIMIT and 1.
"""
from __future__ import annotations

import asyncio
import time

from websockets.asyncio.client import connect

from .delegated_protocol import pack, unpack
from .replicated_protocol import FRAME_LIMIT, STREAM_MEMORY

TRANSPORT = "peer-stream-v1"
# Six directed endpoints: receive queue, wire/decoded message, serialization and
# transport write buffers. Deliberately conservative; not a native capacity bump.


class _DirectedConnect(connect):
    def process_redirect(self, error):
        # Credentials belong to the exact admitted IP origin. Never follow even
        # same-origin redirects; in particular, don't forward headers to DNS or
        # another role. websockets 15 follows redirects by default.
        return error


async def directed_connect(*args, **kwargs):
    opening = _DirectedConnect(*args, **kwargs)
    try:
        return await opening
    except BaseException:
        connection = getattr(opening, "connection", None)
        if connection is not None:
            connection.transport.abort()
            draining = asyncio.create_task(connection.wait_closed())
            while not draining.done():
                try:
                    await asyncio.shield(draining)
                except asyncio.CancelledError:
                    continue
            draining.result()
        raise


def binding(request, source, configuration, sender, recipient, epochs):
    return {"transport": TRANSPORT, "request": request, "source": source,
            "configuration": configuration, "sender": sender, "recipient": recipient,
            "sender_epoch": epochs[sender], "recipient_epoch": epochs[recipient]}


def decode(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= FRAME_LIMIT:
        raise ValueError("replicated stream message exceeds admission")
    return unpack(raw, limit=FRAME_LIMIT)


class PeerStream:
    """Serialized RPC; caller holds attempt admission until aclose drains sends."""

    def __init__(self, origin, key, hello, expires, counters, connector=directed_connect):
        self.url = ("wss://" + origin[8:] if origin.startswith("https://")
                    else "ws://" + origin[7:]) + f"/v1/replicated/{hello['request']}/stream"
        self.key, self.hello, self.expires = key, hello, expires
        self.counters, self.connector = counters, connector
        self.lock = asyncio.Lock()
        self.socket = None
        self.sequence = 0
        self.closed = False
        self.caller = None
        self.users = 0
        self.close_task = None

    async def exchange(self, value, phase):
        raw = pack(value, limit=FRAME_LIMIT)
        await self.socket.send(raw)
        self.counters[f"{phase}_upload_bytes"] += len(raw)
        raw = await self.socket.recv()
        if type(raw) is bytes:
            self.counters[f"{phase}_download_bytes"] += len(raw)
        return decode(raw)

    async def call(self, action, payload):
        if self.closed:
            raise ValueError("replicated stream is burned")
        # Coordinator control and its local frozen data wave share a direction.
        # Permit both concurrently, not an unbounded queue of payload-owning tasks.
        if self.users >= 2:
            self.closed = True
            raise ValueError("replicated stream call queue exceeds admission")
        self.users += 1
        try:
            async with self.lock:
                if self.closed:
                    raise ValueError("replicated stream is burned")
                self.caller = asyncio.current_task()
                try:
                    if self.expires <= time.monotonic():
                        raise TimeoutError("replicated stream expired")
                    async with asyncio.timeout(max(0.0, self.expires - time.monotonic())):
                        if self.socket is None:
                            self.socket = await self.connector(self.url,
                                additional_headers={"authorization": f"Bearer {self.key}"},
                                max_size=FRAME_LIMIT, max_queue=1, write_limit=FRAME_LIMIT,
                                compression=None, proxy=None, open_timeout=60, close_timeout=5,
                                ping_interval=None)
                            if self.closed:
                                # Even a connector that finishes during cancellation
                                # cannot send a new handshake after retirement.
                                raise ValueError("replicated stream is burned")
                            if await self.exchange(self.hello, "setup") != {"accepted": True}:
                                raise ValueError("replicated stream binding differs")
                        sequence = self.sequence
                        if sequence >= 1 << 63:
                            raise ValueError("replicated stream sequence exhausted")
                        self.sequence += 1
                        phase = "setup" if action in {"public", "receive_seed"} else "online"
                        result = await self.exchange({"sequence": sequence, "action": action,
                                                      "payload": payload}, phase)
                        if (set(result) != {"sequence", "result"} or type(result["sequence"]) is not int
                                or result["sequence"] != sequence or type(result["result"]) is not dict):
                            raise ValueError("replicated stream response sequence differs")
                        return result["result"]
                except BaseException:
                    self.closed = True
                    raise
                finally:
                    self.caller = None
        finally:
            self.users -= 1

    async def aclose(self):
        if self.close_task is None:
            self.closed = True
            self.close_task = asyncio.create_task(self._close())
        cancelled = False
        while not self.close_task.done():
            try:
                await asyncio.shield(self.close_task)
            except asyncio.CancelledError:
                cancelled = True
        self.close_task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _close(self):
        self.closed = True
        # Close first to interrupt a blocked recv; lock then proves send retirement.
        if self.caller is not None and self.caller is not asyncio.current_task():
            self.caller.cancel()
        first = self.socket
        try:
            if first is not None:
                await self._close_socket(first)
        finally:
            async with self.lock:
                # A cancelled connector may have completed while cleanup waited
                # for the caller. That socket is still owned and must drain too.
                try:
                    if self.socket is not None and self.socket is not first:
                        await self._close_socket(self.socket)
                finally:
                    self.socket = None

    @staticmethod
    async def _close_socket(socket):
        try:
            await socket.close()
        except BaseException:
            transport = getattr(socket, "transport", None)
            if transport is not None:
                transport.abort()
                # abort schedules connection_lost; do not release the request's
                # wire-buffer charge before that transport callback has drained.
                await socket.wait_closed()
            raise
