"""Origin pinning and cancellation retirement for actual peer WebSocket opens."""
import asyncio
from http import HTTPStatus
from types import SimpleNamespace
import time

import pytest
from websockets.asyncio.server import serve
from websockets.exceptions import InvalidStatus

from pllm.runtime import replicated_peer as peer


@pytest.mark.asyncio
async def test_redirect_cannot_forward_directed_credential_to_another_origin():
    received, forwarded = [], []

    async def target(socket):
        forwarded.append(socket.request.headers.get("authorization"))
        await socket.close()

    async def unexpected(socket):
        pytest.fail("redirecting origin upgraded the connection")

    async with serve(target, "127.0.0.1", 0, compression=None) as destination:
        destination_port = destination.sockets[0].getsockname()[1]

        def redirect(connection, request):
            received.append(request.headers.get("authorization"))
            response = connection.respond(HTTPStatus.TEMPORARY_REDIRECT, "do not follow")
            response.headers["Location"] = f"ws://127.0.0.1:{destination_port}/substituted"
            return response

        async with serve(unexpected, "127.0.0.1", 0, process_request=redirect, compression=None) as origin:
            port = origin.sockets[0].getsockname()[1]
            counters = dict.fromkeys((f"{p}_{d}_bytes" for p in ("setup", "online")
                                      for d in ("upload", "download")), 0)
            stream = peer.PeerStream(f"http://127.0.0.1:{port}", "directed-key-12345678",
                {"request": "a" * 64}, time.monotonic() + 5, counters)
            try:
                with pytest.raises(InvalidStatus):
                    await stream.call("advance", b"")
                with pytest.raises(ValueError, match="burned"):
                    await stream.call("advance", b"")
            finally:
                await stream.aclose()
            assert received == ["Bearer directed-key-12345678"]
            assert not forwarded and not any(counters.values()) and stream.socket is None


@pytest.mark.asyncio
async def test_cancelled_handshake_drains_abort_even_under_repeated_cancellation(monkeypatch):
    entered, aborted, lost = asyncio.Event(), asyncio.Event(), asyncio.Event()
    events = []

    class Transport:
        def abort(self):
            events.append("abort")
            aborted.set()

    async def wait_closed():
        await lost.wait()
        events.append("transport_drained")

    class Opening:
        def __init__(self, *args, **kwargs):
            self.connection = SimpleNamespace(transport=Transport(), wait_closed=wait_closed)

        def __await__(self):
            async def wait():
                entered.set()
                await asyncio.Event().wait()
            return wait().__await__()

    monkeypatch.setattr(peer, "_DirectedConnect", Opening)
    opening = asyncio.create_task(peer.directed_connect("ws://127.0.0.1:1"))
    try:
        await entered.wait()
        opening.cancel()
        await aborted.wait()
        opening.cancel()
        await asyncio.sleep(0)
        assert not opening.done() and events == ["abort"]
        lost.set()
        with pytest.raises(asyncio.CancelledError):
            await opening
        assert events == ["abort", "transport_drained"]
    finally:
        lost.set()
        await asyncio.gather(opening, return_exceptions=True)
