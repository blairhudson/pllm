from __future__ import annotations

import asyncio

from websockets.asyncio.client import ClientConnection, connect

from .preparation_protocol import PreparationAck, validate_attempt_id
from .protocol import ProtocolError


CORRECTION_CHANNEL_PATH = "/v1/he/corrections/ws"
CORRECTION_CHANNEL_SUBPROTOCOL = "pllm-correction-v1"
CORRECTION_CHANNEL_MAX_ACK_BYTES = 1024


class CorrectionChannelError(ProtocolError):
    pass


class CorrectionWebSocketClient:
    """Persistent correction sender with asynchronous acknowledgements."""

    def __init__(
        self,
        url: str,
        api_key: str,
        *,
        timeout: float,
        max_payload_bytes: int,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.timeout = float(timeout)
        self.max_payload_bytes = int(max_payload_bytes)
        self._connect_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._socket: ClientConnection | None = None
        self._receiver: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[bytes]] = {}
        self._closed = False

    async def _connect(self) -> ClientConnection:
        async with self._connect_lock:
            if self._closed:
                raise CorrectionChannelError("correction channel is closed")
            if self._socket is not None:
                return self._socket
            try:
                socket = await connect(
                    self.url,
                    subprotocols=[CORRECTION_CHANNEL_SUBPROTOCOL],
                    additional_headers={"Authorization": f"Bearer {self.api_key}"},
                    compression=None,
                    proxy=None,
                    open_timeout=self.timeout,
                    close_timeout=self.timeout,
                    max_size=None,
                )
            except Exception as exc:
                raise CorrectionChannelError(f"correction channel connection failed: {exc}") from exc
            self._socket = socket
            self._receiver = asyncio.create_task(self._receive_loop(socket))
            return socket

    async def _receive_loop(self, socket: ClientConnection) -> None:
        error: CorrectionChannelError | None = None
        try:
            while True:
                raw = await socket.recv(decode=False)
                if not isinstance(raw, bytes) or len(raw) > CORRECTION_CHANNEL_MAX_ACK_BYTES:
                    raise CorrectionChannelError("invalid correction acknowledgement")
                acknowledgement = PreparationAck.unpack(raw)
                future = self._pending.pop(acknowledgement.attempt_id, None)
                if future is None:
                    raise CorrectionChannelError("unsolicited correction acknowledgement")
                if not future.done():
                    future.set_result(raw)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            error = CorrectionChannelError(f"correction channel failed: {exc}")
        finally:
            if self._socket is socket:
                self._socket = None
                self._receiver = None
            pending = tuple(self._pending.values())
            self._pending.clear()
            if error is not None:
                for future in pending:
                    if not future.done():
                        future.set_exception(error)

    async def push(
        self,
        session_id: str,
        attempt_id: str,
        payload: bytes,
        *,
        wait_for_ack: bool = False,
    ) -> tuple[bytes, int]:
        validate_attempt_id(attempt_id)
        if not session_id or not payload or len(payload) > self.max_payload_bytes:
            raise CorrectionChannelError("correction payload is invalid")
        socket = await self._connect()
        future = asyncio.get_running_loop().create_future()
        if attempt_id in self._pending:
            raise CorrectionChannelError("correction attempt is already in flight")
        self._pending[attempt_id] = future
        try:
            async with self._send_lock:
                if self._socket is not socket:
                    raise CorrectionChannelError("correction channel disconnected before send")
                await socket.send(payload)
            if not wait_for_ack:
                future.add_done_callback(
                    lambda done: None if done.cancelled() else done.exception()
                )
                return b"", len(payload)
            acknowledgement = await asyncio.wait_for(asyncio.shield(future), self.timeout)
            return acknowledgement, len(payload)
        except asyncio.CancelledError:
            self._pending.pop(attempt_id, None)
            await self._discard(socket)
            raise
        except TimeoutError as exc:
            self._pending.pop(attempt_id, None)
            await self._discard(socket)
            raise CorrectionChannelError("correction acknowledgement timed out") from exc
        except Exception as exc:
            self._pending.pop(attempt_id, None)
            await self._discard(socket)
            if isinstance(exc, CorrectionChannelError):
                raise
            raise CorrectionChannelError("correction channel send was ambiguous") from exc

    async def _discard(self, socket: ClientConnection) -> None:
        if self._socket is socket:
            self._socket = None
        receiver = self._receiver
        self._receiver = None
        if receiver is not None and receiver is not asyncio.current_task():
            receiver.cancel()
        try:
            await socket.close()
        except Exception:
            pass

    async def close(self) -> None:
        self._closed = True
        socket = self._socket
        receiver = self._receiver
        self._socket = None
        self._receiver = None
        if receiver is not None:
            receiver.cancel()
        if socket is not None:
            await socket.close()
