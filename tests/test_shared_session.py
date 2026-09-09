from __future__ import annotations

import asyncio
import hashlib

import pytest

from pllm.runtime.shared_session import (
    PeerFrame,
    SessionStatus,
    SharedPeerConnection,
    SharedSessionCommitment,
    SharedSessionError,
    SharedSessionRegistry,
    SharedSessionState,
)


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _commitment(
    session_id: str = "session", maximum_operations: int = 2
) -> SharedSessionCommitment:
    return SharedSessionCommitment(
        session_id=session_id,
        model_id="model",
        model_fingerprint=_fingerprint("model"),
        graph_fingerprint=_fingerprint("graph"),
        scale_fingerprint=_fingerprint("scales"),
        party_fingerprints=(_fingerprint("party-0"), _fingerprint("party-1")),
        maximum_operations=maximum_operations,
    )


def _establish(first: SharedSessionState, second: SharedSessionState) -> None:
    first.establish(
        second.commitment,
        local_nonce=b"a" * 32,
        peer_nonce=b"b" * 32,
        authenticated_peer_fingerprint=first.commitment.party_fingerprints[1 - first.party],
    )
    second.establish(
        first.commitment,
        local_nonce=b"b" * 32,
        peer_nonce=b"a" * 32,
        authenticated_peer_fingerprint=second.commitment.party_fingerprints[1 - second.party],
    )


def test_commitment_and_peer_frame_round_trip() -> None:
    commitment = _commitment()
    assert SharedSessionCommitment.unpack(commitment.pack()) == commitment
    frame = PeerFrame("session", b"c" * 32, 0, 7, "multiply.open", "op-1", b"opaque")
    assert PeerFrame.unpack(frame.pack(), max_payload_bytes=16) == frame
    assert len(commitment.digest) == 64


def test_bidirectional_sequences_and_replay_burn_session() -> None:
    first = SharedSessionState(_commitment(), party=0)
    second = SharedSessionState(_commitment(), party=1)
    _establish(first, second)
    outbound = first.outbound("multiply.open", "first-0", b"first")
    second.accept(
        PeerFrame.unpack(outbound.pack(), max_payload_bytes=16), expected_kind="multiply.open"
    )
    reply = second.outbound("multiply.open", "second-0", b"second")
    first.accept(
        PeerFrame.unpack(reply.pack(), max_payload_bytes=16), expected_kind="multiply.open"
    )

    with pytest.raises(SharedSessionError, match="replayed"):
        second.accept(outbound)
    assert second.status is SessionStatus.ABORTED
    with pytest.raises(SharedSessionError, match="replayed"):
        second.outbound("multiply.open", "second-1", b"never sent")


def test_reflected_outbound_frame_is_rejected_by_sender_identity() -> None:
    state = SharedSessionState(_commitment(), party=0)
    peer = SharedSessionState(_commitment(), party=1)
    _establish(state, peer)
    reflected = state.outbound("multiply.open", "reflection", b"opaque")
    with pytest.raises(SharedSessionError, match="sender"):
        state.accept(reflected)


def test_outbound_operation_ids_and_budget_are_single_use() -> None:
    state = SharedSessionState(_commitment(), party=0)
    peer = SharedSessionState(_commitment(), party=1)
    _establish(state, peer)
    state.outbound("multiply.open", "op-1", b"one")
    with pytest.raises(SharedSessionError, match="reused"):
        state.outbound("multiply.open", "op-1", b"two")
    assert state.status is SessionStatus.ABORTED

    budget = SharedSessionState(_commitment("budget"), party=0)
    budget_peer = SharedSessionState(_commitment("budget"), party=1)
    _establish(budget, budget_peer)
    budget.outbound("multiply.open", "op-1", b"one")
    budget.outbound("multiply.open", "op-2", b"two")
    with pytest.raises(SharedSessionError, match="budget"):
        budget.outbound("multiply.open", "op-3", b"three")


def test_registry_expires_and_caps_sessions() -> None:
    current = [10.0]
    registry = SharedSessionRegistry(capacity=1, idle_seconds=5, now=lambda: current[0])
    registry.create(_commitment("first"), party=0)
    with pytest.raises(SharedSessionError, match="capacity"):
        registry.create(_commitment("second"), party=0)
    current[0] = 16.0
    registry.create(_commitment("second"), party=0)
    with pytest.raises(SharedSessionError, match="expired"):
        registry.get("first")


def test_commitment_mismatch_and_malformed_frames_fail_closed() -> None:
    first = _commitment("same")
    second = SharedSessionCommitment(
        session_id="same",
        model_id="model",
        model_fingerprint=first.model_fingerprint,
        graph_fingerprint=_fingerprint("other-graph"),
        scale_fingerprint=first.scale_fingerprint,
        party_fingerprints=first.party_fingerprints,
        maximum_operations=first.maximum_operations,
    )
    assert first.digest != second.digest
    with pytest.raises(SharedSessionError):
        PeerFrame.unpack(b"not-msgpack", max_payload_bytes=16)
    with pytest.raises(SharedSessionError, match="too large"):
        PeerFrame.unpack(
            PeerFrame("same", b"c" * 32, 0, 0, "kind", "op", b"17-bytes-of-value").pack(),
            max_payload_bytes=16,
        )


def test_bidirectional_peer_connection_exchanges_matching_frames() -> None:
    class Socket:
        def __init__(self) -> None:
            self.incoming: asyncio.Queue[bytes] = asyncio.Queue()
            self.peer: Socket | None = None
            self.closed = False

        async def send(self, payload: bytes) -> None:
            assert self.peer is not None
            await self.peer.incoming.put(payload)

        async def recv(self, maximum_bytes: int) -> bytes:
            del maximum_bytes
            return await self.incoming.get()

        async def close(self) -> None:
            self.closed = True

    async def run() -> None:
        first_socket, second_socket = Socket(), Socket()
        first_socket.peer = second_socket
        second_socket.peer = first_socket
        first = SharedPeerConnection(
            SharedSessionState(_commitment(maximum_operations=4), party=0),
            first_socket,
            max_payload_bytes=1024,
            authenticated_peer_fingerprint=_fingerprint("party-1"),
            timeout_seconds=1,
        )
        second = SharedPeerConnection(
            SharedSessionState(_commitment(maximum_operations=4), party=1),
            second_socket,
            max_payload_bytes=1024,
            authenticated_peer_fingerprint=_fingerprint("party-0"),
            timeout_seconds=1,
        )
        first_result, second_result = await asyncio.gather(
            first.exchange("multiply.open", "multiply-0", b"first"),
            second.exchange("multiply.open", "multiply-0", b"second"),
        )
        assert first_result == b"second"
        assert second_result == b"first"
        assert first.stats.sent_frames == first.stats.received_frames == 1
        assert second.stats.sent_frames == second.stats.received_frames == 1
        first_a, first_b, second_b, second_a = await asyncio.gather(
            first.exchange("multiply.open", "a", b"first-a"),
            first.exchange("multiply.open", "b", b"first-b"),
            second.exchange("multiply.open", "b", b"second-b"),
            second.exchange("multiply.open", "a", b"second-a"),
        )
        assert (first_a, first_b, second_b, second_a) == (
            b"second-a",
            b"second-b",
            b"first-b",
            b"first-a",
        )
        await asyncio.gather(first.close(), second.close())
        assert first_socket.closed and second_socket.closed

    asyncio.run(run())


def test_handshake_rejects_commitment_or_authenticated_peer_mismatch() -> None:
    first = SharedSessionState(_commitment("same"), party=0)
    changed = _commitment("different")
    with pytest.raises(SharedSessionError, match="commitment"):
        first.establish(
            changed,
            local_nonce=b"a" * 32,
            peer_nonce=b"b" * 32,
            authenticated_peer_fingerprint=first.commitment.party_fingerprints[1],
        )
    assert first.status is SessionStatus.ABORTED

    identity = SharedSessionState(_commitment("identity"), party=0)
    with pytest.raises(SharedSessionError, match="fingerprint"):
        identity.establish(
            identity.commitment,
            local_nonce=b"a" * 32,
            peer_nonce=b"b" * 32,
            authenticated_peer_fingerprint=_fingerprint("attacker"),
        )


def test_stale_channel_frame_burns_reused_session_identifier() -> None:
    old_first = SharedSessionState(_commitment("reused"), party=0)
    old_second = SharedSessionState(_commitment("reused"), party=1)
    _establish(old_first, old_second)
    stale = old_first.outbound("multiply.open", "old", b"stale")

    new_first = SharedSessionState(_commitment("reused"), party=0)
    new_second = SharedSessionState(_commitment("reused"), party=1)
    new_first.establish(
        new_second.commitment,
        local_nonce=b"c" * 32,
        peer_nonce=b"d" * 32,
        authenticated_peer_fingerprint=new_first.commitment.party_fingerprints[1],
    )
    with pytest.raises(SharedSessionError, match="channel"):
        new_first.accept(stale)
    assert new_first.status is SessionStatus.ABORTED


def test_peer_timeout_aborts_session_and_closes_transport() -> None:
    class StalledSocket:
        def __init__(self) -> None:
            self.closed = False

        async def send(self, payload: bytes) -> None:
            del payload

        async def recv(self, maximum_bytes: int) -> bytes:
            del maximum_bytes
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def close(self) -> None:
            self.closed = True

    async def run() -> None:
        socket = StalledSocket()
        connection = SharedPeerConnection(
            SharedSessionState(_commitment("timeout"), party=0),
            socket,
            max_payload_bytes=1024,
            authenticated_peer_fingerprint=_fingerprint("party-1"),
            timeout_seconds=0.001,
        )
        with pytest.raises(SharedSessionError, match="timed out"):
            await connection.exchange("multiply.open", "stalled", b"payload")
        assert connection.state.status is SessionStatus.ABORTED
        assert socket.closed

    asyncio.run(run())
