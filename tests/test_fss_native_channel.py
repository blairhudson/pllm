"""Native client framing/lifecycle only; synthetic replies are NOT FSS execution."""
import hashlib
import json
import os
import struct

import pytest

native = pytest.importorskip("pllm._native")
pytestmark = pytest.mark.skipif(not hasattr(native, "FssClientRequest"), reason="native channel not built")


def geometry():
    contract = b"{}"  # Opaque prevalidated-contract digest fixture, not a Source contract.
    manifest = json.dumps({
        "contract": hashlib.sha256(contract).hexdigest(), "initial": 2,
        "input_digest": "02" * 32, "profile": "fusefss.gpt2-complete-ring50-q12.v1",
        "program": "03" * 32, "schema": "pllm.fss.channel.program.v1",
        "selections": 2, "source": "04" * 32, "vocabulary": 7,
    }, sort_keys=True, separators=(",", ":")).encode()
    return contract, bytes([5]) * 32, (manifest,) * 3, [3, 4]


def owner():
    return native.FssClientRequest(*geometry())


def sealed_reply(request, role, words=(1, 2)):
    """Unkeyed corruption-check fixture. No native evaluator or authentication."""
    operation = hashlib.sha256(b"pllm.fss.channel.request.v1\0" + request).digest()
    body = (b"PLLMOU01" + operation + bytes([3]) * 32 + bytes([5]) * 32
            + bytes([role, 50, 1, 0]) + struct.pack("<I", len(words))
            + b"".join(struct.pack("<Q", word) for word in words))
    return body + hashlib.sha256(body).digest()


def test_native_client_bounded_one_recipient_exports_and_final_only_list():
    client = owner()
    request = client.request_bytes()
    assert isinstance(request, bytes) and len(request) == 248
    left, right = client.take_input(0), client.take_input(role=1)
    assert isinstance(left, bytes) and len(left) <= 272 and len(right) == len(left)
    assert left[104] == 0 and right[104] == 1
    assert client.finish(sealed_reply(request, 0), sealed_reply(request, 1)) == [2, 4]
    with pytest.raises(Exception):
        client.request_bytes()
    client.close()  # Idempotent cleanup; exported immutable Python buffers remain caller-owned.
    assert left.startswith(b"PLLMIN01")


@pytest.mark.parametrize("change", [
    lambda args: (b"", *args[1:]),
    lambda args: (b"x" * ((1 << 20) + 1), *args[1:]),
    lambda args: (bytearray(args[0]), *args[1:]),
    lambda args: (args[0], b"\0" * 32, *args[2:]),
    lambda args: (args[0], b"x" * 33, *args[2:]),
    lambda args: (*args[:2], list(args[2]), args[3]),
    lambda args: (*args[:2], args[2][:2], args[3]),
    lambda args: (*args[:3], [True, 1]),
    lambda args: (*args[:3], [-1, 1]),
    lambda args: (*args[:3], [1 << 32, 1]),
    lambda args: (*args[:3], [1, 7]),
    lambda args: (*args[:3], [0] * 17),
    lambda args: (*args[:3], []),
    lambda args: (*args[:3], iter([1, 2])),
])
def test_constructor_rejects_before_unbounded_collection_or_rng(change):
    with pytest.raises(Exception):
        native.FssClientRequest(*change(geometry()))


def test_no_user_iteration_or_integer_coercion():
    class Tokens:
        def __iter__(self):
            pytest.fail("must not iterate arbitrary token providers")

    class Integer(int):
        pass

    for value in (Tokens(), [Integer(1), 2]):
        with pytest.raises(Exception):
            native.FssClientRequest(*geometry()[:3], value)


@pytest.mark.parametrize("call", [
    lambda c: c.take_input(True), lambda c: c.take_input(-1), lambda c: c.take_input(2),
    lambda c: c.take_input(), lambda c: c.take_input(0, role=0),
    lambda c: c.request_bytes(1), lambda c: c.request_bytes(unknown=1),
    lambda c: c.finish(), lambda c: c.finish(bytearray(), b""),
    lambda c: c.finish(b"x" * 177, b""),
    lambda c: c.close(1), lambda c: c.cancel(unknown=1),
])
def test_argument_failures_retire_before_parsing(call):
    client = owner()
    with pytest.raises(Exception):
        call(client)
    with pytest.raises(Exception):
        client.request_bytes()


def test_partial_export_replay_cancel_and_freshness():
    first, second = owner(), owner()
    a, b = first.take_input(0), second.take_input(0)
    assert a[112:-32] != b[112:-32]
    with pytest.raises(Exception):
        first.take_input(0)
    with pytest.raises(Exception):
        first.take_input(1)
    second.cancel()
    with pytest.raises(Exception):
        second.take_input(1)
    assert a.startswith(b"PLLMIN01")


@pytest.mark.parametrize("kind", ["role", "context", "checksum", "ring", "vocabulary"])
def test_final_failures_consume_native_client(kind):
    client = owner()
    request = client.request_bytes()
    client.take_input(0)
    client.take_input(1)
    left = sealed_reply(request, 0)
    right = sealed_reply(request, 1)
    if kind == "role":
        right = sealed_reply(request, 0)
    elif kind == "context":
        other = bytearray(request)
        other[40] ^= 1
        right = sealed_reply(bytes(other), 1)
    elif kind == "checksum":
        right = right[:-1] + bytes([right[-1] ^ 1])
    elif kind == "ring":
        right = sealed_reply(request, 1, (1 << 50, 2))
    else:
        right = sealed_reply(request, 1, (6, 6))
    with pytest.raises(Exception):
        client.finish(left, right)
    with pytest.raises(Exception):
        client.take_input(0)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork unavailable")
def test_fork_retires_child_owner_without_retiring_parent():
    client = owner()
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            try:
                client.request_bytes()
            except Exception:
                try:
                    client.take_input(0)
                except Exception:
                    os.write(write_fd, b"1")
        finally:
            os.close(write_fd)
            os._exit(0)
    os.close(write_fd)
    try:
        message = os.read(read_fd, 1)
        status = os.waitpid(pid, 0)[1]
        assert message == b"1"
        assert status == 0
        assert len(client.request_bytes()) == 248
    finally:
        os.close(read_fd)
        client.close()
