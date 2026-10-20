"""V2 storage saves duplicated metadata only; full original AAD remains required."""
from __future__ import annotations

import hashlib
import struct
import threading
import tracemalloc
from types import SimpleNamespace

import pytest

from pllm.runtime import fss_tape as tape, fss_transport as transport, fss_worker as worker
from test_fss_tape import REQUEST, FileProxy, assert_retired, bytes_on_disk, fill, overwrite
from test_fss_decoder_capacity import CONTRACT, admission
from test_fss_worker import cluster as cluster, double as double
from test_fss_http_lifecycle import (
    test_cancelled_callback_future_keeps_single_live_slot_until_coroutine_drains as _check_callback_drain,
)


def make(tmp_path, **changes):
    options = dict(records=2, packet_bytes=32, total_packet_bytes=64,
                   metadata_storage="external_aad", directory=tmp_path)
    options.update(changes)
    return tape.MaterialTape(REQUEST, "worker_a", **options)


def test_public_qwen_count_price_saves_only_disk_metadata_before_allocation(monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail("pure pricing reached file/key allocation")
    monkeypatch.setattr(tape.tempfile, "TemporaryFile", forbidden)
    monkeypatch.setattr(tape.AESGCM, "generate_key", forbidden)
    limits = dict(records=11_668_738, packet_bytes=tape.MAX_PACKET_BYTES,
        total_packet_bytes=15_206_936_352, metadata_bytes_per_record=1024, capacity="decoder")
    inline = tape.MaterialTape.price(**limits)
    external = tape.MaterialTape.price(**limits, metadata_storage="external_aad")
    assert inline.stored_metadata_bytes_per_record == external.metadata_bytes_per_record == 1024
    assert external.stored_metadata_bytes_per_record == 0
    assert inline.framing_bytes_per_record == external.framing_bytes_per_record == 40
    assert 2 * inline.max_file_bytes == 55_244_947_168
    assert 2 * external.max_file_bytes == 31_347_371_744
    saving = 2 * (inline.max_file_bytes - external.max_file_bytes)
    assert saving == 2 * limits["records"] * 1024 == 23_897_575_424
    # Only the two metadata terms disappear from the supplied topology bound.
    # Live disk availability remains the shared resource owner's responsibility.
    assert 71_527_765_728 - saving == 47_630_190_304
    for field in ("max_ciphertext_bytes", "append_buffer_bytes", "take_buffer_bytes", "max_internal_buffer_bytes",
                  "caller_input_bytes", "max_returned_packet_bytes", "max_retained_output_bytes"):
        assert getattr(inline, field) == getattr(external, field)
    assert external.caller_input_bytes == tape.MAX_PACKET_BYTES + 1024
    assert external.required_disk_free_bytes == external.max_file_bytes + tape.DISK_HEADROOM_BYTES


def test_v2_ceiling_prices_complete_framing_and_ordinals():
    maximum = min((1 << 32) - 1, tape.DECODER_MAX_TOTAL_PACKET_BYTES, tape.DECODER_MAX_FILE_BYTES // 41)
    price = tape.MaterialTape.price(records=maximum, packet_bytes=1, total_packet_bytes=maximum,
                                   capacity="decoder", metadata_storage="external_aad")
    assert price.max_file_bytes == maximum * 41
    assert price.max_file_bytes <= 32 << 30 < price.max_file_bytes + 41
    small = tape.MaterialTape.price(records=1, packet_bytes=1, total_packet_bytes=1,
                                   capacity="decoder", metadata_storage="external_aad")
    assert price.max_internal_buffer_bytes == small.max_internal_buffer_bytes


def test_v2_full_metadata_bound_needs_no_disk_extent_and_empty_tape_still_seals(tmp_path, monkeypatch):
    price = tape.MaterialTape.price(records=1, packet_bytes=1, total_packet_bytes=1,
                                   metadata_storage="external_aad")
    monkeypatch.setattr(tape.shutil, "disk_usage", lambda _: SimpleNamespace(free=price.required_disk_free_bytes))
    with make(tmp_path, records=1, packet_bytes=1, total_packet_bytes=1) as owner:
        metadata = b"m" * 4096
        owner.append(metadata, b"p"); owner.seal()
        assert owner.receipt()["sealed_file_bytes"] == price.max_file_bytes == 41
        assert owner.take(metadata) == b"p"
        owner.finish()
    with make(tmp_path, records=0, packet_bytes=0, total_packet_bytes=0) as owner:
        owner.seal(); owner.finish()
        assert owner.receipt()["encrypted_file_bytes_read"] == owner.receipt()["encrypted_file_bytes_written"] == 0


@pytest.mark.parametrize("bad", [
    {"metadata_storage": None}, {"metadata_storage": True}, {"metadata_storage": "external"},
    {"metadata_bytes_per_record": 0}, {"metadata_bytes_per_record": True}, {"metadata_bytes_per_record": 4097},
    {"records": 1 << 32, "total_packet_bytes": 1 << 32},
    {"records": (32 << 30) // 41 + 1, "packet_bytes": 1, "total_packet_bytes": (32 << 30) // 41 + 1},
    {"packet_bytes": (16 << 20) + 1}, {"total_packet_bytes": (32 << 30) + 1},
    {"records": ((32 << 30) - (2 << 30)) // 40 + 1, "packet_bytes": 16 << 20, "total_packet_bytes": 2 << 30},
])
def test_v2_invalid_policy_and_envelopes_reject_before_file_key(tmp_path, monkeypatch, bad):
    def forbidden(*args, **kwargs): pytest.fail("bad bounds reached file/key allocation")
    monkeypatch.setattr(tape.tempfile, "TemporaryFile", forbidden)
    monkeypatch.setattr(tape.AESGCM, "generate_key", forbidden)
    with pytest.raises(tape.MaterialTapeError): make(tmp_path, capacity="decoder", **bad)


@pytest.mark.parametrize("capacity,mode", [("decoder", "external_aad"), ("capsule", "inline")])
def test_transport_mode_follows_local_capacity_not_metadata_storage_claim(capacity, mode):
    from pllm.runtime.fss_protocol import canonical
    raw = admission(capacity=capacity, primitive_requests_per_party=2,
        cumulative_packet_bytes_per_party_bound=1024, metadata_storage="inline" if capacity == "decoder" else "external_aad")
    price = transport.price_transport(raw, canonical({"capacity": capacity})).tape
    assert price.metadata_storage == mode
    assert price.stored_metadata_bytes_per_record == (0 if capacity == "decoder" else 4096)
    assert price.max_file_bytes == 1024 + 2 * (40 + price.stored_metadata_bytes_per_record)
    other = "capsule" if capacity == "decoder" else "decoder"
    with pytest.raises(ValueError):
        transport.price_transport(raw | {"capacity": other},
                                  CONTRACT if capacity == "decoder" else canonical({"capacity": capacity}))


@pytest.mark.parametrize("mode", ["inline", "external_aad"])
@pytest.mark.parametrize("capacity", ["capsule", "decoder"])
def test_real_completion_framing_follows_local_mode_and_retains_v1_lower_bound(tmp_path, mode, capacity):
    from pllm.runtime import fss_protocol as p
    from pllm.runtime.client import ProtocolError
    with make(tmp_path, records=1, packet_bytes=1, total_packet_bytes=1, metadata_storage=mode) as owner:
        owner.append(b"m", b"p"); owner.seal()
        assert owner.take(b"m") == b"p"
        owner.finish()
        storage = owner.receipt()
    assert storage["encrypted_file_bytes_read"] == (42 if mode == "inline" else 41)
    raw = p.pack({"frame": b"output", "peer_bytes": p.peer_bytes(), "storage": storage})
    if (mode == "external_aad") == (capacity == "decoder"):
        assert transport._receipt(raw, "worker_a", capacity=capacity)[2] == storage
    else:
        with pytest.raises(ProtocolError, match="framing"):
            transport._receipt(raw, "worker_a", capacity=capacity)
    # An unauthenticated storage-mode hint never expands the chosen geometry.
    storage["metadata_storage"] = mode
    with pytest.raises(ProtocolError):
        transport._receipt(p.pack({"frame": b"output", "peer_bytes": p.peer_bytes(), "storage": storage}),
                           "worker_a", capacity=capacity)


def test_v2_stream_uses_full_original_aad_and_constant_resident_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(tape.AESGCM, "generate_key", lambda **_: b"k" * 32)
    count, width = 16385, 24 + 4 + 16
    binding = (b"pllm.fss.offline-material-tape.aesgcm.v2\0" + bytes.fromhex(REQUEST)
               + b"\0" + struct.pack(">IQQ", count, 4, 4 * count))
    aead = tape.AESGCM(b"k" * 32)
    def metadata(i): return b'{"ordinal":' + str(i).encode() + b'}'
    owner = None
    tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]
        owner = make(tmp_path, records=count, packet_bytes=4, total_packet_bytes=4 * count)
        for ordinal in range(count): owner.append(metadata(ordinal), ordinal.to_bytes(4, "big"))
        owner.seal()
        for ordinal in (0, 4095, 4096, count - 1):
            nonce = ordinal.to_bytes(12, "big")
            header = struct.pack(">IQ12s", len(metadata(ordinal)), 20, nonce)
            ciphertext = aead.encrypt(nonce, ordinal.to_bytes(4, "big"),
                binding + ordinal.to_bytes(4, "big") + header + metadata(ordinal))
            owner._file.seek(ordinal * width)
            assert owner._file.read(width) == header + ciphertext
        owner._file.seek(0)
        retained = None
        for ordinal in range(count):
            retained = owner.take(metadata(ordinal))
            assert retained == ordinal.to_bytes(4, "big")
        owner.finish()
        assert retained == (count - 1).to_bytes(4, "big")
        assert tracemalloc.get_traced_memory()[1] - baseline < 128 << 10
    finally:
        tracemalloc.stop()
        if owner is not None: owner.close()
    assert_retired(owner)
    receipt = owner.receipt()
    assert receipt["records_written"] == receipt["records_read"] == count
    assert receipt["encrypted_file_bytes_written"] == receipt["encrypted_file_bytes_read"] == count * width


@pytest.mark.parametrize("expected", [b"native-meta-X", b"short", b"m" * 4096])
def test_wrong_full_caller_metadata_fails_aead_and_burns(tmp_path, expected):
    owner = make(tmp_path); fill(owner)
    with pytest.raises(tape.MaterialTapeError, match="authentication"): owner.take(expected)
    assert_retired(owner)


@pytest.mark.parametrize("operation", ["append", "take"])
@pytest.mark.parametrize("metadata", [b"", bytearray(b"m"), b"m" * 1025])
def test_external_disk_saving_does_not_relax_metadata_input_bound(tmp_path, operation, metadata):
    owner = make(tmp_path, metadata_bytes_per_record=1024)
    if operation == "take": fill(owner)
    def forbidden(*args): pytest.fail("invalid metadata reached AES-GCM")
    owner._aead = SimpleNamespace(encrypt=forbidden, decrypt=forbidden)
    with pytest.raises(tape.MaterialTapeError):
        owner.append(metadata, b"p") if operation == "append" else owner.take(metadata)
    assert_retired(owner)


@pytest.mark.parametrize("target", ["inline", "external_aad"])
def test_cross_mode_substitution_fails_domain_auth_even_after_reframing(tmp_path, monkeypatch, target):
    monkeypatch.setattr(tape.AESGCM, "generate_key", lambda **_: b"k" * 32)
    source = "inline" if target == "external_aad" else "external_aad"
    with make(tmp_path, records=1, metadata_storage=source) as a, make(tmp_path, records=1, metadata_storage=target) as b:
        for owner in (a, b): owner.append(b"m", b"p"); owner.seal()
        raw = bytes_on_disk(a)
        # Normalize layout so only the domain differs. Same key, request, role,
        # limits, header, ordinal and original metadata; tag must still reject.
        forged = raw[:24] + (raw[25:] if source == "inline" else b"m" + raw[24:])
        overwrite(b, forged)
        with pytest.raises(tape.MaterialTapeError, match="authentication"): b.take(b"m")
        assert_retired(b)
        assert a.take(b"m") == b"p"
        a.finish()


@pytest.mark.parametrize("change", [{"role": "worker_b"}, {"request": "13" * 32},
    {"packet_bytes": 33}, {"total_packet_bytes": 63}, {"records": 3}])
def test_v2_binds_immutable_context_and_limits_under_same_test_key(tmp_path, monkeypatch, change):
    monkeypatch.setattr(tape.AESGCM, "generate_key", lambda **_: b"k" * 32)
    options = dict(records=2, packet_bytes=32, total_packet_bytes=64, metadata_storage="external_aad", directory=tmp_path)
    request, role = change.get("request", REQUEST), change.get("role", "worker_a")
    options.update({key: value for key, value in change.items() if key not in ("role", "request")})
    with make(tmp_path) as target, tape.MaterialTape(request, role, **options) as source:
        fill(target)
        source.append(b"native-meta-a", b"private-packet-a")
        source.append(b"native-meta-b", b"private-packet-b")
        if options["records"] == 3: source.append(b"m", b"p")
        source.seal()
        overwrite(target, bytes_on_disk(source))
        with pytest.raises(tape.MaterialTapeError, match="authentication"): target.take(b"native-meta-a")
        assert_retired(target)


@pytest.mark.parametrize("mutation", ["header_truncate", "packet_truncate", "ciphertext", "tag",
    "metadata_length", "zero_metadata", "huge_metadata", "huge_packet", "swap", "replay", "forged_replay_nonce"])
def test_v2_boundary_corruption_replay_and_truncation_remain_one_use(tmp_path, mutation):
    count, boundary, width = 4098, 4096, 41
    owner = make(tmp_path, records=count, packet_bytes=1, total_packet_bytes=count)
    for _ in range(count): owner.append(b"m", b"p")
    owner.seal()
    handle = owner._file
    offset = boundary * width
    handle.seek(offset)
    raw = bytearray(handle.read(2 * width))
    if mutation == "header_truncate": handle.truncate(offset + 23)
    elif mutation == "packet_truncate": handle.truncate(offset + width - 1)
    else:
        if mutation == "ciphertext": raw[24] ^= 1
        elif mutation == "tag": raw[width - 1] ^= 1
        elif mutation == "metadata_length": raw[:4] = (2).to_bytes(4, "big")
        elif mutation == "zero_metadata": raw[:4] = bytes(4)
        elif mutation == "huge_metadata": raw[:4] = (4097).to_bytes(4, "big")
        elif mutation == "huge_packet": raw[4:12] = ((1 << 64) - 1).to_bytes(8, "big")
        elif mutation == "swap": raw = raw[width:] + raw[:width]
        else:
            handle.seek(offset - width)
            raw[:width] = handle.read(width)
            if mutation == "forged_replay_nonce": raw[12:24] = boundary.to_bytes(12, "big")
        handle.seek(offset); handle.write(raw)
    handle.seek(0)
    for _ in range(boundary): assert owner.take(b"m") == b"p"
    sizes, read = [], owner._read_exact
    def observed(size): sizes.append(size); return read(size)
    owner._read_exact = observed
    with pytest.raises(tape.MaterialTapeError): owner.take(b"m")
    assert max(sizes) <= 24  # malformed lengths never allocate attacker-sized buffers
    assert handle.closed
    assert_retired(owner)


@pytest.mark.parametrize("operation", ["take", "finish", "missing", "append_sealed", "seal_twice", "read_twice", "extra_eof"])
def test_v2_order_sealing_and_full_consumption(tmp_path, operation):
    owner = make(tmp_path)
    with pytest.raises(tape.MaterialTapeError):
        if operation == "take": owner.take(b"m")
        elif operation == "finish": owner.finish()
        elif operation == "missing": owner.append(b"m", b"p"); owner.seal()
        else:
            fill(owner)
            if operation == "append_sealed": owner.append(b"m", b"p")
            elif operation == "seal_twice": owner.seal()
            elif operation == "read_twice": owner.take(b"native-meta-a"); owner.take(b"native-meta-a")
            else:
                owner._file.seek(0, 2); owner._file.write(b"extra"); owner._file.seek(0)
                owner.take(b"native-meta-a"); owner.take(b"native-meta-b"); owner.finish()
    assert_retired(owner)


def test_v2_partial_io_exact_counters_and_cancelled_partial_write(tmp_path):
    with make(tmp_path) as owner:
        owner._file = FileProxy(owner._file, partial=True)
        fill(owner)
        raw = bytes_on_disk(owner)
        assert owner.take(b"native-meta-a") == b"private-packet-a"
        assert owner.take(b"native-meta-b") == b"private-packet-b"
        owner.finish()
        receipt = owner.receipt()
        assert receipt["encrypted_file_bytes_read"] == receipt["encrypted_file_bytes_written"] == 112
        assert receipt["sealed_digest"] == hashlib.sha256(raw).hexdigest()
    owner = make(tmp_path)
    handle = owner._file
    class Interrupted(FileProxy):
        def write(self, raw):
            if self.handle.tell() >= 27: raise KeyboardInterrupt()
            return self.handle.write(raw[:3])
    owner._file = Interrupted(handle)
    with pytest.raises(KeyboardInterrupt): owner.append(b"m", b"private-packet")
    assert handle.closed and owner.receipt()["encrypted_file_bytes_written"] == 27
    assert owner.receipt()["records_written"] == 0
    assert_retired(owner)


def test_v2_cancel_drains_take_and_retained_callback_cannot_resume(tmp_path):
    owner = make(tmp_path, records=1, packet_bytes=1, total_packet_bytes=1)
    owner.append(b"m", b"p"); owner.seal()
    callback, handle, aead = owner.take, owner._file, owner._aead
    entered, release, retired = threading.Event(), threading.Event(), threading.Event()
    def decrypt(*args):
        entered.set()
        assert release.wait(5)
        return aead.decrypt(*args)
    owner._aead = SimpleNamespace(decrypt=decrypt)
    results = []
    reader = threading.Thread(target=lambda: results.append(callback(b"m")))
    closer = threading.Thread(target=lambda: (owner.cancel(), retired.set()))
    reader.start()
    try:
        assert entered.wait(5)
        closer.start()
        assert not retired.wait(.02)
    finally:
        release.set(); reader.join(5)
        if closer.ident is not None: closer.join(5)
        owner.close()
    assert not reader.is_alive() and not closer.is_alive()
    assert results == [b"p"] and retired.is_set() and handle.closed
    assert_retired(owner)
    with pytest.raises(tape.MaterialTapeError): callback(b"m")
    assert results == [b"p"]  # immutable returned packet has a separate owner


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [{"capacity": "decoder"}], indirect=True)
async def test_decoder_external_aad_cancel_keeps_callback_payload_and_disk_owners_until_drain(cluster, monkeypatch):
    appended, append = [], worker.MaterialTape.append
    def observe(owner, metadata, packet):
        assert owner.resources.metadata_storage == "external_aad"
        before = owner.receipt()["encrypted_file_bytes_written"]
        append(owner, metadata, packet)
        appended.append(owner.receipt()["encrypted_file_bytes_written"] - before)
        assert appended[-1] == len(packet) + 40
    monkeypatch.setattr(worker.MaterialTape, "append", observe)
    await _check_callback_drain(cluster, monkeypatch)
    assert appended == [len(b"packet") + 40]
