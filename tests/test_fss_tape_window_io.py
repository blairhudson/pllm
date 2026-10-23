"""Real v1/v2 AEAD and bounded I/O controls, not native execution or timing claims."""
from __future__ import annotations

import errno
import os
import sys
import threading
from types import SimpleNamespace

import pytest

from pllm.runtime import fss_tape as tape
from test_fss_tape import FileProxy, REQUEST, assert_retired


pytestmark = pytest.mark.skipif(not hasattr(os, "readv"), reason="OS has no readv")


def make(tmp_path, storage, sizes=(11, 17, 23), *, metadata_size=None):
    rows = tuple((f"context/pc/{i}".encode() if metadata_size is None else
                  bytes([65 + i]) * metadata_size, bytes([97 + i]) * size)
                 for i, size in enumerate(sizes))
    owner = tape.MaterialTape(REQUEST, "worker_a", directory=tmp_path,
        records=len(rows), packet_bytes=max(sizes), total_packet_bytes=sum(sizes),
        metadata_storage=storage)
    for meta, packet in rows:
        owner.append(meta, packet)
    owner.seal()
    return owner, rows


def requests(rows, *, extra=0):
    return tuple((meta, len(packet) + extra) for meta, packet in rows)


def frame_size(row, storage):
    meta, packet = row
    return 24 + (len(meta) if storage == "inline" else 0) + len(packet) + 16


class Counted(FileProxy):
    def __init__(self, handle, *, width=None):
        super().__init__(handle)
        self.reads = []
        self.width = width

    def readinto(self, view):
        self.reads.append(len(view))
        return self.handle.readinto(view if self.width is None else view[:self.width])


def public_progress(owner):
    # Fresh independent keys intentionally produce different file digests.
    return {key: value for key, value in owner.receipt().items() if key != "sealed_digest"}


@pytest.mark.parametrize("storage", ["inline", "external_aad"])
@pytest.mark.parametrize("sizes", [(1,), (11, 17, 23), (8192,) * 8])
def test_exact_final_window_one_real_readv_same_scalar_outputs_and_receipt(tmp_path, monkeypatch, storage, sizes):
    window, rows = make(tmp_path, storage, sizes)
    scalar, originals = make(tmp_path, storage, sizes)
    scalar._file = Counted(scalar._file)
    counter = scalar._file
    real = os.readv
    calls = []

    def readv(fd, views):
        assert len(views) == 1 and type(views[0]) is memoryview
        assert views[0].obj is window._read_window.raw
        assert len(views[0]) == window._sealed_bytes
        assert window.receipt()["records_read"] == window.receipt()["encrypted_file_bytes_read"] == 0
        calls.append(len(views[0]))
        return real(fd, views)

    monkeypatch.setattr(os, "readv", readv)
    try:
        assert window.take_window(requests(rows)) == tuple(packet for _, packet in originals)
        assert tuple(scalar.take(meta) for meta, _ in originals) == tuple(packet for _, packet in rows)
        assert not hasattr(window, "_read_window")
        assert len(calls) == 1
        assert len(counter.reads) == len(rows) * (3 if storage == "inline" else 2)
        assert len(calls) < len(counter.reads)
        assert public_progress(window) == public_progress(scalar)
        window.finish()
        scalar.finish()
        assert public_progress(window) == public_progress(scalar)
    finally:
        window.close()
        scalar.close()


@pytest.mark.parametrize("storage", ["inline", "external_aad"])
def test_nonterminal_window_reads_no_future_byte_or_header_and_never_caches_next_group(tmp_path, monkeypatch, storage):
    owner, rows = make(tmp_path, storage, (4,) * 9)
    handle = owner._file
    real = os.readv
    # Interior windows deliberately stay scalar: no unauthenticated offset walk.
    monkeypatch.setattr(os, "readv", lambda *_: pytest.fail("interior window read ahead"))
    expected_offset = sum(frame_size(row, storage) for row in rows[:8])
    assert owner.take_window(requests(rows[:8])) == tuple(packet for _, packet in rows[:8])
    assert owner._read == handle.tell() == expected_offset
    assert owner._records_read == 8
    # Corrupt next group's header AFTER the earlier callback returned. It must
    # not have been snapshotted, decrypted, consumed or cached by that callback.
    os.pwrite(handle.fileno(), b"\0\0\0\0", expected_offset)
    monkeypatch.setattr(os, "readv", real)
    with pytest.raises(tape.MaterialTapeError, match="record geometry differs"):
        owner.take_window(requests(rows[8:]))
    assert owner._records_read == 8
    assert owner._read == expected_offset + 24
    assert_retired(owner)


@pytest.mark.parametrize("storage", ["inline", "external_aad"])
def test_short_reads_and_eintr_keep_same_os_suffix_and_no_early_cursor_or_auth(tmp_path, monkeypatch, storage):
    owner, rows = make(tmp_path, storage)
    real = os.readv
    offset = 0
    statuses = iter(["interrupt", 1, 23, "interrupt", 7, "rest"])
    aead = owner._aead
    nonces = []

    def decrypt(nonce, ciphertext, aad):
        nonces.append(nonce)
        return aead.decrypt(nonce, ciphertext, aad)

    owner._aead = SimpleNamespace(decrypt=decrypt)

    def short(fd, views):
        nonlocal offset
        assert owner._read == owner._records_read == 0 and not nonces
        assert len(views) == 1 and len(views[0]) == owner._sealed_bytes - offset
        status = next(statuses)
        if status == "interrupt":
            raise InterruptedError(errno.EINTR, "synthetic interruption")
        count = real(fd, views if status == "rest" else (views[0][:status],))
        offset += count
        return count

    monkeypatch.setattr(os, "readv", short)
    assert owner.take_window(requests(rows)) == tuple(packet for _, packet in rows)
    assert offset == owner._sealed_bytes == owner._read
    assert nonces == [i.to_bytes(12, "big") for i in range(len(rows))]
    owner.finish()


@pytest.mark.parametrize("storage", ["inline", "external_aad"])
@pytest.mark.parametrize("fault", ["first_geometry_late_eof", "first_auth_late_geometry",
    "metadata_late_eof", "second_truncated", "long_metadata", "long_ciphertext", "nonce", "reorder"])
def test_fixed_corruptions_match_scalar_first_error_and_consumed_progress(tmp_path, storage, fault):
    outcomes = []
    for coalesced in (False, True):
        owner, rows = make(tmp_path, storage)
        handle = owner._file
        first = frame_size(rows[0], storage)
        if fault == "first_geometry_late_eof":
            os.pwrite(handle.fileno(), bytes(4), 0)
            handle.truncate(first + 3)
        elif fault == "first_auth_late_geometry":
            position = first - 1
            byte = os.pread(handle.fileno(), 1, position)
            os.pwrite(handle.fileno(), bytes([byte[0] ^ 1]), position)
            os.pwrite(handle.fileno(), bytes(4), first)
        elif fault == "metadata_late_eof":
            rows = ((b"wrong/pc/metadata", rows[0][1]),) + rows[1:]
            handle.truncate(first + 3)
        elif fault == "second_truncated":
            handle.truncate(first + frame_size(rows[1], storage) - 1)
        elif fault == "long_metadata":
            os.pwrite(handle.fileno(), (4096).to_bytes(4, "big"), 0)
        elif fault == "long_ciphertext":
            os.pwrite(handle.fileno(), (1 << 63).to_bytes(8, "big"), 4)
        elif fault == "nonce":
            os.pwrite(handle.fileno(), (1).to_bytes(12, "big"), 12)
        else:
            rows = (rows[1], rows[0]) + rows[2:]
        if not coalesced:
            owner._file = Counted(handle)
        with pytest.raises(tape.MaterialTapeError) as caught:
            owner.take_window(requests(rows))
        outcomes.append((str(caught.value), public_progress(owner)))
        assert_retired(owner)
        assert not hasattr(owner, "_read_window")
        with pytest.raises(tape.MaterialTapeError):
            owner.take_window(requests(rows))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("storage", ["inline", "external_aad"])
def test_tamper_between_first_fixed_header_read_and_body_is_not_cached_away(tmp_path, monkeypatch, storage):
    owner, rows = make(tmp_path, storage)
    handle = owner._file
    position = 24 + (len(rows[0][0]) if storage == "inline" else 0)
    original = os.pread(handle.fileno(), 1, position)
    real = os.readv
    calls = []

    def changed(fd, views):
        calls.append(len(views[0]))
        if len(calls) == 1:
            count = real(fd, (views[0][:24],))
            os.pwrite(fd, bytes([original[0] ^ 1]), position)
            return count
        return real(fd, views)

    monkeypatch.setattr(os, "readv", changed)
    with pytest.raises(tape.MaterialTapeError, match="authentication failed"):
        owner.take_window(requests(rows))
    assert len(calls) == 2 and owner._records_read == 0
    assert owner._read == frame_size(rows[0], storage)
    assert_retired(owner)


@pytest.mark.parametrize("storage", ["inline", "external_aad"])
@pytest.mark.parametrize("bad_first", [False, True])
def test_deferred_late_os_error_never_overrides_earlier_auth_failure(tmp_path, monkeypatch, storage, bad_first):
    owner, rows = make(tmp_path, storage)
    handle = owner._file
    first = frame_size(rows[0], storage)
    if bad_first:
        byte = os.pread(handle.fileno(), 1, first - 1)
        os.pwrite(handle.fileno(), bytes([byte[0] ^ 1]), first - 1)
    real = os.readv
    calls = []

    def failing(fd, views):
        calls.append(1)
        if len(calls) == 1:
            return real(fd, (views[0][:first + 7],))
        raise OSError(errno.EIO, "private-not-in-receipt")

    monkeypatch.setattr(os, "readv", failing)
    expected = "authentication failed" if bad_first else "operation failed"
    with pytest.raises(tape.MaterialTapeError, match=expected) as caught:
        owner.take_window(requests(rows))
    assert owner._records_read == (0 if bad_first else 1)
    assert owner._read == first + (0 if bad_first else 7)
    assert caught.value.__cause__ is None
    assert "private" not in repr(owner.receipt())
    assert calls == [1, 1]  # No scalar/seek/suffix retry after the I/O attempt.
    assert_retired(owner)


@pytest.mark.parametrize("storage", ["inline", "external_aad"])
@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit, MemoryError])
def test_partial_unwind_keeps_original_signal_policy_and_completed_record_count(tmp_path, monkeypatch, storage, kind):
    owner, rows = make(tmp_path, storage)
    first = frame_size(rows[0], storage)
    real = os.readv
    calls = []
    original = kind(7)

    def failing(fd, views):
        calls.append(1)
        if len(calls) == 1:
            return real(fd, (views[0][:first + 7],))
        raise original

    monkeypatch.setattr(os, "readv", failing)
    expected = tape.MaterialTapeError if kind is MemoryError else kind
    with pytest.raises(expected) as caught:
        owner.take_window(requests(rows))
    if kind is MemoryError:
        assert str(caught.value) == "material tape operation failed"
    else:
        assert caught.value is original
    assert owner._records_read == 1 and owner._read == first + 7
    assert calls == [1, 1]
    assert not hasattr(owner, "_read_window")
    assert_retired(owner)


@pytest.mark.parametrize("storage", ["inline", "external_aad"])
@pytest.mark.parametrize("result", [0, None, False, -1, 10**9, 1.5])
def test_invalid_read_progress_burns_without_fallback_or_publication(tmp_path, monkeypatch, storage, result):
    owner, rows = make(tmp_path, storage)
    calls = []

    def invalid(*_):
        calls.append(1)
        return result

    monkeypatch.setattr(os, "readv", invalid)
    with pytest.raises(tape.MaterialTapeError, match="truncated"):
        owner.take_window(requests(rows))
    assert calls == [1]
    assert owner._read == owner._records_read == owner._packets_read == 0
    assert_retired(owner)


@pytest.mark.parametrize("storage", ["inline", "external_aad"])
@pytest.mark.parametrize("cancel_at", [0, 1, 3])
def test_cancellation_same_record_boundaries_no_prefix_or_suffix_retry(tmp_path, monkeypatch, storage, cancel_at):
    owner, rows = make(tmp_path, storage)
    real = os.readv
    reads = []
    polls = 0

    def readv(fd, views):
        reads.append(1)
        return real(fd, views)

    def cancelled():
        nonlocal polls
        answer = polls == cancel_at
        polls += 1
        return answer

    monkeypatch.setattr(os, "readv", readv)
    with pytest.raises(tape.MaterialTapeError, match="window retired"):
        owner.take_window(requests(rows), cancelled=cancelled)
    assert owner._records_read == cancel_at
    assert owner._packets_read == sum(len(packet) for _, packet in rows[:cancel_at])
    assert owner._read == sum(frame_size(row, storage) for row in rows[:cancel_at])
    assert len(reads) == (0 if cancel_at == 0 else 1)
    assert_retired(owner)


@pytest.mark.parametrize("storage", ["inline", "external_aad"])
def test_extra_eof_still_fails_original_stream_commitment(tmp_path, storage):
    owner, rows = make(tmp_path, storage)
    os.pwrite(owner._file.fileno(), b"extra", owner._sealed_bytes)
    assert owner.take_window(requests(rows)) == tuple(packet for _, packet in rows)
    with pytest.raises(tape.MaterialTapeError, match="EOF or digest differs"):
        owner.finish()
    assert_retired(owner)


@pytest.mark.parametrize("storage", ["inline", "external_aad"])
def test_scalar_wrappers_and_readv_unavailable_keep_original_path(tmp_path, monkeypatch, storage):
    owner, rows = make(tmp_path, storage)
    owner._file = Counted(owner._file, width=3)
    monkeypatch.setattr(os, "readv", lambda *_: pytest.fail("wrapper was bypassed"))
    assert owner.take_window(requests(rows)) == tuple(packet for _, packet in rows)
    assert len(owner._file.reads) > len(rows) * 3
    owner.finish()
    scalar, rows = make(tmp_path, storage)
    assert tuple(scalar.take(meta) for meta, _ in rows) == tuple(packet for _, packet in rows)
    scalar.finish()
    unvectorized, rows = make(tmp_path, storage)
    monkeypatch.delattr(os, "readv")
    assert unvectorized.take_window(requests(rows)) == tuple(packet for _, packet in rows)
    unvectorized.finish()


@pytest.mark.parametrize("storage", ["inline", "external_aad"])
def test_actual_online_phase_inventory_fits_unchanged_receive_owner(tmp_path, storage):
    owner, rows = make(tmp_path, storage, (65529, 1, 1, 1, 1, 1, 1, 1), metadata_size=4096)
    aead = owner._aead
    observed = []

    def decrypt(nonce, ciphertext, aad):
        packet = aead.decrypt(nonce, ciphertext, aad)
        window = owner._read_window
        returned_and_current = owner._packets_read + len(packet)
        payload = len(window.raw) + len(ciphertext) + returned_and_current + len(aad)
        # Keep the ENTIRE existing metadata/bridge + object staging allowances,
        # rather than charging the blob against unrelated offline/native owners.
        logical = payload + 8192 * 8 + (64 << 10)
        observed.append((logical, sys.getsizeof(window.raw), len(window.raw)))
        assert type(ciphertext) is bytes and type(packet) is bytes
        assert logical < tape.MATERIAL_RECEIVE_PYTHON_BYTES == 655360
        return packet

    owner._aead = SimpleNamespace(decrypt=decrypt)
    assert owner.take_window(requests(rows)) == tuple(packet for _, packet in rows)
    assert len(observed) == 8
    assert all(allocated > payload for _, allocated, payload in observed)
    assert max(payload for _, _, payload in observed) == 65536 + 8 * (40 + (4096 if storage == "inline" else 0))
    # Independent conservative source-phase upper bound, including more AAD
    # temporaries than the captured decrypt call, not quote-versus-same-quote.
    assert 98624 + 65552 + 65536 + 3 * 4096 + 512 + 65536 + 65536 == 373584 < 655360
    owner.finish()


@pytest.mark.parametrize("storage", ["inline", "external_aad"])
def test_close_drains_current_window_io_and_returned_bytes_keep_independent_owner(tmp_path, monkeypatch, storage):
    owner, rows = make(tmp_path, storage)
    entered, release = threading.Event(), threading.Event()
    handle = owner._file
    real = os.readv
    results, errors = [], []

    def blocking(fd, views):
        entered.set()
        assert release.wait(5)
        return real(fd, views)

    def read():
        try:
            results.append(owner.take_window(requests(rows)))
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(os, "readv", blocking)
    reader = threading.Thread(target=read)
    closer = threading.Thread(target=owner.close)
    reader.start()
    try:
        assert entered.wait(5)
        closer.start()
        assert closer.is_alive() and not handle.closed
    finally:
        release.set()
        reader.join(5)
        if closer.ident is not None:
            closer.join(5)
        owner.close()
    assert not errors and not reader.is_alive() and not closer.is_alive()
    assert results == [tuple(packet for _, packet in rows)]
    assert not hasattr(owner, "_read_window")
    assert_retired(owner)
