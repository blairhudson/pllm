"""Small tape I/O regressions, not trained/native startup or performance tests."""
import errno
import hashlib
from io import FileIO
import os
import struct
import sys
import threading

import pytest

from pllm.runtime import fss_tape as tape


def make(tmp_path, storage="external_aad", records=2):
    return tape.MaterialTape("12" * 32, "worker_a", records=records,
        packet_bytes=128, total_packet_bytes=128 * records,
        metadata_bytes_per_record=32, metadata_storage=storage, directory=tmp_path)


def contents(owner):
    handle = owner._file
    offset = handle.tell()
    handle.seek(0)
    raw = handle.read()
    handle.seek(offset)
    return raw


def retired(owner, handle):
    assert owner.closed and handle.closed
    assert owner._file is owner._aead is owner._binding is owner._write_hash is None
    with pytest.raises(tape.MaterialTapeError):
        owner.append(b"no-retry", b"no-suffix")


@pytest.mark.skipif(not hasattr(os, "writev"), reason="OS has no vectored write")
@pytest.mark.parametrize("storage", ["inline", "external_aad"])
def test_actual_writev_one_call_per_record_original_objects_framing_nonce_and_seal(tmp_path, monkeypatch, storage):
    owner = make(tmp_path, storage)
    real_writev, real_sync = os.writev, os.fsync
    calls, encrypted, syncs = [], [], []
    aead = owner._aead
    class Cipher:
        def encrypt(self, nonce, packet, aad):
            result = aead.encrypt(nonce, packet, aad)
            encrypted.append((nonce, packet, aad, result))
            return result
        def decrypt(self, *args):
            return aead.decrypt(*args)
    owner._aead = Cipher()
    def writev(fd, views):
        assert type(owner._file) is FileIO
        assert len(views) == (3 if storage == "inline" else 2)
        assert all(type(view) is memoryview for view in views)
        assert views[-1].obj is encrypted[-1][3]  # No concatenated/copied body.
        calls.append(tuple(view.obj for view in views))
        return real_writev(fd, views)
    def sync(fd):
        syncs.append((fd, owner.receipt()["records_written"], owner._state))
        return real_sync(fd)
    monkeypatch.setattr(os, "writev", writev)
    monkeypatch.setattr(os, "fsync", sync)
    rows = [(b"metadata-a", b"private-a"), (b"metadata-b", b"private-b")]
    try:
        for meta, packet in rows:
            owner.append(meta, packet)
        assert len(calls) == 2 and not syncs
        raw = contents(owner)
        offset = 0
        for ordinal, ((meta, packet), (nonce, clear, aad, ciphertext), buffers) in enumerate(zip(rows, encrypted, calls)):
            header = tape._HEADER.pack(len(meta), len(packet) + 16, ordinal.to_bytes(12, "big"))
            assert nonce == ordinal.to_bytes(12, "big") and clear is packet
            assert aad == owner._aad(ordinal, header, meta)
            assert buffers[0] == header
            if storage == "inline":
                assert buffers[1] is meta
            expected = header + (meta if storage == "inline" else b"") + ciphertext
            assert raw[offset:offset + len(expected)] == expected
            offset += len(expected)
        assert offset == len(raw) == owner._written == owner._file.tell()
        owner.seal()
        assert syncs == [(owner._file.fileno(), 2, "writing")]
        assert owner.receipt()["sealed_digest"] == hashlib.sha256(raw).hexdigest()
        for meta, packet in rows:
            assert owner.take(meta) == packet
        owner.finish()
        assert owner.receipt()["records_read"] == owner.receipt()["records_written"] == 2
    finally:
        owner.close()


@pytest.mark.parametrize("storage", ["inline", "external_aad"])
@pytest.mark.parametrize("width", [1, 23, 24, 25, 31, 8192])
def test_partial_vectors_across_all_boundaries_same_bytes_hash_and_cursor(tmp_path, monkeypatch, storage, width):
    owner = make(tmp_path, storage)
    header, meta, ciphertext = b"h" * 24, b"metadata", b"ciphertext-and-tag" * 3
    buffers = (header, meta, ciphertext) if storage == "inline" else (header, ciphertext)
    expected = b"".join(buffers)
    progress = []
    def short(fd, views):
        assert all(any(view.obj is part for part in buffers) for view in views)
        offered = b"".join(views)  # Test oracle only; production creates no copy.
        assert offered == expected[owner._written:]
        count = os.write(fd, offered[:width])
        progress.append(count)
        return count
    monkeypatch.setattr(os, "writev", short, raising=False)
    try:
        owner._write_record(header, meta if storage == "inline" else None, ciphertext)
        assert contents(owner) == expected
        assert owner._written == owner._file.tell() == sum(progress) == len(expected)
        assert owner._write_hash.hexdigest() == hashlib.sha256(expected).hexdigest()
        assert owner._records_written == owner._packets_written == 0
    finally:
        owner.close()


@pytest.mark.parametrize("storage", ["inline", "external_aad"])
def test_eintr_retries_same_suffix_without_nonce_or_record_advance(tmp_path, monkeypatch, storage):
    owner = make(tmp_path, storage, records=1)
    statuses = iter(["interrupt", 4, "interrupt", 1, "rest"])
    progress, nonces = [], []
    aead = owner._aead
    class Cipher:
        def encrypt(self, nonce, packet, aad):
            nonces.append(nonce)
            return aead.encrypt(nonce, packet, aad)
        def decrypt(self, *args):
            return aead.decrypt(*args)
    owner._aead = Cipher()
    def interrupted(fd, views):
        assert owner._records_written == owner._packets_written == 0
        assert owner._written == sum(progress)
        status = next(statuses)
        if status == "interrupt":
            raise InterruptedError(errno.EINTR, "test interruption")
        raw = b"".join(views)
        count = os.write(fd, raw if status == "rest" else raw[:status])
        progress.append(count)
        return count
    monkeypatch.setattr(os, "writev", interrupted, raising=False)
    try:
        owner.append(b"meta", b"packet")
        assert nonces == [bytes(12)]
        assert owner._records_written == 1 and owner._packets_written == 6
        owner.seal()
        assert owner.take(b"meta") == b"packet"
        owner.finish()
    finally:
        owner.close()


@pytest.mark.parametrize("result", [0, None, False, -1, 10**9, 1.5])
def test_invalid_progress_burns_without_record_commit_or_fallback(tmp_path, monkeypatch, result):
    owner = make(tmp_path)
    handle, calls = owner._file, []
    def invalid(fd, views):
        calls.append(1)
        return result
    monkeypatch.setattr(os, "writev", invalid, raising=False)
    monkeypatch.setattr(owner, "_write_all", lambda *_: pytest.fail("no fallback after vector attempt"))
    with pytest.raises(tape.MaterialTapeError, match="write failed"):
        owner.append(b"meta", b"packet")
    assert calls == [1]
    assert owner.receipt()["records_written"] == owner.receipt()["packet_bytes_written"] == 0
    assert owner.receipt()["encrypted_file_bytes_written"] == 0
    retired(owner, handle)


@pytest.mark.parametrize("error", [OSError(errno.EIO, "private-error"), MemoryError("private-error"),
    tape.MaterialTapeError("fixed-original"), KeyboardInterrupt(), SystemExit(7)])
def test_partial_error_retains_completed_records_burns_key_and_preserves_public_error_policy(tmp_path, monkeypatch, error):
    owner = make(tmp_path)
    owner.append(b"first", b"accepted-first")
    before, handle = owner.receipt(), owner._file
    calls = []
    def failing(fd, views):
        calls.append(1)
        if len(calls) == 1:
            return os.write(fd, views[0][:3])
        raise error
    monkeypatch.setattr(os, "writev", failing, raising=False)
    with pytest.raises(BaseException) as caught:
        owner.append(b"second", b"uncommitted-second")
    if isinstance(error, (tape.MaterialTapeError, KeyboardInterrupt, SystemExit)):
        assert caught.value is error
    else:
        assert type(caught.value) is tape.MaterialTapeError
        assert str(caught.value) == "material tape operation failed" and caught.value.__cause__ is None
    after = owner.receipt()
    assert after["records_written"] == before["records_written"] == 1
    assert after["packet_bytes_written"] == before["packet_bytes_written"]
    assert after["encrypted_file_bytes_written"] == before["encrypted_file_bytes_written"] + 3
    assert after["sealed_digest"] is None and "private-error" not in repr(after)
    retired(owner, handle)


def test_private_helper_rethrows_original_os_error_and_never_retransmits_suffix(tmp_path, monkeypatch):
    owner = make(tmp_path)
    original = OSError(errno.ENOSPC, "private-detail")
    calls = []
    def fail(fd, views):
        calls.append(1)
        if len(calls) == 1:
            return os.write(fd, views[0][:5])
        raise original
    monkeypatch.setattr(os, "writev", fail, raising=False)
    try:
        with pytest.raises(OSError) as caught:
            owner._write_record(b"header", None, b"ciphertext")
        assert caught.value is original and calls == [1, 1]
        assert contents(owner) == b"heade" and owner._written == 5
        assert owner._write_hash.hexdigest() == hashlib.sha256(b"heade").hexdigest()
    finally:
        owner.close()  # Only append's public operation wrapper owns fatal retirement.


def test_hash_error_accounts_entire_accepted_vector_before_burning(tmp_path, monkeypatch):
    owner = make(tmp_path)
    handle, accepted = owner._file, []
    class BrokenHash:
        def update(self, _):
            raise MemoryError("private hash failure")
    owner._write_hash = BrokenHash()
    def write(fd, views):
        count = os.write(fd, b"".join(views))
        accepted.append(count)
        return count
    monkeypatch.setattr(os, "writev", write, raising=False)
    with pytest.raises(tape.MaterialTapeError, match="operation failed"):
        owner.append(b"meta", b"packet")
    assert owner._written == sum(accepted) > 24
    assert owner._records_written == owner._packets_written == 0
    retired(owner, handle)


class Proxy:
    def __init__(self, file):
        self.file, self.calls = file, []
    def __getattr__(self, name):
        return getattr(self.file, name)
    def write(self, raw):
        self.calls.append(len(raw))
        if len(self.calls) == 1:
            raise InterruptedError(errno.EINTR, "test")
        return self.file.write(raw[:3])


@pytest.mark.parametrize("storage", ["inline", "external_aad"])
@pytest.mark.parametrize("wrapper", [False, True])
def test_portable_fallback_and_wrappers_keep_exact_legacy_partial_write_path(tmp_path, monkeypatch, storage, wrapper):
    owner = make(tmp_path, storage, records=1)
    segments = []
    original = owner._write_all
    def scalar(raw):
        segments.append(raw)
        return original(raw)
    monkeypatch.setattr(owner, "_write_all", scalar)
    if wrapper:
        owner._file = Proxy(owner._file)
        monkeypatch.setattr(os, "writev", lambda *_: pytest.fail("wrapper bypassed"), raising=False)
    else:
        monkeypatch.delattr(os, "writev", raising=False)
    try:
        owner.append(b"meta", b"packet")
        assert len(segments) == (3 if storage == "inline" else 2)
        raw = contents(owner)
        assert raw == b"".join(segments)
        assert owner._written == len(raw)
        owner.seal()
        assert owner.take(b"meta") == b"packet"
        owner.finish()
    finally:
        owner.close()


def test_close_waits_for_vector_completion_without_early_refund_or_record_commit(tmp_path, monkeypatch):
    owner = make(tmp_path, records=1)
    handle = owner._file
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    failures = []
    def blocking(fd, views):
        entered.set()
        assert release.wait(5)
        return os.write(fd, b"".join(views))
    monkeypatch.setattr(os, "writev", blocking, raising=False)
    def append():
        try:
            owner.append(b"meta", b"packet")
        except BaseException as error:
            failures.append(error)
    writer = threading.Thread(target=append)
    closer = threading.Thread(target=lambda: (owner.close(), closed.set()))
    try:
        writer.start()
        assert entered.wait(5)
        closer.start()
        assert not closed.wait(.05) and not handle.closed
        assert owner._records_written == owner._packets_written == 0
    finally:
        release.set()
        writer.join(5)
        if closer.ident is not None:
            closer.join(5)
        owner.close()
    assert not writer.is_alive() and not closer.is_alive() and not failures
    assert owner.receipt()["records_written"] == 1
    assert owner.receipt()["sealed_digest"] is None and closed.is_set()
    retired(owner, handle)


def test_fixed_control_inventory_stays_within_existing_append_staging_quote():
    # Enumerated shallow controls + CPython public Py_buffer / POSIX iovec ABI.
    # Sanity check only: NOT complete stack/allocator/crypto/RSS or aggregate proof.
    views = [memoryview(b"h" * 24), memoryview(b"meta"), memoryview(b"ciphertext")]
    tail, hash_slice = views[0][1:], views[0][:1]
    controls = (sys.getsizeof(views) + sum(map(sys.getsizeof, views))
                + sys.getsizeof(tail) + sys.getsizeof(hash_slice)
                + 8 * sys.getsizeof(tape.MAX_PACKET_BYTES)
                + 3 * (struct.calcsize("@PPnniiPPPPP") + struct.calcsize("@Pn")))
    r = tape.MaterialTape.price(records=2, packet_bytes=128, total_packet_bytes=256,
                                metadata_bytes_per_record=32)
    assert controls < tape._BUFFER_SLACK == 64 << 10
    assert r.append_buffer_bytes == r.max_ciphertext_bytes + 2 * r.metadata_bytes_per_record + tape._BUFFER_SLACK
