"""Temporary offline material storage controls, not role/transport qualification."""
from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from pllm.runtime import fss_tape as tape


REQUEST = "12" * 32


def make(tmp_path, *, records=2, packet_bytes=32, total_packet_bytes=64,
         request=REQUEST, role="worker_a"):
    return tape.MaterialTape(request, role, records=records, packet_bytes=packet_bytes,
                             total_packet_bytes=total_packet_bytes, directory=tmp_path)


def fill(owner):
    owner.append(b"native-meta-a", b"private-packet-a")
    owner.append(b"native-meta-b", b"private-packet-b")
    owner.seal()


def assert_retired(owner):
    assert owner.closed
    assert owner._file is None
    assert owner._aead is None
    assert owner._binding is None
    owner.close()
    owner.cancel()


def bytes_on_disk(owner):
    handle = owner._file
    offset = handle.tell()
    handle.seek(0)
    raw = handle.read()
    handle.seek(offset)
    return raw


def overwrite(owner, raw):
    owner._file.seek(0)
    owner._file.write(raw)
    owner._file.truncate(len(raw))
    owner._file.seek(0)


def test_roundtrip_private_unlinked_and_exact_receipt(tmp_path):
    owner = make(tmp_path)
    info = os.fstat(owner._file.fileno())
    assert info.st_nlink == 0
    assert info.st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == []
    assert not hasattr(owner, "path") and not hasattr(owner, "key")
    fill(owner)
    raw = bytes_on_disk(owner)
    assert b"private-packet-a" not in raw
    assert b"private-packet-b" not in raw
    before = owner.receipt()
    expected_size = 2 * (24 + 16 + len(b"native-meta-a") + len(b"private-packet-a"))
    assert before["sealed_file_bytes"] == before["encrypted_file_bytes_written"] == expected_size
    assert before["sealed_digest"] == hashlib.sha256(raw).hexdigest()
    assert owner.take(b"native-meta-a") == b"private-packet-a"
    assert owner.take(b"native-meta-b") == b"private-packet-b"
    owner.finish()
    assert_retired(owner)
    receipt = owner.receipt()
    assert receipt["state"] == "finished"
    assert receipt["records_written"] == receipt["records_read"] == 2
    assert receipt["packet_bytes_written"] == receipt["packet_bytes_read"] == 32
    assert receipt["encrypted_file_bytes_read"] == expected_size
    assert "private-packet" not in repr(receipt) and "exception" not in repr(receipt)


def test_pure_pricing_exact_limits_and_copies(monkeypatch):
    def forbidden(*_, **__):
        pytest.fail("pricing must not allocate a file or key")
    monkeypatch.setattr(tape.tempfile, "TemporaryFile", forbidden)
    monkeypatch.setattr(tape.AESGCM, "generate_key", forbidden)
    r = tape.MaterialTape.price(records=4096, packet_bytes=16 << 20,
                                total_packet_bytes=256 << 20)
    assert r.max_file_bytes == (256 << 20) + 4096 * (4096 + 24 + 16)
    assert r.max_file_bytes < 1 << 30
    assert r.required_disk_free_bytes == r.max_file_bytes + tape.DISK_HEADROOM_BYTES
    assert r.max_ciphertext_bytes == (16 << 20) + 16
    assert r.take_buffer_bytes >= 3 * (16 << 20)
    assert r.max_retained_output_bytes == 256 << 20
    assert r.caller_input_bytes == (16 << 20) + 4096
    with pytest.raises((AttributeError, TypeError)):
        r.records = 1
    # Price partial appends too, not only batches that can eventually seal.
    assert tape.MaterialTape.price(records=2, packet_bytes=100, total_packet_bytes=2).max_returned_packet_bytes == 2


@pytest.mark.parametrize("limits", [
    dict(records=4097, packet_bytes=1, total_packet_bytes=4097),
    dict(records=1, packet_bytes=(16 << 20) + 1, total_packet_bytes=1),
    dict(records=1, packet_bytes=1, total_packet_bytes=(256 << 20) + 1),
    dict(records=True, packet_bytes=1, total_packet_bytes=1),
    dict(records=-1, packet_bytes=1, total_packet_bytes=1),
    dict(records=1, packet_bytes=0, total_packet_bytes=1),
    dict(records=2, packet_bytes=1, total_packet_bytes=1),
    dict(records=0, packet_bytes=1, total_packet_bytes=0),
])
def test_invalid_budgets_before_file_or_key(monkeypatch, tmp_path, limits):
    def forbidden(*_, **__):
        pytest.fail("invalid limits reached allocation")
    monkeypatch.setattr(tape.tempfile, "TemporaryFile", forbidden)
    monkeypatch.setattr(tape.AESGCM, "generate_key", forbidden)
    with pytest.raises(tape.MaterialTapeError):
        tape.MaterialTape(REQUEST, "worker_a", directory=tmp_path, **limits)


@pytest.mark.parametrize("request_id,role", [("0" * 64, "worker_a"), ("g" * 64, "worker_a"),
    ("1" * 63, "worker_a"), (REQUEST, "dealer"), (REQUEST, "worker_c")])
def test_request_role_validation_precedes_creation(monkeypatch, tmp_path, request_id, role):
    monkeypatch.setattr(tape.tempfile, "TemporaryFile", lambda **_: pytest.fail("unexpected file"))
    with pytest.raises(tape.MaterialTapeError):
        make(tmp_path, request=request_id, role=role)


@pytest.mark.parametrize("kind", ["large_packet", "empty_packet", "metadata", "empty_metadata", "type", "total", "records"])
def test_append_rejects_budget_before_encrypt_and_burns(tmp_path, kind):
    owner = make(tmp_path, records=1, packet_bytes=4, total_packet_bytes=4)
    if kind == "records":
        owner.append(b"m", b"ok")
    if kind == "total":
        owner.close()
        owner = make(tmp_path, records=2, packet_bytes=4, total_packet_bytes=4)
        owner.append(b"m", b"123")
    calls = []
    owner._aead = SimpleNamespace(encrypt=lambda *_: calls.append(1))
    meta, packet = b"m", b"ok"
    if kind == "large_packet": packet = b"12345"
    if kind == "empty_packet": packet = b""
    if kind == "metadata": meta = b"x" * 4097
    if kind == "empty_metadata": meta = b""
    if kind == "type": packet = bytearray(b"ok")
    with pytest.raises(tape.MaterialTapeError):
        owner.append(meta, packet)
    assert not calls
    assert_retired(owner)


@pytest.mark.parametrize("operation", ["take", "finish", "missing", "append_sealed", "seal_twice"])
def test_phase_failures_burn_owner(tmp_path, operation):
    owner = make(tmp_path)
    with pytest.raises(tape.MaterialTapeError):
        if operation == "take": owner.take(b"m")
        elif operation == "finish": owner.finish()
        elif operation == "missing": owner.append(b"m", b"p"); owner.seal()
        elif operation == "append_sealed": fill(owner); owner.append(b"m", b"p")
        else: fill(owner); owner.seal()
    assert_retired(owner)


@pytest.mark.parametrize("mutation", ["ciphertext", "tag", "reorder", "truncate", "nonce", "metadata", "huge", "empty"])
def test_corrupted_records_authentication_and_size_preflight(tmp_path, mutation):
    owner = make(tmp_path)
    fill(owner)
    raw = bytearray(bytes_on_disk(owner))
    first_len = 24 + len(b"native-meta-a") + len(b"private-packet-a") + 16
    if mutation == "ciphertext": raw[24 + len(b"native-meta-a")] ^= 1
    elif mutation == "tag": raw[first_len - 1] ^= 1
    elif mutation == "reorder": raw = raw[first_len:] + raw[:first_len]
    elif mutation == "truncate": raw = raw[:first_len - 1]
    elif mutation == "nonce": raw[23] ^= 1
    elif mutation == "metadata": raw[24] ^= 1
    elif mutation == "huge": raw[4:12] = (1 << 63).to_bytes(8, "big")
    else: raw[0:4] = (0).to_bytes(4, "big")
    overwrite(owner, raw)
    with pytest.raises(tape.MaterialTapeError):
        owner.take(b"native-meta-a")
    assert_retired(owner)


@pytest.mark.parametrize("substitution", ["role", "request", "metadata", "ordinal"])
def test_aead_binds_session_role_metadata_and_ordinal_even_under_same_test_key(tmp_path, monkeypatch, substitution):
    # Force only the test key to isolate AAD protection from ordinary fresh-key
    # mismatch. No production API exports/reuses this key.
    monkeypatch.setattr(tape.AESGCM, "generate_key", lambda **_: b"k" * 32)
    owner = make(tmp_path)
    fill(owner)
    if substitution == "role":
        other = make(tmp_path, role="worker_b"); fill(other)
        overwrite(owner, bytes_on_disk(other)); other.close()
    elif substitution == "request":
        other = make(tmp_path, request="13" * 32); fill(other)
        overwrite(owner, bytes_on_disk(other)); other.close()
    elif substitution == "metadata":
        # Rewrite stored metadata to match caller while retaining original tag.
        raw = bytearray(bytes_on_disk(owner)); raw[24] = ord("N"); overwrite(owner, raw)
    else:
        raw = bytearray(bytes_on_disk(owner))
        width = len(raw) // 2
        raw = raw[width:] + raw[:width]
        raw[12:24] = bytes(12)  # forge expected first nonce as well
        overwrite(owner, raw)
    expected = b"Native-meta-a" if substitution == "metadata" else b"native-meta-a"
    if substitution == "ordinal": expected = b"native-meta-b"
    with pytest.raises(tape.MaterialTapeError, match="authentication"):
        owner.take(expected)
    assert_retired(owner)


def test_wrong_expected_metadata_replay_unread_and_extra_eof(tmp_path):
    owner = make(tmp_path); fill(owner)
    with pytest.raises(tape.MaterialTapeError): owner.take(b"wrong")
    assert_retired(owner)
    owner = make(tmp_path); fill(owner)
    assert owner.take(b"native-meta-a") == b"private-packet-a"
    with pytest.raises(tape.MaterialTapeError): owner.take(b"native-meta-a")
    assert_retired(owner)
    owner = make(tmp_path); fill(owner)
    with pytest.raises(tape.MaterialTapeError): owner.finish()
    assert_retired(owner)
    owner = make(tmp_path); fill(owner)
    owner._file.seek(0, 2); owner._file.write(b"extra"); owner._file.seek(0)
    owner.take(b"native-meta-a"); owner.take(b"native-meta-b")
    with pytest.raises(tape.MaterialTapeError): owner.finish()
    assert_retired(owner)


def test_returned_packet_owners_survive_retirement_no_resume(tmp_path):
    owner = make(tmp_path); fill(owner)
    first = owner.take(b"native-meta-a")
    second = owner.take(b"native-meta-b")
    owner.finish()
    assert first + second == b"private-packet-aprivate-packet-b"
    for action in [lambda: owner.take(b"native-meta-a"), lambda: owner.append(b"m", b"p"), owner.finish]:
        with pytest.raises(tape.MaterialTapeError): action()
        assert_retired(owner)
    with pytest.raises(TypeError): pickle.dumps(owner)
    assert first == b"private-packet-a"


def test_fresh_keys_and_nonce_sequence(tmp_path):
    a = make(tmp_path); b = make(tmp_path)
    fill(a); fill(b)
    raw_a, raw_b = bytes_on_disk(a), bytes_on_disk(b)
    assert raw_a != raw_b
    first_len = len(raw_a) // 2
    assert raw_a[12:24] == bytes(12)
    assert raw_a[first_len + 12:first_len + 24] == (1).to_bytes(12, "big")
    a.close(); b.close()


def test_disk_headroom_and_creation_failures_cleanup_before_key(tmp_path, monkeypatch):
    key_calls = []
    monkeypatch.setattr(tape.AESGCM, "generate_key", lambda **_: key_calls.append(1))
    monkeypatch.setattr(tape.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(tape.MaterialTapeError, match="headroom"): make(tmp_path)
    assert not key_calls
    monkeypatch.setattr(tape.shutil, "disk_usage", lambda _: (_ for _ in ()).throw(OSError("private-detail")))
    with pytest.raises(tape.MaterialTapeError, match="creation failed"): make(tmp_path)
    assert not key_calls
    monkeypatch.setattr(tape.shutil, "disk_usage", lambda _: SimpleNamespace(free=1 << 40))
    monkeypatch.setattr(tape.tempfile, "TemporaryFile", lambda **_: (_ for _ in ()).throw(OSError("private-detail")))
    with pytest.raises(tape.MaterialTapeError, match="creation failed"): make(tmp_path)
    assert not key_calls


class FileProxy:
    def __init__(self, handle, *, fail=None, partial=False):
        self.handle, self.fail, self.partial = handle, fail, partial
    def __getattr__(self, name): return getattr(self.handle, name)
    def write(self, raw):
        if self.fail == "write": raise OSError("secret-do-not-log")
        return self.handle.write(raw[:3] if self.partial else raw)
    def readinto(self, raw):
        if self.fail == "read": raise OSError("secret-do-not-log")
        return self.handle.readinto(raw[:3] if self.partial else raw)


@pytest.mark.parametrize("failure", ["write", "read", "sync", "encrypt", "decrypt"])
def test_io_and_crypto_errors_release_file_key_without_error_receipts(tmp_path, monkeypatch, failure):
    owner = make(tmp_path)
    handle = owner._file
    if failure == "write": owner._file = FileProxy(handle, fail="write")
    if failure == "encrypt": owner._aead = SimpleNamespace(encrypt=lambda *_: (_ for _ in ()).throw(RuntimeError("secret")))
    with pytest.raises(tape.MaterialTapeError):
        if failure in ("write", "encrypt"): owner.append(b"m", b"p")
        elif failure == "sync":
            owner.append(b"a", b"a"); owner.append(b"b", b"b")
            monkeypatch.setattr(tape.os, "fsync", lambda _: (_ for _ in ()).throw(OSError("secret")))
            owner.seal()
        else:
            fill(owner)
            if failure == "read": owner._file = FileProxy(handle, fail="read")
            else: owner._aead = SimpleNamespace(decrypt=lambda *_: (_ for _ in ()).throw(RuntimeError("secret")))
            owner.take(b"native-meta-a")
    assert handle.closed
    assert_retired(owner)
    assert "secret" not in repr(owner.receipt())


def test_partial_io_is_bounded_and_counted(tmp_path):
    owner = make(tmp_path)
    owner._file = FileProxy(owner._file, partial=True)
    fill(owner)
    owner.take(b"native-meta-a"); owner.take(b"native-meta-b"); owner.finish()
    assert owner.receipt()["encrypted_file_bytes_read"] == owner.receipt()["encrypted_file_bytes_written"]


def test_close_waits_for_inflight_io_then_outputs_remain_owned(tmp_path):
    owner = make(tmp_path, records=1, packet_bytes=1, total_packet_bytes=1)
    handle = owner._file
    entered, release, retired = threading.Event(), threading.Event(), threading.Event()
    class Blocking(FileProxy):
        def write(self, raw):
            entered.set()
            assert release.wait(5)
            return self.handle.write(raw)
    owner._file = Blocking(handle)
    errors = []
    def append():
        try: owner.append(b"m", b"p")
        except BaseException as e: errors.append(type(e))
    writer = threading.Thread(target=append)
    closer = threading.Thread(target=lambda: (owner.close(), retired.set()))
    writer.start(); assert entered.wait(5); closer.start()
    assert not retired.wait(0.05)
    release.set(); writer.join(5); closer.join(5)
    assert not writer.is_alive() and not closer.is_alive() and not errors
    assert retired.is_set() and handle.closed
    assert_retired(owner)


def test_empty_tape_context_cleanup_and_failed_key_creation(tmp_path, monkeypatch):
    with make(tmp_path, records=0, packet_bytes=0, total_packet_bytes=0) as owner:
        owner.seal(); owner.finish()
    assert_retired(owner)
    real = tape.tempfile.TemporaryFile
    handles = []
    def capture(**kwargs):
        result = real(**kwargs); handles.append(result); return result
    monkeypatch.setattr(tape.tempfile, "TemporaryFile", capture)
    monkeypatch.setattr(tape.AESGCM, "generate_key", lambda **_: (_ for _ in ()).throw(RuntimeError("secret")))
    with pytest.raises(tape.MaterialTapeError): make(tmp_path)
    assert handles[0].closed and list(tmp_path.iterdir()) == []


def test_unsafe_original_file_and_copy_attempt_release_owner(tmp_path, monkeypatch):
    named = tmp_path / "not-unlinked"
    handle = named.open("w+b", buffering=0)
    monkeypatch.setattr(tape.tempfile, "TemporaryFile", lambda **_: handle)
    monkeypatch.setattr(tape.AESGCM, "generate_key", lambda **_: pytest.fail("unsafe file reached key generation"))
    with pytest.raises(tape.MaterialTapeError, match="private and unlinked"):
        make(tmp_path)
    assert handle.closed


def test_active_tape_copy_attempt_burns_without_exporting_key(tmp_path):
    owner = make(tmp_path)
    handle = owner._file
    with pytest.raises(TypeError):
        pickle.dumps(owner)
    assert_retired(owner)
    assert handle.closed
