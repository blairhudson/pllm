"""Seeded scalar/window differentials with real AES, no native material/jobs.

Only synthetic opaque packets and canonical metadata are used. This checks tape
binding and logical progress, not worker/native context authority or performance.
"""
from contextlib import ExitStack
import json
import os
import random
from types import SimpleNamespace

import pytest

from pllm.runtime import fss_tape as tape
from test_fss_tape import REQUEST, assert_retired


pytestmark = pytest.mark.skipif(not hasattr(os, "readv"), reason="OS has no readv")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def workload(seed):
    rng = random.Random(seed)
    count, tail = 23 + seed % 4, 2 + seed % 7
    sizes = [rng.choice((1, 2, 15, 16, 17, 63, 64, 255, 256, 513, 1024)) for _ in range(count)]
    sizes[:3], sizes[-3:] = (1, 17, 1024), (1, 17, 513)
    rows = tuple((canonical({"ordinal": i, "pc": i, "role": 0,
                            "context": {"session": "a" * 64, "operation": "b" * 64},
                            "pad": "κ" * rng.randrange(24)}), rng.randbytes(size))
                 for i, size in enumerate(sizes))
    widths, remaining = [], count - tail
    while remaining:
        width = rng.randint(1, min(8, remaining))
        widths.append(width)
        remaining -= width
    return rows, (*widths, tail)


def make(tmp_path, storage, rows, *, request=REQUEST, role="worker_a"):
    owner = tape.MaterialTape(request, role, records=len(rows),
        packet_bytes=max(len(packet) for _, packet in rows),
        total_packet_bytes=sum(len(packet) for _, packet in rows),
        metadata_storage=storage, directory=tmp_path)
    try:
        for meta, packet in rows:
            owner.append(meta, packet)
        owner.seal()
        return owner
    except BaseException:
        owner.close()
        raise


def frame_size(row, storage):
    # Literal v1/v2 framing oracle, not the production header/quote helper.
    meta, packet = row
    return 24 + (len(meta) if storage == "inline" else 0) + len(packet) + 16


def requests(rows):
    return tuple((meta, len(packet)) for meta, packet in rows)


def take(owner, rows, coalesced):
    if coalesced:
        return owner.take_window(requests(rows))
    # Original public scalar reader, not _take_locked or the new window helper.
    return tuple(owner.take(meta) for meta, _ in rows)


def progress(owner):
    return {k: v for k, v in owner.receipt().items() if k != "sealed_digest"}


@pytest.mark.parametrize("storage", ["inline", "external_aad"], ids=["v1", "v2"])
@pytest.mark.parametrize("seed", [2, 17, 91])
def test_random_partitions_and_short_final_io_match_original_scalar(tmp_path, monkeypatch, storage, seed):
    rows, widths = workload(seed)
    real_readv = os.readv
    rng = random.Random(seed ^ 0x712)
    reads, interrupted = [], False
    with ExitStack() as stack:
        scalar = stack.enter_context(make(tmp_path, storage, rows))
        window = stack.enter_context(make(tmp_path, storage, rows))

        def short(fd, views):
            nonlocal interrupted
            assert fd == window._file.fileno()
            assert window._records_read == len(rows) - widths[-1]
            assert len(views) == 1 and views[0].obj is window._read_window.raw
            reads.append(len(views[0]))
            if not interrupted:
                interrupted = True
                raise InterruptedError()
            return real_readv(fd, (views[0][:rng.randint(1, 257)],))

        monkeypatch.setattr(os, "readv", short)
        offset, retained = 0, []
        for width in widths:
            group = rows[offset:offset + width]
            actual = take(window, group, True)
            assert actual == take(scalar, group, False) == tuple(packet for _, packet in group)
            retained.append(actual)
            offset += width
            assert window._read == scalar._read == sum(frame_size(row, storage) for row in rows[:offset])
            assert progress(window) == progress(scalar)
            if offset < len(rows):
                assert not reads  # No read-ahead into a later logical window.
        assert len(reads) > 2 and reads[0] == reads[1]  # EINTR repeats only untouched OS suffix.
        assert all(a > b for a, b in zip(reads[1:], reads[2:]))
        window.finish()
        scalar.finish()
        assert progress(window) == progress(scalar)
        assert tuple(packet for group in retained for packet in group) == tuple(packet for _, packet in rows)
        assert not hasattr(window, "_read_window")
        assert_retired(window)


@pytest.mark.parametrize("storage", ["inline", "external_aad"], ids=["v1", "v2"])
@pytest.mark.parametrize("seed", [2, 17])
@pytest.mark.parametrize("fault", ["pc", "context-aad", "role-aad", "request-aad"])
@pytest.mark.parametrize("later", ["geometry", "eof"])
def test_random_final_error_winner_preserves_previous_windows_and_consumed_cursors(
        tmp_path, monkeypatch, storage, seed, fault, later):
    # Test-only equal keys isolate full AAD binding from fresh-key mismatch.
    # Never export or reuse production keys. The original owner's binding stays
    # unchanged; role/request attacks transplant a foreign owner's ciphertext.
    monkeypatch.setattr(tape.AESGCM, "generate_key", lambda **_: b"k" * 32)
    rows, widths = workload(seed)
    prefix = len(rows) - widths[-1]
    bad = prefix + random.Random(seed ^ 0x34).randrange(1, widths[-1] - 1)
    starts = [sum(frame_size(row, storage) for row in rows[:i]) for i in range(len(rows))]
    outcomes = []
    for coalesced in (False, True):
        with ExitStack() as stack:
            owner = stack.enter_context(make(tmp_path, storage, rows))
            handle = owner._file
            retained, offset = [], 0
            for width in widths[:-1]:
                group = rows[offset:offset + width]
                retained.append(take(owner, group, coalesced))
                offset += width
            assert offset == prefix
            assert tuple(p for group in retained for p in group) == tuple(p for _, p in rows[:prefix])
            requested = list(rows[prefix:])
            if fault in ("pc", "context-aad"):
                value = json.loads(rows[bad][0])
                if fault == "pc":
                    value["pc"] += 100_000
                else:
                    value["context"]["operation"] = "c" * 64
                meta = canonical(value)
                requested[bad - prefix] = (meta, rows[bad][1])
                if fault == "context-aad" and storage == "inline":
                    # Even matching rewritten inline bytes cannot replace the
                    # original authenticated context. Ciphertext/tag unchanged.
                    assert len(meta) == len(rows[bad][0])
                    assert os.pwrite(handle.fileno(), meta, starts[bad] + 24) == len(meta)
            else:
                foreign = stack.enter_context(make(tmp_path, storage, rows,
                    request="13" * 32 if fault == "request-aad" else REQUEST,
                    role="worker_b" if fault == "role-aad" else "worker_a"))
                size = frame_size(rows[bad], storage)
                raw = os.pread(foreign._file.fileno(), size, starts[bad])
                assert len(raw) == size
                assert os.pwrite(handle.fileno(), raw, starts[bad]) == size
            if later == "geometry":
                assert os.pwrite(handle.fileno(), bytes(4), starts[-1]) == 4
            else:
                handle.truncate(starts[-1] + 23)

            decrypts, reads = [], []
            cipher, real_readv = owner._aead, os.readv

            def decrypt(nonce, ciphertext, aad):
                decrypts.append(int.from_bytes(nonce, "big"))
                return cipher.decrypt(nonce, ciphertext, aad)

            def readv(fd, views):
                reads.append(len(views[0]))
                return real_readv(fd, views)

            owner._aead = SimpleNamespace(decrypt=decrypt)
            result = None
            with monkeypatch.context() as scoped:
                scoped.setattr(os, "readv", readv)
                with pytest.raises(tape.MaterialTapeError) as caught:
                    result = take(owner, tuple(requested), coalesced)
                assert result is None  # No new window prefix escapes.
                metadata_error = fault == "pc" and storage == "inline"
                expected = ("material tape metadata differs" if metadata_error
                            else "material tape authentication failed")
                assert str(caught.value) == expected and caught.value.__cause__ is None
                assert owner._records_read == bad
                assert owner._packets_read == sum(len(packet) for _, packet in rows[:bad])
                consumed_bad = (24 + len(rows[bad][0]) if metadata_error
                                else frame_size(rows[bad], storage))
                assert owner._read == starts[bad] + consumed_bad
                assert decrypts == list(range(prefix, bad + (not metadata_error)))
                assert bool(reads) == coalesced
                assert_retired(owner)
                assert handle.closed and not hasattr(owner, "_read_window")
                before = (progress(owner), len(reads), tuple(decrypts))
                # Retry the failed callback AND its otherwise-valid suffix:
                # neither can do I/O/authentication or publish another packet.
                for retry in (tuple(requested), rows[bad + 1:]):
                    with pytest.raises(tape.MaterialTapeError):
                        take(owner, retry, coalesced)
                assert (progress(owner), len(reads), tuple(decrypts)) == before
                assert tuple(p for group in retained for p in group) == tuple(p for _, p in rows[:prefix])
                outcomes.append((expected, progress(owner), tuple(decrypts)))
    assert outcomes[0] == outcomes[1]
