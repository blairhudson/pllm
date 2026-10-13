"""Independent wire parsing and adversarial publication tests for research codec."""
import hashlib
import io
import json
import struct
import zlib

import pytest

from scripts import probe_artifact_update_delta as probe


def setup_pair(tmp_path, old=None, new=None):
    old = old if old is not None else bytes(range(256)) * 600
    new = new if new is not None else old[:100] + b"changed" + old[107:]
    base, target = tmp_path / "base", tmp_path / "target"
    base.write_bytes(old)
    target.write_bytes(b"existing committed destination")
    rows = probe.descriptor(old), probe.descriptor(new)
    patch = probe.make_patch(old, new, *rows)
    return old, new, base, target, rows, patch


def apply(pair, patch=None, rows=None, digest=None):
    _, _, base, target, original_rows, original_patch = pair
    patch = original_patch if patch is None else patch
    rows = original_rows if rows is None else rows
    return probe.reconstruct(base, io.BytesIO(patch), target, base_row=rows[0],
                             new_row=rows[1], patch_sha256=digest or probe.sha(patch))


@pytest.mark.parametrize("codec", ["xor", "replace"])
@pytest.mark.parametrize("kind", ["sparse", "dense", "dense_low_entropy", "shifted", "append"])
def test_independent_decoder_and_exact_bytes(tmp_path, codec, kind):
    _, old, new = next(row for row in probe.fixtures(333, 131072) if row[0] == kind)
    pair = setup_pair(tmp_path, old, new)
    patch = probe.make_patch(old, new, *pair[4], codec)
    header_size = int.from_bytes(patch[8:12], "big")
    header = json.loads(patch[12:12 + header_size])
    assert header["new"]["content_sha256"] == hashlib.sha256(new).hexdigest()
    cursor, output = 12 + header_size, bytearray()
    while cursor < len(patch):
        tag, size, count = struct.unpack(">BII", patch[cursor:cursor + 9])
        cursor += 9
        base = old[len(output):len(output) + size].ljust(size, b"\0")
        value = base if tag == 0 else zlib.decompress(patch[cursor:cursor + count])
        if tag == 1:
            value = bytes(x ^ y for x, y in zip(value, base))
        output.extend(value)
        cursor += count
    assert bytes(output) == new
    apply(pair, patch)
    assert pair[3].read_bytes() == new
    assert pair[2].read_bytes() == old


@pytest.mark.parametrize("failure", ["base", "truncate", "trailing", "corrupt", "oversize",
                                     "header", "wrong_new", "commitment", "bomb", "copy",
                                     "raw_oversize", "base_oversize", "base_truncated", "scalar_header"])
def test_failures_preserve_base_destination_and_retire_pending(tmp_path, failure):
    pair = setup_pair(tmp_path)
    old, _, base, target, rows, patch = pair
    previous = target.read_bytes()
    digest = None
    cursor = 12 + int.from_bytes(patch[8:12], "big")
    if failure == "base":
        base.write_bytes(b"x" + old[1:])
    elif failure == "base_oversize":
        base.write_bytes(old + b"x")
    elif failure == "base_truncated":
        base.write_bytes(old[:-1])
    elif failure == "truncate":
        patch = patch[:-1]
    elif failure == "trailing":
        patch += b"x"
    elif failure == "corrupt":
        patch = patch[:cursor + 10] + bytes([patch[cursor + 10] ^ 128]) + patch[cursor + 11:]
    elif failure == "oversize":
        patch = patch[:cursor] + struct.pack(">BII", 1, probe.BLOCK, probe.BLOCK + 1025) + patch[cursor + 9:]
    elif failure == "header":
        patch = patch[:8] + struct.pack(">I", probe.MAX_HEADER + 1) + patch[12:]
    elif failure == "wrong_new":
        rows = rows[0], probe.descriptor(b"z" * rows[1]["size"])
    elif failure == "commitment":
        digest = "0" * 64
    elif failure == "bomb":
        encoded = zlib.compress(b"x" * (probe.BLOCK + 1))
        patch = patch[:cursor] + struct.pack(">BII", 2, probe.BLOCK, len(encoded)) + encoded
    elif failure == "copy":
        patch = patch[:cursor] + struct.pack(">BII", 0, probe.BLOCK, 1) + patch[cursor + 9:]
    elif failure == "raw_oversize":
        patch = patch[:cursor] + struct.pack(">BII", 0, probe.BLOCK + 1, 0) + patch[cursor + 9:]
    elif failure == "scalar_header":
        patch = patch[:8] + struct.pack(">I", 1) + b"1" + patch[cursor:]
    before_base = base.read_bytes()
    with pytest.raises((ValueError, zlib.error)):
        apply(pair, patch, rows, digest)
    assert base.read_bytes() == before_base
    assert target.read_bytes() == previous
    assert not list(tmp_path.glob(".delta-*"))


def test_final_content_not_only_patch_digest(tmp_path):
    pair = setup_pair(tmp_path)
    old, _, _, target, rows, _ = pair
    other = b"z" * len(old)
    patch = probe.make_patch(old, other, rows[0], probe.descriptor(other))
    length = int.from_bytes(patch[8:12], "big")
    header = json.loads(patch[12:12 + length])
    header["new"] = rows[1]
    encoded = probe.canonical(header)
    patch = patch[:8] + struct.pack(">I", len(encoded)) + encoded + patch[12 + length:]
    with pytest.raises(ValueError, match="new object mismatch"):
        apply(pair, patch)
    assert target.read_bytes() == b"existing committed destination"


def test_domain_binding_and_publication_alias(tmp_path):
    pair = setup_pair(tmp_path)
    row = dict(pair[4][1], metadata={"wrong": "domain"})
    with pytest.raises(ValueError, match="domain mismatch"):
        apply(pair, rows=(pair[4][0], row))
    with pytest.raises(ValueError, match="publication target"):
        probe.reconstruct(pair[2], io.BytesIO(pair[5]), pair[2], base_row=pair[4][0],
                          new_row=pair[4][1], patch_sha256=probe.sha(pair[5]))
    alias = tmp_path / "alias"
    alias.hardlink_to(pair[2])
    with pytest.raises(ValueError, match="publication target"):
        probe.reconstruct(pair[2], io.BytesIO(pair[5]), alias, base_row=pair[4][0],
                          new_row=pair[4][1], patch_sha256=probe.sha(pair[5]))


def test_atomic_publication_failure(tmp_path, monkeypatch):
    pair = setup_pair(tmp_path)

    def denied(*args):
        raise OSError("publication denied")

    monkeypatch.setattr(probe.os, "replace", denied)
    with pytest.raises(OSError, match="publication denied"):
        apply(pair)
    assert pair[2].read_bytes() == pair[0]
    assert pair[3].read_bytes() == b"existing committed destination"
    assert not list(tmp_path.glob(".delta-*"))


def test_independent_object_domain_identity():
    row = probe.descriptor(b"public bytes", {"dtype": "i1", "source": "public-lock"})
    domain = hashlib.sha256(b'{"dtype":"i1","source":"public-lock"}').digest()
    content = hashlib.sha256(b"public bytes").digest()
    assert row["sha256"] == hashlib.sha256(b"pllm-public-object/1\0" + domain + content).hexdigest()


def test_strongest_control_independent_lengths_and_regression():
    old, new = b"abcd" * 20000, b"\0" * 80000
    values = probe.full_controls(new)
    assert values["zlib-9-1048576"] == len(zlib.compress(new, 9)) + 8
    patch = probe.make_patch(old, new, probe.descriptor(old), probe.descriptor(new))
    assert len(patch) > min(values.values())


def test_short_transport_reads_and_object_limit(tmp_path):
    pair = setup_pair(tmp_path)

    class Short(io.BytesIO):
        def read(self, count=-1):
            return super().read(min(count, 7))

    probe.reconstruct(pair[2], Short(pair[5]), pair[3], base_row=pair[4][0],
                      new_row=pair[4][1], patch_sha256=probe.sha(pair[5]))
    assert pair[3].read_bytes() == pair[1]
    with pytest.raises(ValueError, match="size bound"):
        probe.validate_row(dict(pair[4][1], size=probe.MAX_OBJECT + 1))


def test_deterministic_report_and_accounting():
    first, second = probe.run(), probe.run()
    assert first == second
    for row in first["cases"]:
        assert row["full_bytes"] - row["delta_hit_bytes"] == row["net_hit_saving_bytes"]
        assert row["delta_miss_then_full_bytes"] > row["full_bytes"]
        assert row["strongest_full_bytes"] == min(row["full_controls_bytes"].values())
    assert first["whole_client_low_eligibility"] == "unknown"
