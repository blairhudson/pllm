"""Strict public boundary identities and frames; no checkpoint or serving build."""

import hashlib
from dataclasses import FrozenInstanceError, replace

import msgpack
import pytest

from pllm.runtime.private_boundary_protocol import (
    BoundaryAdmission, BoundaryDescriptor, BoundaryError, ROLES, canonical,
    context_bytes, contextual_lookup_digest, pack_frame, parse_control,
    public_lookup_digest, unpack_frame,
)


def descriptor():
    return BoundaryDescriptor(**{name: "a" * 64 for name in (
        "source_digest", "checkpoint_digest", "model_config_digest", "composition_digest",
        "schedule_digest", "head_weight_digest",
        "head_scales_digest", "lookup_digest",
    )}, vocabulary=7, hidden=3, signed_output_bound=3 * 127 * 127)


def test_descriptor_roundtrip_is_strict_and_immutable():
    d = descriptor()
    assert BoundaryDescriptor.from_dict(d.to_dict()) == d
    assert len(d.digest) == 64
    with pytest.raises(FrozenInstanceError):
        d.hidden = 4
    with pytest.raises(BoundaryError):
        BoundaryDescriptor.from_dict({**d.to_dict(), "uncommitted": 1})
    for removed in ("body_fingerprint", "runtime_config_digest", "head_stage_commitment"):
        assert removed not in d.to_dict()
        with pytest.raises(BoundaryError):
            BoundaryDescriptor.from_dict({**d.to_dict(), removed: "a" * 64})
    assert replace(d, schedule_digest="b" * 64).digest != d.digest


@pytest.mark.parametrize("change", [
    {"activation_bits": True}, {"activation_bits": 6}, {"weight_bits": 4},
    {"ring": "prime"}, {"wire_bits": 24}, {"modulus": 1 << 24}, {"leaf_bits": 1},
    {"hidden": False}, {"vocabulary": 0}, {"signed_output_bound": 1 << 31},
    {"source_digest": "A" * 64}, {"head_stage_id": "lm_head/escape"},
])
def test_descriptor_rejects_malformed_profile(change):
    with pytest.raises(BoundaryError):
        replace(descriptor(), **change)


def test_a4_explicit_and_bounded_admission():
    d = replace(descriptor(), activation_bits=4, signed_output_bound=3 * 127 * 7)
    admitted = BoundaryAdmission(d.digest, "b" * 64, 4096, 4096)
    assert BoundaryAdmission.from_dict(admitted.to_dict()) == admitted
    for key, value in (("lookup_rows", 4097), ("head_calls", 0), ("head_calls", True)):
        with pytest.raises(BoundaryError):
            replace(admitted, **{key: value})


def test_context_binds_source_plan_session_epoch_and_same_role_pair():
    d = descriptor()
    a = BoundaryAdmission(d.digest, "b" * 64, 100, 4)
    context = context_bytes(d, a, "c" * 32, 0)
    assert context == context_bytes(d, a, "c" * 32, 0)
    assert context != context_bytes(d, a, "d" * 32, 0)
    assert context != context_bytes(d, a, "c" * 32, 1)
    assert context != context_bytes(d, replace(a, plan_digest="e" * 64), "c" * 32, 0)
    assert context != context_bytes(d, replace(a, lookup_rows=101), "c" * 32, 0)
    changed = replace(d, source_digest="f" * 64)
    assert context != context_bytes(changed, replace(a, descriptor_digest=changed.digest), "c" * 32, 0)
    expected = hashlib.sha256(b"pllm/private-page-context/v1\0" + bytes.fromhex(d.lookup_digest) + context).digest()
    assert contextual_lookup_digest(d, context) == expected
    with pytest.raises(BoundaryError):
        context_bytes(changed, a, "c" * 32, 0)


def test_streaming_lookup_hash_matches_original_native_table():
    from pllm import _native

    rows = [(bytes([i, i + 1, i + 2]), bytes.fromhex("0000803f")) for i in range(7)]
    raw = b"".join(w + s for w, s in rows)
    server = _native.PrivatePageServer(raw, 7, 0)
    assert public_lookup_digest(iter(rows), vocabulary=7, hidden=3) == server.descriptor()[0].hex()
    with pytest.raises(BoundaryError):
        public_lookup_digest(iter(rows[:-1]), vocabulary=7, hidden=3)
    with pytest.raises(BoundaryError):
        public_lookup_digest(iter(rows + rows[:1]), vocabulary=7, hidden=3)


def test_frames_reject_duplicates_extra_fields_wrong_role_and_limits():
    raw = pack_frame(role=ROLES[0], paired_session="c" * 32, epoch=0,
                     sequence=0, operation="lookup", frames=[b"native-only"])
    row = unpack_frame(raw)
    assert row["frames"] == [b"native-only"]
    assert b"token_ids" not in raw and b"activation_scales" not in raw
    for change in ({"role": "worker_a"}, {"sequence": True}, {"epoch": -1},
                   {"frames": [b"x"] * 65}, {"frames": ["plaintext"]},
                   {"operation": "head", "frames": [b"x", b"y"]}, {"clear_ids": [1]}):
        with pytest.raises(BoundaryError):
            unpack_frame(msgpack.packb({**row, **change}, use_bin_type=True))
    duplicate = msgpack.Packer(use_bin_type=True).pack_map_header(8)
    for k, v in [*row.items(), ("sequence", 1)]:
        duplicate += msgpack.packb(k) + msgpack.packb(v, use_bin_type=True)
    with pytest.raises(BoundaryError):
        unpack_frame(duplicate)
    with pytest.raises(BoundaryError):
        unpack_frame(raw, response=True)


def test_control_rejects_duplicate_non_object_and_oversized():
    for raw in (b'{"role":"boundary_a","role":"boundary_b"}', b"[]", b"x" * 16385):
        with pytest.raises(BoundaryError):
            parse_control(raw)
    assert parse_control(canonical({"public": "only"})) == {"public": "only"}
