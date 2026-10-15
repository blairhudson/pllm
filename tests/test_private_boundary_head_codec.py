"""Bounded synthetic oracles for the production boundary head wire codec."""

import asyncio
import hashlib
import struct
from dataclasses import asdict, replace

import httpx
import numpy as np
import pytest

from pllm import _native
from pllm.runtime.private_boundary_head_codec import DOMAIN, HEADER, HeadOutputCodec
from pllm.runtime.private_boundary_protocol import (
    BoundaryAdmission, BoundaryDescriptor, BoundaryError, ROLES, SCHEMA, canonical,
    pack_frame, unpack_frame,
)
from pllm.runtime.quantization import dequantize_matmul


def descriptor(**changes):
    args = {name: "a" * 64 for name in (
        "source_digest", "checkpoint_digest", "model_config_digest", "composition_digest",
        "schedule_digest", "head_weight_digest", "head_scales_digest", "lookup_digest",
    )}
    args.update(vocabulary=5, hidden=3, signed_output_bound=3 * 128 * 127,
                head_output_encoding="uniform_residues")
    args.update(changes)
    return BoundaryDescriptor(**args)


def args(d):
    return dict(admission=BoundaryAdmission(d.digest, "b" * 64, 1, 4),
                paired_session="c" * 32, epoch=2, sequence=3)


def shares(values):
    # Independent fixed fixture mask. Production input sharing owns fresh RNG.
    b = [((i + 7) * 2654435761) & 0xffffffff for i in range(len(values))]
    a = [(v - mask) & 0xffffffff for v, mask in zip(values, b, strict=True)]
    return [struct.pack("<" + "I" * len(v), *v) for v in (a, b)]


def oracle_pack(raw, width):
    values = struct.unpack("<" + "I" * (len(raw) // 4), raw)
    value = sum((x & ((1 << width) - 1)) << (i * width) for i, x in enumerate(values))
    return value.to_bytes((len(values) * width + 7) // 8, "little")


def encode_pair(codec, raw, kw):
    return [codec.encode(value, role=role, **kw) for value, role in zip(raw, ROLES, strict=True)]


def test_default_digest_and_wire_are_historical():
    d = descriptor(head_output_encoding="u32")
    old = asdict(d)
    del old["head_output_encoding"], old["head_dispatch"], old["head_kernel"]
    expected = {"schema": SCHEMA + ".descriptor", **old}
    assert d.to_dict() == expected
    assert d.digest == hashlib.sha256(b"pllm/private-boundary-descriptor/v1\0" + canonical(expected)).hexdigest()
    assert BoundaryDescriptor.from_dict(expected) == d
    assert BoundaryDescriptor.from_dict({**expected, "head_dispatch": "serial", "head_output_encoding": "u32"}) == d
    assert BoundaryDescriptor.from_dict({**expected, "head_kernel": "existing"}) == d
    codec = HeadOutputCodec(d)
    raw = shares([0, -1, 123, -(1 << 31), (1 << 31) - 1])
    assert encode_pair(codec, raw, args(d)) == raw
    assert codec.decode(*raw, **args(d)).tolist() == [[0, -1, 123, -(1 << 31), (1 << 31) - 1]]
    outer = pack_frame(role=ROLES[0], paired_session="c" * 32, epoch=2, sequence=3,
                       operation="head", frames=[raw[0]], response=True)
    assert unpack_frame(outer, response=True)["frames"] == [raw[0]]


@pytest.mark.parametrize("bits", [4, 8])
def test_geometry_extremes_negative_wrap_and_float32_parity(bits):
    bound = 3 * 128 * ((1 << (bits - 1)) - 1)
    d = descriptor(activation_bits=bits, signed_output_bound=bound)
    codec = HeadOutputCodec(d)
    assert codec.bound == d.hidden * 128 * 127
    assert codec.width == (2 * d.hidden * 128 * 127).bit_length()
    values = [-bound, bound, -1, 0, 1]
    raw = shares(values)
    frames = encode_pair(codec, raw, args(d))
    assert [f[HEADER.size:] for f in frames] == [oracle_pack(r, codec.width) for r in raw]
    result = codec.decode(*frames, **args(d))
    assert result.dtype == np.int64 and result.tolist() == [values]
    scales = np.asarray([0.001, 0.5, 0.03, 1, 0.007], np.float32)
    activation_scale = np.asarray([0.03125], np.float32)
    legacy = HeadOutputCodec(replace(d, head_output_encoding="u32"))
    baseline = legacy.decode(*raw, **args(legacy.descriptor))
    assert dequantize_matmul(result, activation_scale, scales).tobytes() == dequantize_matmul(baseline, activation_scale, scales).tobytes()
    qwen = descriptor(hidden=896, signed_output_bound=896 * 128 * 127)
    assert HeadOutputCodec(qwen).width == 25


@pytest.mark.parametrize("width", range(1, 33))
def test_native_all_supported_widths_against_independent_oracle(width):
    values = [-(1 << (width - 1)), (1 << (width - 1)) - 1, -1, 0, 0]
    raw = shares(values)
    widths = bytes([width]) * len(values)
    packed = [_native.offset_pack_rows(r, widths, 1) for r in raw]
    assert packed == [oracle_pack(r, width) for r in raw]
    reconstructed = _native.offset_reconstruct_rows(*packed, widths, 1)
    assert list(struct.unpack("<5q", reconstructed)) == values
    for bad in (packed[0][:-1], packed[0] + b"\0"):
        with pytest.raises(ValueError):
            _native.offset_reconstruct_rows(bad, packed[1], widths, 1)


@pytest.mark.parametrize("change", [
    {"head_dispatch": "threads"}, {"head_output_encoding": "residues"},
    {"head_dispatch": True}, {"head_output_encoding": 32}, {"activation_bits": 7},
    {"weight_bits": 4}, {"hidden": 0}, {"source_digest": "forged"},
])
def test_forged_descriptor_profiles_rejected(change):
    with pytest.raises(BoundaryError):
        descriptor(**change)


def test_optional_descriptor_fields_strict_and_committed():
    d = descriptor(head_dispatch="parallel")
    assert BoundaryDescriptor.from_dict(d.to_dict()) == d
    assert d.digest != replace(d, head_dispatch="serial").digest
    for extra in ({"width": 1}, {"certificate": {}}, {"quantizer": "self-asserted"}):
        with pytest.raises(BoundaryError):
            BoundaryDescriptor.from_dict({**d.to_dict(), **extra})


def test_replay_substitution_corruption_padding_and_downgrade_rejected():
    d = descriptor()
    codec = HeadOutputCodec(d)
    kw = args(d)
    frames = encode_pair(codec, shares([1, -1, 0, 7, -8]), kw)
    for change in ({"paired_session": "d" * 32}, {"epoch": 3}, {"sequence": 2},
                   {"admission": replace(kw["admission"], plan_digest="d" * 64)}):
        with pytest.raises(BoundaryError):
            codec.decode(*frames, **{**kw, **change})
    for changed in (replace(d, source_digest="f" * 64), replace(d, hidden=4),
                    replace(d, head_dispatch="parallel"), replace(d, activation_bits=4, signed_output_bound=100)):
        with pytest.raises(BoundaryError):
            HeadOutputCodec(changed).decode(*frames, **args(changed))
    with pytest.raises(BoundaryError):
        codec.decode(*reversed(frames), **kw)
    with pytest.raises(BoundaryError):
        codec.decode(*shares([1, -1, 0, 7, -8]), **kw)
    for position in (0, 8, 9, 10, HEADER.size - 1, HEADER.size):
        bad = bytearray(frames[0])
        bad[position] ^= 1
        with pytest.raises(BoundaryError):
            codec.decode(bytes(bad), frames[1], **kw)
    for bad in (frames[0][:-1], frames[0] + b"\0"):
        with pytest.raises(BoundaryError):
            codec.decode(bad, frames[1], **kw)
    # Recompute checksum, proving padding validation independent of checksum.
    magic, party, width, binding, _ = HEADER.unpack_from(frames[0])
    payload = bytearray(frames[0][HEADER.size:])
    assert d.vocabulary * width % 8
    payload[-1] |= 0x80
    bad = HEADER.pack(magic, party, width, binding, hashlib.sha256(DOMAIN + binding + payload).digest()) + payload
    with pytest.raises(BoundaryError, match="padding"):
        codec.decode(bad, frames[1], **kw)


@pytest.mark.parametrize("value", [-48769, 48769])
def test_reconstruction_rejects_outside_admitted_range(value):
    d = descriptor()
    codec = HeadOutputCodec(d)
    with pytest.raises(BoundaryError, match="signed bound"):
        codec.decode(*encode_pair(codec, shares([value] * 5), args(d)), **args(d))


def test_worker_admits_only_committed_encoding_and_returns_exact_residues():
    from test_private_boundary_worker import KEYS, KEY_DIGESTS, app_for, fixture

    d, head, scales, roots = fixture()
    d = replace(d, head_output_encoding="uniform_residues")
    # Factory verifies real immutable weights, scales and bound before admission.
    for changed in (replace(d, signed_output_bound=1), replace(d, head_weight_digest="f" * 64),
                    replace(d, head_scales_digest="f" * 64)):
        with pytest.raises(BoundaryError):
            app_for(changed, head, scales, roots[0], lambda actual, requested: requested)
    if not hasattr(roots[0], "bind_context"):
        pytest.skip("installed native lacks context binding; no fallback")
    admitted = BoundaryAdmission(d.digest, "b" * 64, 1, 1)
    apps = [app_for(d, head, scales, roots[p], lambda actual, requested: admitted, ROLES[p]) for p in (0, 1)]
    sid = "c" * 32

    async def scenario():
        frames = []
        raw = shares([127, -127, 1])
        for p in (0, 1):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(apps[p]), base_url="http://localhost") as client:
                headers = {"authorization": "Bearer " + KEYS[ROLES[p]]}
                row = dict(schema=SCHEMA + ".open", role=ROLES[p], session=sid, epoch=0,
                           admission=admitted.to_dict(), credential_digests=KEY_DIGESTS)
                downgraded = replace(d, head_output_encoding="u32")
                bad = {**row, "admission": replace(admitted, descriptor_digest=downgraded.digest).to_dict()}
                assert (await client.post("/v1/private-boundary/sessions", content=canonical(bad), headers=headers)).status_code == 400
                assert (await client.post("/v1/private-boundary/sessions", content=canonical(row), headers=headers)).status_code == 200
                request = pack_frame(role=ROLES[p], paired_session=sid, epoch=0, sequence=0,
                                     operation="head", frames=[raw[p]])
                reply = await client.post(f"/v1/private-boundary/sessions/{sid}/execute", content=request, headers=headers)
                assert reply.status_code == 200, reply.text
                frames.append(unpack_frame(reply.content, response=True)["frames"][0])
        result = HeadOutputCodec(d).decode(*frames, admission=admitted, paired_session=sid, epoch=0, sequence=0)
        expected = [[sum(int(w) * q for w, q in zip(row, [127, -127, 1], strict=True)) for row in head.weight_view()]]
        assert result.tolist() == expected
        activation_scale = np.asarray([0.03125], np.float32)
        baseline = np.asarray(expected, dtype=np.int64)
        assert dequantize_matmul(result, activation_scale, scales).tobytes() == dequantize_matmul(baseline, activation_scale, scales).tobytes()

    asyncio.run(scenario())
