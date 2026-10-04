import asyncio
import hashlib
from dataclasses import replace

import httpx
import msgpack
import numpy as np
import pytest

import pllm
from pllm import _native
from pllm.profiles import TwoOnlineOffsetCpu
from pllm.protocols import TwoOnlineOffsetLinear
from pllm.quantization import SymmetricPerRow
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.offset_codec import SEED_MAGIC, context, pack_seed, seed_header, unpack_seed
from pllm.runtime.offset_worker import create_offset_worker_app
from pllm.runtime.stage_protocol import MaskedStageRequest, MaskedStageResponse, ProtocolError
from pllm.runtime.transformer_client import ClientBundle
from test_offset_worker import _fixture, _request, _session_body


@pytest.mark.parametrize("bits", [16, 24, 32])
def test_native_seeded_split_is_exact_and_bound(bits):
    value = np.arange(-128, 128, dtype=np.int8)
    seed, binding = bytes(range(32)), b"a" * 32
    a = np.frombuffer(
        _native.offset_seeded_share(seed, binding, len(value), bits, value.tobytes()), dtype="<u4"
    )
    b = np.frombuffer(_native.offset_seeded_share(seed, binding, len(value), bits), dtype="<u4")
    np.testing.assert_array_equal(
        (a.astype(np.uint64) + b) % (1 << bits), value.astype(np.int64) % (1 << bits)
    )
    assert _native.offset_seeded_share(seed, b"b" * 32, len(value), bits) != b.tobytes()
    with pytest.raises(ValueError):
        _native.offset_seeded_share(seed, binding, 4_000_001, bits)


def test_seeded_encoding_is_opt_in_and_compiler_bound():
    raw = TwoOnlineOffsetLinear()
    assert dict(raw.params) == {}
    assert raw.with_params(input_encoding="seeded").params["input_encoding"] == "seeded"
    with pytest.raises(ValueError):
        TwoOnlineOffsetLinear(input_encoding="public")
    with pytest.raises(ValueError, match="requires seeded"):
        TwoOnlineOffsetLinear(dispatch="seed_first")
    assert dict(TwoOnlineOffsetLinear(dispatch="sequential").params) == {}
    concurrent = TwoOnlineOffsetLinear(input_encoding="seeded", dispatch="seed_first")
    assert concurrent.params["dispatch"] == "seed_first"


@pytest.mark.parametrize("count", [1, 31, 32, 33, 257])
def test_batched_aes_counter_encoding_matches_independent_cipher(count):
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    seed, binding, bits = bytes(range(32)), b"j" * 32, 24
    key = hashlib.sha256(
        b"pllm/offset-input-share/aes256/v1\0"
        + seed
        + binding
        + count.to_bytes(8, "little")
        + bytes([bits])
    ).digest()
    encryptor = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    counters = b"".join(
        counter.to_bytes(8, "little") + bytes(8) for counter in range((count + 3) // 4)
    )
    encrypted = encryptor.update(counters) + encryptor.finalize()
    expected = (np.frombuffer(encrypted, dtype="<u4")[:count] & 0xFFFFFF).tobytes()
    assert _native.offset_seeded_share(seed, binding, count, bits) == expected


@pytest.mark.parametrize(
    "role,selected", [("worker_b", True), ("worker_a", True), ("worker_b", False)]
)
def test_seed_request_role_scope_replay_and_exact_worker_output(tmp_path, role, selected):
    base, workers = _fixture(tmp_path)
    worker = workers[0]
    pipeline = TwoOnlineOffsetCpu(
        pllm.Model("offset-model"),
        quantization=SymmetricPerRow(weight_bits=4, activation_bits=4),
        linear=TwoOnlineOffsetLinear(input_encoding="seeded" if selected else "raw"),
    )
    bundle = ClientBundle.unpack(worker.client_bundle("offset-model", placement="offset"))
    compiled = compile_runtime_model(base._plan, bundle, composition=pipeline)
    key = "a" * 32

    async def scenario():
        app = create_offset_worker_app(
            worker, model_id="offset-model", role_id=role, api_key=key, composition=pipeline
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://worker"
        ) as client:
            auth = {"authorization": f"Bearer {key}"}
            response = await client.post(
                "/v1/offset-reference/sessions", json=_session_body(compiled, role), headers=auth
            )
            assert response.status_code == 200, response.text
            session = response.json()["id"]
            stage_id, raw = _request(compiled, session)
            request = MaskedStageRequest.unpack(raw)
            header = seed_header(
                model=request.model,
                stage=stage_id,
                body=request.body_fingerprint,
                weight=request.weight_digest,
                plan=compiled._plan.digest,
                composition=pipeline.digest(),
                session=session,
                ticket=request.correlation_id,
                rows=1,
                columns=request.masked_input.shape[1],
                bits=request.wire_bits,
            )
            seed = b"c" * 32
            packed = pack_seed(header, seed)
            assert unpack_seed(packed, max_rows=4) == (header, seed)
            path = f"/v1/offset-reference/sessions/{session}/stages/{stage_id}"
            result = await client.post(path, headers=auth, content=packed)
            if role != "worker_b" or not selected:
                assert result.status_code == 400
                return
            assert result.status_code == 200, result.text
            expanded = np.frombuffer(
                _native.offset_seeded_share(
                    seed, context(header), header["columns"], request.wire_bits
                ),
                dtype="<u4",
            ).reshape(1, -1)
            entry = worker._model("offset-model").stages[stage_id]
            expected = await worker.execute_stage(
                "offset-model", entry.spec, [replace(request, masked_input=expanded).pack()]
            )
            np.testing.assert_array_equal(
                MaskedStageResponse.unpack(result.content).masked_output,
                MaskedStageResponse.unpack(expected[0]).masked_output,
            )
            assert (await client.post(path, headers=auth, content=packed)).status_code == 400
            assert (await client.post(path, headers=auth, content=packed)).status_code == 409

    asyncio.run(scenario())


def test_seed_codec_rejects_forged_size_duplicate_fields_and_unknown_keys():
    header = seed_header(
        model="a",
        stage="b",
        body="a" * 64,
        weight="b" * 64,
        plan="c" * 64,
        composition="d" * 64,
        session="e" * 32,
        ticket="f" * 32,
        rows=1,
        columns=4,
        bits=24,
    )
    for changes in ({"rows": True}, {"columns": 4_000_001}, {"bits": 8}, {"extra": 1}):
        payload = SEED_MAGIC + msgpack.packb([{**header, **changes}, b"z" * 32], use_bin_type=True)
        with pytest.raises(ProtocolError):
            unpack_seed(payload, max_rows=4)
    with pytest.raises(ProtocolError):
        unpack_seed(SEED_MAGIC + b"x" * 8193, max_rows=4)
    packer = msgpack.Packer(use_bin_type=True)
    duplicate = packer.pack_array_header(2) + packer.pack_map_header(len(header) + 1)
    for key, value in [*header.items(), ("model", "forged")]:
        duplicate += packer.pack(key) + packer.pack(value)
    with pytest.raises(ProtocolError):
        unpack_seed(SEED_MAGIC + duplicate + packer.pack(b"z" * 32), max_rows=4)


@pytest.mark.parametrize("rows,columns", [(1, 1), (3, 97), (11, 32)])
def test_public_row_rings_preserve_exact_integer_projections(rows, columns):
    rng = np.random.default_rng(13)
    weights = rng.integers(-127, 128, (33, columns), dtype=np.int8)
    weights[0] = 0
    values = rng.integers(-127, 128, (rows, columns), dtype=np.int8)
    mask = rng.integers(0, 1 << 32, values.shape, dtype=np.uint32)
    other = ((values.astype(np.int64) - mask.astype(np.int64)) % (1 << 32)).astype("<u4")
    a = ((other.astype(np.int64) @ weights.T.astype(np.int64)) % (1 << 32)).astype("<u4")
    b = ((mask.astype(np.int64) @ weights.T.astype(np.int64)) % (1 << 32)).astype("<u4")
    widths = _native.offset_row_bits(weights.tobytes(), columns, 127)
    bounds = np.abs(weights.astype(np.int64)).sum(axis=1) * 127
    assert list(widths) == [max(1, int(bound).bit_length() + 1) for bound in bounds]
    pa = _native.offset_pack_rows(a.tobytes(), widths, rows)
    pb = _native.offset_pack_rows(b.tobytes(), widths, rows)
    decoded = np.frombuffer(
        _native.offset_reconstruct_rows(pa, pb, widths, rows), dtype="<i8"
    ).reshape(rows, -1)
    np.testing.assert_array_equal(decoded, values.astype(np.int64) @ weights.T.astype(np.int64))
    assert len(pa) == (rows * sum(widths) + 7) // 8
    with pytest.raises(ValueError, match="length|padding"):
        _native.offset_reconstruct_rows(pa + b"\0", pb, widths, rows)
    with pytest.raises(ValueError, match="padding"):
        _native.offset_reconstruct_rows(b"\x80", b"\0", b"\x01", 1)


def test_residue_layout_is_source_bound_before_session_admission(tmp_path):
    from pllm.runtime.offset_codec import compiled_row_layout

    base, workers = _fixture(tmp_path)
    worker = workers[0]
    pipeline = TwoOnlineOffsetCpu(
        pllm.Model("offset-model"),
        quantization=SymmetricPerRow(weight_bits=4, activation_bits=4),
        linear=TwoOnlineOffsetLinear(output_encoding="row_residues"),
    )
    packed = worker.client_bundle(
        "offset-model", placement="offset", output_encoding="row_residues"
    )
    bundle = ClientBundle.unpack(packed)
    compiled = compile_runtime_model(base._plan, bundle, composition=pipeline)
    digest = compiled_row_layout(compiled)
    key = "a" * 32

    async def scenario():
        app = create_offset_worker_app(
            worker, model_id="offset-model", role_id="worker_a", api_key=key, composition=pipeline
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://worker"
        ) as client:
            body = _session_body(compiled, "worker_a")
            auth = {"authorization": f"Bearer {key}"}
            for value in (None, "0" * 64):
                forged = {**body, **({"residue_layout_digest": value} if value else {})}
                assert (
                    await client.post("/v1/offset-reference/sessions", json=forged, headers=auth)
                ).status_code == 409
            response = await client.post(
                "/v1/offset-reference/sessions",
                json={**body, "residue_layout_digest": digest},
                headers=auth,
            )
            assert response.status_code == 200, response.text
            assert response.json()["residue_layout_digest"] == digest
        assert all(stage.calls == 0 for stage in worker._model("offset-model").stages.values())

    asyncio.run(scenario())
    stage = next(s for s in bundle.stages.values() if s.output_residue_bits is not None)
    compiled._bundle.stages[stage.id] = replace(
        stage, output_residue_bits=bytes([1]) * stage.out_features
    )
    with pytest.raises(ProtocolError, match="differs"):
        compiled_row_layout(compiled)


def test_row_response_rejects_downgrade_ticket_and_layout_changes():
    from pllm.runtime.offset_codec import pack_row_response, unpack_row_response

    response = MaskedStageResponse(
        correlation_id="a" * 32,
        stage_id="test",
        masked_output=np.array([[17, 31]], dtype=np.uint32),
        modulus=1 << 16,
        wire_bits=16,
        ring="u16",
        server_ns=0,
    )
    widths = bytes([5, 6])
    packed = pack_row_response(response, widths)
    kwargs = dict(ticket="a" * 32, stage="test", rows=1, widths=widths, bits=16)
    assert unpack_row_response(packed, **kwargs)[1] == 0
    for changes in ({"ticket": "b" * 32}, {"widths": bytes([4, 6])}, {"rows": 2}):
        with pytest.raises(ProtocolError):
            unpack_row_response(packed, **{**kwargs, **changes})
    with pytest.raises(ProtocolError):
        unpack_row_response(response.pack(), **kwargs)
