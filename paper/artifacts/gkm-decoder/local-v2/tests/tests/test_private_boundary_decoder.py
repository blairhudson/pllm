"""Authenticated actual boundary workers preserve full generated decoder state."""
import asyncio
import hashlib
import json
import secrets

import httpx
import msgpack
import numpy as np
import pytest

from pllm import _native
from pllm.runtime.model_binding import RuntimeBindingError, compile_runtime_model
from pllm.runtime.preparation_protocol import PreparationRequest, derive_online_attempt_id
from pllm.runtime.private_boundary_protocol import BoundaryError, ROLES
from pllm.runtime.private_boundary_transport import PrivateBoundaryTransport
from pllm.runtime.private_boundary_worker import create_private_boundary_worker_app
from pllm.runtime.stage_protocol import PreparedStageBatchRequest, PreparedStageBatchResponse
from pllm.runtime.transformer_client import ClientBundle, PreparedInventory, PreparedRemoteLinear, PreparedStageRows

from test_private_boundary_bundle import generated_boundary


def prepared_body(engine, compiled):
    """Real seeded preparation/native products, no decoder or SDK substitution."""
    bundle = compiled._bundle
    demands = compiled.prepared_stage_rows(5, 3)
    stages, corrections = {}, {}
    for stage_id, count in demands.items():
        stage = bundle.stages[stage_id]
        profile = stage.seeded_profile
        request = PreparationRequest(attempt_id=secrets.token_hex(16), session_id="decoder-test",
            model=bundle.model_id, body_fingerprint=bundle.privacy["body_fingerprint"],
            stage_id=stage_id, weight_digest=stage.weight_digest, rows=count,
            in_features=stage.in_features, out_features=stage.out_features,
            weight_bits=8, activation_bits=8, signed_output_bound=profile.signed_output_bound,
            ring=profile.ring, modulus=profile.modulus, wire_bits=profile.wire_bits, seed=secrets.token_bytes(32))
        correction = asyncio.run(engine.prepare_seeded_stage(request))
        corrections[stage_id] = {derive_online_attempt_id(request, i): correction.correction[i] for i in range(count)}
        stages[stage_id] = PreparedStageRows(request)
    inventory = PreparedInventory("decoder-test", max(demands.values()), stages, demand=True)
    lease = inventory.reserve(inventory.capacity)

    def inference(stage_id, payloads):
        assert stage_id not in {"token_lookup", "lm_head"}
        runtime = engine.models[bundle.model_id].stages[stage_id]
        matrix = runtime.compiled_weight
        replies = []
        for payload in payloads:
            request = PreparedStageBatchRequest.unpack(payload)
            correction = np.stack([corrections[stage_id].pop(key) for key in request.correlation_ids])
            product = matrix.modular(request.masked_input, runtime.seeded_profile.modulus)
            output = (product.astype(np.uint64) + correction.astype(np.uint64)) % runtime.seeded_profile.modulus
            replies.append(PreparedStageBatchResponse(request.batch_id, output.astype(np.uint32),
                runtime.seeded_profile.wire_bits, 0).pack())
        return replies

    return PreparedRemoteLinear(bundle.model_id, bundle.privacy["body_fingerprint"], bundle.stages,
        lease, inference, request_encoding="compact")


def boundary_workers(engine, compiled, admission):
    from pllm.configuration import Pipeline
    pin = Pipeline.from_spec(json.loads(compiled._canonical_composition)).components["boundary"].params.get("head_certificate")
    descriptor = compiled.private_boundary_descriptor()
    head = engine.models[compiled._bundle.model_id].stages["lm_head"]
    keys = {ROLES[0]: "a" * 32, ROLES[1]: "b" * 32}

    def admit(actual, requested):
        # Derive independently from locked compiled model and public workload.
        expected = compiled.private_boundary_admission(5, 3)
        if actual != compiled.private_boundary_descriptor() or requested != expected:
            raise BoundaryError("compiled admission differs")
        return expected

    clients = []
    for index, role in enumerate(ROLES):
        app = create_private_boundary_worker_app(descriptor=descriptor, head=head.compiled_weight,
            head_scales=head.weight.scales, role_id=role, api_key=keys[role],
            peer_credential_digest=hashlib.sha256(keys[ROLES[1 - index]].encode()).hexdigest(),
            admit=admit, head_certificate=pin)

        def exchange(request, app=app):
            async def call():
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url=str(request.url)) as client:
                    response = await client.request(request.method, str(request.url),
                        headers=request.headers, content=request.read())
                    return httpx.Response(response.status_code, headers=response.headers, content=response.content)
            return asyncio.run(call())

        clients.append(httpx.Client(base_url=f"http://localhost:{8100 + index}", transport=httpx.MockTransport(exchange)))
    transport = PrivateBoundaryTransport(worker_a=clients[0], worker_b=clients[1],
        api_key_a=keys[ROLES[0]], api_key_b=keys[ROLES[1]], descriptor=descriptor, head_scales=head.weight.scales,
        head_certificate=pin)
    return transport.open(admission), clients


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("reduction", [None, "prefix_f32"])
@pytest.mark.parametrize("boundary_options", [{}, {"head_dispatch": "parallel"},
    {"head_output_encoding": "uniform_residues"},
    {"head_dispatch": "parallel", "head_output_encoding": "uniform_residues"},
    {"head_kernel": "byte_dot"},
    {"head_kernel": "byte_dot", "head_dispatch": "parallel", "head_output_encoding": "uniform_residues"},
    *[{"head_dispatch": dispatch, "head_output_encoding": "row_residues", "head_kernel": kernel}
      for dispatch in ("serial", "parallel") for kernel in ("existing", "byte_dot")]],
    ids=["serial-u32", "parallel-u32", "serial-residues", "parallel-residues", "byte-dot", "byte-dot-parallel-residues",
         "row-serial-existing", "row-serial-byte-dot", "row-parallel-existing", "row-parallel-byte-dot"])
def test_full_logits_kv_tokens_rng_and_repeated_feedback(tmp_path, family, reduction, boundary_options, monkeypatch):
    from pllm.runtime.native import CompiledMatrix
    integer_outputs = []
    native_byte_dot = CompiledMatrix.wrap32_byte_dot
    def checked_byte_dot(matrix, share):
        result = native_byte_dot(matrix, share)
        np.testing.assert_array_equal(result, matrix.wrap32(share))
        integer_outputs.append(result.copy())
        return result
    monkeypatch.setattr(CompiledMatrix, "wrap32_byte_dot", checked_byte_dot)
    engine, original, document, plan, composition = generated_boundary(tmp_path, family, reduction,
        boundary_options=boundary_options)
    if not hasattr(_native.PrivatePageServer, "bind_context"):
        pytest.skip("native authenticated context API not installed")
    from pllm import Model
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    private = compile_runtime_model(plan, ClientBundle.unpack(msgpack.packb(document, use_bin_type=True)), composition=composition)
    ordinary = compile_runtime_model(plan, ClientBundle.unpack(original),
        composition=MaskedLinearCpu(Model("private-boundary"), quantization=SymmetricPerRow(causal_reduction=reduction)))
    boundary, clients = boundary_workers(engine, private, private.private_boundary_admission(5, 3))
    body = prepared_body(engine, private)
    runtime = private.runtime(body, boundary=boundary)
    baseline_body = prepared_body(engine, ordinary)
    baseline = ordinary.runtime(baseline_body)
    try:
        for candidate in (runtime, baseline):
            candidate.rng = np.random.default_rng(417)
        traces = []
        for candidate in (baseline, runtime):
            _, logits, _ = candidate.prepare_ids([0, 2, 2, 5, 2])
            phases = []
            for index in range(4):
                assert logits.shape[-1] == document["config"]["vocab_size"]
                token = candidate.sample(logits, temperature=0.7, top_p=0.9)
                phases.append((logits.tobytes(), token, json.dumps(candidate.rng.bit_generator.state, sort_keys=True),
                    [(row.key[:row.length].tobytes(), row.value[:row.length].tobytes()) for row in candidate.caches]))
                if index < 3:
                    logits = candidate.forward_ids([token])[-1]
            traces.append(phases)
        assert traces[0] == traces[1]
        assert len(integer_outputs) == (8 if boundary_options.get("head_kernel") == "byte_dot" else 0)
        assert boundary.operation_counters["lookup"]["rows"] == 8
        assert boundary.operation_counters["head"]["rows"] == 4
        assert runtime.token_cache == {}  # Repetition and generated feedback never deduplicate.
        with pytest.raises(RuntimeBindingError):
            private.runtime(body)
        with pytest.raises(RuntimeBindingError):
            private.runtime(lambda *_: None, boundary=boundary)
        with pytest.raises(RuntimeBindingError):
            private.runtime(body, boundary=boundary)
        boundary.complete()
        runtime.close()
        runtime.close()
    finally:
        boundary.close()
        body.inventory.close()
        baseline_body.inventory.close()
        for client in clients:
            client.close()
