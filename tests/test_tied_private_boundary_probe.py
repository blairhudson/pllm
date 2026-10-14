"""Independent exact integer, numeric, privacy-shape and coupled decoder gates."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("tied_private_probe", SCRIPTS / "probe_tied_private_boundary.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
sys.path.pop(0)


def fixture(tmp_path, n=129, d=17):
    from pllm import _native
    from pllm.runtime.quantization import quantize_weight_per_row
    weights = np.random.default_rng(441).standard_normal((n, d), dtype=np.float32)
    weights[0] = 0
    weights[-1] = np.linspace(-127, 127, d, dtype=np.float32)
    q = quantize_weight_per_row(weights, bits=8)
    records = np.empty((n, d + 4), np.uint8)
    records[:, :d] = q.values.view(np.uint8)
    records[:, d:] = q.scales.astype("<f4").view(np.uint8).reshape(n, 4)
    q.values.tofile(tmp_path / "head.i8")
    q.scales.astype("<f4").tofile(tmp_path / "scales.f32")
    (tmp_path / "records.bin").write_bytes(records.tobytes())
    (tmp_path / "prompt.json").write_text(json.dumps([i % n for i in range(150)]))
    server = _native.PrivatePageServer(records.tobytes(), d + 4, 0)
    files = {name: {"bytes": (tmp_path / name).stat().st_size, "sha256": probe.digest(tmp_path / name)}
             for name in ("head.i8", "scales.f32", "records.bin", "prompt.json")}
    public = {"shape": [n, d], "files": files, "descriptor": server.descriptor()[0].hex()}
    for name in ("scale_artifact", "worker_artifact", "full_boundary_artifact"):
        public[name] = {"zlib_framed_bytes": 12345}
    return q, records, public


@pytest.mark.parametrize("n", [127, 128, 129, 257])
def test_packed_record_original_scale_bytes_edges_freshness_and_replay(tmp_path, n):
    from pllm import _native
    q, records, public = fixture(tmp_path, n)
    servers = [_native.PrivatePageServer(records.tobytes(), 21, p) for p in (0, 1)]
    issued = [_native.private_page_issue(*servers[0].descriptor(), i, leaf_bits=128)
              for i in (0, n - 1, n - 1, min(127, n - 1))]
    assert issued[1][0] != issued[2][0] and issued[1][1] != issued[2][1]
    replies = [s.evaluate_batch([query[p] for query in issued]) for p, s in enumerate(servers)]
    lengths = set()
    for pos, (index, query) in enumerate(zip((0, n - 1, n - 1, min(127, n - 1)), issued, strict=True)):
        record = query[2].decode(replies[0][pos], replies[1][pos])
        assert record == records[index].tobytes()
        expected = q.values[index].astype(np.float32) * q.scales[index]
        np.testing.assert_array_equal(probe.embedding(record, 17).view("u4"), expected.view("u4"))
        lengths.add(tuple(map(len, (query[0], query[1], replies[0][pos], replies[1][pos]))))
        with pytest.raises(ValueError):
            query[2].decode(replies[0][pos], replies[1][pos])
    assert len(lengths) == 1
    for p, server in enumerate(servers):
        with pytest.raises(ValueError):
            server.evaluate(issued[0][p])


def test_head_independent_signed_integer_and_original_float_order(tmp_path):
    from pllm.native import PagedGEMM
    from pllm.runtime.quantization import quantize_activation_per_row
    qweight, _, public = fixture(tmp_path)
    x = probe.hidden_rows(17)
    x[3] = np.linspace(-127, 127, 17, dtype=np.float32)
    q = quantize_activation_per_row(x, bits=8)
    first, second = probe.split_head(q.values)
    other, _ = probe.split_head(q.values)
    assert not np.array_equal(first, other)
    np.testing.assert_array_equal(np.add(first, second, dtype=np.uint32).view(np.int32), q.values.astype(np.int32))
    kernels = [PagedGEMM.from_raw(tmp_path / "head.i8", public["files"]["head.i8"]["sha256"], (129, 17))
               for _ in (0, 1)]
    try:
        integer, logits = probe.restore_head(kernels[0].wrap32(first), kernels[1].wrap32(second), q.scales, qweight.scales)
        # Neither native clear nor reconstructed output supplies the oracle.
        expected = np.asarray([[sum(int(a) * int(b) for a, b in zip(row, w, strict=True))
                                for w in qweight.values] for row in q.values], np.int64)
        np.testing.assert_array_equal(integer, expected)
        numeric = expected.astype(np.float32) * q.scales[:, None] * qweight.scales[None, :]
        np.testing.assert_array_equal(logits.view("u4"), numeric.view("u4"))
        original = probe.make_bundle(kernels[0], qweight.scales, 17).local_linear("lm_head", x)
        np.testing.assert_array_equal(logits.view("u4"), original.view("u4"))
        assert probe.select(logits) == probe.select(original)
    finally:
        for kernel in kernels:
            kernel.close()


def test_complete_tiny_party_runner_and_byte_ledger(tmp_path):
    from pllm import _native
    from pllm.runtime.quantization import quantize_activation_per_row
    _, _, public = fixture(tmp_path)
    local = probe.baseline({"root": str(tmp_path), "public": public, "prompt_rows": 16})
    q = quantize_activation_per_row(probe.hidden_rows(17), bits=8)
    calls = []
    def worker(payload):
        # Assert each process payload owns only its one role's input, never token ids.
        assert "ids" not in payload and "tokens" not in payload
        assert len(payload["shares"]) in (0, 8)
        calls.append(payload)
        completed = subprocess.run([sys.executable, str(SCRIPTS / "probe_tied_private_boundary.py"), "--worker"],
            input=json.dumps(payload), text=True, capture_output=True, check=True, timeout=30,
            env=dict(os.environ, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
        return json.loads(completed.stdout)
    parties, _, _ = probe.lookup_round(worker, tmp_path, public, list(range(16)),
                                       probe.split_head(q.values), probe.digest(_native.__file__))
    outputs = [np.concatenate([np.frombuffer(bytes.fromhex(s), "<u4").reshape(1, 129)
                               for s in p.pop("head_outputs")]) for p in parties]
    _, logits = probe.restore_head(*outputs, q.scales, np.fromfile(tmp_path / "scales.f32", "<f4"))
    assert logits.astype("<f4").tobytes().hex() == local["logits"]
    tokens, state = probe.select(logits)
    assert tokens == local["tokens"] and state == local["rng_state"]
    more, _, _ = probe.lookup_round(worker, tmp_path, public, tokens[:-1],
                                    (np.empty((0, 17), np.uint32),) * 2, probe.digest(_native.__file__), batch_size=1)
    parties += more
    bill = probe.ledger(public, probe.workload(16), parties)
    assert len(calls) == 4
    assert [p["lookup_rows"] for p in parties] == [16, 16, 7, 7]
    assert [p["head_rows"] for p in parties] == [8, 8, 0, 0]
    assert [p["lookup_batch_size"] for p in parties] == [64, 64, 1, 1]
    assert bill["head_raw_u32_input_bytes"] == 2 * 8 * 17 * 4
    assert bill["head_raw_u32_output_bytes"] == 2 * 8 * 129 * 4
    assert bill["both_workers_retained_table_plus_snapshot_bytes"] == 2 * 129 * (21 + 17)
    assert bill["source_present"]["worker_derived_payload_delivery_bytes"] == 0
    assert bill["source_missing"]["both_workers_raw_table_and_head_delivery_bytes"] == 2 * 129 * 38
    assert bill["physical_wire_bytes"] is None
    assert all(p["replay_rejected"] for p in parties)
    with pytest.raises(ValueError, match="rows"):
        probe.ledger(public, probe.workload(150), parties)


@pytest.mark.parametrize("bad", [True, 0, 15, 151, 16.0])
def test_schedule_admission(bad):
    with pytest.raises(ValueError):
        probe.workload(bad)


def test_shape_ring_record_and_native_identity_admission(tmp_path):
    from pllm import _native
    _, _, public = fixture(tmp_path)
    with pytest.raises(ValueError, match="geometry"):
        probe.admit({"shape": [1, 140000]})
    with pytest.raises(ValueError, match="record"):
        probe.embedding(b"short", 17)
    with pytest.raises(ValueError, match="rank-two"):
        probe.split_head(np.ones((1, 17), np.float32))
    with pytest.raises(ValueError, match="matching"):
        probe.restore_head(np.ones((1, 2), np.uint32), np.ones((2, 2), np.uint32), [1], [1, 1])
    with pytest.raises(ValueError, match="identity"):
        probe.party({"root": str(tmp_path), "public": public, "party": 0, "keys": ["00"],
                     "shares": [], "native_sha256": "0" * 64})
    assert probe.digest(_native.__file__) != "0" * 64


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("reduction", [None, "prefix_f32"])
def test_coupled_full_decoder_logits_kv_rng_and_generated_lookups(tmp_path, monkeypatch, family, reduction):
    from pllm import Model, _native, lower_model
    from pllm.native import PagedGEMM
    from pllm.profiles import ClientOnlyCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.loaders import load_hf_directory
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.runtime.quantization import quantize_activation_per_row
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    path = create_tiny_llama_checkpoint(tmp_path / "model", model_type=family,
        num_hidden_layers=2, with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3")
    config = json.loads((path / "config.json").read_text())
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    asyncio.run(engine.load(load_hf_directory(path, model_id="tied-parity")))
    bundle = ClientBundle.unpack(engine.client_bundle("tied-parity", placement="client"))
    lookup = bundle.stages["token_lookup"]
    head = bundle.stages["lm_head"]
    np.testing.assert_array_equal(lookup.client_weight, head.client_weight)
    np.testing.assert_array_equal(lookup.client_weight_scales, head.client_weight_scales)
    n, d = head.client_weight.shape
    records = np.empty((n, d + 4), np.uint8)
    records[:, :d] = lookup.client_weight.view(np.uint8)
    records[:, d:] = lookup.client_weight_scales.astype("<f4").view(np.uint8).reshape(n, 4)
    servers = [_native.PrivatePageServer(records.tobytes(), d + 4, p) for p in (0, 1)]
    raw = tmp_path / "head.i8"
    head.client_weight.tofile(raw)
    kernels = [PagedGEMM.from_raw(raw, probe.digest(raw), (n, d)) for _ in (0, 1)]
    plan = lower_model(config, batch=1, max_input_tokens=16, max_new_tokens=4)
    pipeline = ClientOnlyCpu(Model("tied-parity"), quantization=SymmetricPerRow(causal_reduction=reduction))
    compiled = compile_runtime_model(plan, bundle, composition=pipeline)
    executor = compiled.client_linear_executor(engine)
    original_lookup, original_linear = ClientBundle.local_token_lookup, ClientBundle.local_linear
    lookup_calls, head_calls = [], []

    def private_lookup(owner, ids):
        if owner is not bundle:
            return original_lookup(owner, ids)
        indices = np.asarray(ids, np.int64).reshape(-1)
        queries = [_native.private_page_issue(*servers[0].descriptor(), int(i), leaf_bits=128) for i in indices]
        try:
            replies = [s.evaluate_batch([q[p] for q in queries]) for p, s in enumerate(servers)]
            result = np.stack([probe.embedding(q[2].decode(replies[0][i], replies[1][i]), d)
                               for i, q in enumerate(queries)])
            np.testing.assert_array_equal(result.view("u4"), original_lookup(owner, ids).view("u4"))
            lookup_calls.append(indices.tolist())
            return result
        finally:
            for q in queries:
                q[2].cancel()

    def private_head(owner, stage_id, x, *args, **kwargs):
        if owner is not bundle or stage_id != "lm_head":
            return original_linear(owner, stage_id, x, *args, **kwargs)
        q = quantize_activation_per_row(x, bits=head.activation_bits)
        shares = probe.split_head(q.values)
        _, logits = probe.restore_head(kernels[0].wrap32(shares[0]), kernels[1].wrap32(shares[1]),
                                      q.scales, head.client_weight_scales)
        expected = original_linear(owner, stage_id, x, *args, **kwargs)
        np.testing.assert_array_equal(logits.view("u4"), expected.view("u4"))
        head_calls.append(logits.shape)
        return logits

    results = []
    try:
        for private in (False, True):
            if private:
                monkeypatch.setattr(ClientBundle, "local_token_lookup", private_lookup)
                monkeypatch.setattr(ClientBundle, "local_linear", private_head)
            runtime = compiled.runtime(executor)
            runtime.rng = np.random.default_rng(418)
            _, logits, _ = runtime.prepare_ids([0, 2, 3, 5, 7])
            phases, tokens = [], []
            for step in range(4):
                assert logits.shape[-1] == n
                token = runtime.sample(logits, temperature=0.7, top_p=0.9)
                tokens.append(token)
                phases.append((logits.tobytes(), token, json.dumps(runtime.rng.bit_generator.state, sort_keys=True),
                    [(c.key[:c.length].tobytes(), c.value[:c.length].tobytes()) for c in runtime.caches]))
                if step < 3:
                    logits = runtime.forward_ids([token])[-1]
            results.append((phases, tokens))
        assert results[0] == results[1]
        assert lookup_calls == [[0, 2, 3, 5, 7], *[[t] for t in results[1][1][:-1]]]
        assert len(head_calls) == 4
    finally:
        for kernel in kernels:
            kernel.close()
