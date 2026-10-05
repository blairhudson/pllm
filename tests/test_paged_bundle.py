"""File-backed public delivery and the ordinary compiled SDK execution contract."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

import pllm
from pllm.kernels import AppleMetal
from pllm.native import PagedGEMM
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu, TwoOnlineOffsetCpu, VerifiedMaskedLinearCpu
from pllm.protocols import ClientBundleTransport, MaskedLinear, TwoOnlineOffsetLinear
from pllm.quantization import SymmetricPerRow
from pllm.roles import ClientLinearRoles
from pllm.runtime.bundle_storage import FilePayload
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.state import ClientPrefixReuse


def test_paged_sdk_tiles_large_linear_and_gather_calls(tmp_path):
    from pllm.runtime.native import MaskedGEMM
    from pllm.runtime.transformer_client import _paged_clear_rows, _paged_gather_rows
    rng = np.random.default_rng(72)
    weights = rng.integers(-7, 8, (4096, 64), dtype=np.int8)
    path = tmp_path / "raw"
    path.write_bytes(weights.tobytes())
    with PagedGEMM.from_raw(path, hashlib.sha256(weights.tobytes()).hexdigest(), weights.shape) as page:
        values = rng.integers(-11, 12, (1025, 64), dtype=np.int8)
        expected = MaskedGEMM(threads=1).compile(weights).clear(values)
        np.testing.assert_array_equal(_paged_clear_rows(page, values), expected)
        ids = np.arange(4101, dtype=np.int64) % weights.shape[0]
        np.testing.assert_array_equal(_paged_gather_rows(page, ids), weights[ids])
        ids = np.arange(1031, dtype=np.int64) % weights.shape[1]
        np.testing.assert_array_equal(_paged_gather_rows(page, ids, columns=True), weights[:, ids].T)


def test_verified_file_payload_cache_repairs_corruption_and_survives_eviction(tmp_path):
    from pllm.runtime.bundle_artifacts import ArtifactObjectCache, _canonical, _key
    data = bytes(range(256)) * 8193  # larger than a framed object group
    metadata = {"dtype": "i1", "shape": [len(data), 1], "orientation": "linear",
                "source": {}, "numeric": {}}
    content, domain = hashlib.sha256(data).hexdigest(), hashlib.sha256(_canonical(metadata)).hexdigest()
    row = {"sha256": _key(content, domain), "size": len(data), "metadata": metadata,
           "content_sha256": content, "domain_hash": domain}
    cache = ArtifactObjectCache(tmp_path / "cache", max_bytes=4 << 20)
    payload = FilePayload((data[i:i + 65536] for i in range(0, len(data), 65536)),
                          size=len(data), digest=row["content_sha256"])
    cache.put(row, payload)
    payload.close()
    snapshot = cache.get(row, file_backed=True)
    assert isinstance(snapshot, FilePayload)
    object_path = tmp_path / "cache" / (row["sha256"] + ".blob")
    object_path.write_bytes(b"!" * len(data))
    assert cache.get(row, file_backed=True) is None
    assert cache.stats.corruptions == 1 and not object_path.exists()
    assert b"".join(snapshot.chunks()) == data
    snapshot.close()


@pytest.mark.integration
def test_client_storage_probe_fixture(tmp_path):
    root = create_tiny_llama_checkpoint(tmp_path / "source")
    directory = tmp_path / "objects"
    directory.mkdir()
    probe = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/probe_client_storage.py"))
    probe["publish"](root, directory)
    samples = [probe["worker"](mode, directory) for mode in ("eager-resident", "stream-paged")]
    for field in ("bundle_sha256", "model_plan_digest", "mask_digest", "boundary_digest"):
        assert samples[0][field] == samples[1][field]


def test_failed_bundle_admission_closes_native_snapshots(tmp_path, monkeypatch):
    from pllm.runtime.bundle_artifacts import parse_manifest, reconstruct_document
    from pllm.runtime.transformer_client import ClientBundle, TransformerClientError

    root = create_tiny_llama_checkpoint(tmp_path / "source")
    directory = tmp_path / "objects"
    directory.mkdir()
    probe = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/probe_client_storage.py"))
    probe["publish"](root, directory)
    descriptor = json.loads((directory / "descriptor.json").read_bytes())["raw"]
    manifest = parse_manifest((directory / "manifest").read_bytes(),
                              fingerprint=descriptor["sha256"], size=descriptor["size"])
    payloads, snapshots = [], []

    def read(row):
        data = (directory / row["sha256"]).read_bytes()
        if row["metadata"]["dtype"] != "i1":
            return data
        value = FilePayload([data], size=row["size"], digest=row["content_sha256"])
        payloads.append(value)
        return value

    original = PagedGEMM._from_raw_executor.__func__

    def record(cls, *args):
        matrix = original(cls, *args)
        snapshots.append(matrix)
        return matrix

    monkeypatch.setattr(PagedGEMM, "_from_raw_executor", classmethod(record))
    try:
        value = reconstruct_document(manifest, read)
        value["stages"]["lm_head"]["client_weight"]["ref"] = "missing"
        with pytest.raises(TransformerClientError, match="reference"):
            ClientBundle._from_document(value, storage="paged")
        assert snapshots
        for matrix in snapshots:
            with pytest.raises(ValueError, match="closed"):
                _ = matrix.shape
    finally:
        for payload in payloads:
            payload.close()


def test_raw_paged_import_preserves_integer_output_and_private_column_gathers(tmp_path):
    rng = np.random.default_rng(471)
    weights = rng.integers(-127, 128, (517, 41), dtype=np.int8)
    raw = weights.tobytes()
    payload = FilePayload([raw], size=len(raw), digest=hashlib.sha256(raw).hexdigest())
    try:
        matrix = PagedGEMM.from_raw(payload.path, payload.digest, weights.shape)
    finally:
        payload.close()
    x = rng.integers(-127, 128, (2, 41), dtype=np.int8)
    np.testing.assert_array_equal(matrix.clear(x), x.astype(np.int64) @ weights.astype(np.int64).T)
    np.testing.assert_array_equal(matrix.gather_columns([40, 0, 40]), weights[:, [40, 0, 40]].T)
    np.testing.assert_array_equal(matrix.gather_rows([516, 0]), weights[[516, 0]])
    assert matrix.weight_digest == hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        matrix.gather_columns([41])
    matrix.close()


def test_file_payload_rejects_incomplete_oversized_and_bad_digest():
    digest = hashlib.sha256(b"abcd").hexdigest()
    for blocks in ([b"abc"], [b"abcde"], [b"abce"], [b""]):
        with pytest.raises(ValueError):
            FilePayload(blocks, size=4, digest=digest)


def test_paged_configuration_is_explicit_and_rejects_unimplemented_residency():
    from pllm.configuration import ConfigurationError
    from pllm.profiles import resolve_runtime_composition
    ordinary = ClientBundleTransport("artifacts")
    assert ordinary == ClientBundleTransport("artifacts", storage="memory")
    selected = ClientBundleTransport("artifacts", compression="zlib", batch_objects=64, storage="paged")
    assert selected.params["storage"] == "paged" and selected != ordinary
    for encoding in ("none", "zlib"):
        with pytest.raises(ConfigurationError):
            ClientBundleTransport(encoding, storage="paged")
    with pytest.raises(ConfigurationError):
        ClientBundleTransport("artifacts", storage="mmap")
    assert resolve_runtime_composition(MaskedLinearCpu(pllm.Model.hf("test/model"),
        delivery=selected, kernels=AppleMetal(),
        placement=ClientLinearRoles(("qkv_projection",)))) is None


@pytest.mark.integration
@pytest.mark.parametrize("topology,family,tied,placement", [
    ("prepared", "qwen3", True, False),
    ("prepared", "qwen2", False, True),
    ("verified", "qwen2", True, False),
    ("offset", "qwen3", False, False),
])
def test_paged_sdk_matches_all_logits_kv_cache_and_gateway(tmp_path, monkeypatch,
                                                         topology, family, tied, placement):
    from fastapi.testclient import TestClient
    from pllm.runtime.model_binding import RuntimeBindingError

    root = create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=2,
        model_type=family, qk_norm=family == "qwen3", with_qkv_bias=family == "qwen2",
        tie_word_embeddings=tied)
    source = pllm.Model.path(str(root), model_id="paged-sdk")
    trajectories, outputs, current = [], [], []
    prefill, decode = SemanticDecoderRuntime.prepare_ids, SemanticDecoderRuntime.decode_step

    def capture(runtime, logits):
        current.append((logits.copy(), [(c.key[:c.length].copy(), c.value[:c.length].copy())
                                       for c in runtime.caches]))

    def prefill_capture(runtime, *args, **kwargs):
        result = prefill(runtime, *args, **kwargs)
        capture(runtime, result[1])
        return result

    def decode_capture(runtime, *args, **kwargs):
        result = decode(runtime, *args, **kwargs)
        capture(runtime, result[0])
        return result

    monkeypatch.setattr(SemanticDecoderRuntime, "prepare_ids", prefill_capture)
    monkeypatch.setattr(SemanticDecoderRuntime, "decode_step", decode_capture)
    for storage in ("memory", "paged"):
        common = dict(quantization=SymmetricPerRow(causal_reduction="prefix_f32"),
            delivery=ClientBundleTransport("artifacts", compression="zlib", batch_objects=64, storage=storage),
            cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=64, generated_prefixes=True))
        if topology == "offset":
            pipeline = TwoOnlineOffsetCpu(source, **common, linear=TwoOnlineOffsetLinear(
                input_encoding="seeded", output_encoding="row_residues", dispatch="seed_first"))
        else:
            factory = VerifiedMaskedLinearCpu if topology == "verified" else MaskedLinearCpu
            pipeline = factory(source, **common, linear=MaskedLinear(output_encoding="row_residues"),
                inventory=PreparedInventory("request-sized", rows=1, refill="on-demand"),
                **({"placement": ClientLinearRoles(("attention_output", "qkv_projection"), prefix_layers=1)}
                   if placement else {}))
        experiment = pllm.Experiment("paged-sdk", pipeline, pllm.Deployment.local(root=str(tmp_path)),
            pllm.ExecutionBudget(requests=3, max_input_tokens=64, max_new_tokens=2))
        current.clear()
        with build_roles(experiment, engine_threads=1) as roles:
            with roles.client(bundle_cache_dir=tmp_path / f"cache-{storage}") as client:
                results = [client.responses.create(input="abc", max_output_tokens=2, temperature=0)
                           for _ in range(2)]
                outputs.append([(r.output_text, r.usage) for r in results])
                audit = client.privacy_audit.to_dict()
                assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0
                assert audit["inference_stage_calls"] > 0
                state = client._core._transformer_states[roles.model_id]
                assert state.bundle.weight_storage == storage
                for stage in state.bundle.stages.values():
                    if stage.client_weight is not None:
                        assert isinstance(stage.client_weight, PagedGEMM) == (storage == "paged")
                if storage == "paged":
                    compiled = client._core._compiled_public_decoder(state, max_input_tokens=4, max_new_tokens=2)
                    head = state.bundle.stages["lm_head"]
                    state.bundle.stages["lm_head"] = dataclasses.replace(head, weight_digest="0" * 64)
                    with pytest.raises(RuntimeBindingError, match="digest"):
                        compiled.validate()
                    state.bundle.stages["lm_head"] = head
            trajectories.append(current.copy())
            if storage == "paged":
                with TestClient(roles.gateway_app(local_api_key="paged-local")) as api:
                    response = api.post("/v1/responses", headers={"Authorization": "Bearer paged-local"},
                        json={"model": roles.model_id, "input": "abc", "max_output_tokens": 2, "temperature": 0})
                    assert response.status_code == 200, response.text
                    text = "".join(part["text"] for item in response.json()["output"]
                                   if item["type"] == "message" for part in item["content"]
                                   if part["type"] == "output_text")
                    assert text == outputs[0][0][0]
    assert outputs[0] == outputs[1]
    assert len(trajectories[0]) == len(trajectories[1]) >= 3
    for (left, caches), (right, others) in zip(*trajectories, strict=True):
        np.testing.assert_array_equal(left, right)
        for pair, other in zip(caches, others, strict=True):
            for a, b in zip(pair, other, strict=True):
                np.testing.assert_array_equal(a, b)
