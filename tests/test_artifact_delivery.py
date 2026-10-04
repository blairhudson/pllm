"""Authenticated artifact delivery, exact import, locality, and failure ordering."""
from __future__ import annotations

import asyncio
import hashlib

import httpx
import msgpack
import pytest
from fastapi.testclient import TestClient

from pllm import Model
from pllm.model_loader import resolve_model
from pllm.runtime import GatewayConfig, create_app
from pllm.runtime.bundle_artifacts import ENCODING, export_bundle, parse_manifest
from pllm.runtime.bundle_compression import ENCODING as FRAME_ENCODING, encode_bundle_frames, decode_bundle_frames
from pllm.runtime.client import ProtocolError, RuntimeClient
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_engine import MaskedTransformerEngine


@pytest.fixture
def delivery(tmp_path):
    root = create_tiny_llama_checkpoint(tmp_path / "weights", hidden_size=128,
                                       intermediate_size=512, head_dim=32)
    payloads = []
    for roles in ((), ("attention_output", "qkv_projection")):
        engine = MaskedTransformerEngine(threads=1, client_linear_roles=roles)
        asyncio.run(engine.load(resolve_model(Model.path(str(root), model_id="artifact-model")).manifest))
        payloads.append(engine.client_bundle("artifact-model"))
    return root, payloads


def mocked_core(tmp_path, payload, *, mutate=None, unsupported=False, redirect=False,
                encoding="artifacts", damage_object=None):
    exported = export_bundle(payload)
    digest = hashlib.sha256(payload).hexdigest()
    calls = []

    def handler(request):
        calls.append(request.url.path)
        assert request.url.host == "inference.test"
        assert request.headers["Authorization"] == "Bearer key"
        if request.url.path.endswith("/client-bundle"):
            if redirect:
                return httpx.Response(302, headers={"Location": "https://other.test/private"})
            body = payload if unsupported else exported.manifest
            if mutate is not None:
                body = mutate(body)
            return httpx.Response(200, content=body, headers={
                "X-PLLM-Bundle-SHA256": digest,
                **({} if unsupported else {"X-PLLM-Bundle-Encoding": ENCODING}),
            })
        if "/client-bundle-objects/" in request.url.path:
            key = request.url.path.rsplit("/", 1)[1]
            data = exported.objects[key]
            headers = {
                "X-PLLM-Bundle-SHA256": digest, "X-PLLM-Object-SHA256": key,
            }
            if request.headers.get("X-PLLM-Accept-Object-Encoding") == FRAME_ENCODING:
                headers.update({"X-PLLM-Object-Encoding": FRAME_ENCODING,
                                "X-PLLM-Object-Raw-Size": str(len(data))})
                data = b"".join(encode_bundle_frames(data))
            if damage_object:
                data, headers = damage_object(data, headers)
            return httpx.Response(200, content=data, headers=headers)
        return httpx.Response(200, json={"client_bundle": {
            "schema": 2, "size": len(payload), "sha256": digest, "etag": f'"{digest}"',
        }})

    core = RuntimeClient(base_url="https://inference.test", api_key="key",
                         bundle_compression=encoding,
                         bundle_cache_dir=tmp_path / "cache",
                         http_client=httpx.Client(base_url="https://inference.test",
                                                  transport=httpx.MockTransport(handler)))
    return core, calls


def test_sdk_cache_baseline_attention_baseline_and_corruption(delivery, tmp_path):
    _, payloads = delivery
    cores = []
    for payload in (payloads[0], payloads[1], payloads[0]):
        core, calls = mocked_core(tmp_path, payload)
        cores.append(core)
        assert core._load_client_bundle("artifact-model").schema_version == 2
        stats = core.artifact_cache_stats
        assert stats.manifest_requests == 1
        assert core.audit.bundle_network_bytes == stats.manifest_download_bytes + stats.object_download_bytes
        assert len(calls) == 2 + stats.object_requests
        core.close()
    cold, attention, warm = [c.artifact_cache_stats for c in cores]
    assert attention.hits >= cold.object_requests
    assert attention.object_requests == len(set(export_bundle(payloads[1]).objects) -
                                            set(export_bundle(payloads[0]).objects)) > 0
    assert warm.object_requests == warm.object_download_bytes == 0
    assert warm.manifest_download_bytes > 0
    assert not list((tmp_path / "cache").rglob("*.msgpack"))
    corrupt = next((tmp_path / "cache" / "public-artifacts").glob("*.blob"))
    # Choose an object present in baseline, rather than an attention-only matrix.
    baseline_keys = export_bundle(payloads[0]).objects
    corrupt = next(p for p in corrupt.parent.glob("*.blob") if p.stem in baseline_keys)
    corrupt.write_bytes(b"bad")
    core, _ = mocked_core(tmp_path, payloads[0])
    try:
        core._load_client_bundle("artifact-model")
        assert core.artifact_cache_stats.corruptions == core.audit.bundle_cache_corruptions == 1
        assert core.artifact_cache_stats.object_requests == 1
        assert corrupt.read_bytes() == baseline_keys[corrupt.stem]
    finally:
        core.close()


@pytest.mark.parametrize("damage", ["unsupported", "redirect", "digest", "domain", "graph"])
def test_explicit_artifacts_fail_closed_before_material_reservation(delivery, tmp_path, damage):
    _, payloads = delivery

    def mutate(body):
        value = msgpack.unpackb(body, raw=False)
        if damage == "digest":
            value["raw"]["sha256"] = "0" * 64
        elif damage == "domain":
            value["objects"][0]["metadata"]["shape"] = [True]
        elif damage == "graph":
            graph = msgpack.unpackb(value["skeleton"], raw=False)
            graph["config"]["hidden_size"] += 1
            value["skeleton"] = msgpack.packb(graph, use_bin_type=True)
        return msgpack.packb(value, use_bin_type=True)

    core, calls = mocked_core(tmp_path, payloads[0],
                              mutate=mutate if damage in {"digest", "domain", "graph"} else None,
                              unsupported=damage == "unsupported", redirect=damage == "redirect")
    try:
        with pytest.raises(ProtocolError):
            core._load_client_bundle("artifact-model")
        assert core.audit.preparation_upload_bytes == 0
        assert core.audit.session_authorization_upload_bytes == 0
        assert not core._transformer_states
        if damage != "graph":
            assert not any("/client-bundle-objects/" in p for p in calls)
    finally:
        core.close()


def test_http_manifest_objects_bearer_and_raw_binding(delivery):
    root, _ = delivery
    engine = MaskedTransformerEngine(threads=1)
    app = create_app(GatewayConfig(api_keys=("key",), allow_insecure_local_correlations=True),
                     engines={engine.capabilities.name: engine})
    headers = {"Authorization": "Bearer key"}
    with TestClient(app) as client:
        loaded = client.post("/v1/runtime/models/load", headers=headers, json={
            "engine": engine.capabilities.name, "kind": "huggingface", "path": str(root),
            "model_id": "artifact-model",
        })
        assert loaded.status_code == 200
        descriptor = client.get("/v1/runtime/models/artifact-model", headers=headers).json()["client_bundle"]
        url = "/v1/runtime/models/artifact-model/client-bundle-artifacts"
        assert client.get(url).status_code == 401
        response = client.get(url, headers=headers)
        assert response.status_code == 200, response.text
        manifest = parse_manifest(response.content, fingerprint=descriptor["sha256"], size=descriptor["size"])
        row = manifest["objects"][0]
        url = f"/v1/runtime/models/artifact-model/client-bundle-objects/{descriptor['sha256']}/{row['sha256']}"
        assert client.get(url).status_code == 401
        response = client.get(url, headers=headers)
        assert response.status_code == 200
        assert hashlib.sha256(response.content).hexdigest() == row["content_sha256"]
        compressed = client.get(url, headers={**headers, "X-PLLM-Accept-Object-Encoding": FRAME_ENCODING})
        assert compressed.headers["X-PLLM-Object-Encoding"] == FRAME_ENCODING
        decoded = decode_bundle_frames(
            [compressed.content], expected_size=row["size"], on_wire_bytes=lambda _: None)
        assert decoded == response.content
        assert client.get(url, headers={**headers, "X-PLLM-Accept-Object-Encoding": "unknown"}).status_code == 400
        assert client.get(url.replace(descriptor["sha256"], "0" * 64), headers=headers).status_code == 404
        assert client.get("/v1/runtime/models/artifact-model/client-bundle",
                           headers={**headers, "X-PLLM-Accept-Bundle-Encoding": ENCODING}).status_code == 200


def test_compressed_artifacts_share_raw_cache_and_count_encoded_bodies(delivery, tmp_path):
    _, payloads = delivery
    raw, _ = mocked_core(tmp_path / "control", payloads[0])
    compressed, _ = mocked_core(tmp_path / "shared", payloads[0], encoding="artifacts-zlib")
    try:
        raw_bundle = raw._load_client_bundle("artifact-model")
        compressed_bundle = compressed._load_client_bundle("artifact-model")
        assert raw_bundle.model_id == compressed_bundle.model_id
        assert compressed.audit.bundle_network_bytes < raw.audit.bundle_network_bytes
        assert compressed.audit.bundle_network_bytes == (
            compressed.artifact_cache_stats.object_download_bytes + compressed.artifact_cache_stats.manifest_download_bytes)
        exported = export_bundle(payloads[0])
        for key, content in exported.objects.items():
            assert (tmp_path / "shared/cache/public-artifacts" / f"{key}.blob").read_bytes() == content
    finally:
        raw.close()
        compressed.close()
    warm, _ = mocked_core(tmp_path / "shared", payloads[0])
    try:
        warm._load_client_bundle("artifact-model")
        assert warm.artifact_cache_stats.object_requests == 0
        assert warm.audit.bundle_network_bytes == warm.artifact_cache_stats.manifest_download_bytes
    finally:
        warm.close()


@pytest.mark.parametrize("damage", ["ack", "size", "truncated", "trailing", "inflation", "content"])
def test_compressed_artifact_rejects_bad_frames_before_material(delivery, tmp_path, damage):
    def mutate(data, headers):
        if damage == "ack":
            headers.pop("X-PLLM-Object-Encoding")
        elif damage == "size":
            headers["X-PLLM-Object-Raw-Size"] = "1"
        elif damage == "truncated":
            data = data[:-1]
        elif damage == "trailing":
            data += b"extra"
        elif damage == "inflation":
            data = (2**32 - 1).to_bytes(4, "big") + data[4:]
        else:
            data = b"".join(encode_bundle_frames(b"x" * int(headers["X-PLLM-Object-Raw-Size"])))
        return data, headers

    core, _ = mocked_core(tmp_path, delivery[1][0], encoding="artifacts-zlib", damage_object=mutate)
    try:
        with pytest.raises(ProtocolError):
            core._load_client_bundle("artifact-model")
        assert not core._transformer_states
        assert core.audit.preparation_upload_bytes == core.audit.session_authorization_upload_bytes == 0
        assert not list((tmp_path / "cache").rglob("*.blob"))
    finally:
        core.close()
