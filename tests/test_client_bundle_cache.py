from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
from pathlib import Path

import httpx
import msgpack
import pytest
from fastapi.testclient import TestClient

from pllm.runtime import GatewayConfig, create_app
from pllm.runtime.client import HEClientCore
from pllm.runtime.client import HEAPIError
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.tiny_gemma import create_tiny_gemma4_checkpoint
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def _bundle_payload(tmp_path: Path, model_id: str = "cache-model") -> bytes:
    root = create_tiny_gemma4_checkpoint(tmp_path / "model", num_hidden_layers=1)
    manifest = load_hf_directory(root, model_id=model_id)
    engine = MaskedTransformerEngine(threads=1)
    asyncio.run(engine.load(manifest))
    return engine.client_bundle(model_id)


def _mock_client(
    payload: bytes,
    *,
    base_url: str,
    calls: dict[str, int],
) -> httpx.Client:
    fingerprint = hashlib.sha256(payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/client-bundle"):
            calls["bundle"] = calls.get("bundle", 0) + 1
            return httpx.Response(
                200,
                content=payload,
                headers={"X-PLLM-Bundle-SHA256": fingerprint},
            )
        calls["descriptor"] = calls.get("descriptor", 0) + 1
        return httpx.Response(
            200,
            json={
                "id": "cache-model",
                "client_bundle": {
                    "schema": 2,
                    "sha256": fingerprint,
                    "size": len(payload),
                    "etag": f'"{fingerprint}"',
                },
            },
        )

    return httpx.Client(base_url=base_url, transport=httpx.MockTransport(handler))


def _revision(payload: bytes, marker: str, *, proprietary: bool = False) -> bytes:
    value = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    value["tokenizer"]["chat_template"] = marker
    if proprietary:
        value["client_weights"] = {}
        for stage in value["stages"].values():
            stage.pop("client_weight", None)
            stage.pop("client_aux_weight", None)
        value["privacy"].update(
            {
                "mode": "proprietary",
                "protocol": "direct_bfv_w4a4",
                "dense_weights_in_bundle": False,
                "local_quantized_stages": [],
            }
        )
    return msgpack.packb(value, use_bin_type=True)


def _core(
    payload: bytes,
    cache_dir: Path,
    calls: dict[str, int],
    *,
    base_url: str = "https://EXAMPLE.test:443/api/",
    api_key: str = "credential-a",
    mode: str = "read-write",
) -> HEClientCore:
    http = _mock_client(payload, base_url=base_url, calls=calls)
    return HEClientCore(
        base_url=base_url,
        api_key=api_key,
        http_client=http,
        bundle_cache_mode=mode,
        bundle_cache_dir=cache_dir,
    )


def test_bundle_cache_cold_warm_corruption_and_audit(tmp_path: Path):
    payload = _bundle_payload(tmp_path)
    cache = tmp_path / "cache"
    calls: dict[str, int] = {}

    cold = _core(payload, cache, calls)
    cold._load_client_bundle("cache-model")
    assert cold.audit.bundle_network_bytes == len(payload)
    assert cold.audit.bundle_cache_misses == 1
    assert cold.audit.bundle_cache_hits == 0
    assert next(cache.rglob("*.msgpack")).stat().st_mode & 0o777 == 0o600
    if os.name == "posix":
        assert cache.stat().st_mode & 0o777 == 0o700
        assert next(path for path in cache.iterdir() if path.is_dir()).stat().st_mode & 0o777 == 0o700
        assert (cache / ".identity-key").stat().st_mode & 0o777 == 0o600
    cold.close()

    warm = _core(payload, cache, calls, base_url="https://example.test/api")
    warm._load_client_bundle("cache-model")
    assert warm.audit.bundle_network_bytes == 0
    assert warm.audit.bundle_cache_hits == 1
    assert calls["bundle"] == 1
    warm.close()

    cached = next(cache.rglob("*.msgpack"))
    cached.write_bytes(b"corrupt")
    repaired = _core(payload, cache, calls)
    repaired._load_client_bundle("cache-model")
    assert repaired.audit.bundle_cache_corruptions == 1
    assert repaired.audit.bundle_cache_misses == 0
    assert repaired.audit.bundle_network_bytes == len(payload)
    assert cached.read_bytes() == payload
    repaired.close()


def test_bundle_cache_isolates_origin_and_credential(tmp_path: Path):
    payload = _bundle_payload(tmp_path)
    cache = tmp_path / "cache"
    calls: dict[str, int] = {}
    identities = (
        ("https://one.example/api", "credential-a"),
        ("https://two.example/api", "credential-a"),
        ("https://one.example/api", "credential-b"),
    )
    for origin, credential in identities:
        core = _core(payload, cache, calls, base_url=origin, api_key=credential)
        core._load_client_bundle("cache-model")
        assert core.audit.bundle_cache_misses == 1
        core.close()
    assert calls["bundle"] == 3
    assert len(list(cache.rglob("*.msgpack"))) == 3


def test_bundle_cache_modes(tmp_path: Path):
    payload = _bundle_payload(tmp_path)
    cache = tmp_path / "cache"
    calls: dict[str, int] = {}

    off = _core(payload, cache, calls, mode="off")
    off._load_client_bundle("cache-model")
    assert not cache.exists()
    assert off.audit.bundle_network_bytes == len(payload)
    off.close()

    refresh = _core(payload, cache, calls, mode="refresh")
    refresh._load_client_bundle("cache-model")
    refresh.close()
    assert len(list(cache.rglob("*.msgpack"))) == 1

    read_only = _core(payload, cache, calls, mode="read-only")
    read_only._load_client_bundle("cache-model")
    assert read_only.audit.bundle_cache_hits == 1
    assert read_only.audit.bundle_network_bytes == 0
    read_only.close()

    refreshed = _core(payload, cache, calls, mode="refresh")
    refreshed._load_client_bundle("cache-model")
    assert refreshed.audit.bundle_network_bytes == len(payload)
    assert refreshed.audit.bundle_cache_hits == 0
    refreshed.close()

    empty = tmp_path / "read-only-empty"
    read_only_miss = _core(payload, empty, calls, mode="read-only")
    read_only_miss._load_client_bundle("cache-model")
    assert read_only_miss.audit.bundle_cache_misses == 1
    assert read_only_miss.audit.bundle_network_bytes == len(payload)
    assert not empty.exists()
    read_only_miss.close()

    with pytest.raises(ValueError, match="bundle_cache_mode"):
        _core(payload, cache, calls, mode="invalid")


def test_bundle_cache_defaults_under_xdg_cache_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _bundle_payload(tmp_path)
    calls: dict[str, int] = {}
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    http = _mock_client(payload, base_url="https://example.test", calls=calls)
    core = HEClientCore(
        base_url="https://example.test",
        api_key="credential",
        http_client=http,
        bundle_cache_mode="off",
    )
    try:
        assert core.bundle_cache_dir == tmp_path / "xdg" / "pllm" / "client-bundles"
    finally:
        core.close()


def test_bundle_cache_lock_prevents_duplicate_concurrent_downloads(tmp_path: Path):
    payload = _bundle_payload(tmp_path)
    cache = tmp_path / "cache"
    calls: dict[str, int] = {}
    calls_lock = threading.Lock()
    fingerprint = hashlib.sha256(payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/client-bundle"):
            with calls_lock:
                calls["bundle"] = calls.get("bundle", 0) + 1
            time.sleep(0.05)
            return httpx.Response(
                200,
                content=payload,
                headers={"X-PLLM-Bundle-SHA256": fingerprint},
            )
        return httpx.Response(
            200,
            json={
                "client_bundle": {
                    "schema": 2,
                    "sha256": fingerprint,
                    "size": len(payload),
                    "etag": f'"{fingerprint}"',
                }
            },
        )

    cores = [
        HEClientCore(
            base_url="https://example.test/api",
            api_key="credential",
            http_client=httpx.Client(
                base_url="https://example.test/api", transport=httpx.MockTransport(handler)
            ),
            bundle_cache_dir=cache,
        )
        for _ in range(2)
    ]
    threads = [
        threading.Thread(target=core._load_client_bundle, args=("cache-model",))
        for core in cores
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for core in cores:
        core.close()

    assert calls["bundle"] == 1
    assert sorted(core.audit.bundle_cache_hits for core in cores) == [0, 1]


def test_bundle_revision_replaces_stable_record_and_rebuilds_transformer_state(
    tmp_path: Path,
):
    original = _bundle_payload(tmp_path)
    revisions = [
        _revision(original, "revision-one", proprietary=True),
        _revision(original, "revision-two", proprietary=True),
    ]
    selected = 0
    calls: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = revisions[selected]
        fingerprint = hashlib.sha256(payload).hexdigest()
        if request.method == "POST" and request.url.path.endswith("/v1/he/sessions"):
            calls["session"] = calls.get("session", 0) + 1
            return httpx.Response(200, json={"id": f"session-{calls['session']}"})
        if request.url.path.endswith("/client-bundle"):
            calls["bundle"] = calls.get("bundle", 0) + 1
            return httpx.Response(
                200,
                content=payload,
                headers={"X-PLLM-Bundle-SHA256": fingerprint},
            )
        calls["descriptor"] = calls.get("descriptor", 0) + 1
        return httpx.Response(
            200,
            json={
                "metadata": {"client_runtime": "direct_fhe_transformer_v1"},
                "client_bundle": {
                    "schema": 2,
                    "sha256": fingerprint,
                    "size": len(payload),
                    "etag": f'"{fingerprint}"',
                },
            },
        )

    core = HEClientCore(
        base_url="https://example.test",
        api_key="credential",
        http_client=httpx.Client(
            base_url="https://example.test", transport=httpx.MockTransport(handler)
        ),
        bundle_cache_dir=tmp_path / "cache",
    )
    try:
        _, first, _ = core._open_transformer_session("cache-model", max_output_tokens=1)
        selected = 1
        _, second, _ = core._open_transformer_session("cache-model", max_output_tokens=1)
    finally:
        core.close()

    assert first is not second
    assert first.bundle.privacy["body_fingerprint"] == second.bundle.privacy["body_fingerprint"]
    assert first.bundle.tokenizer_descriptor["chat_template"] == "revision-one"
    assert second.bundle.tokenizer_descriptor["chat_template"] == "revision-two"
    cached = list((tmp_path / "cache").rglob("*.msgpack"))
    assert len(cached) == 1
    assert cached[0].read_bytes() == revisions[1]
    assert cached[0].stat().st_size <= len(revisions[1])
    assert calls == {"descriptor": 2, "bundle": 2, "session": 2}
    assert core.audit.bundle_cache_misses == 2
    assert core.audit.bundle_cache_corruptions == 0


def test_unavailable_default_cache_falls_back_but_explicit_cache_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _bundle_payload(tmp_path)
    blocked = tmp_path / "not-a-directory"
    blocked.write_bytes(b"file")
    monkeypatch.setenv("XDG_CACHE_HOME", str(blocked))

    calls: dict[str, int] = {}
    implicit = HEClientCore(
        base_url="https://example.test",
        api_key="credential",
        http_client=_mock_client(payload, base_url="https://example.test", calls=calls),
    )
    try:
        assert implicit._load_client_bundle("cache-model").model_id == "cache-model"
        assert implicit.audit.bundle_network_bytes == len(payload)
    finally:
        implicit.close()

    explicit = _core(payload, blocked / "cache", calls)
    try:
        with pytest.raises(HEAPIError, match="configured bundle cache is unavailable"):
            explicit._load_client_bundle("cache-model")
    finally:
        explicit.close()


def test_descriptor_download_race_retries_once(tmp_path: Path):
    first = _revision(_bundle_payload(tmp_path), "old")
    second = _revision(first, "new")
    descriptors = [first, second]
    descriptor_calls = 0
    bundle_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal descriptor_calls, bundle_calls
        if request.url.path.endswith("/client-bundle"):
            bundle_calls += 1
            fingerprint = hashlib.sha256(second).hexdigest()
            return httpx.Response(
                200,
                content=second,
                headers={"X-PLLM-Bundle-SHA256": fingerprint},
            )
        payload = descriptors[min(descriptor_calls, 1)]
        descriptor_calls += 1
        fingerprint = hashlib.sha256(payload).hexdigest()
        return httpx.Response(
            200,
            json={
                "client_bundle": {
                    "schema": 2,
                    "sha256": fingerprint,
                    "size": len(payload),
                    "etag": f'"{fingerprint}"',
                }
            },
        )

    core = HEClientCore(
        base_url="https://example.test",
        api_key="credential",
        http_client=httpx.Client(
            base_url="https://example.test", transport=httpx.MockTransport(handler)
        ),
        bundle_cache_dir=tmp_path / "cache",
    )
    try:
        bundle = core._load_client_bundle("cache-model")
    finally:
        core.close()
    assert bundle.tokenizer_descriptor["chat_template"] == "new"
    assert descriptor_calls == 2
    assert bundle_calls == 2


def test_network_integrity_failure_does_not_retry_unchanged_revision(tmp_path: Path):
    payload = _bundle_payload(tmp_path)
    calls: dict[str, int] = {}
    fingerprint = hashlib.sha256(payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/client-bundle"):
            calls["bundle"] = calls.get("bundle", 0) + 1
            return httpx.Response(
                200,
                content=payload + b"tampered",
                headers={"X-PLLM-Bundle-SHA256": fingerprint},
            )
        calls["descriptor"] = calls.get("descriptor", 0) + 1
        return httpx.Response(
            200,
            json={
                "client_bundle": {
                    "schema": 2,
                    "sha256": fingerprint,
                    "size": len(payload),
                    "etag": f'"{fingerprint}"',
                }
            },
        )

    core = HEClientCore(
        base_url="https://example.test",
        api_key="credential",
        http_client=httpx.Client(
            base_url="https://example.test", transport=httpx.MockTransport(handler)
        ),
        bundle_cache_dir=tmp_path / "cache",
    )
    try:
        with pytest.raises(HEAPIError, match="fingerprint mismatch"):
            core._load_client_bundle("cache-model")
    finally:
        core.close()
    assert calls == {"descriptor": 2, "bundle": 1}
    assert not list((tmp_path / "cache").rglob("*.msgpack"))


def test_absurd_descriptor_size_is_rejected_before_download(tmp_path: Path):
    calls: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/client-bundle"):
            calls["bundle"] = calls.get("bundle", 0) + 1
            return httpx.Response(500)
        calls["descriptor"] = calls.get("descriptor", 0) + 1
        return httpx.Response(
            200,
            json={
                "client_bundle": {
                    "schema": 2,
                    "sha256": "0" * 64,
                    "size": 10**30,
                    "etag": '"oversized"',
                }
            },
        )

    core = HEClientCore(
        base_url="https://example.test",
        api_key="credential",
        http_client=httpx.Client(
            base_url="https://example.test", transport=httpx.MockTransport(handler)
        ),
        bundle_cache_dir=tmp_path / "cache",
    )
    try:
        with pytest.raises(HEAPIError, match="descriptor is invalid"):
            core._load_client_bundle("cache-model")
    finally:
        core.close()
    assert calls == {"descriptor": 1}


def test_server_exposes_fingerprint_etag_and_memoizes_bundle(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(tmp_path / "server-model", num_hidden_layers=1)
    engine = MaskedTransformerEngine(threads=1)
    calls = 0
    original = engine.client_bundle

    def counted(model_id: str, *, include_local_weights: bool = True) -> bytes:
        nonlocal calls
        calls += 1
        return original(model_id, include_local_weights=include_local_weights)

    engine.client_bundle = counted  # type: ignore[method-assign]
    app = create_app(
        GatewayConfig(api_keys=("key",), allow_insecure_local_correlations=True),
        engines={engine.capabilities.name: engine},
    )
    headers = {"Authorization": "Bearer key"}
    with TestClient(app) as client:
        loaded = client.post(
            "/v1/he/models/load",
            headers=headers,
            json={
                "engine": engine.capabilities.name,
                "kind": "huggingface",
                "path": str(root),
                "model_id": "server-model",
            },
        )
        assert loaded.status_code == 200
        descriptor = client.get("/v1/he/models/server-model", headers=headers).json()[
            "client_bundle"
        ]
        response = client.get("/v1/he/models/server-model/client-bundle", headers=headers)
        assert response.status_code == 200
        assert response.headers["etag"] == descriptor["etag"]
        assert response.headers["x-pllm-bundle-sha256"] == descriptor["sha256"]
        assert hashlib.sha256(response.content).hexdigest() == descriptor["sha256"]
        unchanged = client.get(
            "/v1/he/models/server-model/client-bundle",
            headers={**headers, "If-None-Match": descriptor["etag"]},
        )
        assert unchanged.status_code == 304
        listed = client.get("/v1/models", headers=headers).json()["data"]
        model = next(row for row in listed if row["id"] == "server-model")
        assert model["client_bundle"] == descriptor
        assert calls == 1
