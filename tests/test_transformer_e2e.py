from pathlib import Path

import httpx
import pytest

from conftest import start_gateway
from pllm.runtime import OpenAI
from pllm.runtime.client import HEAPIError
from pllm.runtime.tiny_gemma import create_tiny_gemma4_checkpoint
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def test_tiny_gemma_responses_api_keeps_prompt_local(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(tmp_path / "tiny")
    engine = MaskedTransformerEngine(threads=1)
    gateway = start_gateway(engines={engine.capabilities.name: engine})
    canary = "PRIVATE-CANARY-4b6a8739"
    try:
        with httpx.Client(base_url=gateway.base_url, timeout=30) as admin:
            loaded = admin.post(
                "/v1/he/models/load",
                headers={"Authorization": f"Bearer {gateway.api_key}"},
                json={
                    "engine": engine.capabilities.name,
                    "kind": "huggingface",
                    "path": str(root),
                    "model_id": "tiny-gemma-he",
                },
            )
            assert loaded.status_code == 200, loaded.text
            assert loaded.json()["status"] == "ready"

        with OpenAI(
            api_key=gateway.api_key,
            base_url=gateway.base_url,
            correlation_mode="local-test",
            correlation_prefetch=1,
        ) as client:
            response = client.responses.create(
                model="tiny-gemma-he",
                input=canary,
                max_output_tokens=2,
                temperature=0,
            )
            assert response.status == "completed"
            assert response.usage is not None
            assert response.usage.output_tokens <= 2
            audit = client.privacy_audit.to_dict()
            assert audit["plaintext_prompt_bytes_sent"] == 0
            assert audit["plaintext_token_ids_sent"] == 0
            assert audit["online_steps"] > 0
            assert audit["correlation_count"] > 0

        raw_audit = b"\n".join(payload for _, payload in gateway.audit)
        assert canary.encode() not in raw_audit
        assert engine.stats()["execute_items"] > 0
    finally:
        gateway.close()


def test_two_provider_execution_matches_clear_w8_and_sends_independent_shares(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(
        tmp_path / "tiny-two-provider",
        num_hidden_layers=1,
        ple_dim=0,
    )
    engines = [MaskedTransformerEngine(threads=1), MaskedTransformerEngine(threads=1)]
    gateways = [start_gateway(engines={engine.capabilities.name: engine}) for engine in engines]
    model_id = "tiny-two-provider"
    try:
        for gateway, engine in zip(gateways, engines):
            with httpx.Client(base_url=gateway.base_url, timeout=30) as admin:
                loaded = admin.post(
                    "/v1/he/models/load",
                    headers={"Authorization": f"Bearer {gateway.api_key}"},
                    json={
                        "engine": engine.capabilities.name,
                        "kind": "huggingface",
                        "path": str(root),
                        "model_id": model_id,
                    },
                )
                assert loaded.status_code == 200, loaded.text
        prompt = "PRIVATE-TWO-PROVIDER-CANARY"
        with OpenAI(
            api_key=gateways[0].api_key,
            base_url=gateways[0].base_url,
            correlation_mode="local-test",
            correlation_prefetch=1,
        ) as clear_client:
            clear_response = clear_client.responses.create(
                model=model_id,
                input=prompt,
                max_output_tokens=2,
                temperature=0,
            )
        with OpenAI(
            api_key=gateways[0].api_key,
            base_url=gateways[0].base_url,
            execution_strategy="two-provider",
            secondary_base_url=gateways[1].base_url,
            secondary_api_key=gateways[1].api_key,
        ) as client:
            response = client.responses.create(
                model=model_id,
                input=prompt,
                max_output_tokens=2,
                temperature=0,
            )
            assert response.status == "completed"
            assert response.output_text == clear_response.output_text
            assert response.usage is not None
            assert response.usage.output_tokens <= 2
            audit = client.privacy_audit.to_dict()
            assert audit["plaintext_prompt_bytes_sent"] == 0
            assert audit["plaintext_token_ids_sent"] == 0
            assert audit["correlation_count"] == 0
            assert audit["direct_share_primary_upload_bytes"] > 0
            assert audit["direct_share_secondary_upload_bytes"] > 0
            assert audit["online_steps"] > 0
        for gateway, engine in zip(gateways, engines):
            raw_audit = b"\n".join(payload for _, payload in gateway.audit)
            assert prompt.encode() not in raw_audit
            assert engine.stats()["execute_items"] > 0
    finally:
        for gateway in gateways:
            gateway.close()


def test_two_provider_rejects_mismatched_body_weights(tmp_path: Path):
    roots = [
        create_tiny_gemma4_checkpoint(tmp_path / "primary", seed=17),
        create_tiny_gemma4_checkpoint(tmp_path / "secondary", seed=18),
    ]
    engines = [MaskedTransformerEngine(threads=1), MaskedTransformerEngine(threads=1)]
    gateways = [start_gateway(engines={engine.capabilities.name: engine}) for engine in engines]
    try:
        for root, gateway, engine in zip(roots, gateways, engines):
            with httpx.Client(base_url=gateway.base_url, timeout=30) as admin:
                loaded = admin.post(
                    "/v1/he/models/load",
                    headers={"Authorization": f"Bearer {gateway.api_key}"},
                    json={
                        "engine": engine.capabilities.name,
                        "kind": "huggingface",
                        "path": str(root),
                        "model_id": "tiny-mismatch",
                    },
                )
                assert loaded.status_code == 200, loaded.text
        with OpenAI(
            api_key=gateways[0].api_key,
            base_url=gateways[0].base_url,
            execution_strategy="two-provider",
            secondary_base_url=gateways[1].base_url,
            secondary_api_key=gateways[1].api_key,
        ) as client:
            with pytest.raises(HEAPIError, match="commitments do not match"):
                client.responses.create(
                    model="tiny-mismatch",
                    input="mismatched providers",
                    max_output_tokens=1,
                    temperature=0,
                )
    finally:
        for gateway in gateways:
            gateway.close()


def test_previous_response_id_reuses_private_kv_and_token_cache(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(
        tmp_path / "tiny-continuation",
        num_hidden_layers=1,
        ple_dim=4,
    )
    engine = MaskedTransformerEngine(threads=1)
    gateway = start_gateway(engines={engine.capabilities.name: engine})
    try:
        with httpx.Client(base_url=gateway.base_url, timeout=30) as admin:
            loaded = admin.post(
                "/v1/he/models/load",
                headers={"Authorization": f"Bearer {gateway.api_key}"},
                json={
                    "engine": engine.capabilities.name,
                    "kind": "huggingface",
                    "path": str(root),
                    "model_id": "tiny-continuation-he",
                },
            )
            assert loaded.status_code == 200, loaded.text

        with OpenAI(
            api_key=gateway.api_key,
            base_url=gateway.base_url,
            correlation_mode="local-test",
            correlation_prefetch=1,
            token_cache_size=128,
        ) as client:
            first = client.responses.create(
                model="tiny-continuation-he",
                input="repeat repeat repeat",
                max_output_tokens=1,
                temperature=0,
            )
            before_qkv_rows = engine.models["tiny-continuation-he"].stages[
                "layers.0.self_attn.qkv_proj"
            ].rows
            second = client.responses.create(
                model="tiny-continuation-he",
                previous_response_id=first.id,
                input="repeat again",
                max_output_tokens=1,
                temperature=0,
            )
            after_qkv_rows = engine.models["tiny-continuation-he"].stages[
                "layers.0.self_attn.qkv_proj"
            ].rows
            audit = client.privacy_audit.to_dict()
            assert second.status == "completed"
            assert audit["kv_continuation_hits"] == 1
            assert audit["kv_continuation_misses"] == 0
            assert audit["token_lookup_cache_hits"] == 0
            assert engine.models["tiny-continuation-he"].stages["token_lookup"].calls == 0
            assert engine.models["tiny-continuation-he"].stages["lm_head"].calls == 0
            # The continuation executes only the newly appended template suffix,
            # not the full context represented by the public usage count.
            assert after_qkv_rows - before_qkv_rows < second.usage.input_tokens
    finally:
        gateway.close()


def test_transformer_preprocess_populates_per_stage_inventory(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(
        tmp_path / "tiny-preprocess",
        num_hidden_layers=1,
        ple_dim=0,
    )
    engine = MaskedTransformerEngine(threads=1)
    gateway = start_gateway(engines={engine.capabilities.name: engine})
    try:
        with httpx.Client(base_url=gateway.base_url, timeout=30) as admin:
            loaded = admin.post(
                "/v1/he/models/load",
                headers={"Authorization": f"Bearer {gateway.api_key}"},
                json={
                    "engine": engine.capabilities.name,
                    "kind": "huggingface",
                    "path": str(root),
                    "model_id": "tiny-preprocess-he",
                },
            )
            assert loaded.status_code == 200, loaded.text
        with OpenAI(
            api_key=gateway.api_key,
            base_url=gateway.base_url,
            correlation_mode="local-test",
            correlation_prefetch=1,
        ) as client:
            result = client.he.preprocess(
                model="tiny-preprocess-he",
                correlations=3,
                stages=["layers.0.self_attn.qkv_proj"],
            )
            assert result["generated"] == 3
            assert result["available_per_stage"] == {
                "layers.0.self_attn.qkv_proj": 3
            }
            again = client.he.preprocess(
                model="tiny-preprocess-he",
                correlations=3,
                stages=["layers.0.self_attn.qkv_proj"],
            )
            assert again["generated"] == 0
    finally:
        gateway.close()


def test_official_sdk_factory_forwards_transformer_cache_options(monkeypatch):
    import sys
    from types import SimpleNamespace
    import pllm.runtime.official as official

    captured = {}

    class FakeTransport:
        def __init__(self, **kwargs):
            captured["transport"] = kwargs

    class FakeOfficialClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

    monkeypatch.setattr(official, "HETransport", FakeTransport)
    monkeypatch.setattr(official._httpx, "Client", lambda *, transport: ("http-client", transport))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOfficialClient))
    result = official.create_openai_client(
        gateway_url="http://gateway",
        gateway_api_key="secret",
        correlation_prefetch=9,
        token_cache_size=2048,
    )
    assert isinstance(result, FakeOfficialClient)
    assert captured["transport"]["correlation_prefetch"] == 9
    assert captured["transport"]["token_cache_size"] == 2048
    assert captured["client"]["http_client"][0] == "http-client"
