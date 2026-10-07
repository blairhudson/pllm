"""Selected client KV lifetime, compiler admission and actual SDK role paths."""
import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from pllm import Deployment, ExecutionBudget, Experiment, Model, lower_model
from pllm.configuration import ComponentRef, ConfigurationError, Pipeline
from pllm.passes import KvCacheEviction
from pllm.profiles import ClientOnlyCpu, MaskedLinearCpu, TwoOnlineOffsetCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.servers import build_roles
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.state import ClientLocalKv, ClientPrefixReuse
from pllm.verification import FreivaldsVerify


def retention(keep=(1, 2)):
    return ClientLocalKv(retention=KvCacheEviction(static_keep=keep,
        observation_window=(1, 4), dynamic_keep=(1, 1), share_adjacent_layers=False))


def experiment(tmp_path, *, prepared=False, keep=(1, 2)):
    path = create_tiny_llama_checkpoint(tmp_path / "checkpoint", num_hidden_layers=1,
        model_type="qwen2", with_qkv_bias=True)
    pipeline = MaskedLinearCpu if prepared else ClientOnlyCpu
    return Experiment(name="retained-sdk", pipeline=pipeline(Model.path(str(path), model_id="retained-sdk"),
        state=retention(keep), quantization=SymmetricPerRow(weight_bits=8, activation_bits=8)),
        deployment=Deployment.local(root="local://retained-sdk"),
        budget=ExecutionBudget(requests=4, max_input_tokens=256, max_new_tokens=3))


def test_component_roundtrip_native_schedule_and_memory_owners(tmp_path):
    value = experiment(tmp_path)
    restored = Experiment.from_spec(json.loads(json.dumps(value.to_spec())))
    assert restored == value
    assert restored.resolve().composition_digest == value.pipeline.digest()
    with pytest.raises(TypeError):
        value.pipeline.state.params["retention"]["params"]["static_keep"]["numerator"] = 9
    with pytest.raises(ConfigurationError, match="static"):
        ClientLocalKv(retention=KvCacheEviction())
    changed = value.pipeline.with_params(state__retention__params__static_keep={"numerator": 1, "denominator": 1})
    assert changed != value.pipeline
    plan = lower_model((tmp_path / "checkpoint" / "config.json").read_bytes(), batch=1,
        max_input_tokens=32, max_new_tokens=3)
    native = plan.runtime_schedule(value.pipeline)
    contract = native.to_dict()["retained_state"]
    from pllm.runtime.decoder_memory import decoder_memory
    memory = decoder_memory(plan, native)
    assert memory["mode"] == "retained_kv_live_roots_v1"
    assert memory["working_bytes"] + memory["state_bytes"] == contract["owned_array_bound_bytes"] + memory["rotary_coefficient_bytes"]
    assert contract["policy"] == value.pipeline.state.to_spec()["params"]["retention"]
    assert plan.runtime_schedule(changed).digest != native.digest
    for slot, component in (("verification", FreivaldsVerify()),
                            ("cache", ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=32)),
                            ("quantization", SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"))):
        wrong = Pipeline(value.pipeline.model, dict(value.pipeline.components) | {slot: component})
        with pytest.raises(ValueError):
            plan.runtime_schedule(wrong)
    with pytest.raises(ValueError):
        Pipeline(value.pipeline.model, dict(value.pipeline.components) | {
            "state": ComponentRef("pllm/client-local-kv", {"retention": {"component": "bad", "params": {}}})})
    with pytest.raises(ValueError):
        plan.runtime_schedule(Pipeline(value.pipeline.model,
            dict(TwoOnlineOffsetCpu(value.pipeline.model).components) | {"state": retention()}))
    large_config = json.loads((tmp_path / "checkpoint" / "config.json").read_bytes())
    large_config["max_position_embeddings"] = 1024
    oversized = lower_model(large_config, batch=1,
        max_input_tokens=513, max_new_tokens=3)
    with pytest.raises(ValueError, match="512"):
        oversized.runtime_schedule(value.pipeline)


@pytest.mark.parametrize("prepared", [False, True])
def test_sdk_prefill_decode_state_and_history_retire_on_completion(tmp_path, monkeypatch, prepared):
    value = experiment(tmp_path, prepared=prepared)
    recorded, native_owners = [], []
    run = SemanticDecoderRuntime._forward

    def observe(self, ids, **kwargs):
        result = run(self, ids, **kwargs)
        if self._retained is not None:
            recorded.append((self.position, result.copy(), tuple(row.gather() for row in self.caches)))
            native_owners.extend(self.caches)
        return result

    monkeypatch.setattr(SemanticDecoderRuntime, "_forward", observe)
    with build_roles(value) as roles, roles.client() as client:
        first = client.responses.create(input="A", max_output_tokens=3, temperature=0)
        assert first.usage.output_tokens == 3
        assert native_owners and all(owner.allocated_bytes == 0 for owner in native_owners)
        assert not client._core._transformer_conversations
        second = client.responses.create(input="B", previous_response_id=first.id,
            max_output_tokens=2, temperature=0)
        assert second.usage.input_tokens > first.usage.input_tokens
        assert all(owner.allocated_bytes == 0 for owner in native_owners)
        if prepared:
            assert len(roles.statuses) == 2
            assert client.privacy_audit.inference_stage_calls > 0
            assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
        else:
            assert client.privacy_audit.inference_stage_calls == 0
    assert recorded
    for _, _, layers in recorded:
        assert all(len(positions) < 256 for _, _, positions in layers)


def test_default_tiny_identity_agrees_with_experiment_before_gateway_start():
    value = Experiment(
        name="tiny-retained",
        pipeline=MaskedLinearCpu(Model.tiny(), state=retention()),
        deployment=Deployment.local(root="local://tiny-retained"),
        budget=ExecutionBudget(requests=1, max_input_tokens=32, max_new_tokens=2),
    )
    with pytest.raises(ValueError, match="model_id conflicts"):
        build_roles(value, model_id="wrong")
    with build_roles(value) as roles:
        with TestClient(roles.gateway_app(local_api_key="test-retained-key")) as gateway:
            reply = gateway.post('/v1/responses', headers={"Authorization": "Bearer test-retained-key"},
                json={"model": value.resolve().model, "input": "A", "max_output_tokens": 2, "temperature": 0})
            assert reply.status_code == 200, reply.text
            result = reply.json()
            assert result["model"] == value.resolve().model
            assert result["usage"]["output_tokens"] == 2
            assert result["status"] == "incomplete"
            assert result["incomplete_details"] == {"reason": "max_output_tokens"}


def test_gateway_full_retention_matches_dense_and_stream_cancel_releases_rows(tmp_path, monkeypatch):
    value = experiment(tmp_path, keep=(1, 1))
    base = dataclasses.replace(value, pipeline=value.pipeline.with_params(state=None))
    with build_roles(base) as roles, roles.client() as client:
        expected = client.responses.create(input="A", max_output_tokens=3, temperature=0)
    owners = []
    observe = SemanticDecoderRuntime._install_retained

    def install(self, contract):
        observe(self, contract)
        owners.append(self._retained)

    monkeypatch.setattr(SemanticDecoderRuntime, "_install_retained", install)
    with build_roles(value) as roles, TestClient(roles.gateway_app(local_api_key="test-retained")) as gateway:
        actual = gateway.post("/v1/responses", headers={"Authorization": "Bearer test-retained"},
            json={"model": value.resolve().model, "input": "A", "max_output_tokens": 3, "temperature": 0})
        assert actual.status_code == 200, actual.text
        text = "".join(part["text"] for item in actual.json()["output"] for part in item.get("content", [])
            if part["type"] == "output_text")
        assert text == expected.output_text
        assert actual.json()["usage"] == expected.usage.to_dict()
        with roles.client() as client:
            stream = client.responses.create(input="A", max_output_tokens=3, temperature=0, stream=True)
            for event in stream:
                if event.type == "response.output_text.delta":
                    break
            stream.close()
    assert owners and all(owner.failed and owner.allocated_bytes == 0 for owner in owners)


@pytest.mark.asyncio
async def test_async_sdk_cancel_retires_retained_rows(tmp_path, monkeypatch):
    from pllm import AsyncOpenAI
    owners = []
    observe = SemanticDecoderRuntime._install_retained

    def install(self, contract):
        observe(self, contract)
        owners.append(self._retained)

    monkeypatch.setattr(SemanticDecoderRuntime, "_install_retained", install)
    async with AsyncOpenAI(experiment=experiment(tmp_path)) as client:
        stream = await client.responses.create(input="A", max_output_tokens=3, temperature=0, stream=True)
        async for event in stream:
            if event.type == "response.output_text.delta":
                break
        assert owners and any(owner.allocated_bytes > 0 for owner in owners)
        await stream.aclose()
        assert all(owner.failed and owner.allocated_bytes == 0 for owner in owners)


@pytest.mark.asyncio
async def test_async_cancellation_drains_iteration_before_scoped_cleanup():
    import asyncio
    from contextvars import ContextVar
    import threading
    from types import SimpleNamespace
    from pllm.runtime.client import AsyncResponsesResource, ResponseStream

    owner = ContextVar("test_stream_owner", default=None)
    entered, release, retired = threading.Event(), threading.Event(), threading.Event()

    def events():
        token = owner.set("request")
        try:
            yield "first"
            assert owner.get() == "request"
            entered.set()
            assert release.wait(5)
            yield "second"
        finally:
            owner.reset(token)
            retired.set()

    resource = AsyncResponsesResource(SimpleNamespace(create=lambda **_: ResponseStream(events())))
    stream = await resource.create(stream=True)
    assert await anext(stream) == "first"
    pending = asyncio.create_task(anext(stream))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        pending.cancel()
        await asyncio.sleep(0)
        assert not retired.is_set()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, 5)
    assert retired.is_set() and owner.get() is None
