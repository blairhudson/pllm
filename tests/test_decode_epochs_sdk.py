"""Generated, real-provider epoch lifecycle checks; no checkpoint cohorts.

Runtime-only tests select the private core switch after admitting a compatible
baseline. Public component/native admission is tested separately by integration.
This lets runtime regressions run against an older installed native extension.
"""
import numpy as np
import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.state import ClientPrefixReuse


def epoch_experiment(tmp_path, family="qwen2", epoch=None):
    root = create_tiny_llama_checkpoint(tmp_path / "model", model_type=family, seed=179,
        hidden_size=32, intermediate_size=64, head_dim=8, num_hidden_layers=1,
        with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3", vocab_size=258)
    inventory = dict(allocation="demand", refill="on-demand")
    if epoch is not None:
        inventory["decode_epoch_tokens"] = epoch
    return Experiment("epoch-sdk", MaskedLinearCpu(Model.path(str(root), model_id="epoch-sdk"),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
        inventory=PreparedInventory(**inventory),
        cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=128)),
        Deployment.local(root=str(tmp_path / "roles")),
        ExecutionBudget(requests=16, max_input_tokens=128, max_new_tokens=7))


def numeric_state(saved):
    return [saved.next_logits, *(value[:row.length] for row in saved.snapshot.caches
                                for value in (row.key, row.value))]


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("epoch", [1, 3])
@pytest.mark.parametrize("transport", ["http", "websocket"])
def test_same_response_exact_state_rng_fresh_sessions_and_accounting(tmp_path, family, epoch, transport):
    exp = epoch_experiment(tmp_path, family)
    with build_roles(exp, engine_threads=1) as roles, roles.client(
        session_transport=transport, bundle_cache_dir=tmp_path / "cache") as client:
        core = client._core
        control = client.responses.create(input="abc", max_output_tokens=7, temperature=0.8, top_p=0.9)
        control_state = core._transformer_conversations[control.id]
        core._transformer_states["epoch-sdk"].prefill_cache.clear()
        core.preparation_decode_epoch_tokens = epoch
        records, sessions, inventories = [], [], []
        original = core.http.post
        original_get = core.http.get
        control_bodies = [0, 0]

        def record_get(path, **kwargs):
            result = original_get(path, **kwargs)
            if path.startswith("/v1/runtime/inventories/"):
                control_bodies[1] += len(result.content)
            return result

        def record(path, **kwargs):
            if path == "/v1/runtime/inventories":
                assert core._online_active == 0
                assert core._transformer_states["epoch-sdk"].active_prepared_responses == 0
            result = original(path, **kwargs)
            if (path == "/v1/runtime/inventories" or path == "/v1/runtime/sessions"
                    or path.endswith(("/ready", "/complete", "/cancel"))):
                control_bodies[0] += len(result.request.content)
                control_bodies[1] += len(result.content)
            records.append((path, kwargs.get("json"), result.status_code))
            if path == "/v1/runtime/sessions" and result.is_success:
                sessions.append(result.json()["id"])
            if path == "/v1/runtime/inventories" and result.is_success:
                inventories.append(result.json()["id"])
            return result

        core.http.post = record
        core.http.get = record_get
        before = core.audit.to_dict()
        events = list(client.responses.create(input="abc", max_output_tokens=7,
            temperature=0.8, top_p=0.9, stream=True))
        final = events[-1].response
        saved = core._transformer_conversations[final["id"]]
        assert final["output"] == [replace_output_id(control.output[0], final["output"][0]["id"])]
        assert final["usage"] == control.usage.to_dict()
        assert [event.sequence_number for event in events] == list(range(len(events)))
        assert {event.response["id"] for event in events if "response" in event.data} == {final["id"]}
        assert saved.token_ids == control_state.token_ids
        assert saved.pending_token_ids == control_state.pending_token_ids
        assert saved.snapshot.position == control_state.snapshot.position
        for actual, expected in zip(numeric_state(saved), numeric_state(control_state), strict=True):
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        count = (control.usage.output_tokens + epoch - 1) // epoch
        assert len(sessions) == len(set(sessions)) == count
        assert len(inventories) == len(set(inventories)) == count
        for index, session in enumerate(sessions[:-1]):
            completed = next(i for i, row in enumerate(records) if row[0] == f"/v1/runtime/sessions/{session}/complete")
            next_inventory = next(i for i, row in enumerate(records) if row[0] == "/v1/runtime/inventories"
                                  and i > completed)
            assert completed < next_inventory
        audit = core.audit.to_dict()
        reserved = audit["prepared_stage_rows_reserved"] - before["prepared_stage_rows_reserved"]
        claimed = audit["prepared_stage_rows_claimed"] - before["prepared_stage_rows_claimed"]
        burned = audit["prepared_stage_rows_burned"] - before["prepared_stage_rows_burned"]
        assert reserved == claimed + burned
        assert audit["decode_epoch_pauses"] - before["decode_epoch_pauses"] == count - 1
        assert audit["decode_epochs_started"] - before["decode_epochs_started"] == count
        assert audit["generation_decode_output_tokens"] - before["generation_decode_output_tokens"] == control.usage.output_tokens - 1
        online = sum(audit[name] - before[name] for name in
                     ("inference_upload_bytes", "inference_download_bytes"))
        generation_online = sum(audit[name] - before[name] for name in
                                ("generation_prefill_online_body_bytes", "generation_decode_online_body_bytes"))
        assert online == generation_online > 0
        assert audit["preparation_requests_during_online"] == 0
        assert audit["inference_control_download_bytes"] > before["inference_control_download_bytes"]
        assert audit["inference_control_upload_bytes"] - before["inference_control_upload_bytes"] == control_bodies[0]
        assert audit["inference_control_download_bytes"] - before["inference_control_download_bytes"] == control_bodies[1]
        assert core._online_active == 0
        assert core._transformer_states["epoch-sdk"].prepared_inventory is None


def replace_output_id(output, item_id):
    value = output.to_dict() if hasattr(output, "to_dict") else dict(output)
    return {**value, "id": item_id}


@pytest.mark.parametrize("epoch", [1, 3])
def test_controlled_eos_selection_at_boundary_preserves_executed_state(tmp_path, monkeypatch, epoch):
    # Force a public EOS predicate after three ordinary selections. The runtime
    # still calls its actual stochastic sampler, including on the EOS selection.
    exp = epoch_experiment(tmp_path)
    original = SemanticDecoderRuntime.sample
    calls = {}

    def sample(runtime, logits, **kwargs):
        token = original(runtime, logits, **kwargs)
        n = calls.get(runtime, 0)
        calls[runtime] = n + 1
        return int(runtime.cfg["eos_token_id"]) if n == 3 else (3 if token == runtime.cfg["eos_token_id"] else token)

    monkeypatch.setattr(SemanticDecoderRuntime, "sample", sample)
    with build_roles(exp, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / "cache") as client:
        core = client._core
        control = client.responses.create(input="abc", max_output_tokens=7, temperature=0.8)
        core._transformer_states["epoch-sdk"].prefill_cache.clear()
        core.preparation_decode_epoch_tokens = epoch
        result = client.responses.create(input="abc", max_output_tokens=7, temperature=0.8)
        assert result.status == control.status == "completed"
        assert result.usage.output_tokens == control.usage.output_tokens == 3
        first, second = (core._transformer_conversations[r.id] for r in (control, result))
        assert first.pending_token_ids == second.pending_token_ids == []
        for actual, expected in zip(numeric_state(second), numeric_state(first), strict=True):
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        assert core.audit.decode_epoch_pauses == 3 // epoch


def test_stream_cancel_and_failed_next_preparation_burn_material(tmp_path, monkeypatch):
    exp = epoch_experiment(tmp_path)
    with build_roles(exp, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / "cache") as client:
        core = client._core
        core.preparation_decode_epoch_tokens = 1
        stream = client.responses.create(input="abc", max_output_tokens=7, temperature=0, stream=True)
        created = next(stream)
        client.responses.cancel(created.response["id"])
        with pytest.raises(Exception, match="cancelled"):
            list(stream)
        assert not core._decode_epoch_responses
        assert core._online_active == 0
        original = core._prepare_inventory_locked
        calls = 0

        def fail(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("next epoch failed")
            return original(*args, **kwargs)

        monkeypatch.setattr(core, "_prepare_inventory_locked", fail)
        with pytest.raises(RuntimeError, match="next epoch failed"):
            client.responses.create(input="abc", max_output_tokens=7, temperature=0)
        assert core._online_active == 0
        assert not core._decode_epoch_responses
        assert core._transformer_states["epoch-sdk"].prepared_inventory is None
        audit = core.audit
        assert audit.prepared_stage_rows_reserved == audit.prepared_stage_rows_claimed + audit.prepared_stage_rows_burned


def test_response_owned_previous_ids_and_store_false(tmp_path):
    exp = epoch_experiment(tmp_path)
    with build_roles(exp, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / "cache") as client:
        core = client._core
        control = client.responses.create(input="abc", max_output_tokens=3, temperature=0)
        continued = client.responses.create(input="d", previous_response_id=control.id,
            max_output_tokens=3, temperature=0)
        expected = core._transformer_conversations[continued.id]
        core._transformer_states["epoch-sdk"].prefill_cache.clear()
        core.preparation_decode_epoch_tokens = 1
        first = client.responses.create(input="abc", max_output_tokens=3, temperature=0)
        result = client.responses.create(input="d", previous_response_id=first.id,
            max_output_tokens=3, temperature=0)
        saved = core._transformer_conversations[result.id]
        assert saved.token_ids == expected.token_ids
        assert saved.pending_token_ids == expected.pending_token_ids
        for actual, oracle in zip(numeric_state(saved), numeric_state(expected), strict=True):
            np.testing.assert_array_equal(actual.view(np.uint32), oracle.view(np.uint32))
        ephemeral = client.responses.create(input="abc", max_output_tokens=3, temperature=0, store=False)
        assert ephemeral.output_text == control.output_text
        assert ephemeral.id not in core._transformer_conversations
        assert not core._decode_epoch_responses


def test_cancel_during_idle_preparation_retires_unseen_committed_epoch(tmp_path, monkeypatch):
    exp = epoch_experiment(tmp_path)
    with build_roles(exp, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / "cache") as client:
        core = client._core
        core.preparation_decode_epoch_tokens = 1
        prepare = core._prepare_inventory_locked
        calls = 0
        committed = []

        def cancel_after_prepare(*args, **kwargs):
            nonlocal calls
            inventory = prepare(*args, **kwargs)
            calls += 1
            if calls == 2:
                assert core._online_active == 0
                committed.append(inventory)
                response_id = next(iter(core._decode_epoch_responses))
                client.responses.cancel(response_id)
            return inventory

        monkeypatch.setattr(core, "_prepare_inventory_locked", cancel_after_prepare)
        with pytest.raises(Exception, match="cancelled"):
            client.responses.create(input="abc", max_output_tokens=7, temperature=0)
        assert committed[0].status()["status"] == "cancelled"
        assert core.audit.decode_epochs_started == 1
        assert core._online_active == 0
        assert not core._decode_epoch_responses
        assert core._transformer_states["epoch-sdk"].prepared_inventory is None


def test_overbound_rejected_before_material_issuance(tmp_path):
    exp = epoch_experiment(tmp_path)
    with build_roles(exp, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / "cache") as client:
        core = client._core
        core.preparation_decode_epoch_tokens = 1
        with pytest.raises(Exception):
            client.responses.create(input="abc" * 100, max_output_tokens=7, temperature=0)
        assert core.audit.preparation_attempts == core.audit.decode_epochs_started == 0


def test_fresh_session_continuation_ack_cannot_downgrade(tmp_path, monkeypatch):
    import httpx

    exp = epoch_experiment(tmp_path)
    with build_roles(exp, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / "cache") as client:
        core = client._core
        core.preparation_decode_epoch_tokens = 1
        original = core.http.post
        calls = 0

        def downgrade(path, **kwargs):
            nonlocal calls
            response = original(path, **kwargs)
            if path == "/v1/runtime/sessions" and response.is_success:
                calls += 1
                if calls == 2:
                    value = response.json()
                    value.pop("decoder_continuation")
                    return httpx.Response(response.status_code, json=value, request=response.request)
            return response

        monkeypatch.setattr(core.http, "post", downgrade)
        with pytest.raises(Exception, match="did not admit"):
            client.responses.create(input="abc", max_output_tokens=7, temperature=0)
        assert calls == core.audit.decode_epochs_started == 2
        assert core._online_active == 0
        assert not core._decode_epoch_responses
        assert core._transformer_states["epoch-sdk"].prepared_inventory is None
        assert core.audit.prepared_stage_rows_reserved == (
            core.audit.prepared_stage_rows_claimed + core.audit.prepared_stage_rows_burned)


def test_native_sampler_natural_eos_without_override(tmp_path):
    # A generated source declares token zero as EOS and has an all-zero public
    # head. Actual greedy sampling selects EOS; no injected termination callback.
    import json
    import torch
    from safetensors.torch import load_file, save_file

    exp = epoch_experiment(tmp_path)
    root = tmp_path / "model"
    config = json.loads((root / "config.json").read_text())
    config["eos_token_id"] = 0
    config["tie_word_embeddings"] = False
    (root / "config.json").write_text(json.dumps(config))
    weights = load_file(str(root / "model.safetensors"))
    weights["lm_head.weight"] = torch.zeros_like(weights["model.embed_tokens.weight"])
    save_file(weights, str(root / "model.safetensors"))
    with build_roles(exp, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / "cache") as client:
        core = client._core
        control = client.responses.create(input="abc", max_output_tokens=7, temperature=0)
        core.preparation_decode_epoch_tokens = 1
        result = client.responses.create(input="abc", max_output_tokens=7, temperature=0)
        assert result.status == control.status == "completed"
        assert result.usage.output_tokens == control.usage.output_tokens == 0
        assert core.audit.decode_epochs_started == 1
        assert core.audit.decode_epoch_pauses == 0
        assert core._transformer_conversations[result.id].pending_token_ids == []


@pytest.mark.integration
@pytest.mark.parametrize("preissued", [False, True])
def test_public_decode_epoch_option_lifecycle(tmp_path, preissued):
    """Fresh-wheel integration: public component must reach the runtime unchanged."""
    exp = epoch_experiment(tmp_path, epoch=2)
    with build_roles(exp, engine_threads=1) as roles, roles.client(bundle_cache_dir=tmp_path / "cache") as client:
        assert client._core.preparation_decode_epoch_tokens == 2
        inventory = None
        if preissued:
            client.prepare_response("abc", max_output_tokens=7)
            inventory = client._core._transformer_states["epoch-sdk"].prepared_inventory
            assert inventory is not None and inventory.claimed == 0
        result = client.responses.create(input="abc", max_output_tokens=7, temperature=0)
        assert result.usage.output_tokens == 7
        assert client.privacy_audit.decode_epoch_pauses == 3
        assert client.privacy_audit.prepared_stage_rows_discarded == 0
        assert client.privacy_audit.prepared_stage_rows_reserved == (
            client.privacy_audit.prepared_stage_rows_claimed
            + client.privacy_audit.prepared_stage_rows_burned)
        if inventory is not None:
            assert inventory.claimed == inventory.capacity
            assert inventory.status()["reserved"] == 0
