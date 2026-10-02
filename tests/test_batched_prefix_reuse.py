from __future__ import annotations

import numpy as np
import pytest
import dataclasses
import os
from pathlib import Path

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.prefill_cache import ExactPrefillCache, prefill_key
from pllm.runtime.responses import normalize_input
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.state import ClientPrefixReuse


def prefix_experiment(tmp_path, family="qwen2", *, bound=248, vocab_size=258):
    # Ordinary randomly initialized heads; token round-trip is checked separately.
    root = create_tiny_llama_checkpoint(
        tmp_path / "model", model_type=family, seed=179, vocab_size=vocab_size,
        hidden_size=48, intermediate_size=96, head_dim=12, num_hidden_layers=2,
        with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3", tie_word_embeddings=False,
    )
    model_id = "evaluated-prefix"
    return model_id, Experiment(
        model_id,
        MaskedLinearCpu(
            Model.path(str(root), model_id=model_id),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
            inventory=PreparedInventory("request-sized", rows=1),
            cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=bound),
        ),
        Deployment.local(root=str(tmp_path / "roles")),
        ExecutionBudget(requests=32, max_input_tokens=bound, max_new_tokens=4),
    )


def cached_state(client, model_id, ids, max_tokens):
    state = client._core._transformer_states[model_id]
    compiled = client._core._compiled_public_decoder(
        state, max_input_tokens=len(ids), max_new_tokens=max_tokens
    )
    return state, compiled, state.prefill_cache.get(
        prefill_key(compiled.digest, state.bundle_fingerprint, ids),
        position=len(ids), layers=int(state.bundle.cfg["num_hidden_layers"]),
    )


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
@pytest.mark.parametrize("reduction", [None, "prefix_f32"])
def test_sdk_provider_admitted_prefix_batches_suffix_and_reserves_one_use_rows(tmp_path, monkeypatch, family, reduction):
    root = create_tiny_llama_checkpoint(
        tmp_path / "model", model_type=family, seed=83,
        num_hidden_layers=2, with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3",
        tie_word_embeddings=False,
    )
    model_id = "batched-prefix"
    experiment = Experiment(
        model_id,
        MaskedLinearCpu(
            Model.path(str(root), model_id=model_id),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction=reduction),
            inventory=PreparedInventory("request-sized", rows=1),
            cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=248),
        ),
        Deployment.local(root=str(tmp_path / "roles")),
        ExecutionBudget(requests=6, max_input_tokens=248, max_new_tokens=2),
    )
    continued, prefills = [], []
    original = SemanticDecoderRuntime.continue_ids
    prepare = SemanticDecoderRuntime.prepare_ids

    def capture(runtime, ids, **kwargs):
        position = runtime.position
        result = original(runtime, ids, **kwargs)
        continued.append((position, len(ids), runtime.position, result.copy()))
        return result

    monkeypatch.setattr(SemanticDecoderRuntime, "continue_ids", capture)

    def capture_prefill(runtime, ids):
        result = prepare(runtime, ids)
        prefills.append(result[1].copy())
        return result

    monkeypatch.setattr(SemanticDecoderRuntime, "prepare_ids", capture_prefill)
    with build_roles(experiment, engine_threads=1) as topology, topology.client(
        background_inventory_refill=False
    ) as client:
        base = "abcde " * 24
        client.responses.create(model=model_id, input=base, temperature=0, max_output_tokens=1)
        branch = (base[:96] + "some changed suffix").ljust(len(base), "X")
        if reduction:
            branch += " and growing context"
        total = client.prepared_rows_for_response(branch, 1, model=model_id, store=False)
        suffix = client.prepared_rows_for_response(branch, 1, model=model_id)
        assert 0 < suffix < total
        audit = client.privacy_audit
        before_calls = audit.inference_stage_calls
        before_bytes = audit.masked_online_upload_bytes + audit.masked_online_download_bytes
        cached = client.responses.create(model=model_id, input=branch, temperature=0, max_output_tokens=1)
        assert len(continued) == 1
        prefix, query, end, logits = continued[0]
        assert prefix + query == end == total and query == suffix
        assert audit.inference_stage_calls - before_calls == 8
        cached_bytes = audit.masked_online_upload_bytes + audit.masked_online_download_bytes - before_bytes
        before_bytes = audit.masked_online_upload_bytes + audit.masked_online_download_bytes
        before_calls = audit.inference_stage_calls
        fresh = client.responses.create(model=model_id, input=branch, temperature=0, max_output_tokens=1, store=False)
        np.testing.assert_array_equal(logits[-1], prefills[-1])
        assert fresh.output_text == cached.output_text and fresh.usage == cached.usage
        assert audit.inference_stage_calls - before_calls == 8
        assert cached_bytes < audit.masked_online_upload_bytes + audit.masked_online_download_bytes - before_bytes
        assert np.all(np.isfinite(logits))
        assert audit.prefill_prefix_tokens_reused == prefix
        assert audit.preparation_requests_during_online == 0
        assert audit.plaintext_prompt_bytes_sent == audit.plaintext_token_ids_sent == 0
        cache = client._core._transformer_states[model_id].prefill_cache
        assert cache.size_bytes <= 1 << 20
        # A one-token response has no evaluated generated token to promote.
        assert end == cached.usage.input_tokens
        before_calls = audit.inference_stage_calls
        again = client.responses.create(model=model_id, input=branch, temperature=0, max_output_tokens=1)
        assert again.output_text == cached.output_text
        assert audit.inference_stage_calls == before_calls
        assert len(continued) == 1
        # Reserved-but-unused row burns on cancellation; unreserved inventory
        # remains idle and reusable. No preparation is triggered online.
        client.preprocess(model_id, count=16)
        before_inventory = client.prepared_inventory_status(model_id)
        stream = client.responses.create(
            model=model_id, input=branch, temperature=0, max_output_tokens=1, stream=True
        )
        next(stream)
        stream.close()
        after_inventory = client.prepared_inventory_status(model_id)
        assert after_inventory["burned"] - before_inventory["burned"] == 1
        assert after_inventory["available"] == before_inventory["available"] - 1
        assert after_inventory["consumed"] == before_inventory["consumed"]
        assert audit.preparation_requests_during_online == 0


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
def test_generated_state_is_excluded_from_fresh_prefix_cache(tmp_path, monkeypatch, family):
    model_id, experiment = prefix_experiment(tmp_path, family)
    continued, prefills = [], []
    continuation = SemanticDecoderRuntime.continue_ids
    prepare = SemanticDecoderRuntime.prepare_ids

    def capture(runtime, ids, **kwargs):
        prefix = runtime.position
        result = continuation(runtime, ids, **kwargs)
        continued.append((prefix, list(ids), result[-1].copy()))
        return result

    def fresh_capture(runtime, ids):
        result = prepare(runtime, ids)
        prefills.append(result[1].copy())
        return result

    monkeypatch.setattr(SemanticDecoderRuntime, "continue_ids", capture)
    monkeypatch.setattr(SemanticDecoderRuntime, "prepare_ids", fresh_capture)
    with build_roles(experiment, engine_threads=1) as topology, topology.client(
        background_inventory_refill=False
    ) as client:
        prompt = "A short ordinary prompt"
        first = client.responses.create(model=model_id, input=prompt, temperature=0, max_output_tokens=4)
        conversation = client._core._transformer_conversations[first.id]
        assert first.usage.output_tokens == 4
        evaluated = conversation.token_ids[:-1]
        assert conversation.snapshot.position == len(evaluated)
        state, compiled, saved = cached_state(client, model_id, evaluated, 4)
        assert saved is None
        assert conversation.snapshot.state_basis.phase == "incremental"
        assert conversation.snapshot.state_basis.owner_response_id == first.id
        assert not state.prefill_cache.put("generated-alias", conversation.snapshot, conversation.next_logits)
        assert not state.prefill_cache.put_prefixes(compiled.digest, state.bundle_fingerprint, evaluated, conversation.snapshot)
        # The pending final sample is absent, including as a proper-prefix entry.
        _, _, pending = cached_state(client, model_id, conversation.token_ids, 4)
        assert pending is None
        retained = state.prefill_cache.longest_prefix(
            compiled.digest, state.bundle_fingerprint, conversation.token_ids,
            layers=2,
        )
        assert retained is None or retained[0] <= first.usage.input_tokens
        # Ordinary retokenized message history must actually match evaluated IDs.
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": first.output_text},
            {"role": "user", "content": "Continue with one more point"},
        ]
        before = client.privacy_audit.inference_stage_calls
        cached = client.responses.create(model=model_id, input=messages, temperature=0, max_output_tokens=4)
        assert not continued  # Longer context changes reduction geometry: miss.
        assert client.privacy_audit.inference_stage_calls - before == 8 * 4
        actual_logits = prefills[-1].copy()
        reference = client.responses.create(
            model=model_id, input=messages, temperature=0, max_output_tokens=4, store=False
        )
        np.testing.assert_array_equal(actual_logits, prefills[-1])
        assert cached.output_text == reference.output_text and cached.usage == reference.usage
        assert client.privacy_audit.preparation_requests_during_online == 0
        assert state.prefill_cache.size_bytes <= 1 << 20
        # Workload/source, bundle and cap identity shifts cannot expose old state.
        _, other, shifted = cached_state(client, model_id, evaluated, 3)
        assert other.digest != compiled.digest and shifted is None
        assert state.prefill_cache.longest_prefix(compiled.digest, "other-bundle", evaluated, layers=2) is None
        smaller = ExactPrefillCache(1)
        assert not smaller.put(prefill_key(compiled.digest, state.bundle_fingerprint, evaluated), conversation.snapshot, conversation.next_logits)
        assert smaller.size_bytes == 0


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
def test_prior_pending_and_new_suffix_use_admitted_batch_and_match_fresh(tmp_path, monkeypatch, family):
    model_id, experiment = prefix_experiment(tmp_path, family)
    batches, fresh_logits = [], []
    continuation = SemanticDecoderRuntime.continue_ids
    prepare = SemanticDecoderRuntime.prepare_ids

    def capture(runtime, ids, **kwargs):
        before = runtime.position
        result = continuation(runtime, ids, **kwargs)
        batches.append((before, list(ids), result[-1].copy()))
        return result

    def prepare_capture(runtime, ids):
        result = prepare(runtime, ids)
        fresh_logits.append(result[1].copy())
        return result

    monkeypatch.setattr(SemanticDecoderRuntime, "continue_ids", capture)
    monkeypatch.setattr(SemanticDecoderRuntime, "prepare_ids", prepare_capture)
    with build_roles(experiment, engine_threads=1) as topology, topology.client(background_inventory_refill=False) as client:
        prompt = "Exact state and a pending sampled token"
        first = client.responses.create(model=model_id, input=prompt, temperature=0, max_output_tokens=3)
        prior = client._core._transformer_conversations[first.id]
        assert prior.pending_token_ids == prior.token_ids[-1:]
        before = client.privacy_audit.inference_stage_calls
        continued = client.responses.create(
            model=model_id, input="Explain the next detail", previous_response_id=first.id,
            temperature=0, max_output_tokens=3,
        )
        assert batches[-1][0] == prior.snapshot.position
        assert batches[-1][1][:1] == prior.pending_token_ids
        assert client.privacy_audit.inference_stage_calls - before == 8 * 3
        assert client.privacy_audit.kv_continuation_batched_hits == 1
        assert client.privacy_audit.kv_continuation_legacy_sequential_hits == 0
        actual = batches[-1][2]
        # Replay exact sampled IDs. Re-encoding byte fallback text can legitimately
        # produce another cohort; that is not a fresh-reference numeric comparison.
        from test_decoder_continuation import bound_checkpoint, runtime_for

        engine, golden_compiled, _, _ = bound_checkpoint(tmp_path / "model", bound=248)
        golden, _ = runtime_for(golden_compiled, engine)
        exact_ids = [*prior.token_ids[:prior.snapshot.position], *batches[-1][1]]
        golden_logits = golden.prepare_ids(exact_ids)[1]
        np.testing.assert_allclose(actual, golden_logits, atol=1e-5, rtol=0)
        expected_ids = []
        for step in range(3):
            token = golden.sample(golden_logits, temperature=0)
            if token == int(golden.cfg["eos_token_id"]):
                break
            expected_ids.append(token)
            if step < 2:
                golden_logits = golden.decode_step(token, golden.caches)[0]
        produced = client._core._transformer_conversations[continued.id]
        assert produced.token_ids[len(exact_ids):] == expected_ids


def test_generated_basis_never_aliases_prefill_after_eos_cancel_store_false_or_bound(tmp_path, monkeypatch):
    model_id, experiment = prefix_experiment(tmp_path, bound=128)
    sampled, runtimes = [], []
    sample = SemanticDecoderRuntime.sample

    def sampling(runtime, *args, **kwargs):
        token = sample(runtime, *args, **kwargs)
        sampled.append(token)
        runtimes.append(runtime)
        return token

    monkeypatch.setattr(SemanticDecoderRuntime, "sample", sampling)
    with build_roles(experiment, engine_threads=1) as topology, topology.client(background_inventory_refill=False) as client:
        one = client.responses.create(model=model_id, input="one pending", temperature=0, max_output_tokens=1)
        one_prior = client._core._transformer_conversations[one.id]
        assert one_prior.snapshot.position == one.usage.input_tokens
        assert one_prior.snapshot.state_basis.owner_response_id == one.id
        private = client.responses.create(model=model_id, input="do not store", temperature=0, max_output_tokens=3, store=False)
        assert private.id not in client._core._transformer_conversations
        assert private.usage.output_tokens == 3
        assert runtimes[-1].snapshot().state_basis.phase == "incremental"
        state = client._core._transformer_states[model_id]
        assert not state.prefill_cache.put("store-false-alias", runtimes[-1].snapshot(), np.zeros(258, np.float32))
        stream = client.responses.create(model=model_id, input="cancel after evaluation", temperature=0, max_output_tokens=4, stream=True)
        deltas = 0
        for event in stream:
            if event.type == "response.output_text.delta":
                deltas += 1
                if deltas == 2:
                    break
        stream.close()
        runtime = runtimes[-1]
        rendered_ids = runtime.encode_prompt(runtime.bundle.render_prompt(
            [{"role": "user", "content": "cancel after evaluation"}], add_generation_prompt=True
        ))
        assert runtime.position == len(rendered_ids) + 1
        assert runtime.snapshot().state_basis.phase == "incremental"
        assert not state.prefill_cache.put("cancel-alias", runtime.snapshot(), np.zeros(258, np.float32))
        assert cached_state(client, model_id, [*rendered_ids, sampled[-2]], 4)[2] is None
        # Immediate EOS is sampled, never appended/evaluated or promoted.
        monkeypatch.setattr(SemanticDecoderRuntime, "sample", lambda runtime, *a, **kw: int(runtime.cfg["eos_token_id"]))
        response = client.responses.create(model=model_id, input="EOS", temperature=0, max_output_tokens=4)
        assert response.status == "completed" and response.usage.output_tokens == 0
        eos_state = client._core._transformer_conversations[response.id]
        assert not eos_state.pending_token_ids and eos_state.snapshot.position == response.usage.input_tokens
        # EOS after evaluated output differs from a token-limit response: every
        # emitted token is already evaluated, and only the unappended EOS is absent.
        samples_before_eos = [0]

        def late_eos(runtime, *args, **kwargs):
            samples_before_eos[0] += 1
            if samples_before_eos[0] == 3:
                return int(runtime.cfg["eos_token_id"])
            return sampling(runtime, *args, **kwargs)

        monkeypatch.setattr(SemanticDecoderRuntime, "sample", late_eos)
        eos_response = client.responses.create(model=model_id, input="evaluated before EOS", temperature=0, max_output_tokens=4)
        eos_prior = client._core._transformer_conversations[eos_response.id]
        assert eos_response.status == "completed" and eos_response.usage.output_tokens == 2
        assert not eos_prior.pending_token_ids
        assert eos_prior.snapshot.position == len(eos_prior.token_ids)
        assert eos_prior.snapshot.state_basis.phase == "incremental"
        assert cached_state(client, model_id, eos_prior.token_ids, 4)[2] is None
        monkeypatch.setattr(SemanticDecoderRuntime, "sample", sampling)
        # Input still fits; evaluated generation can exceed fixed input capacity.
        bounded = client.responses.create(model=model_id, input="a" * 107, temperature=0, max_output_tokens=4)
        bounded_prior = client._core._transformer_conversations[bounded.id]
        assert bounded_prior.snapshot.position > 128
        assert not state.prefill_cache.put("over-bound-alias", bounded_prior.snapshot, bounded_prior.next_logits)


def test_non_round_trip_text_cannot_claim_generated_prefix(tmp_path):
    model_id, experiment = prefix_experiment(tmp_path, vocab_size=258)
    with build_roles(experiment, engine_threads=1) as topology, topology.client(background_inventory_refill=False) as client:
        response = client.responses.create(model=model_id, input="Byte fallback state", temperature=0, max_output_tokens=4)
        conversation = client._core._transformer_conversations[response.id]
        state = client._core._transformer_states[model_id]
        rendered = state.bundle.render_prompt([
            {"role": "user", "content": "Byte fallback state"},
            {"role": "assistant", "content": response.output_text},
            {"role": "user", "content": "next"},
        ], add_generation_prompt=True)
        encoded = state.bundle.tokenizer().encode(rendered, add_bos=True)
        evaluated = conversation.token_ids[:conversation.snapshot.position]
        assert encoded[:len(evaluated)] != evaluated
        compiled = client._core._compiled_public_decoder(state, max_input_tokens=len(encoded), max_new_tokens=4)
        hit = state.prefill_cache.longest_prefix(compiled.digest, state.bundle_fingerprint, encoded, layers=2)
        assert hit is None or hit[0] < len(evaluated)
        assert cached_state(client, model_id, evaluated, 4)[2] is None


def test_prior_snapshot_rejects_wrong_owner_source_or_basis_before_stages(tmp_path):
    model_id, experiment = prefix_experiment(tmp_path)
    with build_roles(experiment, engine_threads=1) as topology, topology.client(background_inventory_refill=False) as client:
        response = client.responses.create(model=model_id, input="state identity", temperature=0, max_output_tokens=3)
        prior = client._core._transformer_conversations[response.id]
        state, compiled, _ = cached_state(client, model_id, prior.token_ids[:-1], 3)
        runtime = compiled.runtime(lambda *args: pytest.fail("promotion must not execute stages"))
        from pllm.runtime.transformer_client import TransformerClientError

        for forged, owner in (
            (prior.snapshot, "wrong-response"),
            (dataclasses.replace(prior.snapshot, state_basis=None), response.id),
            (dataclasses.replace(prior.snapshot, state_basis=dataclasses.replace(
                prior.snapshot.state_basis, source_binding_digest="a" * 64)), response.id),
            (dataclasses.replace(prior.snapshot, state_basis=dataclasses.replace(
                prior.snapshot.state_basis, native_phase_digest="a" * 64)), response.id),
            (dataclasses.replace(prior.snapshot, state_basis=dataclasses.replace(
                prior.snapshot.state_basis, execution_digest="a" * 64)), response.id),
        ):
            with pytest.raises(TransformerClientError, match="owner/source"):
                runtime.restore_for_response(forged, owner)
            assert runtime.position == 0


@pytest.mark.slow
@pytest.mark.skipif(os.getenv("PLLM_RUN_CACHED_CONTINUATION") != "1", reason="opt-in pinned offline HTTP checkpoint")
def test_pinned_generated_alias_blocked_and_fresh_cache_preserves_every_logit(tmp_path, monkeypatch, record_property):
    from huggingface_hub import hf_hub_download

    root = Path(hf_hub_download(
        "Qwen/Qwen2.5-0.5B-Instruct", "config.json", revision="7ae557604adf67be50417f59c2c2f167def9a775",
        cache_dir=os.getenv("PLLM_CACHED_HF_DIR", str(Path.home() / ".cache/huggingface/hub")), local_files_only=True,
    )).parent
    model_id = "pinned-promoted-prefix"
    experiment = Experiment(
        model_id,
        MaskedLinearCpu(
            Model.path(str(root), model_id=model_id),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
            inventory=PreparedInventory("request-sized", rows=1),
            cache=ClientPrefixReuse(max_bytes=64 << 20, fixed_input_tokens=128),
        ),
        Deployment.local(root=str(tmp_path / "roles")),
        ExecutionBudget(requests=8, max_input_tokens=128, max_new_tokens=4),
    )
    batches, prefills = [], []
    continuation = SemanticDecoderRuntime.continue_ids
    prepare = SemanticDecoderRuntime.prepare_ids

    def capture(runtime, ids, **kwargs):
        position = runtime.position
        result = continuation(runtime, ids, **kwargs)
        batches.append((position, list(ids), result[-1].copy()))
        return result

    def fresh_capture(runtime, ids):
        result = prepare(runtime, ids)
        prefills.append(result[1].copy())
        return result

    monkeypatch.setattr(SemanticDecoderRuntime, "continue_ids", capture)
    monkeypatch.setattr(SemanticDecoderRuntime, "prepare_ids", fresh_capture)
    # Independently replay the actual singleton protocol's activations through
    # the same checkpoint's clear integer kernel, before any float attention.
    from test_decoder_continuation import bound_checkpoint, QuantizedRemote
    from pllm.runtime.transformer_client import PreparedRemoteLinear

    reference_engine, reference_compiled, _, _ = bound_checkpoint(root, bound=128)
    clear_remote = QuantizedRemote(reference_engine, reference_compiled._bundle.model_id)
    remote_call = PreparedRemoteLinear.__call__

    def verify_integer_stage(remote, stage_id, value):
        actual = remote_call(remote, stage_id, value)
        if np.asarray(value).shape[0] == 1:
            expected = clear_remote(stage_id, value)
            np.testing.assert_array_equal(actual, expected, err_msg=f"singleton prepared stage {stage_id}")
        return actual

    monkeypatch.setattr(PreparedRemoteLinear, "__call__", verify_integer_stage)
    with build_roles(experiment, engine_threads=1) as topology, topology.client(background_inventory_refill=False) as client:
        prompt = "Explain why private inference uses fresh masks."
        first = client.responses.create(model=model_id, input=prompt, temperature=0, max_output_tokens=4)
        prior = client._core._transformer_conversations[first.id]
        assert first.usage.output_tokens == 4
        evaluated = prior.token_ids[:-1]
        state, compiled, saved = cached_state(client, model_id, evaluated, 4)
        assert saved is None
        assert prior.snapshot.state_basis.phase == "incremental"
        assert prior.snapshot.state_basis.owner_response_id == first.id
        assert not state.prefill_cache.put("forbidden-generated", prior.snapshot, prior.next_logits)
        assert not state.prefill_cache.put_prefixes(compiled.digest, state.bundle_fingerprint, evaluated, prior.snapshot)
        assert cached_state(client, model_id, prior.token_ids, 4)[2] is None
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": first.output_text},
            {"role": "user", "content": "Describe one reason briefly."},
        ]
        encoded = state.bundle.tokenizer().encode(
            client._core._render_cached_decoder_prompt(state, normalize_input(messages), add_generation_prompt=True),
            add_bos=bool(state.bundle.tokenizer_descriptor.get("add_bos_token", True)),
        )
        assert encoded[:len(evaluated)] == evaluated
        assert state.prefill_cache.longest_prefix(compiled.digest, state.bundle_fingerprint, encoded, layers=24) is None
        before = client.privacy_audit.inference_stage_calls
        cached = client.responses.create(model=model_id, input=messages, temperature=0, max_output_tokens=4)
        assert not batches  # Neither generated lineage nor different geometry is eligible.
        assert client.privacy_audit.inference_stage_calls - before == 96 * 4
        canonical_logits = prefills[-1].copy()
        reference = client.responses.create(model=model_id, input=messages, temperature=0, max_output_tokens=4, store=False)
        # Cache-enabled ordinary input must retain canonical full-prefill numerics,
        # not just an unchanged argmax despite quantization-amplified drift.
        np.testing.assert_array_equal(canonical_logits, prefills[-1])
        assert cached.output_text == reference.output_text and cached.usage == reference.usage
        record_property("ordinary_cache_vs_full_prefill_array_equal_every_logit", True)
        # Fresh checkpoint replay of the actual source execution: input prefill,
        # fixed generated teacher tokens, then the declared query block. This
        # isolates snapshot/promotion correctness from inherited decode-vs-full-
        # prefill float/quantizer drift, which is reported separately below.
        from test_decoder_continuation import runtime_for

        golden, _ = runtime_for(reference_compiled, reference_engine)
        golden.prepare_ids(prior.token_ids[:first.usage.input_tokens])
        for token in prior.token_ids[first.usage.input_tokens:prior.snapshot.position]:
            golden.decode_step(token, golden.caches)
        for actual_state, expected_state in zip(prior.snapshot.caches, golden.snapshot().caches, strict=True):
            np.testing.assert_array_equal(actual_state.key, expected_state.key)
            np.testing.assert_array_equal(actual_state.value, expected_state.value)
        golden.install_continuation(reference_compiled._plan.continuation_schedule(
            MaskedLinearCpu(Model(reference_compiled._bundle.model_id), quantization=SymmetricPerRow(weight_bits=8, activation_bits=8))
        ))
        # Independent incremental replay explicitly belongs to a response owner.
        golden.restore_for_response(golden.snapshot_for_response("clear-reference"), "clear-reference")
        expected_logits = golden.continue_ids(encoded[len(evaluated):])[-1]
        incremental_logits = expected_logits.copy()
        drift = float(np.max(np.abs(incremental_logits - canonical_logits)))
        assert drift > 1e-5  # Demonstrates why generated token-hash aliasing is vetoed.
        expected_ids = []
        for step in range(4):
            token = golden.sample(expected_logits, temperature=0)
            if token == int(golden.cfg["eos_token_id"]):
                break
            expected_ids.append(token)
            if step < 3:
                expected_logits = golden.decode_step(token, golden.caches)[0]
        record_property("generated_fresh_reuse_numeric_gate_failed_max_abs_difference", drift)
        record_property("generated_fresh_reuse_admitted", False)
        before = client.privacy_audit.inference_stage_calls
        conversation = client.responses.create(
            model=model_id, previous_response_id=first.id, input="Describe one reason briefly.",
            temperature=0, max_output_tokens=4,
        )
        assert batches[-1][0] == prior.snapshot.position
        assert batches[-1][1][:1] == prior.pending_token_ids
        assert client.privacy_audit.inference_stage_calls - before == 96 * 4
        np.testing.assert_array_equal(batches[-1][2], incremental_logits)
        assert client._core._transformer_conversations[conversation.id].token_ids[len(encoded):] == expected_ids
        assert conversation.usage == cached.usage
        assert client.privacy_audit.kv_continuation_batched_hits == 1
        assert state.prefill_cache.size_bytes <= 64 << 20
        assert client.privacy_audit.preparation_requests_during_online == 0
