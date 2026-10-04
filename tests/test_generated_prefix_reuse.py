from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration

from pllm import Experiment
from pllm.profiles import TwoOnlineOffsetCpu, VerifiedMaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.prefill_cache import prefill_key
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.servers import build_roles
from pllm.state import ClientPrefixReuse
from test_batched_prefix_reuse import prefix_experiment


@pytest.mark.parametrize("topology_name", ["prepared", "offset", "verified"])
def test_generated_prefix_uses_only_completed_executed_tokens(tmp_path, monkeypatch, topology_name):
    model_id, base = prefix_experiment(tmp_path)
    options = dict(quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
        cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=248, generated_prefixes=True))
    pipeline = (TwoOnlineOffsetCpu(base.pipeline.model, **options) if topology_name == "offset" else
        VerifiedMaskedLinearCpu(base.pipeline.model, inventory=base.pipeline.inventory, **options)
        if topology_name == "verified" else base.pipeline.with_params(**options))
    experiment = Experiment(base.name, pipeline, base.deployment, base.budget)
    promoted, prefills, continued = [], [], []
    promote = SemanticDecoderRuntime.canonical_generated_snapshot
    prepare = SemanticDecoderRuntime.prepare_ids
    continuation = SemanticDecoderRuntime.continue_ids

    def capture(runtime):
        result = promote(runtime)
        promoted.append(result)
        return result

    def capture_prefill(runtime, ids):
        result = prepare(runtime, ids)
        prefills.append(result[1].copy())
        return result

    def capture_suffix(runtime, ids, **kwargs):
        position = runtime.position
        result = continuation(runtime, ids, **kwargs)
        continued.append((position, result[-1].copy()))
        return result

    monkeypatch.setattr(SemanticDecoderRuntime, "canonical_generated_snapshot", capture)
    monkeypatch.setattr(SemanticDecoderRuntime, "prepare_ids", capture_prefill)
    monkeypatch.setattr(SemanticDecoderRuntime, "continue_ids", capture_suffix)
    # Keep byte-tokenizer round-trip exact while testing all logits and KV math.
    monkeypatch.setattr(SemanticDecoderRuntime, "sample", lambda runtime, *a, **kw:
        runtime.tokenizer.encode("X", add_bos=False)[0])
    client_options = {} if topology_name == "offset" else {"background_inventory_refill": False}
    with build_roles(experiment, engine_threads=1) as roles, roles.client(**client_options) as client:
        prompt = "A short ordinary prompt"
        first = client.responses.create(model=model_id, input=prompt, temperature=0, max_output_tokens=4)
        prior = client._core._transformer_conversations[first.id]
        state = client._core._transformer_states[model_id]
        compiled = client._core._compiled_public_decoder(state, max_input_tokens=first.usage.input_tokens, max_new_tokens=4)
        ids = prior.token_ids[:prior.snapshot.position]
        assert len(promoted) == 1 and len(ids) == first.usage.input_tokens + 3
        assert prior.snapshot.state_basis.owner_response_id == first.id
        assert prior.snapshot.state_basis.phase == "incremental"
        key = prefill_key(compiled.digest, state.bundle_fingerprint, ids, causal_reduction="prefix_f32")
        saved = state.prefill_cache.get(key, position=len(ids), layers=2)
        assert saved is not None and saved[0].state_basis.phase == "canonical_incremental"
        pending = prefill_key(compiled.digest, state.bundle_fingerprint, prior.token_ids, causal_reduction="prefix_f32")
        assert state.prefill_cache.get(pending, position=len(prior.token_ids), layers=2) is None
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": first.output_text},
                    {"role": "user", "content": "Continue"}]
        cached = client.responses.create(model=model_id, input=messages, temperature=0, max_output_tokens=4)
        assert continued[-1][0] == len(ids)
        actual = continued[-1][1]
        before = len(promoted)
        fresh = client.responses.create(model=model_id, input=messages, temperature=0, max_output_tokens=4, store=False)
        assert len(promoted) == before
        assert np.array_equal(actual.view(np.uint32), prefills[-1].view(np.uint32))
        assert cached.output_text == fresh.output_text and cached.usage == fresh.usage
        stream = client.responses.create(model=model_id, input="cancel a different context", temperature=0,
            max_output_tokens=4, stream=True)
        deltas = 0
        for event in stream:
            if event.type == "response.output_text.delta":
                deltas += 1
                if deltas == 2:
                    break
        stream.close()
        assert len(promoted) == before
        assert client.privacy_audit.plaintext_prompt_bytes_sent == client.privacy_audit.plaintext_token_ids_sent == 0
        assert client.privacy_audit.preparation_requests_during_online == 0
