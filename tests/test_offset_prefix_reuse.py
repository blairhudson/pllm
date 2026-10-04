"""Two independently admitted worker sessions retain exact client-owned prefixes."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import TwoOnlineOffsetCpu
from pllm.protocols import ClientBundleTransport, TwoOnlineOffsetLinear
from pllm.quantization import SymmetricPerRow
from pllm.runtime.offset_reference import OffsetReferenceError
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.state import ClientPrefixReuse


def experiment_at(tmp_path, family="qwen2", encoding="combined"):
    root = create_tiny_llama_checkpoint(
        tmp_path / "model", model_type=family, seed=83, num_hidden_layers=2,
        with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3", tie_word_embeddings=False,
    )
    return Experiment(
        "offset-prefix",
        TwoOnlineOffsetCpu(
            Model.path(str(root), model_id="offset-prefix"),
            linear=TwoOnlineOffsetLinear(
                input_encoding="seeded" if encoding == "combined" else "raw",
                output_encoding="row_residues" if encoding == "combined" else "raw",
            ),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
            cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=248),
            delivery=ClientBundleTransport("artifacts", compression="zlib"),
        ),
        Deployment.local(root=str(tmp_path / "roles")),
        ExecutionBudget(requests=12, max_input_tokens=248, max_new_tokens=2),
    )


def online_bytes(client):
    return sum(value for key, value in client.privacy_audit.to_dict().items()
               if key.startswith("role_link.") and key.endswith(("online_upload_bytes", "online_download_bytes")))


def capture_state(runtime, logits):
    return np.asarray(logits).copy(), [
        (cache.key[:cache.length].copy(), cache.value[:cache.length].copy())
        for cache in runtime.caches
    ]


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
@pytest.mark.parametrize("encoding", ["raw", "combined"])
def test_offset_compressed_artifacts_prefix_and_transport_preserve_all_logits_and_kv(tmp_path, monkeypatch, family, encoding):
    experiment = experiment_at(tmp_path, family, encoding)
    continued, prefills = [], []
    original, prepare = SemanticDecoderRuntime.continue_ids, SemanticDecoderRuntime.prepare_ids

    def continuation(runtime, ids, **kwargs):
        result = original(runtime, ids, **kwargs)
        continued.append(capture_state(runtime, result[-1]))
        return result

    def prefill(runtime, ids):
        result = prepare(runtime, ids)
        prefills.append(capture_state(runtime, result[1]))
        return result

    monkeypatch.setattr(SemanticDecoderRuntime, "continue_ids", continuation)
    monkeypatch.setattr(SemanticDecoderRuntime, "prepare_ids", prefill)
    with build_roles(experiment, engine_threads=1) as topology, topology.client(
        bundle_cache_dir=tmp_path / "cache"
    ) as client:
        base = "abcde " * 24
        request = dict(model="offset-prefix", temperature=0, max_output_tokens=2)
        client.responses.create(input=base, **request)
        assert client._core.artifact_cache_stats.object_requests > 0
        branch = base[:96] + "different growing suffix" * 2
        before = online_bytes(client)
        cached = client.responses.create(input=branch, **request)
        cached_bytes = online_bytes(client) - before
        assert len(continued) == 1
        before = online_bytes(client)
        fresh = client.responses.create(input=branch, store=False, **request)
        assert cached.output_text == fresh.output_text and cached.usage == fresh.usage
        assert cached_bytes < online_bytes(client) - before
        np.testing.assert_array_equal(continued[0][0], prefills[-1][0])
        for actual, expected in zip(continued[0][1], prefills[-1][1], strict=True):
            for left, right in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(left, right)
        before = client.privacy_audit.inference_stage_calls
        repeat = client.responses.create(input=branch, **request)
        assert repeat.output_text == fresh.output_text
        assert client.privacy_audit.inference_stage_calls - before == 8  # decode only
        assert client.privacy_audit.prefill_cache_hits == 2
        assert client.privacy_audit.prefill_prefix_tokens_reused > 0
        assert client.privacy_audit.plaintext_prompt_bytes_sent == client.privacy_audit.plaintext_token_ids_sent == 0
        assert client.privacy_audit.preparation_requests_during_online == 0
        state = client._core._transformer_states["offset-prefix"]
        assert state.prefill_cache.size_bytes <= 1 << 20
        # Generated lineage is never promoted into ordinary cache keys.
        saved = client._core._transformer_conversations[repeat.id]
        assert not state.prefill_cache.put("forged-generated", saved.snapshot, saved.next_logits)


@pytest.mark.parametrize("failure", ["missing", "forged", "cancel"])
def test_offset_continuation_requires_both_acknowledgements_and_burns_both_sessions(tmp_path, failure):
    experiment = experiment_at(tmp_path)
    sessions, frames, requests = {}, [], {}
    with build_roles(experiment, engine_threads=1) as topology, topology.client(
        bundle_cache_dir=tmp_path / "cache"
    ) as client:
        for role, (worker, _) in client._core._offset_workers.items():
            def observe(response, role=role):
                path = response.request.url.path
                if "/stages/" in path:
                    frames.append(path)
                if path == "/v1/offset-reference/sessions" and response.status_code == 200:
                    response.read()
                    value = response.json()
                    sessions[role] = value["id"]
                    requests[role] = json.loads(response.request.content)
                    if role == "worker_b" and failure != "cancel":
                        if failure == "missing":
                            value.pop("decoder_continuation")
                        else:
                            value["decoder_continuation"]["numeric_digest"] = "f" * 64
                        response._content = json.dumps(value).encode()
                        response.headers["content-length"] = str(len(response._content))
            worker.event_hooks["response"].append(observe)
        options = dict(model="offset-prefix", input="A", temperature=0, max_output_tokens=2)
        if failure == "cancel":
            stream = client.responses.create(**options, stream=True)
            next(stream)
            stream.close()
        else:
            with pytest.raises(OffsetReferenceError, match="admission response differs"):
                client.responses.create(**options)
        assert set(sessions) == {"worker_a", "worker_b"} and not frames
        for role, (worker, key) in client._core._offset_workers.items():
            response = worker.post(f"/v1/offset-reference/sessions/{sessions[role]}/complete",
                                   headers={"authorization": f"Bearer {key}"})
            assert response.status_code == 409
            worker.event_hooks["response"].clear()
            for field in ("numeric_digest", "source_schedule_digest", "composition_digest", "token_bound"):
                forged = copy.deepcopy(requests[role])
                forged["decoder_continuation"]["contract"][field] = 247 if field == "token_bound" else "f" * 64
                assert worker.post("/v1/offset-reference/sessions", json=forged,
                                   headers={"authorization": f"Bearer {key}"}).status_code == 409
        assert client.privacy_audit.inference_stage_calls == 0
