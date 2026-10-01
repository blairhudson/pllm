from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, Pipeline
from pllm.profiles import MaskedLinearCpu, VerifiedMaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.roles import PreparedProviderRoles
from pllm.state import ClientPrefixReuse
from pllm.runtime.prefill_cache import ExactPrefillCache, prefill_key
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import LayerCache, RuntimeSnapshot


def _snapshot(value: float) -> RuntimeSnapshot:
    return RuntimeSnapshot(
        2,
        [
            LayerCache(
                np.full((2, 1, 2), value, dtype=np.float32),
                np.full((2, 1, 2), value, dtype=np.float32),
                2,
            )
        ],
        {},
    )


def test_cache_key_and_eviction_keep_only_bounded_independent_snapshots() -> None:
    key = prefill_key("a" * 64, "b" * 64, [1, 2])
    assert key != prefill_key("a" * 64, "b" * 64, [2, 1])
    assert key != prefill_key("c" * 64, "b" * 64, [1, 2])
    assert key != prefill_key("a" * 64, "c" * 64, [1, 2])
    with pytest.raises(ValueError, match="token IDs"):
        prefill_key("a" * 64, "b" * 64, [-1])
    snapshot = _snapshot(3.0)
    cache = ExactPrefillCache(40)
    assert cache.put(key, snapshot, np.ones(2, dtype=np.float32))
    fetched = cache.get(key, position=2, layers=1)
    assert fetched is not None
    fetched[0].caches[0].key[0, 0, 0] = 99
    assert cache.get(key, position=2, layers=1)[0].caches[0].key[0, 0, 0] == 3
    assert cache.put("other", _snapshot(4.0), np.ones(2, dtype=np.float32))
    assert cache.get(key, position=2, layers=1) is None
    # Cache owns immutable copies, never caller/runtime storage.
    assert np.all(snapshot.caches[0].key == 3)
    cache.clear()
    assert cache.size_bytes == 0


def _long_snapshot(tokens: int) -> RuntimeSnapshot:
    key = np.arange(tokens * 4, dtype=np.float32).reshape(tokens, 2, 2)
    return RuntimeSnapshot(tokens, [LayerCache(key, key + 1, tokens)], {})


def test_shared_blocks_keep_all_checkpoints_without_quadratic_payloads() -> None:
    snapshot = _long_snapshot(128)
    ids = list(range(128))
    payload = 128 * 4 * 4 * 2
    cache = ExactPrefillCache(payload + 8)
    assert cache.put_prefixes("a" * 64, "b" * 64, ids, snapshot) == 19
    assert cache.size_bytes == payload
    assert cache.block_count == 16
    assert cache.entry_count == 19
    assert cache.put(prefill_key("a" * 64, "b" * 64, ids), snapshot, np.ones(2, np.float32))
    assert cache.size_bytes == payload + 8
    for position in (1, 2, 4, 8, 64, 120, 127):
        found = cache.longest_prefix("a" * 64, "b" * 64, ids[:position] + [999], layers=1)
        assert found is not None and found[0] == position
        np.testing.assert_array_equal(found[1].caches[0].key, snapshot.caches[0].key[:position])
    assert cache.longest_prefix("c" * 64, "b" * 64, ids + [999], layers=1) is None
    assert cache.longest_prefix("a" * 64, "c" * 64, ids + [999], layers=1) is None


def test_shared_block_eviction_erases_only_last_reference_and_hits_are_independent() -> None:
    snapshot = _long_snapshot(16)
    cache = ExactPrefillCache(16 * 4 * 4 * 2)
    ids = list(range(16))
    cache.put_prefixes("a" * 64, "b" * 64, ids, snapshot)
    shared = list(cache._blocks.values())
    # Removing one checkpoint must not corrupt longer ones sharing its block.
    with cache._lock:
        cache._remove(prefill_key("a" * 64, "b" * 64, ids[:1]))
    found = cache.longest_prefix("a" * 64, "b" * 64, ids, layers=1)
    assert found is not None and found[0] == 15
    found[1].caches[0].key.fill(-99)
    original = cache.longest_prefix("a" * 64, "b" * 64, ids, layers=1)
    np.testing.assert_array_equal(original[1].caches[0].key, snapshot.caches[0].key[:15])
    cache.clear()
    assert cache.size_bytes == cache.entry_count == cache.block_count == 0
    assert all(not value.any() for block in shared for pair in block.arrays for value in pair)


def test_branch_blocks_share_prefix_and_remain_correct_under_eviction() -> None:
    first, branch = _long_snapshot(32), _long_snapshot(32)
    branch.caches[0].key[24:] += 50
    branch.caches[0].value[24:] += 50
    # Five physical blocks, rather than eight, store both branches.
    cache = ExactPrefillCache(5 * 8 * 4 * 4 * 2 + 8)
    ids = list(range(32))
    other = ids[:24] + list(range(100, 108))
    assert cache.put_prefixes("a" * 64, "b" * 64, ids, first)
    assert cache.put_prefixes("a" * 64, "b" * 64, other, branch)
    assert cache.block_count == 5
    for tokens, snapshot in ((ids, first), (other, branch)):
        found = cache.longest_prefix("a" * 64, "b" * 64, tokens + [999], layers=1)
        assert found is not None and found[0] == 31
        np.testing.assert_array_equal(found[1].caches[0].key, snapshot.caches[0].key[:31])
    # A larger unrelated branch forces eviction of shared prefixes. Candidate
    # copies must not be zeroized when old live blocks lose their last reference.
    replacement = _long_snapshot(40)
    replacement.caches[0].key += 1000
    replacement.caches[0].value += 1000
    assert cache.put("replacement", replacement, np.ones(2, np.float32))
    fetched = cache.get("replacement", position=40, layers=1)
    np.testing.assert_array_equal(fetched[0].caches[0].key, replacement.caches[0].key)
    assert cache.size_bytes <= cache.max_bytes


def test_oversized_prefill_retains_bounded_prefix_without_allocating_full_candidates() -> None:
    snapshot = _long_snapshot(128)
    cache = ExactPrefillCache(16 * 4 * 4 * 2)
    ids = list(range(128))
    assert cache.put_prefixes("a" * 64, "b" * 64, ids, snapshot) == 5
    assert cache.size_bytes <= cache.max_bytes
    found = cache.longest_prefix("a" * 64, "b" * 64, ids, layers=1)
    assert found is not None and found[0] == 16
    np.testing.assert_array_equal(found[1].caches[0].key, snapshot.caches[0].key[:16])
    before = cache.size_bytes
    assert not cache.put("too-large", snapshot, np.ones(2, np.float32))
    assert cache.size_bytes == before


def test_checkpoint_keeps_logits_only_for_bit_identical_state() -> None:
    cache = ExactPrefillCache(4096)
    ids = list(range(16))
    snapshot = _long_snapshot(8)
    key = prefill_key("a" * 64, "b" * 64, ids[:8])
    assert cache.put(key, snapshot, np.array([3, 4], np.float32))
    longer = _long_snapshot(16)
    cache.put_prefixes("a" * 64, "b" * 64, ids, longer)
    fetched = cache.get(key, position=8, layers=1)
    assert fetched is not None
    np.testing.assert_array_equal(fetched[1], [3, 4])
    longer.caches[0].key[0, 0, 0] += 1
    cache.put_prefixes("a" * 64, "b" * 64, ids, longer)
    assert cache.get(key, position=8, layers=1) is None


@pytest.mark.parametrize("bad", ["length", "nonfinite", "shape", "shared"])
def test_block_cache_rejects_invalid_state_without_disturbing_live_entries(bad: str) -> None:
    cache = ExactPrefillCache(1024)
    good = _snapshot(3)
    assert cache.put("valid", good, np.ones(2, np.float32))
    malformed = _snapshot(4)
    if bad == "length":
        malformed.caches[0].length = 1
    elif bad == "nonfinite":
        malformed.caches[0].key[0, 0, 0] = np.nan
    elif bad == "shape":
        malformed.caches[0].value = np.ones((2, 2, 2), np.float32)
    else:
        malformed.shared_kv["other"] = (np.ones(2), np.ones(2))
    assert not cache.put("bad", malformed, np.ones(2, np.float32))
    assert cache.get("valid", position=2, layers=1) is not None


@pytest.mark.parametrize("model_type", ["qwen2", "llama"])
@pytest.mark.parametrize("max_tokens", [1, 2])
def test_exact_prefill_skips_repeated_masked_stages_without_reusing_material(
    tmp_path: Path,
    model_type: str,
    max_tokens: int,
) -> None:
    model_id = f"prefill-{model_type}"
    root = create_tiny_llama_checkpoint(
        tmp_path / "model",
        num_hidden_layers=1,
        model_type=model_type,
        with_qkv_bias=model_type == "qwen2",
    )
    selected = MaskedLinearCpu(
        Model.path(str(root), model_id=model_id),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    experiment = Experiment(
        name=model_id,
        pipeline=Pipeline(
            model=selected.model,
            components={**selected.components, "topology": PreparedProviderRoles()},
        ),
        deployment=Deployment.local(root=str(tmp_path / "roles")),
        budget=ExecutionBudget(requests=5, max_input_tokens=128, max_new_tokens=2),
    )
    with build_roles(experiment, engine_threads=1) as topology:
        with topology.client(
            prefill_cache_bytes=1 << 20,
            prepared_inventory_rows=1,
            background_inventory_refill=False,
        ) as client:
            first = client.responses.create(
                model=model_id,
                input="a repeatable prompt",
                max_output_tokens=max_tokens,
                temperature=0,
            )
            assert first.usage is not None
            steps = client.privacy_audit.online_steps
            assert steps > 0
            upload = client.privacy_audit.masked_online_upload_bytes
            download = client.privacy_audit.masked_online_download_bytes
            stage_calls = client.privacy_audit.inference_stage_calls
            assert client.prepared_rows_for_response(
                "a repeatable prompt",
                max_tokens,
                model=model_id,
            ) == max(1, max_tokens - 1)
            second = client.responses.create(
                model=model_id,
                input="a repeatable prompt",
                max_output_tokens=max_tokens,
                temperature=0,
            )
            assert second.output_text == first.output_text
            assert second.usage == first.usage
            assert client.privacy_audit.online_steps - steps <= (max_tokens - 1) * 16
            assert client.privacy_audit.inference_stage_calls - stage_calls < stage_calls
            assert client.privacy_audit.masked_online_upload_bytes - upload < upload
            assert client.privacy_audit.masked_online_download_bytes - download < download
            if max_tokens == 1:
                assert client.privacy_audit.online_steps == steps
                assert client.privacy_audit.masked_online_upload_bytes == upload
                assert client.privacy_audit.masked_online_download_bytes == download
            assert client.privacy_audit.prefill_cache_hits == 1
            assert client.privacy_audit.prefill_cache_misses == 1
            assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
            assert client.privacy_audit.plaintext_token_ids_sent == 0
            assert client.privacy_audit.preparation_requests_during_online == 0
            assert (
                client.prepared_rows_for_response(
                    "a repeatable prompt",
                    max_tokens,
                    model=model_id,
                    store=False,
                )
                == first.usage.input_tokens + max_tokens - 1
            )
            client.responses.create(
                model=model_id,
                input="a repeatable prompt",
                max_output_tokens=max_tokens,
                temperature=0,
                store=False,
            )
            assert client.privacy_audit.online_steps > steps
            uncached = client.privacy_audit.online_steps
            client.responses.create(
                model=model_id,
                input="a different prompt",
                max_output_tokens=max_tokens,
                temperature=0,
            )
            assert client.privacy_audit.online_steps > uncached
            assert client.privacy_audit.prefill_cache_hits == 1
            continued = client.responses.create(
                model=model_id,
                previous_response_id=second.id,
                input="follow up",
                max_output_tokens=1,
                temperature=0,
            )
            assert continued.usage is not None
            assert client.privacy_audit.kv_continuation_hits == 1
            assert client.privacy_audit.prefill_cache_hits == 1


def test_verified_placement_does_not_activate_unverified_prefill_reuse(tmp_path: Path) -> None:
    root = create_tiny_llama_checkpoint(
        tmp_path / "model",
        num_hidden_layers=1,
        model_type="qwen2",
        with_qkv_bias=True,
    )
    model_id = "verified-prefill"
    experiment = Experiment(
        name=model_id,
        pipeline=VerifiedMaskedLinearCpu(
            Model.path(str(root), model_id=model_id),
            topology=PreparedProviderRoles(),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
        ),
        deployment=Deployment.local(root=str(tmp_path / "roles")),
        budget=ExecutionBudget(requests=2, max_input_tokens=64, max_new_tokens=1),
    )
    with build_roles(experiment, engine_threads=1) as topology:
        with topology.client(
            prefill_cache_bytes=1 << 20,
            background_inventory_refill=False,
        ) as client:
            first = client.responses.create(
                model=model_id,
                input="same prompt",
                max_output_tokens=1,
                temperature=0,
            )
            first_steps = client.privacy_audit.online_steps
            second = client.responses.create(
                model=model_id,
                input="same prompt",
                max_output_tokens=1,
                temperature=0,
            )
            assert first.output_text == second.output_text
            assert client.privacy_audit.online_steps == 2 * first_steps
            assert client.privacy_audit.prefill_cache_hits == 0
            assert client.privacy_audit.prefill_cache_misses == 0


def test_prefix_reuse_restores_only_causal_kv_and_reserves_suffix_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = create_tiny_llama_checkpoint(
        tmp_path / "model",
        num_hidden_layers=1,
        model_type="qwen2",
        with_qkv_bias=True,
    )
    model_id = "prefix-qwen2"
    selected = MaskedLinearCpu(
        Model.path(str(root), model_id=model_id),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
        cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=248),
    )
    experiment = Experiment(
        name=model_id,
        pipeline=Pipeline(
            model=selected.model,
            components={**selected.components, "topology": PreparedProviderRoles()},
        ),
        deployment=Deployment.local(root=str(tmp_path / "roles")),
        budget=ExecutionBudget(requests=5, max_input_tokens=248, max_new_tokens=2),
    )
    with build_roles(experiment, engine_threads=1) as topology:
        with topology.client(
            prepared_inventory_rows=1,
            background_inventory_refill=False,
        ) as client:
            base = "a repeatable prompt " * 8
            first = client.responses.create(
                model=model_id,
                input=base,
                max_output_tokens=1,
                temperature=0,
            )
            assert first.usage is not None
            following = base + "more"
            full_rows = client.prepared_rows_for_response(
                following,
                1,
                model=model_id,
                store=False,
            )
            suffix_rows = client.prepared_rows_for_response(following, 1, model=model_id)
            # Tiny matrices cannot amortize per-token HTTP envelopes. Admission
            # rejects reuse even though a matching checkpoint exists.
            assert suffix_rows == full_rows
            monkeypatch.setattr(client._core, "_profitable_cached_prefix", lambda *_: True)
            suffix_rows = client.prepared_rows_for_response(following, 1, model=model_id)
            assert 0 < suffix_rows < full_rows
            before_bytes = client.privacy_audit.masked_online_upload_bytes
            reused = client.responses.create(
                model=model_id,
                input=following,
                max_output_tokens=1,
                temperature=0,
            )
            reused_bytes = client.privacy_audit.masked_online_upload_bytes - before_bytes
            assert client.privacy_audit.prefill_prefix_tokens_reused == full_rows - suffix_rows
            assert client.privacy_audit.prefill_cache_hits == 1
            assert reused.usage is not None
            before_bytes = client.privacy_audit.masked_online_upload_bytes
            fresh = client.responses.create(
                model=model_id,
                input=following,
                max_output_tokens=1,
                temperature=0,
                store=False,
            )
            assert fresh.output_text == reused.output_text
            assert fresh.usage == reused.usage
            assert client.privacy_audit.masked_online_upload_bytes - before_bytes != reused_bytes
            assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
            assert client.privacy_audit.plaintext_token_ids_sent == 0
            assert client.privacy_audit.preparation_requests_during_online == 0
