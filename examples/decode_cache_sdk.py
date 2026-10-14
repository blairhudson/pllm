"""Generated prepared SDK decode-cache validation, not a performance benchmark.

Run after the coordinated fresh-wheel build:
    python examples/decode_cache_sdk.py --root target/decode-sdk-example --family qwen3

The root must be new. This example generates a tiny model; it downloads nothing.
Keys/accepted heads stay with the caller. No provider credentials enter cache files.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import weakref

import numpy as np

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.state import ClientPrefixReuse, DecodeCache, DecodeCacheHead


@contextmanager
def seeded_runtime_sampler(seed):
    """Validation harness only: seed each runtime's RNG, not np.random.seed."""
    original, seen = SemanticDecoderRuntime.sample, weakref.WeakSet()

    def sample(runtime, logits, **kwargs):
        if runtime not in seen:
            runtime.rng = np.random.default_rng(seed)
            seen.add(runtime)
        return original(runtime, logits, **kwargs)

    SemanticDecoderRuntime.sample = sample
    try:
        yield
    finally:
        SemanticDecoderRuntime.sample = original


def numeric_commitment(client, response):
    saved = client._core._transformer_conversations[response.id]
    values = [saved.next_logits, *(v[:row.length] for row in saved.snapshot.caches for v in (row.key, row.value))]
    return (response.output_text, response.usage.to_dict(), saved.token_ids.copy(),
            saved.pending_token_ids.copy(), saved.snapshot.position,
            hashlib.sha256(b"".join(v.view(np.uint32).tobytes() for v in values)).hexdigest())


def run(root, family, seed):
    root.mkdir(parents=True, exist_ok=False)
    source = create_tiny_llama_checkpoint(root / "model", model_type=family, seed=179,
        vocab_size=258, hidden_size=48, intermediate_size=96, head_dim=12, num_hidden_layers=2,
        with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3", tie_word_embeddings=False)
    model_id = "decode-sdk-example"
    experiment = Experiment(model_id, MaskedLinearCpu(Model.path(str(source), model_id=model_id),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
        inventory=PreparedInventory(allocation="demand", refill="on-demand", decode_epoch_tokens=1),
        cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=128,
            generated_prefixes=True, decode=True, decode_max_bytes=32 << 20)),
        Deployment.local(root=str(root / "roles")),
        ExecutionBudget(requests=4, max_input_tokens=128, max_new_tokens=4))
    key = os.urandom(32)  # Caller custody; do not write beside encrypted cache.
    directory = root / "decode"
    owner_options = {"max_snapshot_bytes": 1 << 20, "max_transient_bytes": 16 << 20}
    request = {"input": "abc", "max_output_tokens": 4, "temperature": 0.85, "top_p": 0.9}
    with seeded_runtime_sampler(seed), build_roles(experiment, engine_threads=1) as roles:
        cache = DecodeCache(directory, key, **owner_options)
        with roles.client(decode_cache=cache, background_inventory_refill=False,
                          bundle_cache_dir=root / "bundle") as client:
            cold = client.responses.create(**request)
            expected = numeric_commitment(client, cold)
            # Remove resident initial-prefix reuse from this check: hot request
            # must find durable root and every executed generated child itself.
            client._core._transformer_states[model_id].prefill_cache.clear()
            before = client.privacy_audit.to_dict()
            hot = client.responses.create(**request)
            assert numeric_commitment(client, hot) == expected
            after = client.privacy_audit.to_dict()
            for metric in ("prepared_stage_rows_issued", "prepared_stage_rows_reserved",
                           "inference_upload_bytes", "inference_download_bytes"):
                assert after[metric] == before[metric], metric
            assert client.privacy_audit.prepared_stage_rows_reserved == (
                client.privacy_audit.prepared_stage_rows_claimed + client.privacy_audit.prepared_stage_rows_burned)
            trusted_head = DecodeCacheHead.from_spec(cache.head.to_spec())
        assert cache._runtime is cache._store.aead is None
        # Explicit caller-selected accepted head. No directory scan chooses latest.
        restored_owner = DecodeCache(directory, key, head=trusted_head, **owner_options)
        with roles.client(decode_cache=restored_owner, background_inventory_refill=False,
                          bundle_cache_dir=root / "bundle") as client:
            restored = client.responses.create(**request)
            assert numeric_commitment(client, restored) == expected
            assert client.privacy_audit.prepared_stage_rows_issued == 0
            assert client.privacy_audit.prepared_stage_rows_reserved == 0
            assert client.privacy_audit.inference_upload_bytes == client.privacy_audit.inference_download_bytes == 0
            assert client.privacy_audit.plaintext_prompt_bytes_sent == client.privacy_audit.plaintext_token_ids_sent == 0
            assert restored_owner.storage_metrics["staged_count"] == 0
        print({"family": family, "seed": seed, "exact_state": True,
               "hot_provider_rows": 0, "restart_provider_rows": 0,
               "scope": "generated SDK validation; no RSS/ALL-LINK qualification"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", choices=("qwen2", "qwen3", "llama"), default="qwen2")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    run(args.root, args.family, args.seed)


if __name__ == "__main__":
    main()
