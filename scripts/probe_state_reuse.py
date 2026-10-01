"""Block-shared cache payload control and one canonical tiny branching request."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

from pllm.runtime.prefill_cache import ExactPrefillCache, prefill_key
from pllm.runtime.transformer_client import LayerCache, RuntimeSnapshot


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def positions(total: int) -> list[int]:
    return sorted(
        {
            total - 1,
            *(1 << bit for bit in range(total.bit_length()) if 1 << bit < total),
            *range(8, total, 8),
        }
    )


def payload_control() -> dict:
    from huggingface_hub import hf_hub_download

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    revision = "7ae557604adf67be50417f59c2c2f167def9a775"
    config_path = Path(
        hf_hub_download(model_id, "config.json", revision=revision, local_files_only=True)
    )
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    # Dimensions from the cached pinned config, not a checkpoint trajectory.
    # Synthetic finite KV values, not real model states or peak-memory samples.
    total = 512
    layers, heads = config["num_hidden_layers"], config["num_key_value_heads"]
    width = config["hidden_size"] // config["num_attention_heads"]
    row_bytes = layers * heads * width * 2 * 4
    cap, logit_bytes = 64 << 20, config["vocab_size"] * 4
    ids = list(range(total))
    blocks = []
    for layer in range(layers):
        key = (
            np.arange(total * heads * width, dtype=np.float32).reshape(total, heads, width) + layer
        )
        blocks.append(LayerCache(key, key + 1, total))
    snapshot = RuntimeSnapshot(total, blocks, {})
    cache = ExactPrefillCache(cap)
    saved = cache.put_prefixes("a" * 64, "b" * 64, ids, snapshot)
    assert cache.put(
        prefill_key("a" * 64, "b" * 64, ids), snapshot, np.zeros(config["vocab_size"], np.float32)
    )
    # Independent payload-only model of old ascending checkpoint copies/LRU.
    old: OrderedDict[int, int] = OrderedDict()
    old_bytes = 0
    for position in [*positions(total), total]:
        size = position * row_bytes + (logit_bytes if position == total else 0)
        while old and old_bytes + size > cap:
            _, removed = old.popitem(last=False)
            old_bytes -= removed
        old[position] = size
        old_bytes += size
    target = ids[:384] + [999]
    hit = cache.longest_prefix("a" * 64, "b" * 64, target, layers=layers)
    assert hit is not None and hit[0] == 384
    full_copy_demand = sum(positions(total)) * row_bytes + total * row_bytes + logit_bytes
    result = {
        "scope": "synthetic dimensions; old LRU payload model, not a sampled old-process RAM peak",
        "source": {
            "model_id": model_id,
            "revision": revision,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "cache_payload_budget_bytes": cap,
        "bytes_per_kv_row": row_bytes,
        "checkpoint_payload_before_eviction_bytes": full_copy_demand,
        "old_retained_payload_bytes": old_bytes,
        "old_retained_checkpoints": len(old),
        "old_common_384_prefix_hit": 384 in old,
        "new_retained_payload_bytes": cache.size_bytes,
        "new_retained_checkpoints": cache.entry_count,
        "new_physical_blocks": cache.block_count,
        "new_common_384_prefix_hit": hit[0] == 384,
        "saved_prefix_checkpoints": saved,
        "metadata_and_transient_copies_included": False,
    }
    cache.clear()
    return result


def tiny_branch() -> dict:
    from pllm import Deployment, ExecutionBudget, Experiment, Model
    from pllm.preparation import PreparedInventory
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.servers import build_roles
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    from pllm.state import ClientPrefixReuse

    with tempfile.TemporaryDirectory(prefix="pllm-state-reuse-") as directory:
        root = create_tiny_llama_checkpoint(
            Path(directory) / "model",
            hidden_size=128,
            intermediate_size=256,
            head_dim=32,
            num_hidden_layers=8,
            model_type="qwen2",
            with_qkv_bias=True,
        )
        experiment = Experiment(
            "block-shared-branch",
            MaskedLinearCpu(
                Model.path(str(root), model_id="block-shared-branch"),
                quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
                inventory=PreparedInventory("request-sized", rows=1),
                cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=248),
            ),
            Deployment.local(root=str(Path(directory) / "roles")),
            ExecutionBudget(requests=3, max_input_tokens=248, max_new_tokens=1),
        )
        base = "abcde " * 32
        branch = base[:128] + "new"
        with build_roles(experiment, engine_threads=1) as topology:
            with topology.client(background_inventory_refill=False) as client:
                first = client.responses.create(
                    model="block-shared-branch", input=base, temperature=0, max_output_tokens=1
                )
                audit = client.privacy_audit
                before = (
                    audit.masked_online_upload_bytes,
                    audit.masked_online_download_bytes,
                    audit.inference_stage_calls,
                    audit.prefill_prefix_tokens_reused,
                )
                reused = client.responses.create(
                    model="block-shared-branch", input=branch, temperature=0, max_output_tokens=1
                )
                reused_cost = (
                    audit.masked_online_upload_bytes - before[0],
                    audit.masked_online_download_bytes - before[1],
                    audit.inference_stage_calls - before[2],
                )
                reused_tokens = audit.prefill_prefix_tokens_reused - before[3]
                before = (
                    audit.masked_online_upload_bytes,
                    audit.masked_online_download_bytes,
                    audit.inference_stage_calls,
                )
                fresh = client.responses.create(
                    model="block-shared-branch",
                    input=branch,
                    temperature=0,
                    max_output_tokens=1,
                    store=False,
                )
                fresh_cost = (
                    audit.masked_online_upload_bytes - before[0],
                    audit.masked_online_download_bytes - before[1],
                    audit.inference_stage_calls - before[2],
                )
                assert reused.output_text == fresh.output_text and reused.usage == fresh.usage
                assert reused_tokens > 0 and sum(reused_cost[:2]) < sum(fresh_cost[:2])
                assert audit.plaintext_prompt_bytes_sent == audit.plaintext_token_ids_sent == 0
                assert audit.preparation_requests_during_online == 0
                return {
                    "scope": "one generated-checkpoint two-child loopback sample; online masked bodies only",
                    "experiment_digest": experiment.configuration_digest(),
                    "prompt_cohort_digest": digest([base, branch]),
                    "first_input_tokens": first.usage.input_tokens,
                    "branch_input_tokens": reused.usage.input_tokens,
                    "reused_prefix_tokens": reused_tokens,
                    "cached_online_upload_download_bytes": list(reused_cost[:2]),
                    "fresh_online_upload_download_bytes": list(fresh_cost[:2]),
                    "cached_stage_calls": reused_cost[2],
                    "fresh_stage_calls": fresh_cost[2],
                    "same_selected_output_and_usage": True,
                    "no_plaintext_prompt_or_token_ids": True,
                    "no_online_preparation": True,
                }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-tiny", action="store_true")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    result = {
        "schema": "pllm.block_shared_state_reuse.v1",
        "date": "2026-10-01",
        "code_sha256": {
            path: hashlib.sha256(
                (Path(__file__).resolve().parents[1] / path).read_bytes()
            ).hexdigest()
            for path in (
                "scripts/probe_state_reuse.py",
                "python/pllm/runtime/prefill_cache.py",
                "python/pllm/runtime/tiny_llama.py",
            )
        },
        "payload_control": payload_control(),
        "tiny_branch": tiny_branch() if args.verify_tiny else None,
        "coverage": {
            "completed_prefill_checkpoints": True,
            "generated_prefix_promotion": False,
            "batched_suffix_phase": False,
            "compiler_gate": "decode query sequence is one; no admitted chunked-continuation phase",
        },
        "limitations": [
            "No real-checkpoint quality/traffic cohort",
            "No full wire/peak RAM measurement",
            "Generated tokens remain in existing explicit response continuations",
            "Suffix execution remains one-token decode; byte admission unchanged",
        ],
    }
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.archive:
        args.archive.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
