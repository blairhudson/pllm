"""Matched fresh / explicit legacy sequential / native batched continuation evidence.

HTTP metrics count covered protocol bodies, never full wire traffic. Numeric
controls use the same checkpoint's W8A8 integer kernels, not float weight dots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.prefill_cache import prefill_key
from pllm.runtime.responses import normalize_input
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.state import ClientPrefixReuse


@contextmanager
def capture_prefill(mode, results):
    prepare = SemanticDecoderRuntime.prepare_ids
    continuation = SemanticDecoderRuntime.continue_ids

    def saved_prepare(runtime, ids):
        value = prepare(runtime, ids)
        results.append(np.asarray(value[1]).copy())
        return value

    def saved_continuation(runtime, ids, **kwargs):
        if mode == "legacy_sequential_suffix_control":
            # Explicit experiment control only; production has no fallback.
            value = runtime.forward_ids(ids)
        else:
            value = continuation(runtime, ids, **kwargs)
        results.append(np.asarray(value[-1]).copy())
        return value

    with patch.object(SemanticDecoderRuntime, "prepare_ids", saved_prepare), patch.object(
        SemanticDecoderRuntime, "continue_ids", saved_continuation
    ):
        yield


def inventory_ledger(inventories):
    statuses = [item.status() for item in inventories.values()]
    return {
        "issued_per_stage_rows": sum(item.capacity * len(item.stages) for item in inventories.values()),
        "complete_reserved_rows": sum(item.claimed for item in inventories.values()),
        "complete_consumed_rows": sum(int(row["consumed"]) for row in statuses),
        "complete_burned_rows": sum(int(row["burned"]) for row in statuses),
        "idle_unreserved_rows": sum(int(row["available"]) for row in statuses),
    }


def http_cohort(root: Path, work: Path, *, base: str, branches: list[str], bound: int,
                 cache_bytes: int, max_new_tokens: int = 1, generated_prompt: str | None = None,
                 match_first_branch_geometry: bool = False) -> dict:
    model_id = "batched-continuation-probe"
    experiment = Experiment(
        "batched-continuation-probe",
        MaskedLinearCpu(
            Model.path(str(root), model_id=model_id),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
            inventory=PreparedInventory("request-sized", rows=1),
            cache=ClientPrefixReuse(max_bytes=cache_bytes, fixed_input_tokens=bound),
        ),
        Deployment.local(root=str(work)),
        ExecutionBudget(requests=4 * (len(branches) + 1) + 4, max_input_tokens=bound,
                         max_new_tokens=max(max_new_tokens, 4 if generated_prompt else 1)),
    )
    samples, goldens = [], {}
    contract_binding = None
    generated = None
    with build_roles(experiment, engine_threads=1) as topology:
        processes = [{"role": item.role, "pid": item.pid, "running": item.running} for item in topology.statuses]
        assert len(processes) == 2 and all(item["running"] and item["pid"] != os.getpid() for item in processes)
        assert len({item["pid"] for item in processes}) == 2
        for mode in ("fresh_prefill", "legacy_sequential_suffix_control", "native_batched_suffix"):
            # Independent client cache/material horizons; same warm-up charged
            # separately. Keep every installed inventory for exact residual ledger.
            with topology.client(background_inventory_refill=False) as client:
                core = client._core
                inventories = {}
                install = core._install_prepared_inventory_locked

                def save_inventory(state, item):
                    inventories[item.id] = item
                    return install(state, item)

                with patch.object(core, "_install_prepared_inventory_locked", save_inventory):
                    initial = client.responses.create(
                        model=model_id, input=base, temperature=0, max_output_tokens=max_new_tokens
                    )
                    state = core._transformer_states[model_id]
                    compiled = core._compiled_public_decoder(
                        state, max_input_tokens=bound, max_new_tokens=max_new_tokens
                    )
                    contract_binding = core._decoder_continuation(compiled).handshake_spec()
                    warm_audit = client.privacy_audit.to_dict()
                    warm_material = inventory_ledger(inventories)
                    if match_first_branch_geometry:
                        # Match numerical reduction geometry by token count, not
                        # numeric outcomes. Other branches retain changed lengths
                        # and must safely miss this original prefill lineage.
                        candidate = branches[0]
                        count = client.prepared_rows_for_response(candidate, 1, model=model_id, store=False)
                        while count < initial.usage.input_tokens:
                            candidate += " x"
                            count = client.prepared_rows_for_response(candidate, 1, model=model_id, store=False)
                        assert count == initial.usage.input_tokens
                        branches[0] = candidate
                    for index, prompt in enumerate(branches):
                        audit = client.privacy_audit
                        before = audit.to_dict()
                        before_material = inventory_ledger(inventories)
                        reserved = client.prepared_rows_for_response(
                            prompt, max_new_tokens, model=model_id, store=mode != "fresh_prefill"
                        )
                        logits = []
                        started = time.perf_counter()
                        with capture_prefill(mode, logits):
                            response = client.responses.create(
                                model=model_id, input=prompt, temperature=0,
                                max_output_tokens=max_new_tokens, store=mode != "fresh_prefill",
                            )
                        elapsed = time.perf_counter() - started
                        after = audit.to_dict()
                        delta = {key: after[key] - before[key] for key in before}
                        material = inventory_ledger(inventories)
                        material_delta = {key: material[key] - before_material[key] for key in material}
                        actual = logits[-1]
                        if mode == "fresh_prefill":
                            goldens[index] = (actual, response.output_text)
                        golden, text = goldens[index]
                        error = float(np.max(np.abs(actual - golden)))
                        same_token = int(np.argmax(actual)) == int(np.argmax(golden))
                        same_text = response.output_text == text
                        assert delta["preparation_requests_during_online"] == 0
                        assert delta["plaintext_prompt_bytes_sent"] == delta["plaintext_token_ids_sent"] == 0
                        state = core._transformer_states[model_id]
                        cache = state.prefill_cache
                        samples.append({
                            "mode": mode, "branch": index, "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens, "reserved_rows": reserved,
                            "reused_prefix_tokens": delta["prefill_prefix_tokens_reused"],
                            "stage_calls": delta["inference_stage_calls"],
                            "evaluated_per_stage_rows": delta["online_steps"],
                            "covered_online_upload_bytes": delta["masked_online_upload_bytes"],
                            "covered_online_download_bytes": delta["masked_online_download_bytes"],
                            "new_preparation_upload_bytes": delta["preparation_upload_bytes"],
                            "new_preparation_download_bytes": delta["preparation_download_bytes"],
                            "new_correction_push_bytes": delta["correction_push_bytes"],
                            "new_session_authorization_upload_bytes": delta["session_authorization_upload_bytes"],
                            "new_session_authorization_download_bytes": delta["session_authorization_download_bytes"],
                            "latency_seconds_including_request_preparation": elapsed,
                            "array_equal_all_logits": bool(np.array_equal(actual, golden)),
                            "max_abs_logit_difference": error,
                            "same_selected_token": same_token,
                            "expected_selected_token": int(np.argmax(golden)),
                            "actual_selected_token": int(np.argmax(actual)),
                            "same_response_text": same_text,
                            "numeric_gate_pass": error <= 1e-5 and same_token and same_text,
                            "cache_cap_bytes": cache_bytes,
                            "retained_cache_payload_bytes": 0 if cache is None else cache.size_bytes,
                            "material_delta": material_delta,
                            "material_horizon": material,
                            "online_preparation_requests": 0,
                             "unevaluated_final_generated_token_promoted": False,
                             "source_prefill_extent": initial.usage.input_tokens,
                             "actual_execution_mode": (mode if mode == "fresh_prefill" or delta["prefill_prefix_tokens_reused"]
                                                       else "fresh_prefill_geometry_miss"),
                            "warmup": {"input_tokens": initial.usage.input_tokens,
                                       "audit": warm_audit, "material": warm_material},
                         })
        if generated_prompt:
            generated = generated_cohort(topology, model_id, generated_prompt, cache_bytes)
    return {
        "experiment_digest": experiment.configuration_digest(),
        "native_contract": contract_binding,
        "source_config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
        "cohort_sha256": hashlib.sha256(json.dumps([base, *branches]).encode()).hexdigest(),
        "samples": samples,
        "child_processes": processes,
        "generated_state": generated,
        "scope": "ordinary SDK; independent inference/preparation child processes; native provider extension admission; one loopback repetition; covered bodies, not full wire or peak RSS",
        "production_extension_handshake_admitted": True,
    }


def generated_cohort(topology, model_id, prompt, cache_bytes):
    """Fresh prompts exclude generated lineage; explicit prior owner still batches."""
    with topology.client(background_inventory_refill=False) as client:
        core = client._core
        first = client.responses.create(model=model_id, input=prompt, temperature=0, max_output_tokens=4)
        prior = core._transformer_conversations[first.id]
        state = core._transformer_states[model_id]
        evaluated = prior.token_ids[:prior.snapshot.position]
        compiled = core._compiled_public_decoder(state, max_input_tokens=len(evaluated), max_new_tokens=4)
        cache = state.prefill_cache
        saved = cache.get(prefill_key(compiled.digest, state.bundle_fingerprint, evaluated), position=len(evaluated), layers=int(state.bundle.cfg["num_hidden_layers"]))
        assert first.usage.output_tokens == 4 and saved is None
        assert prior.pending_token_ids == prior.token_ids[-1:]
        assert cache.get(prefill_key(compiled.digest, state.bundle_fingerprint, prior.token_ids), position=len(prior.token_ids), layers=int(state.bundle.cfg["num_hidden_layers"])) is None
        assert not cache.put("generated-alias", prior.snapshot, prior.next_logits)
        assert not cache.put_prefixes(compiled.digest, state.bundle_fingerprint, evaluated, prior.snapshot)
        question = "Describe one reason briefly."
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": first.output_text}, {"role": "user", "content": question}]
        rendered = core._render_cached_decoder_prompt(state, normalize_input(messages), add_generation_prompt=True)
        ids = state.bundle.tokenizer().encode(rendered, add_bos=bool(state.bundle.tokenizer_descriptor.get("add_bos_token", True)))
        assert ids[:len(evaluated)] == evaluated
        records, vectors, texts = [], [], []
        for mode, body in (
            ("ordinary_fresh_prompt_generated_alias_blocked", {"input": messages}),
            ("fresh_one_shot_full_prefill_control", {"input": messages, "store": False}),
            ("native_prior_pending_suffix", {"input": question, "previous_response_id": first.id}),
        ):
            before = client.privacy_audit.to_dict()
            inventory = state.prepared_inventory
            before_status = None if inventory is None else inventory.status()
            results = []
            start = time.perf_counter()
            with capture_prefill(mode, results):
                response = client.responses.create(model=model_id, temperature=0, max_output_tokens=4, **body)
            elapsed = time.perf_counter() - start
            after = client.privacy_audit.to_dict()
            delta = {key: after[key] - before[key] for key in before}
            assert delta["preparation_requests_during_online"] == 0
            actual = results[-1]
            vectors.append(actual)
            texts.append(response.output_text)
            current_inventory = state.prepared_inventory
            status = current_inventory.status()
            material = {key: int(status[key]) - (int(before_status[key]) if current_inventory is inventory and before_status else 0) for key in ("consumed", "burned")}
            records.append({
                "mode": mode, "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
                "latency_seconds_including_request_preparation": elapsed,
                "stage_calls": delta["inference_stage_calls"], "evaluated_per_stage_rows": delta["online_steps"],
                "covered_online_upload_bytes": delta["masked_online_upload_bytes"],
                "covered_online_download_bytes": delta["masked_online_download_bytes"],
                "issued_per_stage_rows": delta["preparation_rows"],
                "complete_consumed_rows": material["consumed"], "complete_burned_rows": material["burned"],
                "idle_unreserved_rows": status["available"], "reused_prefix_tokens": delta["prefill_prefix_tokens_reused"],
                "actual_selected_token": int(np.argmax(actual)), "online_preparation_requests": delta["preparation_requests_during_online"],
                "cache_cap_bytes": cache_bytes, "retained_cache_payload_bytes": cache.size_bytes,
            })
        for index, record in enumerate(records):
            record.update({
                "max_abs_logit_difference_vs_one_shot_full_prefill": float(np.max(np.abs(vectors[index] - vectors[1]))),
                "array_equal_vs_one_shot_full_prefill": bool(np.array_equal(vectors[index], vectors[1])),
                "same_text_vs_one_shot_full_prefill": texts[index] == texts[1],
                "fresh_reuse_numeric_gate_pass": bool(np.array_equal(vectors[index], vectors[1])),
                "eligible_ordinary_prefill_basis": index != 2,
            })
        assert np.array_equal(vectors[0], vectors[1]) and texts[0] == texts[1]
        assert records[0]["reused_prefix_tokens"] == 0
        assert prior.snapshot.state_basis.owner_response_id == first.id
        return {
            "evaluated_generated_tokens_promoted": 0,
            "evaluated_generated_tokens_retained_in_private_conversation": len(evaluated) - first.usage.input_tokens,
            "fresh_generated_reuse_admitted": False,
            "fresh_generated_reuse_numeric_gate_pass": False,
            "evaluated_position": len(evaluated), "pending_final_token_promoted": False,
            "round_trip_token_prefix_matches": True, "native_contract": compiled._plan.continuation_schedule(
                experiment_pipeline(core)).handshake_spec(),
            "samples": records,
            "numeric_reference": "ordinary prompt compares every logit against canonical one-shot prefill; explicit prior owner compares independent clear teacher-fold in opt-in test; incremental state is not eligible for fresh reuse",
        }


def experiment_pipeline(core):
    from pllm.configuration import Pipeline

    return Pipeline.from_spec(json.loads(core.experiment.canonical_composition))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--cached-qwen", action="store_true", help="pinned cached weights, offline only")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pllm-batched-continuation-") as directory:
        work = Path(directory)
        root = create_tiny_llama_checkpoint(
            work / "tiny", hidden_size=128, intermediate_size=256, head_dim=32,
            num_hidden_layers=8, model_type="qwen2", with_qkv_bias=True,
        )
        base = "abcde " * 22
        tiny = http_cohort(root, work / "tiny-roles", base=base,
                           branches=[base[:-23] + "new" + "y" * 20], bound=248, cache_bytes=1 << 20)
        real = None
        if args.cached_qwen:
            from huggingface_hub import hf_hub_download

            revision = "7ae557604adf67be50417f59c2c2f167def9a775"
            root = Path(hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct", "config.json", revision=revision,
                                        local_files_only=True)).parent
            base = "Explain why private inference uses fresh masks. " * 4
            real = http_cohort(
                root, work / "qwen-roles", base=base,
                branches=[base[:94] + "Give one reason.", base[:94] + "Describe two reasons briefly.",
                          base[:94] + "What changes for a longer prompt with different tokens?"],
                bound=128, cache_bytes=64 << 20,
                generated_prompt="Explain why private inference uses fresh masks.",
                match_first_branch_geometry=True,
            )
            real["source"] = {"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "revision": revision,
                              "local_files_only": True}
        result = {
            "schema": "pllm.batched_continuation_evidence.v1", "date": "2026-10-01",
            "tiny": tiny, "pinned_qwen": real,
            "generated_state_promotion": False,
            "generated_fresh_promotion_status": "vetoed_pending_canonical_numeric_compatibility",
            "fresh_prefix_admission": "sealed completed prefill basis; same original full attention-reduction extent",
            "production_extension_handshake_admitted": True,
            "code_sha256": {
                name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                for name in (
                    "crates/pllm-compiler/src/decoder_continuation.rs",
                    "crates/pllm-python/src/continuation.rs", "python/pllm/modeling.py",
                    "python/pllm/runtime/semantic_executor.py", "python/pllm/runtime/client.py",
                     "python/pllm/runtime/prefill_cache.py", "scripts/probe_batched_continuation.py",
                     "python/pllm/runtime/transformer_client.py",
                    "python/pllm/runtime/continuation_admission.py", "python/pllm/runtime/server.py",
                )
            },
            "limitations": ["single repetition; no wire/RSS claim",
                            "hybrid/recurrent/windowed/shared states rejected", "legacy sequential is explicit control only"],
        }
        if args.archive and args.archive.exists():
            previous = json.loads(args.archive.read_text())
            historical = previous.get("historical_reference")
            if historical is None and previous.get("production_extension_handshake_admitted") is False:
                historical = previous
            if historical is not None:
                result["historical_reference"] = historical
            research = previous.get("historical_pre_lineage_veto")
            if research is None and previous.get("generated_state_promotion") is True:
                research = {key: previous[key] for key in ("tiny", "pinned_qwen", "code_sha256", "date")}
                research.update({
                    "scope": "historical research before lineage/geometry veto; not selectable fresh-prompt optimization",
                    "fresh_generated_reuse_numeric_gate_pass": False,
                    "shipping_optimization_evidence": False,
                })
            if research is not None:
                result["historical_pre_lineage_veto"] = research
        text = json.dumps(result, sort_keys=True, indent=2) + "\n"
        if args.archive:
            args.archive.write_text(text)
        print(text)


if __name__ == "__main__":
    main()
