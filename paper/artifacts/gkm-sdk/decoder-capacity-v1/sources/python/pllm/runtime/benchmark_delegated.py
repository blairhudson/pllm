"""Public metadata/admission for ordinary GKM benchmarks; no execution runner."""

import json


DELIVERY = "buffered_until_verified"
TPS_UNAVAILABLE = "request_end_verification_buffers_all_output"


def is_delegated(experiment):
    if experiment is None:
        return False
    from pllm.profiles import resolve_runtime_composition

    options = resolve_runtime_composition(experiment.pipeline)
    return options is not None and options.client_runtime == "compiled_gkm_v1"


def admit_options(experiment, *, backend=None, docker=False, docker_image=None,
                  docker_network=None, cache_bytes=0, cache_mode="exact", cache_bound=None,
                  inventory_policy=None, bundle_compression=None, temperature=None,
                  output_tokens, requests=1):
    """Reject incompatible benchmark choices before source inspection or role startup."""
    if (backend not in (None, "native") or docker or docker_image is not None
            or docker_network is not None):
        raise ValueError("GKM benchmarking requires the native role backend")
    if cache_bytes or cache_mode != "exact" or cache_bound is not None:
        raise ValueError("GKM benchmarking does not support prefix/KV reuse")
    if inventory_policy not in (None, "none") or bundle_compression not in (None, "none"):
        raise ValueError("GKM benchmarking has no preparation inventory or ordinary client bundle")
    if temperature is not None and (type(temperature) not in (int, float) or temperature != 0):
        raise ValueError("GKM benchmarking requires fixed-length greedy generation")
    budget = experiment.budget
    expanded = experiment.pipeline.components["linear"].params.get("capacity") == "decoder"
    if (experiment.pipeline.model.kind == "tiny"
            or not 1 <= budget.max_input_tokens <= (512 if expanded else 32)
            or not 1 <= budget.max_new_tokens <= (65 if expanded else 9)
            or not 1 <= budget.requests <= 64 or type(output_tokens) is not int
            or not 1 <= output_tokens <= budget.max_new_tokens
            or type(requests) is not int or not 1 <= requests <= budget.requests):
        raise ValueError("GKM benchmark exceeds its saved source/workload envelope")


def inspect_runtime(experiment):
    """Native full-envelope contract and actual source pins, without tensor values."""
    from pllm._benchmark_contract import digest
    from .delegated_source import admit_source

    source = admit_source(experiment)
    # Admission already bound the complete saved envelope. Preserve its native
    # canonical spelling and domain-separated identity instead of binding again.
    raw = source.runtime_contract_bytes
    contract = json.loads(raw)
    decoder = contract["decoder"]
    numeric = {"numeric_profile": decoder["numeric_profile"],
               "local_numeric_profile": decoder["local_numeric_profile"],
               "lookup": "original_float32", "output_head": "full", "kv": "full"}
    pins = {"checkpoint_digest": source.checkpoint_digest,
            "source_lock_digest": source.source_lock_digest,
            "tokenizer_digest": source.tokenizer_digest}
    metadata = {
        "schema": "pllm.gkm.benchmark_contract.v1", "runtime_kind": "gkm",
        "client_runtime": "compiled_gkm_v1", "numeric": numeric, **pins,
        "model_fingerprint": digest({"schema": "pllm.gkm.numeric_source.v1", **pins, **numeric}),
        "native_runtime_contract_digest": source.runtime_contract_digest,
        "model_config_digest": decoder["plan"]["config_digest"],
        "input_tokens_bound": decoder["input_rows"],
        "output_tokens_bound": decoder["decode_steps"] + 1,
        "vocabulary": decoder["vocabulary"], "width": decoder["width"],
        "matrix_count": len(decoder["matrices"]),
        "source_float32_bytes": decoder["source_float32_bytes"],
        "execution_payload_estimate_bytes": decoder["execution_payload_estimate_bytes"],
        "sharing": "full", "generation_policy": "fixed_length_greedy",
        "eos_policy": "fixed_count", "verification_window": "request_end",
        "output_delivery": DELIVERY, "decode_tps_supported": False,
        "decode_tps_unavailable_reason": TPS_UNAVAILABLE,
        "ttft_scope": "online_start_to_first_accepted_text",
        "decode_body_scope": "semantic_decode_before_request_end_verification",
        "kv_reuse": False, "preparation_inventory": False,
    }
    return metadata, contract
