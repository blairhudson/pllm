"""Public admission for ordinary final-output benchmarks; no execution runner."""

import json
import hashlib
import math


DELIVERY = "buffered_until_verified"
TPS_UNAVAILABLE = "request_end_verification_buffers_all_output"
REPLICATED_DELIVERY = "buffered_final_ids"
REPLICATED_TPS_UNAVAILABLE = "final_only_private_token_disclosure"

# Cumulative worker-local wall clocks, sampled by the existing metrics path.
# These are neither remote compute counters nor protocol/body measurements.
FSS_PHASE_COUNTERS = frozenset({
    "preparation_calls", "preparation_ns", "delivery_callback_ns", "seal_callback_ns",
    "online_calls", "online_ns",
})


def fss_phase_timing_delta(before, after):
    """Difference complete same-worker snapshots; absent/reset samples stay unknown.

    Both arguments are the worker metrics' ``phase_timing`` objects. Completed
    native calls include failed/cancelled calls, after their native thread drains.
    Callback times are serial subsets of preparation_ns, including bridge/HTTP
    waiting and seal acknowledgements. The remainder is not pure native compute.
    """
    def valid(value):
        return (type(value) is dict and set(value) == FSS_PHASE_COUNTERS
                and all(type(n) is int and 0 <= n < 1 << 63 for n in value.values())
                and value["preparation_calls"] <= 64 and value["online_calls"] <= 64
                and value["delivery_callback_ns"] + value["seal_callback_ns"] <= value["preparation_ns"]
                and (value["preparation_calls"] != 0 or value["preparation_ns"] == 0)
                and (value["online_calls"] != 0 or value["online_ns"] == 0))
    if not valid(before) or not valid(after):
        return None
    delta = {key: after[key] - before[key] for key in FSS_PHASE_COUNTERS}
    return delta if valid(delta) else None


def fss_phase_timing(record):
    """Summarize measured per-role deltas; never infer phases from disclosure time.

    ``processes[role].phase_timing`` holds fss_phase_timing_delta's counters.
    Independent holder run clocks overlap: their maximum is a role-run envelope,
    not the client's complete online wall interval, and must not be summed.
    """
    from .fss_protocol import HOLDERS, ROLES
    processes = record.get("processes")
    processes = processes if type(processes) is dict else {}
    zero = dict.fromkeys(FSS_PHASE_COUNTERS, 0)
    counters = {role: fss_phase_timing_delta(zero, value.get("phase_timing"))
                if type(value := processes.get(role)) is dict else None for role in ROLES}
    dealer = counters["dealer"]
    # Ordinary run records cover one request. Never combine maxima of unrelated
    # multi-request holder totals into a purported request latency.
    prepared = dealer is not None and dealer["preparation_calls"] == 1 and dealer["online_calls"] == 0
    online = {role: (value["online_ns"] / 1e9 if value is not None
                      and value["online_calls"] == 1 and value["preparation_calls"] == 0 else None)
              for role in HOLDERS for value in [counters[role]]}
    return {
        "preparation_seconds": dealer["preparation_ns"] / 1e9 if prepared else None,
        "delivery_callback_seconds": dealer["delivery_callback_ns"] / 1e9 if prepared else None,
        "seal_callback_seconds": dealer["seal_callback_ns"] / 1e9 if prepared else None,
        "preparation_outside_callbacks_seconds": (
            dealer["preparation_ns"] - dealer["delivery_callback_ns"] - dealer["seal_callback_ns"]
        ) / 1e9 if prepared else None,
        "online_seconds_by_holder": online,
        "online_seconds": max(online.values()) if all(value is not None for value in online.values()) else None,
        "preparation_scope": "dealer_native_run_wall_including_serial_delivery_and_seal_callbacks",
        "outside_callbacks_scope": "native_planning_issuance_encoding_and_bridge_not_pure_compute",
        "online_scope": "maximum_holder_native_run_wall_not_client_online_interval",
        "excludes": ["source_import", "native_role_bind", "client_input_output_transport", "retirement"],
        "includes_failed_calls": True,
    }


def is_delegated(experiment):
    if experiment is None:
        return False
    from pllm.profiles import resolve_runtime_composition

    options = resolve_runtime_composition(experiment.pipeline)
    return options is not None and options.client_runtime == "compiled_gkm_v1"


def is_replicated(experiment):
    if experiment is None:
        return False
    from pllm.profiles import resolve_runtime_composition
    options = resolve_runtime_composition(experiment.pipeline)
    return options is not None and options.client_runtime == "compiled_replicated_v1"


def is_buffered(experiment):
    return is_delegated(experiment) or is_replicated(experiment) or is_fss(experiment)


def is_fss(experiment):
    if experiment is None:
        return False
    from pllm.profiles import resolve_runtime_composition
    options = resolve_runtime_composition(experiment.pipeline)
    return options is not None and options.client_runtime == "compiled_fss_v1"


def output_contract(experiment):
    return ((REPLICATED_DELIVERY, REPLICATED_TPS_UNAVAILABLE) if is_replicated(experiment) or is_fss(experiment)
            else (DELIVERY, TPS_UNAVAILABLE))


def admit_options(experiment, *, backend=None, docker=False, docker_image=None,
                  docker_network=None, emulate_wan=False, cache_bytes=0, cache_mode="exact", cache_bound=None,
                  inventory_policy=None, bundle_compression=None, temperature=None,
                  output_tokens, requests=1):
    """Reject incompatible benchmark choices before source inspection or role startup."""
    name = "FuseFSS" if is_fss(experiment) else "PUMA" if is_replicated(experiment) else "GKM"
    if (backend not in (None, "native") or docker or docker_image is not None
            or docker_network is not None and not emulate_wan):
        raise ValueError(f"{name} benchmarking requires the native role backend")
    if docker_network is not None:
        from .native_wan import validate_native_link
        validate_native_link(docker_network)
    if cache_bytes or cache_mode != "exact" or cache_bound is not None:
        raise ValueError(f"{name} benchmarking does not support prefix/KV reuse")
    if inventory_policy not in (None, "none") or bundle_compression not in (None, "none"):
        raise ValueError(f"{name} benchmarking has no preparation inventory or ordinary client bundle")
    if temperature is not None and (type(temperature) not in (int, float) or temperature != 0):
        raise ValueError(f"{name} benchmarking requires fixed-length greedy generation")
    budget = experiment.budget
    expanded = experiment.pipeline.components["linear"].params.get("capacity") == "decoder"
    if (experiment.pipeline.model.kind == "tiny"
             or not 1 <= budget.max_input_tokens <= (16 if name == "FuseFSS" else 512 if expanded else 32)
             or not 1 <= budget.max_new_tokens <= (4 if name == "FuseFSS" else 8 if name == "PUMA" else 65 if expanded else 9)
            or not 1 <= budget.requests <= 64 or type(output_tokens) is not int
            or not 1 <= output_tokens <= budget.max_new_tokens
            or type(requests) is not int or not 1 <= requests <= budget.requests):
        raise ValueError(f"{name} benchmark exceeds its saved source/workload envelope")
    if name == "FuseFSS" and budget.max_input_tokens + budget.max_new_tokens - 1 > 16:
        raise ValueError("FuseFSS complete-prefix sequence exceeds native envelope")


def inspect_runtime(experiment):
    """Native full-envelope contract and actual source pins, without tensor values."""
    if is_fss(experiment):
        return _inspect_fss(experiment)
    if is_replicated(experiment):
        return _inspect_replicated(experiment)
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
        "sharing": experiment.pipeline.components["linear"].params["sharing"],
        "generation_policy": "fixed_length_greedy",
        "eos_policy": "fixed_count", "verification_window": "request_end",
        "output_delivery": DELIVERY, "decode_tps_supported": False,
        "decode_tps_unavailable_reason": TPS_UNAVAILABLE,
        "ttft_scope": "online_start_to_first_accepted_text",
        "decode_body_scope": "semantic_decode_before_request_end_verification",
        "kv_reuse": False, "preparation_inventory": False,
    }
    return metadata, contract


def _inspect_replicated(experiment):
    from pllm._benchmark_contract import digest
    from .replicated_source import admit_source
    source = admit_source(experiment)
    contract = json.loads(source.runtime_contract_bytes)
    numeric = {"numeric_profile": contract["numeric_profile"], "lookup": "shared_q18",
               "output_head": "full_shared_q18", "state": "replicated_private_full_prefix"}
    pins = {"checkpoint_digest": source.checkpoint_digest,
            "source_lock_digest": source.source_lock_digest, "tokenizer_digest": source.tokenizer_digest}
    return {
        "schema": "pllm.replicated.benchmark_contract.v1", "runtime_kind": "replicated",
        "client_runtime": "compiled_replicated_v1", "numeric": numeric, **pins,
        "model_fingerprint": digest({"schema": "pllm.replicated.numeric_source.v1", **pins, **numeric}),
        "native_runtime_contract_digest": source.runtime_contract_digest,
        "model_config_digest": contract["plan"]["config_digest"],
        "input_tokens_bound": contract["input_rows"], "output_tokens_bound": contract["decode_steps"] + 1,
        "vocabulary": contract["vocabulary"], "width": contract["width"],
        "matrix_count": sum(len(shape) == 2 for shape in contract["source_shapes"].values()),
        "source_float32_bytes": contract["source_float32_bytes"],
        "execution_payload_estimate_bytes": contract["execution_payload_estimate_bytes"],
        "sharing": "three_party_replicated", "generation_policy": "fixed_length_greedy",
        "eos_policy": "fixed_count", "verification_window": None,
        "output_delivery": REPLICATED_DELIVERY, "decode_tps_supported": False,
        "decode_tps_unavailable_reason": REPLICATED_TPS_UNAVAILABLE,
        "ttft_scope": "online_start_to_final_token_disclosure",
        "decode_body_scope": "unmeasured", "kv_reuse": False, "preparation_inventory": False,
    }, contract


def _inspect_fss(experiment):
    from pllm._benchmark_contract import digest
    from .fss_source import resolve_client_source, _material_delivery
    from .fss_protocol import ROLES
    source = resolve_client_source(experiment)
    contract = source.contract
    delivery = _material_delivery(contract)
    if delivery != _material_delivery(experiment.pipeline.components["linear"].params):
        raise ValueError("FuseFSS source material delivery differs from benchmark Experiment")
    numeric = {"numeric_profile": contract["numeric_profile"], "ring_bits": 50,
               "fractional_bits": 12, "external_encoding": "exact_ieee_nearest_even_q12",
               "internal_rescale": "signed_floor", "state": "private_complete_prefix"}
    pins = {"checkpoint_digest": source.checkpoint_digest,
            "source_lock_digest": source.source_lock_digest, "tokenizer_digest": source.tokenizer_digest}
    return {
        "schema": "pllm.fss.benchmark_contract.v1", "runtime_kind": "fss",
        "client_runtime": "compiled_fss_v1", "numeric": numeric, **pins,
        **({"material_delivery": delivery} if delivery != "single" else {}),
        "snapshot_pins": source.snapshot_pins,
        "model_fingerprint": digest({"schema": "pllm.fss.numeric_source.v1", **pins, **numeric}),
        "source_contract_bytes_sha256": hashlib.sha256(source.contract_bytes).hexdigest(),
        "native_runtime_contract_digest": None, "native_program_digest": None,
        "model_config_digest": contract["plan"]["config_digest"],
        "input_tokens_bound": contract["input_rows"], "output_tokens_bound": contract["output_tokens"],
        "vocabulary": contract["vocabulary"], "width": contract["width"],
        "matrix_count": sum(len(s) == 2 for s in contract["source_shapes"].values()),
        "source_float32_bytes": contract["source_float32_bytes"],
        "execution_payload_estimate_bytes": None, "program_resources": None,
        "native_admission": contract["admission"], "complete_program_admitted": False,
        "admission_scope": "metadata_preflight_only; execution_admission_occurs_in_each_role",
        "sharing": "two_holders_offline_dealer_public_weights", "roles": list(ROLES),
        "preparation_roles": ["dealer"], "generation_policy": "fixed_length_greedy", "eos_policy": "fixed_count",
        "verification_window": None, "output_delivery": REPLICATED_DELIVERY,
        "decode_tps_supported": False, "decode_tps_unavailable_reason": REPLICATED_TPS_UNAVAILABLE,
        "ttft_scope": "sdk_call_to_final_disclosure_including_preprocessing", "decode_body_scope": "unmeasured",
        "kv_reuse": False, "preparation_inventory": False, "source_checkpoint_delivery_measured": False,
        "cross_numeric_rankings_supported": False,
    }, contract


def completed_fss_contract(metadata, manifest_bytes):
    """Attach a completed request's native-checked public program identity.

    The client imports no weights. Resource estimates remain unknown unless
    independently obtained from the role's native program; metadata preflight
    keeps its original scope.
    """
    from .fss_protocol import metadata as parse, digest
    manifest = parse(manifest_bytes)
    if (set(manifest) != {"schema", "contract", "program", "source", "input_digest",
                          "profile", "initial", "selections", "vocabulary"}
            or manifest.get("schema") != "pllm.fss.channel.program.v1"
            or manifest.get("profile") != metadata["numeric"]["numeric_profile"]
            or type(manifest.get("initial")) is not int
            or not 0 < manifest["initial"] <= metadata["input_tokens_bound"]
            or type(manifest.get("selections")) is not int
            or not 0 < manifest["selections"] <= metadata["output_tokens_bound"]
            or type(manifest.get("vocabulary")) is not int
            or manifest["vocabulary"] != metadata["vocabulary"]):
        raise ValueError("FuseFSS completed program differs from benchmark contract")
    if (manifest["initial"] == metadata["input_tokens_bound"]
            and manifest["selections"] == metadata["output_tokens_bound"]
            and manifest["contract"] != metadata["source_contract_bytes_sha256"]):
        raise ValueError("FuseFSS completed source contract differs")
    for field in ("contract", "program", "source", "input_digest"):
        digest(manifest[field])
    return {**metadata, "native_runtime_contract_digest": manifest["contract"],
            "native_program_digest": manifest["program"],
            "native_source_digest": manifest["source"],
            "native_input_digest": manifest["input_digest"],
            "completed_program_geometry": {"input_tokens": manifest["initial"],
                                           "output_tokens": manifest["selections"]},
            "complete_program_admitted": True,
            "admission_scope": "matching_three_role_native_manifests_and_completed_request"}


def fss_body_accounting(record):
    """All instrumented application bodies once; disk/timing are separate scopes."""
    from .fss_protocol import ROLES, PEER_COUNTERS
    from .topology_accounting import FSS_COVERAGE
    privacy = record.get("privacy")
    privacy = privacy if isinstance(privacy, dict) else {}
    fields = [f"role_link.{role}.{phase}_{direction}_bytes" for role in ROLES
              for phase in ("setup", "online", "teardown") for direction in ("upload", "download")]
    fields += sorted(PEER_COUNTERS)
    valid = lambda key: type(privacy.get(key)) is int and 0 <= privacy[key] < 1 << 63
    complete = all(valid(key) for key in fields)
    edges = {}
    phases = {"setup": 0, "online": 0, "teardown": 0}
    for role in ROLES:
        for phase in phases:
            for direction in ("upload", "download"):
                key = f"role_link.{role}.{phase}_{direction}_bytes"
                edge = f"client->{role}" if direction == "upload" else f"{role}->client"
                edges.setdefault(edge, {})[phase] = privacy[key] if valid(key) else None
    for source in ROLES:
        for destination in ROLES:
            if source == destination:
                continue
            edge = edges.setdefault(f"{source}->{destination}", {})
            for phase in ("setup", "online"):
                key = f"peer_link.{source}.{destination}.{phase}_bytes"
                edge[phase] = privacy[key] if valid(key) else None
            edge["teardown"] = 0  # No role peer teardown body in this protocol.
    if complete:
        phases = {phase: sum(values[phase] for values in edges.values()) for phase in phases}
    else:
        phases = dict.fromkeys(phases)
    disk_keys = [f"fss_storage.{role}.{operation}_bytes" for role in ("worker_a", "worker_b")
                 for operation in ("read", "write")]
    links = [{"source": edge.split("->")[0], "destination": edge.split("->")[1],
              "phase": phase, "serialized_body_bytes": value}
             for edge, values in edges.items() for phase, value in values.items()
             if phase != "teardown" or "client" in edge.split("->")] if complete else None
    processes = record.get("processes")
    processes = processes if isinstance(processes, dict) else {}
    cpu = {role: (processes[role].get("cpu_seconds") if isinstance(processes.get(role), dict) else None)
           for role in ("client", *ROLES)}
    if not all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in cpu.values()):
        cpu = None
    return {"schema": "pllm.fss.topology_body_accounting.v1",
            "body_counter_coverage": FSS_COVERAGE,
            "tracked_body_counter_set_present": complete, "roles": ["client", *ROLES],
            "preparation_roles": ["dealer"], "preparation_inventory": False,
            "body_bytes_by_edge": links, "body_bytes_by_phase": phases,
            "all_link_serialized_body_bytes": sum(phases.values()) if complete else None,
            "client_serialized_body_bytes": sum(e["serialized_body_bytes"] for e in links
                if "client" in (e["source"], e["destination"])) if complete else None,
            "online_all_link_serialized_body_bytes": phases["online"],
            "online_client_serialized_body_bytes": sum(e["serialized_body_bytes"] for e in links
                if e["phase"] == "online" and "client" in (e["source"], e["destination"])) if complete else None,
            "run_window_cpu_seconds_by_role": cpu,
            "aggregate_run_window_cpu_seconds": sum(cpu.values()) if cpu is not None else None,
            "total_wire_bytes": None, "full_response_compute_cap_checked": False,
            "total_body_bytes": sum(phases.values()) if complete else None,
            "decode_online_body_bytes": None, "decode_output_tokens": None,
             "storage_bytes": {key: privacy[key] if valid(key) else None for key in disk_keys},
             "phase_timing": fss_phase_timing(record),
            "excluded_costs": ["HTTP/TLS framing", "uninstrumented retries", "source checkpoint delivery"],
            "storage_in_network_totals": False, "streaming_ttft_supported": False}


def inspect_fss_imported(loaded):
    """Optional pricing hook for an ALREADY imported native source; never loads it.

    Client metadata preflight must not invent this evidence. Role/report owners
    may call this after import and separately retain its native build identity.
    Program pricing is not request admission or measured memory/traffic.
    """
    from pllm import _native
    from .fss_protocol import METADATA_LIMIT
    from .fss_tape import MAX_PACKET_BYTES, capacity_limits
    from .fss_transport import client_memory_bytes, price_transport, transport_capacity
    from .fss_source import _material_delivery
    contract = json.loads(loaded.contract_bytes)
    capacity = transport_capacity(contract)
    packet_limit, _ = capacity_limits(capacity)
    pricing = json.loads(_native.fss_transport_admission(loaded.native_program, loaded.contract_bytes))
    manifest = json.loads(loaded.manifest_bytes)
    if (pricing.get("schema") != "pllm.fss.program_transport_admission.v1"
            or pricing.get("scope") != "program-pricing-not-request-admission"
            or manifest.get("schema") != "pllm.fss.channel.program.v1"
            or pricing.get("contract_digest") != hashlib.sha256(loaded.contract_bytes).hexdigest()
            or any(pricing.get(key) != manifest.get(field) for key, field in (
                ("contract_digest", "contract"), ("program_digest", "program"),
                ("source_digest", "source"), ("input_digest", "input_digest")))
            or pricing.get("numeric_profile") != contract["numeric_profile"]
            or _material_delivery(pricing) != _material_delivery(contract)
            or pricing.get("initial_tokens") != contract["input_rows"]
            or pricing.get("selections") != contract["output_tokens"]
            or pricing.get("vocabulary") != contract["vocabulary"]):
        raise ValueError("FuseFSS imported native program pricing differs")
    for field, bound in (("largest_packet_bytes", MAX_PACKET_BYTES),
                          ("cumulative_packet_bytes_per_party_bound", packet_limit),
                          ("peer_frame_bytes_bound", MAX_PACKET_BYTES),
                          ("callback_metadata_bytes_bound", METADATA_LIMIT),
                          ("kernels", 4096 if capacity == "capsule" else (1 << 64) - 1),
                          ("input_frame_bytes", 272), ("output_frame_bytes", 272),
                          ("combined_reservation_bytes", 64 << 20),
                          ("client_peak_bytes", client_memory_bytes(capacity))):
        if type(pricing.get(field)) is not int or not 0 <= pricing[field] <= bound:
            raise ValueError("FuseFSS imported program exceeds unchanged transport bounds")
    if pricing["client_peak_bytes"] > 8 << 20:
        raise ValueError("FuseFSS imported client exceeds benchmark native workspace allowance")
    # Same pure storage admission used by actual workers. No files, shares,
    # packet issuance or model execution; metadata-only estimates lack this proof.
    price_transport(pricing, loaded.contract_bytes)
    return pricing
