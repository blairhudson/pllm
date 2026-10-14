"""Matched fresh-process SDK responses: prepared body versus coupled boundary.

No builds, installs or checkpoint-heavy work during planning. Private prompts,
credentials and one cohort salt travel on stdin. Persist commitments, never text
or token IDs. Local children do not establish independent operators. Native WAN
is opt-in and fails closed when the installed five-role adapter is unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile
import threading
import time

from benchmark_conversation_restore import commitment, state_commitment
from benchmark_growing_epochs_sdk import (
    bounded_commitment, freeze_installed_implementation, installed_bootstrap,
)
from frontier_probe_support import fresh_workers

SAMPLER_SEED = 42


def driver_implementation_identity():
    """The installed SDK lock does not cover these external measurement scripts."""
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
        "benchmark_private_boundary.py", "benchmark_conversation_restore.py",
        "benchmark_growing_epochs_sdk.py", "benchmark_epoch_prefix_sdk.py", "frontier_probe_support.py")}


DRIVER_IMPLEMENTATION = driver_implementation_identity()


def require_driver_implementation(expected):
    if expected != DRIVER_IMPLEMENTATION or driver_implementation_identity() != expected:
        raise BenchmarkBlocked("measurement_driver_implementation_changed")


def normalize_sdk_connection(value):
    """Restore SDK tuple contracts after private stdin JSON round-trip."""
    if type(value) is not dict:
        raise ValueError("invalid SDK connection")
    connection = dict(value)
    names = ("boundary_urls", "boundary_api_keys")
    if any(name in connection for name in names):
        for name in names:
            values = connection.get(name)
            if (type(values) not in (list, tuple) or len(values) != 2
                    or any(type(item) is not str or not item for item in values)):
                raise ValueError("paired boundary connection requires two URLs and keys")
            connection[name] = tuple(values)
    return connection


def freeze_sampler_rng(runtime):
    """Public benchmark sampler fixture only; cryptographic RNGs stay untouched."""
    import numpy as np
    runtime.rng = np.random.default_rng(SAMPLER_SEED)


def response_state_commitment(client, response, salt):
    """Compare exact numerical state, not topology-specific snapshot basis tags.

    Shared helper contains logits/KV bytes, token/pending and output commitments.
    Composition/source-binding digests and fresh response IDs are intentionally
    not a numerical parity witness; original body/source are checked separately.
    """
    saved = client._core._transformer_conversations[response.id]
    arrays = [saved.next_logits]
    for cache in saved.snapshot.caches:
        arrays.extend((cache.key[:cache.length], cache.value[:cache.length]))
    return {**state_commitment(client, response, salt),
        "array_dtype_and_shape": array_commitment(salt, arrays),
        "pending_count": len(saved.pending_token_ids)}


def boundary_storage_evidence(bundle):
    """Inspect known SDK matrix owners only; never serialize tensor contents.

    Private bundle admission separately validates source-required local tensors.
    No claim about unknown owners, OS caches or provider/source residency follows.
    """
    if type(bundle.stages) is not dict or type(bundle._local_matrices) is not dict:
        raise RuntimeError("client matrix ownership unobservable")
    local = {stage.id for stage in bundle.stages.values()
             if stage.client_weight is not None or stage.client_aux_weight is not None}
    boundary = {stage.id for stage in bundle.stages.values()
                if stage.role in {"token_lookup", "lm_head"}
                and (stage.client_weight is not None or stage.client_aux_weight is not None)}
    return {"scope": "known SDK stage weights and retained local native matrix bindings only",
        "private_boundary_selected": "private_boundary" in bundle.privacy,
        "tied_embeddings": bundle.manifest.get("tied_embeddings") is True,
        "local_quantized_stage_ids": sorted(local), "boundary_client_weight_stage_ids": sorted(boundary),
        "retained_local_matrix_ids": sorted(bundle._local_matrices)}


def bound_body_fingerprint(bundle):
    """Use the validated numeric body, not the placement-specific manifest ID."""
    fingerprint = bundle.privacy.get("body_fingerprint")
    metadata = bundle.manifest.get("metadata")
    if (not _digest(fingerprint) or type(metadata) is not dict
            or metadata.get("body_fingerprint") != fingerprint):
        raise RuntimeError("ordinary body fingerprint unavailable or inconsistent")
    return fingerprint


def complete_case_cpu(parent_start, parent_end, child_start, child_end):
    """Process CPU includes all parent threads; child rusage is charged once.

    Provider samples and the client's pre-report lifetime sample are subsets of
    reaped child CPU, not extra charges. Parent includes native WAN relay threads,
    admission, samplers and supervision; source distribution remains unmeasured.
    """
    if (not all(_duration(value) for value in (parent_start, parent_end, child_start, child_end))
            or parent_end < parent_start or child_end < child_start):
        raise RuntimeError("complete-case CPU clocks invalid or decreasing")
    parent, children = parent_end - parent_start, child_end - child_start
    return {"schema": "pllm.private_boundary_case_cpu.v1",
        "parent_all_threads_cpu_seconds": parent, "reaped_children_cpu_seconds": children,
        "aggregate_cpu_seconds": parent + children,
        "scope": "case admission through owned client/provider/relay cleanup; CPU only; post-return cohort verification/reporting charged separately",
        "native_wan_cpu_location": "included in parent all-threads process CPU, not client child CPU",
        "sampled_client_and_provider_cpu_added_again": False,
        "synchronized_all_role_request_cpu_seconds": None}


class BenchmarkBlocked(RuntimeError):
    """Public capability blocker; safe to retain without private exception text."""

    def __init__(self, code, *, memory_admission=None):
        self.code = code
        self.memory_admission = memory_admission
        super().__init__(code)


def matched_experiments(base, input_tokens, cap=8):
    from pllm.configuration import Pipeline
    from pllm.roles import PrivateLookupAndFullHead, PreparedBoundaryRoles
    from pllm.profiles import resolve_runtime_composition

    if type(input_tokens) is not int or not 1 <= input_tokens <= 4089 or cap != 8:
        raise ValueError("matched workload requires bounded input and eight-output cap")
    control = base.with_params(budget__requests=1, budget__max_input_tokens=input_tokens,
                               budget__max_new_tokens=cap)
    options = resolve_runtime_composition(control.pipeline)
    if (options is None or options.private_boundary or options.remote_output_head
            or not options.requires_preparation):
        raise ValueError("baseline must be ordinary prepared body with local tied boundary")
    # Optional slots are absent in serialized cold controls. Nested with_params
    # replaces existing paths only; public Pipeline construction adds new slots.
    coupled_pipeline = Pipeline(control.pipeline.model,
        {**control.pipeline.components, "boundary": PrivateLookupAndFullHead(),
         "topology": PreparedBoundaryRoles()}, profile=control.pipeline.profile)
    coupled = control.with_params(pipeline=coupled_pipeline)
    if resolve_runtime_composition(coupled.pipeline) is None:
        raise ValueError("coupled runtime does not support baseline components; do not silently change body/cache")
    # Only boundary placement and explicit role topology may change.
    left, right = control.to_spec(), coupled.to_spec()
    for spec in (left, right):
        components = spec["pipeline"]["components"]
        components.pop("boundary", None)
        components.pop("topology", None)
    if left != right:
        raise RuntimeError("matched model/body/tokenizer/quantization/cap identity changed")
    return {"baseline": control, "coupled": coupled}


def replicate_order(replicates):
    if type(replicates) is not int or not 3 <= replicates <= 12:
        raise ValueError("matched cohort requires 3..12 alternating replicates")
    return [(replicate, variant) for replicate in range(replicates)
            for variant in (("baseline", "coupled") if replicate % 2 == 0
                            else ("coupled", "baseline"))]


def array_commitment(salt, arrays):
    import numpy as np
    digest = hmac.new(salt, digestmod=hashlib.sha256)
    for array in arrays:
        array = np.asarray(array)
        digest.update(json.dumps({"shape": array.shape, "dtype": array.dtype.str},
                                 sort_keys=True).encode())
        for row in np.atleast_2d(array):
            digest.update(np.ascontiguousarray(row).view(np.uint8))
    return digest.hexdigest()


def client_worker(payload):
    import pllm
    from pllm.runtime.benchmark_response_group import inspect_private_cleanup, prefix_owners
    from pllm.runtime.topology_accounting import prepared_body_accounting, private_boundary_body_accounting
    from pllm.runtime.transformer_client import MaskedTransformerClientRuntime
    from pllm.runtime.private_boundary_transport import PrivateBoundaryTransport
    from pllm.runtime.types import Response

    salt = bytes.fromhex(payload["salt"])
    trajectory, times, response, failure, client, owners = [], [], None, None, None, ()
    account = private_boundary_body_accounting if payload["variant"] == "coupled" else prepared_body_accounting
    sample = MaskedTransformerClientRuntime.sample
    boundary_owners = []
    sampler_owners = []
    boundary_init = PrivateBoundaryTransport.__init__

    def audited_boundary_init(owner, *args, **kwargs):
        boundary_init(owner, *args, **kwargs)
        boundary_owners.append(owner)  # Reference only; no table or private-state copy.

    def audited_sample(runtime, logits, **kwargs):
        if all(runtime is not owner for owner in sampler_owners):
            freeze_sampler_rng(runtime)
            sampler_owners.append(runtime)
        # All exact logits, live KV, tokens and RNG states charged to both arms.
        arrays = [logits]
        for cache in runtime.caches:
            arrays.extend((cache.key[:cache.length], cache.value[:cache.length]))
        before = bounded_commitment(salt, runtime.rng.bit_generator.state)
        bits = array_commitment(salt, arrays)
        token = sample(runtime, logits, **kwargs)
        trajectory.append({"position": runtime.position, "logits_and_kv": bits,
            "sampled_token": commitment(salt, token), "rng_before": before,
            "rng_after": bounded_commitment(salt, runtime.rng.bit_generator.state)})
        return token

    MaskedTransformerClientRuntime.sample = audited_sample
    PrivateBoundaryTransport.__init__ = audited_boundary_init
    started = time.perf_counter()
    client_cpu_started = time.process_time()
    request_started = initialized = None
    rows = []
    try:
        require_driver_implementation(payload["driver_implementation"])
        exp = pllm.Experiment.from_spec(payload["experiment"])
        from pllm.model_loader import resolve_model
        if (not payload.get("source_lock_digest")
                or resolve_model(exp.pipeline.model).source_lock_digest != payload["source_lock_digest"]):
            raise RuntimeError("original source identity changed")
        client = pllm.OpenAI(experiment=exp, **normalize_sdk_connection(payload["connection"]),
            bundle_cache_dir=payload["cache"], background_inventory_refill=False, timeout=900)
        with client.tokenizer_scope():
            client.runtime.capabilities()
        initialized = time.perf_counter()
        request_started = time.perf_counter()
        request_cpu_started = time.process_time()
        with client.tokenizer_scope(), client.responses.create(input=payload["prompt"],
                max_output_tokens=8, temperature=0, store=True, stream=True) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    times.append(time.perf_counter())
                elif event.type in {"response.completed", "response.incomplete"}:
                    response = Response.from_dict(event.response) if isinstance(event.response, dict) else event.response
        elapsed = time.perf_counter() - request_started
        request_cpu = time.process_time() - request_cpu_started
        if (response is None or response.usage is None or response.usage.input_tokens != payload["input_tokens"]
                or response.usage.output_tokens != 8 or len(times) != 8
                or response.status != "incomplete"
                or (response.incomplete_details or {}).get("reason") != "max_output_tokens"
                or len(trajectory) != 8):
            raise RuntimeError("declared matched full-cap response not exercised")
        rows = [{"input_tokens": response.usage.input_tokens, "output_tokens": 8,
            "status": response.status, "state": response_state_commitment(client, response, salt),
            "trajectory": trajectory, "request_seconds": elapsed,
            "client_request_cpu_seconds": request_cpu,
            "request_cpu_scope": "client child dispatch through stream drainage; per-step audits included; final state hash excluded",
            "first_text_seconds": times[0] - request_started,
            "worker_entry_to_first_text_seconds": times[0] - started,
            "decode_tps": 7 / (times[-1] - times[0])}]
        states = tuple(client._core._transformer_states.values())
        if len(states) != 1:
            raise RuntimeError("ordinary compiled body owner unavailable")
        rows[0]["body_fingerprint"] = bound_body_fingerprint(states[0].bundle)
        rows[0]["bundle_model_fingerprint"] = states[0].bundle.manifest.get("fingerprint")
        rows[0]["client_boundary_storage"] = boundary_storage_evidence(states[0].bundle)
        states = ()  # Do not retain crypto/bundle state merely for ownership diagnostics.
    except BaseException as error:
        failure = {"stage": "client_execution", "error_type": type(error).__name__}
        if isinstance(error, BenchmarkBlocked):
            failure["blocker"] = error.code
    finally:
        if client is not None:
            owners = prefix_owners(client)
            try:
                client.close()
            except BaseException as error:
                failure = {"stage": "client_retirement", "error_type": type(error).__name__}
        MaskedTransformerClientRuntime.sample = sample
        PrivateBoundaryTransport.__init__ = boundary_init
        frozen_runtime_count = len(sampler_owners)
        sampler_owners.clear()  # Audit must not keep a decoder/KV owner after close.
    audit = client.privacy_audit.to_dict() if client is not None else {}
    cleanup = None
    try:
        if client is not None:
            cleanup = inspect_private_cleanup(client, owners)
            if client._core._prepared_queue.requests or client._core._online_active or client._core._preparation_active:
                raise RuntimeError("prepared owners not retired")
            if payload["variant"] == "coupled" and not boundary_owners:
                raise RuntimeError("coupled transport execution unobservable")
            if payload["variant"] == "baseline" and boundary_owners:
                raise RuntimeError("ordinary baseline unexpectedly executed coupled transport")
            if any(not owner._retired or owner._state != "closed" or not owner.body_coverage_complete
                    or any(owned and not http.is_closed for http, owned in zip(owner._clients, owner._owned))
                    for owner in boundary_owners):
                raise RuntimeError("boundary session, HTTP owner or attempt coverage not retired")
            cleanup["boundary_transports_observed"] = len(boundary_owners)
            cleanup["boundary_transports_retired_and_owned_http_closed"] = True
        ledger = account({"privacy": audit})
        if (not ledger["tracked_body_counter_set_present"]
                or any(audit.get(key) != 0 for key in ("plaintext_prompt_bytes_sent",
                    "plaintext_token_ids_sent", "preparation_requests_during_online"))):
            raise RuntimeError("combined sender-side/private lifecycle evidence incomplete")
    except BaseException as error:
        failure = failure or {"stage": "client_validation", "error_type": type(error).__name__}
        ledger = account({"privacy": audit})
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"failure": failure, "responses": rows, "audit": audit, "ledger": ledger,
        "driver_implementation": DRIVER_IMPLEMENTATION,
        "sampler_diagnostic": {"seed": SAMPLER_SEED, "temperature": 0,
            "frozen_runtime_count": frozen_runtime_count, "cryptographic_randomness_frozen": False,
            "scope": "public sampler fixture and exact-state hashing charged inside client CPU/RSS/wall"},
        "cleanup": cleanup, "client_lifetime_cpu_seconds": usage.ru_utime + usage.ru_stime,
        "boundary_operation_diagnostics": [owner.operation_counters for owner in boundary_owners],
        "client_peak_rss_bytes": int(usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024),
        "client_setup_seconds": initialized - started if initialized is not None else None,
        "client_worker_entry_cpu_seconds": time.process_time() - client_cpu_started,
        "client_cpu_scope": "child process birth through pre-serialization report sample; complete child CPU charged after reap",
        "client_rss_scope": "OS child process lifetime high-water RSS through report sample, not a sampled request peak",
        "client_window_seconds": time.perf_counter() - started}


def admission(exp, wan):
    from pllm.metrics import benchmark_memory
    report = benchmark_memory(exp, backend="native", enforced_wan=wan)
    # Fresh client process, launcher, exact-state hashing, relay and sampler owners.
    required = report["estimate"]["native_total_peak_bytes"] + (320 << 20)
    return {"topology": report, "required_bytes": required,
        "admitted": report["admitted"] and required <= report["host"]["admission_budget_bytes"]}


def run_case(exp, workload, variant, replicate, salt, run, *, wan=False,
             source_case="source-present", source_lock_digest=None, driver_implementation=None):
    import psutil
    from pllm.deployment import WanConditions
    from pllm.runtime.servers import build_roles, TopologyError
    from pllm.runtime.topology_accounting import source_delivery_accounting

    # Start before admission: its source/header checks and geometry CPU belong to
    # this complete case too. process_time includes the native relay's threads.
    wall, cpu = time.perf_counter(), time.process_time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    require_driver_implementation(driver_implementation)
    memory = admission(exp, wan)
    if not memory["admitted"]:
        raise BenchmarkBlocked("complete_case_memory_admission_rejected", memory_admission=memory)
    if source_case == "missing-workers":
        # Local build_roles resolves prepositioned source. It is not a source
        # delivery instrument and cannot manufacture this arm's billed bytes.
        raise BenchmarkBlocked("missing_worker_artifact_and_full_source_meter_unavailable")
    expected = {"inference", "preparation"} | ({"boundary_a", "boundary_b"} if variant == "coupled" else set())
    peak, errors, stop = {}, [], threading.Event()
    network = None
    with tempfile.TemporaryDirectory(prefix="pllm-private-boundary-") as root:
        try:
            topology = build_roles(exp, **({"wan": WanConditions()} if wan else {}), log_dir=Path(root) / "logs")
        except TopologyError as error:
            if variant == "coupled" and wan:
                raise BenchmarkBlocked("native_five_role_wan_unavailable") from error
            raise
        with topology:
            processes = {status.role: psutil.Process(status.pid) for status in topology.statuses}
            if set(processes) != expected:
                raise RuntimeError("all hosted role processes must be counted separately")

            def sample():
                try:
                    while not stop.is_set():
                        for role, process in processes.items():
                            peak[role] = max(peak.get(role, 0), process.memory_info().rss)
                        stop.wait(0.05)
                except psutil.Error as error:
                    errors.append(type(error).__name__)

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            try:
                connection = {"base_url": topology.inference_url,
                    "preparation_base_url": topology.preparation_url,
                    "api_key": topology._inference_key, "preparation_api_key": topology._preparation_key}
                if variant == "coupled":
                    links = topology.boundary_connections()
                    connection.update(boundary_urls=[links[role][0] for role in ("boundary_a", "boundary_b")],
                        boundary_api_keys=[links[role][1] for role in ("boundary_a", "boundary_b")])
                client = run({"experiment": exp.to_spec(), **workload, "variant": variant,
                    "salt": salt.hex(), "connection": connection, "cache": str(Path(root) / "cache"),
                    "source_lock_digest": source_lock_digest, "driver_implementation": driver_implementation})
                require_driver_implementation(driver_implementation)
                role_cpu = {}
                for role, process in processes.items():
                    used = process.cpu_times()
                    role_cpu[role] = used.user + used.system
            finally:
                stop.set()
                sampler.join()
        if wan:
            network = topology._native_network.pacer.snapshot()
            if (network["total_stream_bytes"] != sum(network["directed_stream_bytes"].values())
                    or network["conditions_digest"] != WanConditions().digest or not network["enforced"]):
                raise RuntimeError("native WAN conservation unavailable")
            # Boundary directions must exist; no three-role WAN relabelled five-role.
            boundary_edges = {f"{source}->{destination}"
                for role in ("boundary_a", "boundary_b")
                for source, destination in (("client", role), (role, "client"))}
            if variant == "coupled" and not boundary_edges <= network["directed_stream_bytes"].keys():
                raise RuntimeError("native WAN excludes boundary graph")
    if errors or set(peak) != expected:
        raise RuntimeError("role resource sample coverage incomplete")
    if any(process.is_running() for process in processes.values()):
        raise RuntimeError("hosted role processes not reaped")
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_ledger = complete_case_cpu(cpu, time.process_time(), before.ru_utime + before.ru_stime,
                                   after.ru_utime + after.ru_stime)
    response_rows = client.get("responses", [])
    part = (client.get("ledger") or {}).get("traffic_attribution") or {}
    metrics = {"client_base_traffic_bytes": None, "client_output_traffic_bytes": None,
        "all_link_base_traffic_bytes": part.get("base_all_link_body_bytes"),
        "all_link_output_traffic_bytes": part.get("output_dependent_all_link_body_bytes"),
        "client_peak_rss_bytes": client.get("client_peak_rss_bytes"),
        "ttft_seconds": response_rows[0]["first_text_seconds"] if response_rows else None,
        "decode_tps": response_rows[0]["decode_tps"] if response_rows else None}
    return {"variant": variant, "replicate": replicate, "input_tokens": workload["input_tokens"],
        "client": client, "provider_birth_cpu_seconds": role_cpu,
        "provider_sampled_peak_rss_bytes": peak, "wan": network,
        "aggregate_cleanup_cpu_seconds": cpu_ledger["aggregate_cpu_seconds"], "cpu_accounting": cpu_ledger,
        "provider_cpu_scope": "sampled process birth through client subprocess exit; shutdown included separately in reaped-child CPU",
        "provider_rss_scope": "50 ms samples after role startup through client subprocess exit; startup and shutdown peaks unknown",
        "complete_case_seconds": time.perf_counter() - wall,
        "source_delivery": source_delivery_accounting(source_case),
        "admitted_host_bytes": memory["required_bytes"], "priority_metrics": metrics,
        "provider_processes_reaped": True, "source_lock_digest": source_lock_digest,
        "source_case": source_case, "wan_requested": wan,
        "source_inclusive_all_link_application_bytes": None,
        "source_inclusive_all_link_tcp_stream_bytes": None}


def _count(value, *, positive=False):
    return type(value) is int and value >= int(positive)


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _duration(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0 and (not positive or value > 0)


def validate_result(result, *, source_case="source-present", wan=False, source_lock_digest=None):
    """Revalidate raw SDK evidence; cached flags/ledgers alone cannot certify it."""
    from pllm.runtime.topology_accounting import (
        ACK_COVERAGE, BOUNDARY_COVERAGE, body_coverage_identity,
        prepared_body_accounting, private_boundary_body_accounting, source_delivery_accounting,
    )
    variant, count = result.get("variant"), result.get("input_tokens")
    if variant not in {"baseline", "coupled"} or not _count(count, positive=True) or count > 4089:
        raise RuntimeError("unknown arm or workload")
    if (result.get("source_case") != source_case or result.get("wan_requested") is not wan
            or not _digest(result.get("source_lock_digest"))
            or source_lock_digest is not None and result["source_lock_digest"] != source_lock_digest):
        raise RuntimeError("source or network scope changed")
    client = result.get("client")
    if (type(client) is not dict or client.get("failure") is not None
            or type(client.get("responses")) is not list or len(client["responses"]) != 1):
        raise RuntimeError("SDK response failed; do not rank partial execution")
    if client.get("driver_implementation") != DRIVER_IMPLEMENTATION:
        raise RuntimeError("measurement script identity mismatch")
    row = client["responses"][0]
    if (type(row) is not dict or row.get("input_tokens") != count or not _count(row.get("input_tokens"))
            or type(row.get("output_tokens")) is not int or row["output_tokens"] != 8
            or row.get("status") != "incomplete"
            or not all(_duration(row.get(key), positive=True) for key in (
                "request_seconds", "first_text_seconds", "decode_tps"))
            or row["first_text_seconds"] > row["request_seconds"]
            or type(row.get("body_fingerprint")) is not str or not row["body_fingerprint"]):
        raise RuntimeError("full-cap authoritative response evidence unavailable")
    storage = row.get("client_boundary_storage")
    ids = ("local_quantized_stage_ids", "boundary_client_weight_stage_ids", "retained_local_matrix_ids")
    if (type(storage) is not dict or storage.get("private_boundary_selected") is not (variant == "coupled")
            or storage.get("tied_embeddings") is not True
            or any(type(storage.get(key)) is not list or len(storage[key]) > 1024
                or any(type(value) is not str or not value or len(value) > 128 for value in storage[key])
                or storage[key] != sorted(set(storage[key])) for key in ids)
            or (variant == "coupled" and any(storage[key] for key in ids))
            or (variant == "baseline" and not {"lm_head", "token_lookup"}
                <= set(storage["boundary_client_weight_stage_ids"]))):
        raise RuntimeError("known client tied-head/table ownership does not match selected arm")
    state, trajectory = row.get("state"), row.get("trajectory")
    if (type(state) is not dict or not _count(state.get("position")) or state["position"] != count + 7
            or type(state.get("pending_count")) is not int or state["pending_count"] != 1
            or not _count(state.get("words"), positive=True)
            or any(not _digest(state.get(key)) for key in (
                "bits", "tokens", "pending", "output", "array_dtype_and_shape"))
            or type(trajectory) is not list or len(trajectory) != 8):
        raise RuntimeError("complete exact final state or sampling trajectory unavailable")
    for index, step in enumerate(trajectory):
        if (type(step) is not dict or not _count(step.get("position")) or step["position"] != count + index
                or any(not _digest(step.get(key)) for key in (
                    "logits_and_kv", "sampled_token", "rng_before", "rng_after"))
                or step["rng_before"] != step["rng_after"]
                or step["rng_before"] != trajectory[0]["rng_before"]):
            raise RuntimeError("full logits/KV/token/greedy RNG trajectory mismatch")
    sampler = client.get("sampler_diagnostic")
    if (type(sampler) is not dict or type(sampler.get("seed")) is not int or sampler["seed"] != SAMPLER_SEED
            or type(sampler.get("temperature")) is not int or sampler["temperature"] != 0
            or type(sampler.get("frozen_runtime_count")) is not int or sampler["frozen_runtime_count"] != 1
            or sampler.get("cryptographic_randomness_frozen") is not False):
        raise RuntimeError("public sampler fixture unavailable or crypto randomness changed")
    audit = client.get("audit")
    if type(audit) is not dict:
        raise RuntimeError("raw privacy audit unavailable")
    required = {"plaintext_prompt_bytes_sent": 0, "plaintext_token_ids_sent": 0,
        "preparation_requests_during_online": 0, "private_boundary_body_coverage_failures": 0,
        "generation_completed_responses": 1, "generation_output_tokens": 8,
        "generation_decode_output_tokens": 7, "generation_executed_decode_steps": 7,
        "generation_requested_output_tokens": 8,
        "private_boundary_lookup_rows": count + 7 if variant == "coupled" else 0,
        "private_boundary_head_calls": 8 if variant == "coupled" else 0}
    if any(not _count(audit.get(key)) or audit[key] != expected for key, expected in required.items()):
        raise RuntimeError("private workload/lookup/head/phase counters do not match response")
    if (not _count(audit.get("generation_attribution_supported_responses"))
            or audit["generation_attribution_supported_responses"] > 1):
        raise RuntimeError("unknown phase attribution support")
    rows = {name: audit.get("prepared_stage_rows_" + name)
            for name in ("issued", "reserved", "claimed", "burned", "discarded")}
    if (not all(_count(value) for value in rows.values())
            or rows["issued"] != rows["reserved"] + rows["discarded"]
            or rows["reserved"] != rows["claimed"] + rows["burned"]):
        raise RuntimeError("prepared lifetime row conservation failed")
    account = private_boundary_body_accounting if variant == "coupled" else prepared_body_accounting
    ledger = account({"privacy": audit})
    if (client.get("ledger") != ledger or ledger.get("tracked_body_counter_set_present") is not True
            or body_coverage_identity(ledger) != (BOUNDARY_COVERAGE if variant == "coupled" else ACK_COVERAGE)
            or not all(_count(audit.get("generation_" + phase + "_online_body_bytes")) for phase in ("prefill", "decode"))
            or sum(audit["generation_" + phase + "_online_body_bytes"] for phase in ("prefill", "decode"))
                != ledger["online_all_link_serialized_body_bytes"]):
        raise RuntimeError("raw audit and versioned body/phase ledger disagree")
    diagnostics = client.get("boundary_operation_diagnostics")
    if type(diagnostics) is not list or len(diagnostics) != int(variant == "coupled"):
        raise RuntimeError("boundary operation diagnostic coverage unavailable")
    for operation in ("lookup", "head"):
        observed = diagnostics[0].get(operation) if diagnostics else None
        if diagnostics and (type(observed) is not dict or set(observed) != {
                "calls", "rows", "upload_bytes", "download_bytes"}
                or not all(_count(value, positive=True) for value in observed.values())
                or observed["rows"] != required["private_boundary_" + ("lookup_rows" if operation == "lookup" else "head_calls")]
                or observed["calls"] > observed["rows"]
                or operation == "head" and observed["calls"] != 8):
            raise RuntimeError("boundary operation diagnostics do not match workload")
        for direction in ("upload", "download"):
            expected = observed[direction + "_bytes"] if observed is not None else 0
            key = f"private_boundary_{operation}_{direction}_bytes"
            if not _count(audit.get(key)) or audit[key] != expected:
                raise RuntimeError("operation diagnostic and audit bytes disagree")
    if variant == "coupled":
        for direction in ("upload", "download"):
            if sum(audit[f"private_boundary_{operation}_{direction}_bytes"] for operation in ("lookup", "head")) != sum(
                    audit[f"role_link.{role}.online_{direction}_bytes"] for role in ("boundary_a", "boundary_b")):
                raise RuntimeError("HTTP boundary operation bodies do not conserve directed links")
    cleanup = client.get("cleanup")
    names = {"cache", "histories", "message_histories", "_pending_response_store", "_transformer_states",
             "_transformer_conversations", "_decode_epoch_responses"}
    if (type(cleanup) is not dict or cleanup.get("known_private_containers_empty") is not True
            or type(cleanup.get("container_counts")) is not dict or set(cleanup["container_counts"]) != names
            or any(type(value) is not int or value != 0 for value in cleanup["container_counts"].values())
            or cleanup.get("owner_counters_after_close") != []
            or type(cleanup.get("boundary_transports_observed")) is not int
            or cleanup["boundary_transports_observed"] != int(variant == "coupled")
            or cleanup.get("boundary_transports_retired_and_owned_http_closed") is not True
            or result.get("provider_processes_reaped") is not True):
        raise RuntimeError("private/client/role cleanup evidence incomplete")
    expected_roles = {"inference", "preparation"} | ({"boundary_a", "boundary_b"} if variant == "coupled" else set())
    for key, checker in (("provider_birth_cpu_seconds", _duration),
                         ("provider_sampled_peak_rss_bytes", lambda value: _count(value, positive=True))):
        samples = result.get(key)
        if type(samples) is not dict or set(samples) != expected_roles or not all(checker(value) for value in samples.values()):
            raise RuntimeError("hosted role resource coverage incomplete")
    if (not _count(client.get("client_peak_rss_bytes"), positive=True)
            or not _duration(client.get("client_lifetime_cpu_seconds"))
            or not _duration(result.get("aggregate_cleanup_cpu_seconds"))):
        raise RuntimeError("client or complete-case resource measurement unavailable")
    cpu = result.get("cpu_accounting")
    if (type(cpu) is not dict or cpu.get("schema") != "pllm.private_boundary_case_cpu.v1"
            or not all(_duration(cpu.get(key)) for key in (
                "parent_all_threads_cpu_seconds", "reaped_children_cpu_seconds", "aggregate_cpu_seconds"))
            or cpu["aggregate_cpu_seconds"] != cpu["parent_all_threads_cpu_seconds"] + cpu["reaped_children_cpu_seconds"]
            or cpu["aggregate_cpu_seconds"] != result["aggregate_cleanup_cpu_seconds"]
            or cpu.get("sampled_client_and_provider_cpu_added_again") is not False
            or cpu.get("synchronized_all_role_request_cpu_seconds") is not None):
        raise RuntimeError("complete-case parent/WAN/reaped-child CPU conservation failed")
    # psutil clock ticks and rusage clocks can round differently. This allowance
    # checks sampled subsets only; it never changes measured headline CPU totals.
    sampled_children = client["client_lifetime_cpu_seconds"] + sum(result["provider_birth_cpu_seconds"].values())
    if sampled_children > cpu["reaped_children_cpu_seconds"] + 0.05 * (len(expected_roles) + 1):
        raise RuntimeError("sampled provider/client CPU exceeds charged reaped-child CPU")
    if (not _duration(row.get("client_request_cpu_seconds"))
            or not _duration(client.get("client_worker_entry_cpu_seconds"))
            or row["client_request_cpu_seconds"] > client["client_worker_entry_cpu_seconds"] + 1e-6
            or client["client_worker_entry_cpu_seconds"] > client["client_lifetime_cpu_seconds"] + 1e-6
            or not _duration(row.get("worker_entry_to_first_text_seconds"), positive=True)
            or row["worker_entry_to_first_text_seconds"] < row["first_text_seconds"]
            or not _duration(client.get("client_window_seconds"), positive=True)
            or row["worker_entry_to_first_text_seconds"] > client["client_window_seconds"]
            or not _duration(result.get("complete_case_seconds"), positive=True)
            or result["complete_case_seconds"] < client["client_window_seconds"]):
        raise RuntimeError("request versus client-lifetime/complete-case windows inconsistent")
    part = ledger["traffic_attribution"]
    if result.get("priority_metrics") != {"client_base_traffic_bytes": None, "client_output_traffic_bytes": None,
            "all_link_base_traffic_bytes": part.get("base_all_link_body_bytes"),
            "all_link_output_traffic_bytes": part.get("output_dependent_all_link_body_bytes"),
            "client_peak_rss_bytes": client["client_peak_rss_bytes"],
            "ttft_seconds": row["first_text_seconds"], "decode_tps": row["decode_tps"]}:
        raise RuntimeError("seven priority metrics disagree with raw evidence")
    if source_case != "source-present" or result.get("source_delivery") != source_delivery_accounting(source_case):
        raise RuntimeError("source delivery case cannot be certified by this local driver")
    if any(result.get(key) is not None for key in (
            "source_inclusive_all_link_application_bytes", "source_inclusive_all_link_tcp_stream_bytes")):
        raise RuntimeError("unmeasured source-inclusive totals must remain null")
    if not wan and result.get("wan") is not None:
        raise RuntimeError("unshaped local run cannot claim WAN/TCP measurements")
    if wan:
        from pllm.deployment import WanConditions
        network = result.get("wan")
        expected_edges = {f"{source}->{destination}" for source, destination in (
            ("client", "inference"), ("inference", "client"), ("client", "preparation"),
            ("preparation", "client"), ("preparation", "inference"))}
        if variant == "coupled":
            expected_edges |= {f"{source}->{destination}" for role in ("boundary_a", "boundary_b")
                              for source, destination in (("client", role), (role, "client"))}
        if (type(network) is not dict or network.get("enforced") is not True
                or network.get("conditions_digest") != WanConditions().digest
                or type(network.get("directed_stream_bytes")) is not dict
                or not expected_edges <= network["directed_stream_bytes"].keys()
                or not all(_count(value) for value in network["directed_stream_bytes"].values())
                or not _count(network.get("total_stream_bytes"))
                or network["total_stream_bytes"] != sum(network["directed_stream_bytes"].values())):
            raise RuntimeError("whole-graph native WAN coverage unavailable")
    return {**{key: row[key] for key in (
        "input_tokens", "output_tokens", "status", "state", "trajectory", "body_fingerprint")},
        "prepared_body_rows": rows,
        "phase_attribution_supported": audit["generation_attribution_supported_responses"]}


def verify_results(results, *, replicates=3, input_tokens=(150,), partial=False,
                   source_case="source-present", wan=False, source_lock_digest=None):
    order = replicate_order(replicates)
    if (type(input_tokens) not in (list, tuple) or not input_tokens
            or any(not _count(count, positive=True) or count > 4089 for count in input_tokens)
            or len(set(input_tokens)) != len(input_tokens) or type(results) is not list):
        raise RuntimeError("bounded unique cohort workloads required")
    expected = {(count, replicate, variant) for count in input_tokens for replicate, variant in order}
    references, peaks, role_peaks, seen, source_locks = {}, [], {}, set(), set()
    for result in results:
        if type(result) is not dict or not _count(result.get("replicate")):
            raise RuntimeError("invalid cohort cell")
        cell = (result.get("input_tokens"), result["replicate"], result.get("variant"))
        if cell not in expected or cell in seen:
            raise RuntimeError("duplicate or unexpected cohort cell")
        seen.add(cell)
        witness = validate_result(result, source_case=source_case, wan=wan, source_lock_digest=source_lock_digest)
        references.setdefault(result["input_tokens"], witness)
        if references[result["input_tokens"]] != witness:
            raise RuntimeError("exact full logits/KV/token/output/RNG parity failed")
        peak = result["client"]["client_peak_rss_bytes"]
        peaks.append(peak)
        source_locks.add(result["source_lock_digest"])
        role_peaks[result["variant"]] = max(role_peaks.get(result["variant"], 0), peak)
    complete = seen == expected
    if len(source_locks) > 1 or not partial and not complete:
        raise RuntimeError("source changed or required matched replicates/arms missing")
    return {"complete": complete, "exact_full_state_parity": complete,
        "checked_responses": len(results),
        "maximum_client_rss_across_replicates_bytes": max(peaks) if peaks else None,
        "maximum_client_rss_by_variant_bytes": role_peaks}


def main(argv=None):
    cohort_children_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--baseline-spec", type=Path)
    parser.add_argument("--tiny", "--smoke", action="store_true", help="Generated source transport control; not checkpoint evidence")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--input-tokens", type=int, default=150, help="Authoritative rendered prompt count, 1..4089")
    parser.add_argument("--short-prompt-file", type=Path)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--wan", action="store_true", help="Require native shared 100 Mbps down / 40 Mbps up graph")
    parser.add_argument("--source-case", choices=("source-present", "missing-workers"), default="source-present")
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(client_worker(json.load(sys.stdin))))
        return
    if (args.output is None or args.output.exists() or args.prompt_file is None
            or (args.baseline_spec is None) != args.tiny):
        parser.error("supply baseline spec or --tiny, prompt file and new output path")
    if not 1 <= args.input_tokens <= 4089 or args.short_prompt_file and args.input_tokens == 16:
        parser.error("input tokens must be 1..4089; short workload must be distinct")
    order = replicate_order(args.replicates)
    import pllm
    from pllm.metrics import BuildRef
    from pllm.model_loader import resolve_model
    if args.tiny:
        from pllm.profiles import MaskedLinearCpu
        from pllm.preparation import PreparedInventory
        base = pllm.Experiment("private-boundary-transport-control", MaskedLinearCpu(pllm.Model.tiny(),
            inventory=PreparedInventory(rows=1, refill="on-demand", allocation="demand")),
            pllm.Deployment.local(root=".pllm/private-boundary-smoke"),
            pllm.ExecutionBudget(requests=1, max_input_tokens=args.input_tokens, max_new_tokens=8))
    else:
        base = pllm.Experiment.from_spec(json.loads(args.baseline_spec.read_bytes()))
    workloads = [{"prompt": args.prompt_file.read_text(), "input_tokens": args.input_tokens}]
    if args.short_prompt_file:
        workloads.append({"prompt": args.short_prompt_file.read_text(), "input_tokens": 16})
    experiments = {work["input_tokens"]: matched_experiments(base, work["input_tokens"]) for work in workloads}
    admissions = {f"{count}/{variant}": admission(exp, args.wan)
                  for count, pairs in experiments.items() for variant, exp in pairs.items()}
    build = BuildRef.capture().to_spec()
    freeze_installed_implementation(build)
    salt = os.urandom(32)
    report = {"schema": "pllm.private_boundary_sdk_benchmark.v1", "complete": False,
        "driver_implementation": DRIVER_IMPLEMENTATION,
        "tiny": args.tiny, "scope": "tiny-sdk-transport-control" if args.tiny else "matched-sdk-response",
        "source_delivery_scope": "prepositioned source; distribution unmeasured",
        "network_scope": "native shared 100 Mbps down / 40 Mbps up" if args.wan else "unshaped local child processes",
        "sampler_fixture": {"seed": SAMPLER_SEED, "temperature": 0,
            "cryptographic_randomness_frozen": False, "diagnostic_cost_charged_equally": True},
        "build": build, "source_lock_digest": resolve_model(base.pipeline.model).source_lock_digest,
        "cohort_commitment": commitment(salt, workloads), "replicates": args.replicates,
        "planned_cells": len(workloads) * len(order), "memory_admission": admissions,
        "source_case": args.source_case, "wan_requested": args.wan, "results": [], "attempts": [],
        "priority_metrics": ["client_base_traffic", "client_output_traffic", "all_link_base_traffic",
            "all_link_output_traffic", "client_peak_rss", "ttft", "decode_tps"],
        "limits": ["source measurements absent remain null; prepositioned source is not free",
            "application bodies and TCP stream bytes separate; physical wire unknown",
            "local children share host/operator; no independent-operator claim",
             "provider RSS sampled after startup; startup peaks unknown",
             "request CPU is client-only; parent native WAN and providers are charged in complete-case CPU",
             "fresh client/processes do not imply cold OS source caches or newly distributed checkpoint artifacts",
            "exact trajectory hashing charged equally inside client CPU/RSS/wall windows",
            "missing-worker source delivery currently blocked; no fabricated bytes"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        cohort_children = resource.getrusage(resource.RUSAGE_CHILDREN)
        launcher_cpu = time.process_time()  # All threads, including native WAN; includes process import/bootstrap CPU.
        children_cpu = (cohort_children.ru_utime + cohort_children.ru_stime
                        - cohort_children_start.ru_utime - cohort_children_start.ru_stime)
        report["cohort_cpu_accounting"] = {
            "launcher_process_birth_to_report_sample_cpu_seconds": launcher_cpu,
            "children_reaped_since_driver_entry_cpu_seconds": children_cpu,
            "aggregate_cpu_seconds": launcher_cpu + children_cpu,
            "case_cpu_totals_are_subsets_not_added_again": True,
            "scope": "source/header preflight, case admission, SDK children, relays, verification and prior report writes; through this sample",
            "excluded": ["final report serialization/write and process teardown after this CPU sample",
                         "source distribution before this driver process"]}
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    if args.preflight_only:
        return
    if not all(row["admitted"] for row in admissions.values()):
        report["failure"] = {"stage": "memory_admission"}
        save()
        return
    with fresh_workers(Path(__file__), memory_bytes=max(row["required_bytes"] for row in admissions.values()), timeout=1800) as (run, guard, host):
        for work in workloads:
            for replicate, variant in order:
                try:
                    result = run_case(experiments[work["input_tokens"]][variant], work, variant,
                        replicate, salt, run, wan=args.wan, source_case=args.source_case,
                        source_lock_digest=report["source_lock_digest"], driver_implementation=report["driver_implementation"])
                    report["results"].append(result)
                    report["attempts"].append({"variant": variant, "replicate": replicate, "status": "executed"})
                    report["verification"] = verify_results(report["results"], replicates=args.replicates,
                        input_tokens=[work["input_tokens"] for work in workloads], partial=True,
                        source_case=args.source_case, wan=args.wan, source_lock_digest=report["source_lock_digest"])
                except BaseException as error:
                    # Do not persist exception messages: they may contain private frames.
                    report["failure"] = {"stage": "case_or_parity", "variant": variant,
                        "replicate": replicate, "error_type": type(error).__name__}
                    if isinstance(error, BenchmarkBlocked):
                        report["failure"]["blocker"] = error.code
                        if error.memory_admission is not None:
                            report["failure"]["memory_admission"] = error.memory_admission
                    save()
                    return
                finally:
                    report["swap"] = {"initial_bytes": host.swap_used,
                        "maximum_growth_bytes": guard.maximum_swap_growth_bytes,
                        "minimum_available_bytes": guard.minimum_available_bytes}
                    save()
    report["verification"] = verify_results(report["results"], replicates=args.replicates,
        input_tokens=[work["input_tokens"] for work in workloads], source_case=args.source_case,
        wan=args.wan, source_lock_digest=report["source_lock_digest"])
    report["complete"] = report["verification"]["complete"]
    save()


if __name__ == "__main__":
    installed_bootstrap(__file__)
    main()
