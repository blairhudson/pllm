"""One bounded same-client group through public SDK, with one accounting window.

Unlike dashboard request windows, concurrent branches must never sum individual
audit deltas or RSS peaks. This first slice supports one repeated private prompt,
native local roles, greedy sampling, natural EOS, and fresh client cache lifecycle.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import resource
import threading
import time

from pllm._benchmark_contract import Scenario, TargetUnavailable, digest
from pllm._benchmark_worker import peak_rss


CONCURRENCY_CONTRACT = "prepared_queue_inflight_before_admission_v1"


def group_memory_contract(admission, *, requests, max_output_tokens):
    """Extend shared full-KV estimate without aliasing existing owner capacity.

    Existing concurrency prices active execution/workspace/owner copies. Stored
    completed responses are a distinct lifetime: price one complete native state
    allowance per extra response even when active owners were already priced.
    Queued input/response Python bookkeeping uses the worker's bounded 16 MiB
    private-input envelope per caller plus boxed full-cap token arrays. This is
    conservative allocation admission, not an observed RSS or a text-size proof.
    """
    estimate = admission.get("estimate", {})
    required = admission.get("candidates", {}).get("native", {}).get("required_host_bytes")
    bound = estimate.get("concurrent_response_bound")
    components = estimate.get("components", {})
    state = components.get("client_state_work_bytes")
    live = components.get("client_live_tensor_bytes")
    input_bound = estimate.get("compiled_input_bound")
    output_bound = estimate.get("max_output_tokens")
    slack = estimate.get("allocation_slack_percent")
    if (any(type(value) is not int or value <= 0 for value in (
            required, bound, state, live, input_bound, output_bound))
            or slack != 25 or output_bound < max_output_tokens
            or state % bound or live % bound
            or not estimate.get("tensor_storage", "").startswith("full_kv_bounded_live_roots_v2")):
        raise TargetUnavailable("group_memory_bound_unavailable", status="blocked")
    # Baseline planner may price one caller while candidate prices eight. Do not
    # duplicate the candidate's pipeline concurrency. Queued preparation remains
    # single-active under the required runtime contract.
    unpriced_callers = max(0, requests - bound)
    queued_scratch = unpriced_callers * (live // bound)
    retained_states = (requests - 1) * (state // bound)
    metadata = requests * ((16 << 20) + 64 * (input_bound + output_bound))
    extra = queued_scratch + retained_states + metadata
    extra_with_slack = (extra * 125 + 99) // 100
    return {"scope": "shared_topology_plus_distinct_queued_and_retained_response_owners",
            "shared_topology_bytes": required, "pipeline_concurrent_response_bound": bound,
            "unpriced_caller_count": unpriced_callers, "queued_scratch_bytes": queued_scratch,
            "retained_response_state_bytes": retained_states, "caller_metadata_bytes": metadata,
            "additional_bytes_with_slack": extra_with_slack,
            "required_bytes": required + extra_with_slack + (128 << 20),
            "provider_checkpoints_duplicated": False}


def prefix_owners(client):
    """Keep owner handles until cleanup audit; inspect no keys or private arrays."""
    from .inflight_prefix import InflightPrefixOwner
    found = []
    for value in vars(client._core).values():
        candidates = value.values() if isinstance(value, dict) else (value,)
        for candidate in candidates:
            if isinstance(candidate, InflightPrefixOwner) and all(candidate is not owner for owner in found):
                found.append(candidate)
    return tuple(found)


def inspect_private_cleanup(client, owners=()):
    """Inspect only known container lengths; never serialize keys or payloads."""
    core = client._core
    names = ("cache", "histories", "message_histories", "_pending_response_store",
             "_transformer_states", "_transformer_conversations", "_decode_epoch_responses")
    counts = {}
    for name in names:
        value = getattr(core, name, None)
        if not isinstance(value, dict):
            raise TargetUnavailable("group_private_cleanup_unobservable", status="failed")
        counts[name] = len(value)
    if any(counts.values()):
        raise TargetUnavailable("group_private_cleanup_failed", status="failed")
    owner_counts = [owner.counters() for owner in owners]
    if any(row.get(name) != 0 for row in owner_counts for name in (
            "reserved_bytes", "requests", "nodes", "pending_nodes", "active_producers")):
        raise TargetUnavailable("group_private_owner_cleanup_failed", status="failed")
    return {"known_private_containers_empty": True, "container_counts": counts,
            "owner_counters_after_close": owner_counts}


def require_concurrency_contract(client):
    """Fail closed on old runtimes rather than mislabel serial/racy execution."""
    if getattr(getattr(client, "_core", None), "prepared_concurrency_contract", None) != CONCURRENCY_CONTRACT:
        raise TargetUnavailable("group_runtime_concurrency_contract_unavailable")


def run_group(client, prompt, *, requests, max_output_tokens, timeout_seconds, commitment):
    """Fully drain all branches, including failures, before returning text-free rows.

    No benchmark-only runtime patches. Barrier aligns submissions, not prefill
    execution; completed-cache hits remain valid and receive no invented sharing.
    Worker-process deadline is authoritative if a native call cannot be cancelled.
    """
    barrier = threading.Barrier(requests)

    def branch(index):
        barrier.wait(timeout=timeout_seconds)
        started = time.perf_counter()
        response = client.responses.create(input=prompt, max_output_tokens=max_output_tokens,
                                           temperature=0, store=True)
        usage = response.usage
        if (response.status not in {"completed", "incomplete"} or usage is None
                or (response.status == "incomplete" and (response.incomplete_details or {}).get(
                    "reason") != "max_output_tokens")
                or type(usage.output_tokens) is not int or not 0 <= usage.output_tokens <= max_output_tokens
                or type(usage.input_tokens) is not int or usage.input_tokens < 0):
            raise TargetUnavailable("group_response_incomplete", status="failed")
        return {"index": index, "status": "completed", "response_status": response.status,
                "tokens": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                           "authoritative": True},
                "elapsed_seconds": time.perf_counter() - started,
                "output_digest": commitment(response.output_text)}

    # Context exit joins every submitted call on success or failure. SDK owns
    # session abort/burn; no completed response is fabricated for an aborted call.
    with ThreadPoolExecutor(max_workers=requests, thread_name_prefix="pllm-response-group") as pool:
        futures = [pool.submit(branch, index) for index in range(requests)]
        return [future.result() for future in futures]


def execute_group(p, m, prompt):
    from pllm.configuration import Experiment
    from pllm.metrics import benchmark_memory
    from pllm.model_loader import expected_model_id, resolve_model
    from pllm.runtime.servers import serve_local
    from pllm._search_benchmark import _boundary
    from .benchmark_response_target import INFLIGHT_NONADDITIVE, audit_metrics
    from .topology_accounting import prepared_body_accounting

    scenario, requests = Scenario(**p["scenario"]), p["concurrent_requests"]
    declared = p["model"]
    experiment = Experiment.from_spec(declared["experiment"]).with_params(
        pipeline__profile=declared["profile"], deployment__root=str(m.directory / "roles"))
    if experiment.budget.requests < requests or experiment.budget.max_new_tokens < scenario.max_output_tokens:
        raise TargetUnavailable("scenario_exceeds_experiment_budget", status="blocked")
    # Reuse provider/checkpoint and in-flight owner bounds once. Add separate
    # queued caller and retained response lifetimes; never borrow cache capacity.
    admission = benchmark_memory(experiment, backend="native", enforced_wan=False,
                                 max_output_tokens=scenario.max_output_tokens,
                                 memory_budget_bytes=m.sampling.memory_bytes)
    m.details["topology_admission"] = admission
    if not admission["admitted"]:
        raise TargetUnavailable("topology_memory_limit", status="blocked")
    memory = group_memory_contract(admission, requests=requests, max_output_tokens=scenario.max_output_tokens)
    m.details["group_memory_contract"] = memory
    m.admit(memory["required_bytes"])
    geometry = {"requests": requests, "groups": 1, "prompt_relation": "identical",
                "submission": "thread_barrier", "cache": "fresh_client_then_close",
                "execution": "runtime_prepared_queue_concurrent_callers",
                "runtime_contract": CONCURRENCY_CONTRACT,
                "initialization": "public_capabilities_and_same_cap_sizing_without_prefill",
                "scenario": scenario.to_spec(), "temperature": 0, "store": True}

    def execute(_):
        client = None
        owners = ()
        closed = False
        started, cpu = time.perf_counter(), time.process_time()
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        child_cpu_before = children.ru_utime + children.ru_stime
        try:
            resolved = resolve_model(experiment.pipeline.model)
            source = resolved.source_lock_digest
            if source is None:
                raise TargetUnavailable("group_source_lock_unavailable")
            with serve_local(experiment, startup_timeout=m.sampling.timeout_seconds) as topology:
                client = topology.client(timeout=m.sampling.timeout_seconds,
                                         background_inventory_refill=False)
                try:
                    require_concurrency_contract(client)
                    # Initialize source/lookup/compiled state serially, not cached
                    # prefill. All setup stays inside the aggregate charged window.
                    client.runtime.capabilities()
                    client.prepared_rows_for_response(prompt, scenario.max_output_tokens, store=True)
                    before = client.privacy_audit.to_dict()
                    rows = run_group(client, prompt, requests=requests,
                        max_output_tokens=scenario.max_output_tokens,
                        timeout_seconds=m.sampling.timeout_seconds, commitment=m.commitment)
                finally:
                    owners = prefix_owners(client)
                    client.close()
                    closed = True
                after = client.privacy_audit.to_dict()
                cleanup = inspect_private_cleanup(client, owners)
                if any(after.get(key, 0) < value for key, value in before.items()
                       if key not in INFLIGHT_NONADDITIVE):
                    raise TargetUnavailable("group_audit_counter_regressed", status="failed")
                if any(after.get(key) != 0 for key in (
                        "plaintext_prompt_bytes_sent", "plaintext_token_ids_sent",
                        "preparation_requests_during_online")):
                    raise TargetUnavailable("group_privacy_checks_failed", status="failed")
                return {"schema_version": "pllm.response_group.v1", "geometry": geometry,
                        "source_lock_digest": source, "runs": rows,
                        "privacy": after, "before": before,
                        "private_cleanup": cleanup,
                        "request_delta": {key: value - before.get(key, 0) for key, value in after.items()}}
        finally:
            # Even a topology failure leaves no owned client alive.
            if client is not None and not closed:
                client.close()
            if client is not None:
                m.details["group_lifecycle_audit"] = client.privacy_audit.to_dict()
                try:
                    m.details["group_private_cleanup"] = inspect_private_cleanup(client, owners)
                except TargetUnavailable as failure:
                    m.details["group_private_cleanup"] = {"known_private_containers_empty": False,
                                                          "failure_code": failure.code}
            children = resource.getrusage(resource.RUSAGE_CHILDREN)
            child_cpu = children.ru_utime + children.ru_stime - child_cpu_before
            m.details["group_lifecycle"] = {"elapsed_seconds": time.perf_counter() - started,
                "client_cpu_seconds": time.process_time() - cpu,
                "reaped_role_children_cpu_seconds": child_cpu,
                "aggregate_cpu_seconds": time.process_time() - cpu + child_cpu,
                "single_client_peak_rss_bytes": peak_rss(), "includes_cleanup": True}

    try:
        report, = m.run(execute)
    except BaseException:
        # Failed/cancelled work remains charged, without completed samples or
        # output commitments. Never serialize exception messages/private values.
        lifecycle = m.details.get("group_lifecycle", {})
        m.metric("group.elapsed", lifecycle.get("elapsed_seconds"),
                 window="group_attempt_through_cleanup", scope="failed_sdk_group")
        m.metric("group.client_cpu", lifecycle.get("client_cpu_seconds"), kind="cpu",
                 window="group_attempt_through_cleanup", scope="failed_sdk_group")
        m.metric("group.aggregate_cpu", lifecycle.get("aggregate_cpu_seconds"), kind="cpu",
                 window="group_attempt_through_cleanup", scope="client_and_reaped_role_children",
                 owners=["client", "reaped_role_children"])
        audit = m.details.get("group_lifecycle_audit", {})
        audit_metrics([audit], m, window="client_birth_through_failed_cleanup")
        failed_account = prepared_body_accounting({"privacy": audit}, initial_preparation=True)
        m.details["group_body_accounting"] = failed_account
        m.metric("network.covered_body_bytes", failed_account["all_link_serialized_body_bytes"],
                 kind="bytes", role="all-links", statistic="sum",
                 window="client_birth_through_failed_cleanup",
                 scope=f"covered_application_bodies:{failed_account['body_counter_coverage']}")
        raise
    m.details.update(report=report, target_peak_rss_bytes=peak_rss(),
                     target_cpu_from_birth_seconds=time.process_time())
    audit_metrics([report["privacy"]], m, window="client_birth_through_close")
    account = prepared_body_accounting({"privacy": report["privacy"]}, initial_preparation=True)
    m.details["group_body_accounting"] = account
    m.metric("network.covered_body_bytes", account["all_link_serialized_body_bytes"], kind="bytes",
             role="all-links", statistic="sum", window="client_birth_through_close",
             scope=f"covered_application_bodies:{account['body_counter_coverage']}")
    m.metric("client.peak_rss", m.details["target_peak_rss_bytes"], kind="memory",
             window="process_birth_through_group_cleanup", scope="single_process_lifetime_high_water")
    m.metric("group.elapsed", m.details["timings"]["wall_seconds"][0],
             window="source_resolution_through_role_cleanup", scope="complete_sdk_group")
    m.metric("group.client_cpu", m.details["timings"]["cpu_seconds"][0], kind="cpu",
             window="source_resolution_through_role_cleanup", scope="single_client_process")
    m.metric("group.aggregate_cpu", m.details["group_lifecycle"]["aggregate_cpu_seconds"], kind="cpu",
             window="source_resolution_through_role_cleanup", scope="client_and_reaped_role_children",
             owners=["client", "reaped_role_children"])
    boundary, runs = _boundary(experiment), report["runs"]
    m.comparison.update(source_digest=report["source_lock_digest"],
        numeric_digest=digest(boundary["numeric"]), privacy_digest=digest(boundary["privacy"]),
        state_digest=digest(geometry), workload_digest=digest({"geometry": geometry,
            "tokens": [row["tokens"] for row in runs], "input": m.commitment(prompt)}),
        output_digest=digest([row["output_digest"] for row in runs]),
        measurement_digest=digest({"schema": report["schema_version"], "geometry": geometry,
                                   "body_counter_coverage": account["body_counter_coverage"],
                                   "scope": "setup_group_cleanup_single_client_process"}))
    m.fixture = {"kind": "generated_checkpoint" if experiment.pipeline.model.kind == "tiny" else "checkpoint",
                 "digest": report["source_lock_digest"], "metadata": {"cached_checkpoint": True}}
    m.oracle = {"kind": "ordinary_sdk_terminal_usage", "outcome": "passed",
                "checked_values": sum(row["tokens"]["output_tokens"] for row in runs)}
    m.details.update(model={"id": expected_model_id(experiment.pipeline.model), "checkpoint_digest": None,
                           "source_lock_digest": report["source_lock_digest"]},
                     component_ids=sorted(c.component for c in experiment.pipeline.components.values()))
    m.limitations += ["One repeated private prompt; no speculative prefix splitting, WAN or research adapters.",
        "Concurrent SDK callers use runtime prepared execution queue; no parallel online-kernel throughput claim.",
        "Barrier aligns calls, not prefill; completed-cache hits do not establish in-flight sharing.",
        "Aggregate CPU includes client and reaped role children through cleanup; no per-role CPU attribution.",
        "One client process high-water only; provider RSS and physical wire unmeasured; no summed RSS peaks.",
        "Covered audit bodies include setup and cleanup; missing counters stay unknown.",
        "Natural EOS preserved; output commitments and usage are not full-logit/KV or quality oracles."]
