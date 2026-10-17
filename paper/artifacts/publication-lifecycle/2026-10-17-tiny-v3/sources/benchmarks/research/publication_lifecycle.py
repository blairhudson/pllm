"""Bounded, replayable phase/role accounting for publication experiments.

This module aggregates observations. It does not estimate missing measurements,
assign semantic phases to existing reports, or turn sampled RSS into an enforced
memory bound. Live adapters must supply a complete synchronized observation.
"""
from __future__ import annotations

import math
from typing import Any

LEGACY_PHASES = ("public_setup", "private_preparation", "prefill", "decode", "cleanup")
PHASES = (*LEGACY_PHASES, "request_setup", "unclassified")
MAX_SAMPLES = 32_768
MAX_PROCESSES = 32
MAX_LINKS = 128
MAX_PHASE_EVENTS = 256
TRACE_ADMISSION_BYTES = 64 << 20
MAX_SAMPLE_BYTES = 8192
MAX_TRACE_FILE_BYTES = MAX_SAMPLES * MAX_SAMPLE_BYTES
SCHEMA = "pllm.publication_lifecycle.v2"
CLEANUP_FLAGS = ("dashboard_joined", "client_close_returned", "client_workers_joined", "memory_guard_joined",
    "ordinary_sampler_joined", "observer_joined", "topology_closed", "relay_closed", "relay_joined",
    "relay_empty", "relay_loop_closed", "relay_listeners_closed", "no_live_children")


def recorder_capacity(timeout_seconds: float) -> dict:
    """Admit disk capacity for the entire timeout, including boundary samples.

    Rows stream to a private file; neither the trace nor its replay lives in the
    measured client heap. Replay belongs to the controller after child exit.
    """
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 86400):
        raise InvalidObservation("recorder_timeout")
    grace, boundaries = 30, MAX_PHASE_EVENTS + 32
    interval = max(50_000_000, math.ceil((timeout_seconds + grace) * 1e9 / (MAX_SAMPLES - boundaries)))
    rows = math.ceil((timeout_seconds + grace) * 1e9 / interval) + boundaries
    return {"timeout_seconds": timeout_seconds, "interval_ns": interval,
        "shutdown_grace_seconds": grace, "boundary_sample_allowance": boundaries,
        "maximum_samples": rows, "maximum_sample_bytes": MAX_SAMPLE_BYTES,
        "maximum_trace_disk_bytes": rows * MAX_SAMPLE_BYTES,
        "client_allocation_admission_bytes": TRACE_ADMISSION_BYTES,
        "storage": "private_streamed_jsonl", "replay_owner": "controller_after_worker_exit"}


class InvalidObservation(ValueError):
    """The observations cannot support a complete lifecycle result."""


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise InvalidObservation(name)
    return value


def _name(value: Any, name: str) -> str:
    if (type(value) is not str or not 1 <= len(value) <= 64
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value)):
        raise InvalidObservation(name)
    return value


def _aggregate(document: dict, *, schema="pllm.publication_lifecycle.v1", phase_names=LEGACY_PHASES) -> dict:
    """Recompute phase and lifecycle peaks from the retained observations.

    Process keys identify one process incarnation, not reusable operating-system
    PIDs. Each process belongs to exactly one measurement owner and one host.
    Co-resident roles must use a joint owner. Every sample includes every declared
    process; absence before birth or after reaping is explicit zero residency.
    """
    if document.get("schema") != schema:
        raise InvalidObservation("schema")
    owners = document.get("owners")
    samples = document.get("samples")
    events = document.get("events")
    if not isinstance(owners, dict) or not 1 <= len(owners) <= MAX_PROCESSES:
        raise InvalidObservation("owner_count")
    if not isinstance(samples, list) or not 2 <= len(samples) <= MAX_SAMPLES:
        raise InvalidObservation("sample_count")
    if not isinstance(events, list) or not 2 <= len(events) <= MAX_PHASE_EVENTS:
        raise InvalidObservation("phase_event_count")
    if document.get("complete") is not True or document.get("dropped_samples") != 0:
        raise InvalidObservation("incomplete_trace")
    interval = _count(document.get("nominal_sample_interval_ns"), "sample_interval")
    if interval == 0:
        raise InvalidObservation("sample_interval")
    roles: dict[str, list[str]] = {}
    hosts: dict[str, list[str]] = {}
    clients, providers = [], []
    for process, owner in owners.items():
        _name(process, "process_identity")
        if not isinstance(owner, dict):
            raise InvalidObservation("owner")
        role = _name(owner.get("role"), "role")
        host = _name(owner.get("host"), "host")
        if type(owner.get("client")) is not bool:
            raise InvalidObservation("client_ownership")
        roles.setdefault(role, []).append(process)
        hosts.setdefault(host, []).append(process)
        (clients if owner["client"] else providers).append(process)
    if not clients:
        raise InvalidObservation("client_missing")
    for processes in roles.values():
        if len({owners[p]["client"] for p in processes}) != 1:
            raise InvalidObservation("mixed_client_provider_role")

    previous_end, previous_links = -1, {}
    max_gap, max_skew = 0, 0
    for index, sample in enumerate(samples):
        start = _count(sample.get("start_ns"), "sample_start")
        end = _count(sample.get("end_ns"), "sample_end")
        if end < start or start < previous_end:
            raise InvalidObservation("sample_clock_order")
        max_skew = max(max_skew, end - start)
        if index:
            max_gap = max(max_gap, start - previous_end)
        previous_end = end
        rss = sample.get("rss_bytes")
        if not isinstance(rss, dict) or set(rss) != set(owners):
            raise InvalidObservation("incomplete_rss_vector")
        for value in rss.values():
            _count(value, "rss_bytes")
        alive = sample.get("alive")
        if not isinstance(alive, list) or len(alive) != len(set(alive)) or not set(alive) <= set(owners):
            raise InvalidObservation("process_lifecycle")
        if any(rss[p] != 0 for p in owners if p not in alive):
            raise InvalidObservation("inactive_process_has_rss")
        links = sample.get("directed_stream_bytes")
        if not isinstance(links, dict) or len(links) > MAX_LINKS:
            raise InvalidObservation("link_count")
        for link, size in links.items():
            endpoints = link.split("->")
            if len(endpoints) != 2 or any(x not in roles for x in endpoints) or endpoints[0] == endpoints[1]:
                raise InvalidObservation("directed_link")
            _count(size, "stream_bytes")
        if any(links.get(k, 0) < v for k, v in previous_links.items()):
            raise InvalidObservation("stream_counter_regression")
        previous_links = links
    if any(samples[0]["directed_stream_bytes"].values()):
        raise InvalidObservation("missing_initial_traffic")
    if any(p in samples[-1]["alive"] for p in providers):
        raise InvalidObservation("providers_not_reaped")

    def memory(indices):
        rows = [samples[i]["rss_bytes"] for i in indices]
        if not rows:
            return None

        def group(processes):
            peaks = {p: max(row[p] for row in rows) for p in processes}
            aggregate = [sum(row[p] for p in processes) for row in rows]
            winner = max(range(len(rows)), key=aggregate.__getitem__)
            return {"instances": len(processes), "individual_peak_bytes": peaks,
                "largest_individual_peak_bytes": max(peaks.values(), default=0),
                "simultaneous_peak_bytes": aggregate[winner],
                "peak_sample": indices[winner],
                "vector_at_peak": {p: rows[winner][p] for p in processes}}

        return {"roles": {role: group(p) for role, p in roles.items()},
            "hosts": {host: group(p) for host, p in hosts.items()},
            "client": group(clients), "non_client": group(providers),
            "whole_deployment": group(list(owners))}

    phases = {phase: {"elapsed_seconds": 0.0, "directed_stream_bytes": {},
                      "intervals": [], "sample_indices": set()} for phase in phase_names}
    previous_index = -1
    for index, event in enumerate(events):
        at = _count(event.get("sample"), "event_sample")
        if at <= previous_index or at >= len(samples):
            raise InvalidObservation("event_order")
        previous_index = at
        phase = event.get("phase")
        if index == len(events) - 1:
            if phase != "end" or at != len(samples) - 1:
                raise InvalidObservation("missing_final_boundary")
            break
        if phase not in phase_names or (index == 0 and (phase != "public_setup" or at != 0)):
            raise InvalidObservation("phase_identity")
        until = events[index + 1].get("sample")
        if type(until) is not int or not at < until < len(samples):
            raise InvalidObservation("phase_interval")
        before, after = samples[at], samples[until]
        row = phases[phase]
        row["elapsed_seconds"] += (after["end_ns"] - before["end_ns"]) / 1e9
        row["intervals"].append([at, until])
        # A shared boundary sample belongs to both adjoining memory windows.
        # Traffic uses disjoint counter differences and is never counted twice.
        row["sample_indices"].update(range(at, until + 1))
        for link, size in after["directed_stream_bytes"].items():
            delta = size - before["directed_stream_bytes"].get(link, 0)
            row["directed_stream_bytes"][link] = row["directed_stream_bytes"].get(link, 0) + delta
    if events[-2].get("phase") != "cleanup":
        raise InvalidObservation("missing_cleanup")
    for row in phases.values():
        row["memory"] = memory(sorted(row.pop("sample_indices")))
        row["all_link_stream_bytes"] = sum(row["directed_stream_bytes"].values())
        client_roles = {owners[p]["role"] for p in clients}
        row["client_upload_bytes"] = sum(size for link, size in row["directed_stream_bytes"].items()
            if link.split("->")[0] in client_roles and link.split("->")[1] not in client_roles)
        row["client_download_bytes"] = sum(size for link, size in row["directed_stream_bytes"].items()
            if link.split("->")[1] in client_roles and link.split("->")[0] not in client_roles)
    total = sum(samples[-1]["directed_stream_bytes"].values())
    elapsed = (samples[-1]["end_ns"] - samples[0]["end_ns"]) / 1e9
    if (sum(row["all_link_stream_bytes"] for row in phases.values()) != total
            or not math.isclose(sum(row["elapsed_seconds"] for row in phases.values()), elapsed,
                                rel_tol=1e-12, abs_tol=1e-9)):
        raise InvalidObservation("lifecycle_conservation")
    return {"schema": "pllm.publication_lifecycle_summary.v1", "phases": phases,
        "lifecycle_memory": memory(list(range(len(samples)))),
        "all_link_stream_bytes": total, "elapsed_seconds": elapsed,
        "sampling": {"samples": len(samples), "nominal_interval_ns": interval,
            "maximum_between_sample_gap_ns": max_gap, "maximum_within_sample_skew_ns": max_skew},
        "scope": {"memory": "sampled_process_rss_synchronous_vectors",
            "traffic": "rate_admitted_tcp_payload", "boundary_memory_samples_shared": True,
            "device_memory": None, "provisioned_capacity": None,
             "rss_may_count_shared_pages_multiple_times": True}}


def network_counters(snapshot: dict, roles: list[str]) -> dict:
    """Validate independent sender/receiver counters for distinct local parties."""
    links = snapshot.get("directed_stream_bytes")
    parties = snapshot.get("parties")
    if (not isinstance(links, dict) or len(links) > MAX_LINKS
            or not isinstance(parties, dict) or set(parties) != set(roles)):
        raise InvalidObservation("network_owners")
    uploads, downloads = dict.fromkeys(roles, 0), dict.fromkeys(roles, 0)
    for edge, value in links.items():
        pair = edge.split("->")
        if len(pair) != 2 or pair[0] not in uploads or pair[1] not in downloads or pair[0] == pair[1]:
            raise InvalidObservation("directed_link")
        value = _count(value, "stream_bytes")
        uploads[pair[0]] += value
        downloads[pair[1]] += value
    total = _count(snapshot.get("total_stream_bytes"), "total_stream_bytes")
    if total != sum(links.values()):
        raise InvalidObservation("network_total_conservation")
    for role in roles:
        for direction, expected in (("upload", uploads[role]), ("download", downloads[role])):
            if _count(parties[role].get(direction), "party_bytes") != expected:
                raise InvalidObservation("party_conservation")
    return snapshot


def counters_le(before: dict, after: dict) -> bool:
    """Counter readings at different instants need bounds, never equality."""
    return all(after["directed_stream_bytes"].get(key, 0) >= value
               for key, value in before["directed_stream_bytes"].items())


def clock_tolerance_ns(clock: dict, *seconds: float) -> int:
    resolution = _count(clock.get("resolution_ns"), "clock_resolution")
    if clock.get("name") != "monotonic" or not 0 < resolution <= 1_000_000_000:
        raise InvalidObservation("observation_clock")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in seconds):
        raise InvalidObservation("clock_seconds")
    # Two independently read clock endpoints plus float serialization/arithmetic.
    return 2 * resolution + math.ceil(sum(math.ulp(float(value)) * 1e9 for value in seconds))


def relay_instant(document: dict, elapsed: float) -> tuple[int, int]:
    origin = document["relay_started_monotonic_seconds"]
    tolerance = clock_tolerance_ns(document["clock"], origin, elapsed, origin + elapsed)
    return round((origin + elapsed) * 1e9), tolerance


def request_durations(request: dict) -> dict:
    """Named SDK windows. TTFT begins online, not at request admission."""
    emission_summary(request)
    names = ("admitted_ns", "preparation_started_ns", "preparation_finished_ns", "online_started_ns",
             "first_output_ns", "last_output_ns", "completed_ns")
    previous = 0
    for name in names:
        value = request[name]
        if value is None and name in {"first_output_ns", "last_output_ns"}:
            continue
        value = _count(value, "request_endpoint")
        if value < previous:
            raise InvalidObservation("request_endpoint_order")
        previous = value
    def delta(start, end):
        a, b = request[start], request[end]
        return None if a is None or b is None else (b - a) / 1e9
    return {"full_seconds": delta("admitted_ns", "completed_ns"),
        "preparation_seconds": delta("preparation_started_ns", "preparation_finished_ns"),
        "transition_seconds": delta("preparation_finished_ns", "online_started_ns"),
        "online_seconds": delta("online_started_ns", "completed_ns"),
        "ttft_seconds": delta("online_started_ns", "first_output_ns"),
        "generation_seconds": delta("first_output_ns", "last_output_ns")}


def duration_matches(actual, expected, clock):
    if actual is None or expected is None:
        return actual is expected
    tolerance = clock_tolerance_ns(clock, actual, expected) / 1e9
    return abs(actual - expected) <= tolerance


def reconcile_network_snapshots(document, snapshots):
    """Bracket independent ordinary snapshots by the retained read intervals."""
    rows, roles = document["samples"], document["expected_roles"]
    previous, previous_elapsed = rows[0]["network"], -1
    observations = []
    for raw in snapshots:
        if (raw["conditions"] != document["network_conditions"]
                or raw["role_parties"] != {role: role for role in roles}):
            raise InvalidObservation("ordinary_network_identity")
        counters = network_counters({"directed_stream_bytes": raw["directed_stream_bytes"],
            "total_stream_bytes": raw["total_stream_bytes"],
            "parties": {role: {direction: raw["parties"][role][direction]["admitted_bytes"]
                for direction in ("upload", "download")} for role in roles}}, roles)
        at, tolerance = relay_instant(document, raw["elapsed_seconds"])
        if (raw["elapsed_seconds"] < previous_elapsed or not counters_le(previous, counters)
                or not counters_le(counters, rows[-1]["network"])):
            raise InvalidObservation("ordinary_network_chronology")
        if not rows[0]["start_ns"] - tolerance <= at <= rows[-1]["end_ns"] + tolerance:
            raise InvalidObservation("ordinary_network_window")
        lower, upper = rows[0]["network"], rows[-1]["network"]
        for row in rows:
            if row["end_ns"] < at - tolerance:
                lower = row["network"]
            if row["start_ns"] > at + tolerance:
                upper = row["network"]
                break
        if not counters_le(lower, counters) or not counters_le(counters, upper):
            raise InvalidObservation("ordinary_network_sample_bracket")
        observations.append((at, tolerance))
        previous, previous_elapsed = counters, raw["elapsed_seconds"]
    if len(observations) != 3:
        raise InvalidObservation("ordinary_network_window_count")
    request = document["request"]
    if (any(at > request["admitted_ns"] + tolerance for at, tolerance in observations[:2])
            or observations[-1][0] < request["completed_ns"] - observations[-1][1]):
        raise InvalidObservation("ordinary_network_request_bounds")
    return {"comparison": "monotone counters bracketed by measured read intervals",
        "maximum_clock_tolerance_ns": max(tolerance for _, tolerance in observations)}


def emission_summary(request: dict) -> dict:
    """Use the dashboard's exact monotonic event window, never phase endpoints."""
    outputs = _count(request.get("output_tokens"), "output_tokens")
    inputs = _count(request.get("input_tokens"), "input_tokens")
    start = _count(request.get("admitted_ns"), "request_admission")
    end = _count(request.get("completed_ns"), "request_completion")
    first, last = request.get("first_output_ns"), request.get("last_output_ns")
    if end < start or outputs > 64 or inputs == 0:
        raise InvalidObservation("request_window")
    if outputs:
        if (type(first) is not int or type(last) is not int
                or not start <= first <= last <= end
                or (outputs == 1 and first != last) or (outputs > 1 and first == last)):
            raise InvalidObservation("emission_window")
    elif first is not None or last is not None:
        raise InvalidObservation("zero_output_window")
    intervals = max(0, outputs - 1)
    elapsed = (last - first) / 1e9 if intervals else None
    return {"input_tokens": inputs, "output_tokens": outputs, "decode_intervals": intervals,
        "decode_seconds": elapsed, "decode_tokens_per_second": intervals / elapsed if elapsed else None,
        "request_to_first_output_seconds": (first - start) / 1e9 if outputs else None,
        "request_seconds": (end - start) / 1e9,
        "scope": "ordinary_dashboard_monotonic_emission_events_not_semantic_phase_time"}


def summarize(document: dict) -> dict:
    """Structural observation replay only; archive qualification uses load_trace.

    Historical inline v1 keeps its frozen semantics, without v2 archive guarantees.
    """
    if document.get("schema") == "pllm.publication_lifecycle.v1":
        return _aggregate(document)
    if document.get("schema") != SCHEMA:
        raise InvalidObservation("schema")
    try:
        return _summarize_v2(document)
    except InvalidObservation:
        raise
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        raise InvalidObservation("invalid_lifecycle_evidence") from error


def _summarize_v2(document):
    owners, samples, events = (document[k] for k in ("owners", "samples", "events"))
    expected = document["expected_roles"]
    if (not isinstance(expected, list) or len(expected) != len(set(expected))
            or "client" not in expected or len(expected) > MAX_PROCESSES
            or set(owners) != set(expected) or any(o["role"] != p for p, o in owners.items())):
        raise InvalidObservation("expected_role_graph")
    capacity = recorder_capacity(document["capacity"]["timeout_seconds"])
    if document["capacity"] != capacity or len(samples) > capacity["maximum_samples"]:
        raise InvalidObservation("recorder_capacity")
    if document["nominal_sample_interval_ns"] != capacity["interval_ns"]:
        raise InvalidObservation("sample_interval")
    tolerance = clock_tolerance_ns(document["clock"])
    if (samples[-1]["end_ns"] - samples[0]["start_ns"] >
            (capacity["timeout_seconds"] + capacity["shutdown_grace_seconds"]) * 1e9 + tolerance):
        raise InvalidObservation("trace_duration_exceeds_capacity")
    if document.get("errors") or document.get("target_succeeded") is not True:
        raise InvalidObservation("lifecycle_failed")
    health = document["cleanup_health"]
    if any(health.get(key) is not True for key in CLEANUP_FLAGS) or health.get("relay_error") is not None:
        raise InvalidObservation("cleanup_health")
    if len(document["request_workers"]) != 1:
        raise InvalidObservation("request_worker_retirement")
    worker, = document["request_workers"]
    for field in ("entered_ns", "returned_ns", "joined_ns"):
        _count(worker.get(field), "request_worker_clock")
    if (worker.get("outcome") != "returned" or not _count(worker.get("ident"), "request_worker_ident")
            or not _count(worker.get("native_id"), "request_worker_ident")
            or not samples[0]["start_ns"] <= worker["entered_ns"] <= worker["returned_ns"]
                <= worker["joined_ns"] <= samples[-1]["end_ns"]
            or not document["request"]["admitted_ns"] <= worker["entered_ns"]
                <= document["request"]["completed_ns"] <= worker["returned_ns"]):
        raise InvalidObservation("request_worker_retirement")
    identities = set()
    for key, owner in owners.items():
        pid = _count(owner.get("pid"), "process_pid")
        born = _count(owner.get("create_time_ns"), "process_birth")
        if not pid or not born or (pid, born) in identities:
            raise InvalidObservation("process_incarnation")
        identities.add((pid, born))
        if owner["host"] != "local_host" or owner["client"] != (key == "client"):
            raise InvalidObservation("client_ownership")
        birth = _count(owner.get("first_sample"), "process_first_sample")
        if birth >= len(samples):
            raise InvalidObservation("process_first_sample")
        present = [i for i, row in enumerate(samples) if key in row["alive"]]
        if not present or present[0] != birth or present != list(range(present[0], present[-1] + 1)):
            raise InvalidObservation("process_lifecycle")
        if not any(samples[i]["rss_bytes"][key] > 0 for i in present):
            raise InvalidObservation("process_never_observed")
        if key == "client":
            if birth != 0 or present[-1] != len(samples) - 1:
                raise InvalidObservation("client_lifecycle")
        else:
            if (birth == 0 or type(owner.get("returncode")) is not int or present[-1] == len(samples) - 1
                    or type(owner.get("reaped_ns")) is not int
                    or not samples[present[-1]]["start_ns"] <= owner["reaped_ns"] <= samples[-1]["end_ns"]):
                raise InvalidObservation("providers_not_reaped")
            receipt = owner["child_receipt"]
            if (receipt["pid"] != pid or receipt["create_time_ns"] != born
                    or receipt["ppid"] != owners["client"]["pid"] or receipt["role"] != key
                    or receipt["nonce"] != document["binding"]["nonce"]
                    or receipt["runtime"] != document["runtime_provenance"]
                    or receipt["source_identity"] != document["binding"]["source_identity"]
                    or receipt["command_digest"] != owner["command_digest"]):
                raise InvalidObservation("child_provenance")
    swap_min, swap_growth, previous_elapsed = None, 0, None
    for row in samples:
        network_counters(row["network"], expected)
        if row["network"]["directed_stream_bytes"] != row["directed_stream_bytes"]:
            raise InvalidObservation("network_snapshot_binding")
        swap = _count(row.get("host_swap_used_bytes"), "host_swap_sample")
        swap_min = swap if swap_min is None else min(swap_min, swap)
        swap_growth = max(swap_growth, swap - swap_min)
        elapsed = row["network_elapsed_seconds"]
        if elapsed is None:
            if previous_elapsed is not None or row["network"]["total_stream_bytes"]:
                raise InvalidObservation("relay_clock_missing")
        else:
            at, tolerance = relay_instant(document, elapsed)
            if (not row["start_ns"] - tolerance <= at <= row["end_ns"] + tolerance
                    or previous_elapsed is not None and elapsed < previous_elapsed):
                raise InvalidObservation("relay_clock_window")
            previous_elapsed = elapsed
    if swap_growth:
        raise InvalidObservation("host_swap_grew_in_observed_window")
    if document["final_network"] != samples[-1]["network"]:
        raise InvalidObservation("final_network_binding")
    semantics = document["semantic_work"]
    emission = emission_summary(document["request"])
    request_durations(document["request"])
    outputs = emission["output_tokens"]
    eos = semantics["terminal_eos"]
    if (any(type(v) is not int or v < 0 for k, v in semantics.items() if k != "terminal_eos")
            or type(eos) is not bool or semantics["prefill_forwards"] != 1
            or semantics["selections"] != outputs + int(eos)
            or semantics["decode_forwards"] != semantics["selections"] - 1
            or semantics["text_events"] != outputs or semantics["preparation_calls"] < 1
            or document["terminal_received"] is not True):
        raise InvalidObservation("semantic_work")
    phases = {e["phase"] for e in events}
    if not {"public_setup", "private_preparation", "request_setup", "prefill", "cleanup", "end"} <= phases:
        raise InvalidObservation("missing_required_phase")
    if ("decode" in phases) != (semantics["decode_forwards"] > 0):
        raise InvalidObservation("decode_phase_coverage")
    transitions = {"public_setup": {"private_preparation", "request_setup"},
        "private_preparation": {"public_setup", "request_setup"},
        "request_setup": {"private_preparation", "prefill"},
        "prefill": {"unclassified", "cleanup"}, "unclassified": {"decode", "cleanup"},
        "decode": {"unclassified", "cleanup"}, "cleanup": {"end"}}
    if any(b["phase"] not in transitions.get(a["phase"], set()) for a, b in zip(events, events[1:])):
        raise InvalidObservation("semantic_phase_order")
    for phase, field in (("prefill", "prefill_forwards"), ("decode", "decode_forwards"),
                         ("private_preparation", "preparation_calls")):
        if sum(e["phase"] == phase for e in events) != semantics[field]:
            raise InvalidObservation("semantic_phase_counts")
    if not samples[0]["end_ns"] <= document["request"]["admitted_ns"] <= document["request"]["completed_ns"] <= samples[-1]["end_ns"]:
        raise InvalidObservation("request_outside_trace")
    result = _aggregate(document, schema=SCHEMA, phase_names=PHASES)
    result["schema"] = "pllm.publication_lifecycle_summary.v2"
    result["sampled_window_memory"] = result.pop("lifecycle_memory")
    for name, row in result["phases"].items():
        row["status"] = "observed" if row["intervals"] else "not_entered"
        if not row["intervals"] and name not in {"decode", "unclassified"}:
            raise InvalidObservation("unknown_phase")
    result["emission"] = emission
    result["semantic_work"] = semantics
    result["maximum_observed_host_swap_growth_bytes"] = swap_growth
    result["scope"].update(memory="sampled_process_rss_vectors_with_reported_read_skew",
        qualification="observation structure only; archive receipts and ordinary envelope require load_trace",
        window="observer_start_through_post_serialization_atexit_sample",
        excludes="process startup before observer; final receipt serialization and later interpreter teardown",
        sampling_can_miss_peaks=True, whole_process_high_water_is_separate=True,
        supported_graph="one native client with two prepared providers and optionally two boundary providers; one cold response",
        child_census="spawn-bound known providers plus sampled undeclared-descendant checks during serving; not an OS fork audit",
        provenance="local consistency receipts, not remote worker authentication; dependencies use ordinary parent build inspection",
        phase_boundaries="synchronous counter/RSS snapshot completion; observer overhead included",
        provider_lifetime_high_water=None,
        network_excluded=["telemetry", "checkpoint distribution", "IP/TCP headers, ACKs, retransmissions"],
        provider_profile="resource_unnormalized_per_party_100_down_40_up_local_emulation")
    result["sampling"]["capacity"] = capacity
    return result
