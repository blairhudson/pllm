"""Conservative base/generation attribution of measured application bodies.

Material row payloads are measured by the preparation protocol, not inferred
from total bytes / rows. Mixed envelopes, padding and controls remain residual.
No linear next-token cost, physical wire measurement, or private data is inferred.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCHEMA = "pllm.benchmark_traffic_decomposition.v1"
SOURCE = "sdk_audit_phase_and_prepared_row_payload_counters_v1"
_BASE = ("setup_and_delivery", "prefill_online", "prefill_material")
_OUTPUT = ("decode_online", "decode_material")
_ATTRIBUTION_COUNTERS = frozenset({
    "bundle_network_bytes", "session_authorization_upload_bytes", "session_authorization_download_bytes",
    "preparation_upload_bytes", "preparation_download_bytes", "correction_push_bytes",
    "correction_ack_bytes", "correction_ack_accounted_push_bytes",
    "inference_upload_bytes", "inference_download_bytes",
    "preparation_attributed_body_bytes", "preparation_decode_material_body_bytes",
    "preparation_prefill_material_body_bytes", "generation_attribution_supported_responses",
    "generation_completed_responses", "generation_output_tokens", "generation_decode_output_tokens",
    "generation_executed_decode_steps", "generation_requested_output_tokens",
    "generation_prefill_online_body_bytes", "generation_decode_online_body_bytes",
    "decode_epochs_started", "inflight_producer_registrations", "inflight_joins",
    "public_prefix_artifact_local_bytes",
    *(f"{role}_control_{direction}_bytes" for role in ("inference", "preparation")
      for direction in ("upload", "download")),
    *("preparation_" + phase + "_rows_issued" for phase in ("prefill", "decode", "padding")),
    *("prepared_stage_rows_" + name for name in ("issued", "reserved", "claimed", "burned", "discarded")),
})


def _count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _sum(values: list[int | None]) -> int | None:
    return sum(values) if values and all(value is not None for value in values) else None


def _ledger_total(ledger: Mapping[str, Any]) -> int | None:
    """Reject malformed directed ledgers before using either totals or phases."""
    total = _count(ledger.get("all_link_serialized_body_bytes"))
    edges = ledger.get("body_bytes_by_edge")
    if (ledger.get("tracked_body_counter_set_present") is not True or total is None
            or not isinstance(edges, list) or len(edges) > 32
            or any(not isinstance(edge, Mapping) or _count(edge.get("serialized_body_bytes")) is None
                   or any(type(edge.get(name)) is not str or not 0 < len(edge[name]) <= 128
                          for name in ("source", "destination", "phase")) for edge in edges)
            or sum(edge["serialized_body_bytes"] for edge in edges) != total):
        return None
    return total


def _prepared_edges_match(counters: Mapping[str, Any], ledger: Mapping[str, Any]) -> bool:
    expected = {
        ("client", "preparation", "offline"): ("session_authorization_upload_bytes", "preparation_upload_bytes"),
        ("preparation", "client", "offline"): ("session_authorization_download_bytes", "preparation_download_bytes"),
        ("preparation", "inference", "offline"): ("correction_push_bytes",),
        ("inference", "preparation", "offline"): ("correction_ack_bytes",),
        ("inference", "client", "cold"): ("bundle_network_bytes",),
        ("client", "inference", "online"): ("inference_upload_bytes",),
        ("inference", "client", "online"): ("inference_download_bytes",),
        **{(source, destination, "control"): (f"{role}_control_{direction}_bytes",)
           for role in ("inference", "preparation")
           for direction, source, destination in (("upload", "client", role), ("download", role, "client"))},
    }
    edges = ledger.get("body_bytes_by_edge", [])
    measured = {(edge.get("source"), edge.get("destination"), edge.get("phase")): edge["serialized_body_bytes"] for edge in edges}
    return len(measured) == len(edges) == len(expected) and all(
        _sum([_count(counters.get(name)) for name in fields]) == measured.get(edge)
        for edge, fields in expected.items())


def body_decomposition(
    total_body_bytes: int | None,
    *,
    base_body_bytes: int = 0,
    output_dependent_body_bytes: int = 0,
    unused_output_material_body_bytes: int | None = None,
    attribution_present: bool = True,
    headline_eligible: bool = True,
) -> dict[str, Any]:
    """Reconcile directly attributed, disjoint byte amounts with one measured total.

    Callers own attribution evidence and lifecycle alignment. Unused output
    material is a *subset* of output-dependent bytes, never an extra charge.
    Missing measurements and impossible partitions cannot yield headline rates.
    """
    total = _count(total_body_bytes)
    base = _count(base_body_bytes)
    output = _count(output_dependent_body_bytes)
    unused = _count(unused_output_material_body_bytes)
    valid = (attribution_present is True and base is not None and output is not None
             and (unused_output_material_body_bytes is None or unused is not None)
             and (unused is None or unused <= output)
             and (total is None or base + output <= total))
    if not valid:
        base = output = 0
        unused = None
    residual = None if total is None else total - base - output
    complete = valid and total is not None and residual == 0 and headline_eligible is True
    return {
        "schema": SCHEMA,
        "status": "Attributed" if complete else "Partial" if valid and base + output else "Unattributed",
        "total_all_link_body_bytes": total,
        "base_all_link_body_bytes": base if complete else None,
        "output_dependent_all_link_body_bytes": output if complete else None,
        "known_base_all_link_body_bytes": base,
        "known_output_dependent_all_link_body_bytes": output,
        "unattributed_all_link_body_bytes": residual,
        "unused_output_material_body_bytes": unused,
        "conservation_checked": total is not None,
        "attribution_valid": valid,
        "headline_eligible": headline_eligible is True,
    }


def prepared_window_attribution(counters: Mapping[str, Any], ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Attribute one audit delta; unavailable counters never become measured zero."""
    present = all(key in counters for key in (
        "preparation_attributed_body_bytes", "preparation_decode_material_body_bytes",
        "preparation_prefill_material_body_bytes", "generation_attribution_supported_responses",
    ))
    components: dict[str, int] = {}
    reasons: list[str] = []
    total = _ledger_total(ledger)
    from .topology_accounting import ACK_COVERAGE, body_coverage_identity
    present = (present and body_coverage_identity(ledger) == ACK_COVERAGE and total is not None
               and _prepared_edges_match(counters, ledger))
    row_conserved = False
    if present:
        setup = _sum([_count(counters.get(key)) for key in (
            "bundle_network_bytes", "session_authorization_upload_bytes",
            "session_authorization_download_bytes",
        )])
        if setup is not None:
            components["setup_and_delivery"] = setup
        unsupported = any(key in counters and _count(counters[key]) != 0 for key in (
            "decode_epochs_started", "inflight_producer_registrations", "inflight_joins",
            "public_prefix_artifact_local_bytes"))
        if unsupported:
            reasons.append("epochs, in-flight sharing or public capsules lack exact attribution support")
        online = _count(ledger.get("online_all_link_serialized_body_bytes"))
        prefill = _count(counters.get("generation_prefill_online_body_bytes"))
        decode = _count(counters.get("generation_decode_online_body_bytes"))
        generated = _count(counters.get("generation_output_tokens"))
        executed = _count(counters.get("generation_executed_decode_steps"))
        cap = _count(counters.get("generation_requested_output_tokens"))
        directed_online = _sum([_count(counters.get(key)) for key in (
            "inference_upload_bytes", "inference_download_bytes")])
        if (not unsupported and _count(counters.get("generation_attribution_supported_responses")) == 1
                and _count(counters.get("generation_completed_responses")) == 1
                and generated is not None and generated > 0
                and _count(counters.get("generation_decode_output_tokens")) == generated - 1
                and executed is not None and generated - 1 <= executed <= generated
                and cap is not None and generated <= cap
                and prefill is not None and decode is not None and prefill + decode == online == directed_online):
            components.update(prefill_online=prefill, decode_online=decode)
        elif online == directed_online == 0:
            components.update(prefill_online=0, decode_online=0)
        else:
            reasons.append("online phase attribution unavailable or unreconciled")
        preparation = _sum([_count(counters.get(key)) for key in (
            "preparation_upload_bytes", "preparation_download_bytes", "correction_push_bytes",
        )])
        prefill_material = _count(counters.get("preparation_prefill_material_body_bytes"))
        decode_material = _count(counters.get("preparation_decode_material_body_bytes"))
        rows = _sum([_count(counters.get(key)) for key in (
            "preparation_prefill_rows_issued", "preparation_decode_rows_issued",
            "preparation_padding_rows_issued",
        )])
        inventory = {name: _count(counters.get("prepared_stage_rows_" + name)) for name in
                     ("issued", "reserved", "claimed", "burned", "discarded")}
        # Optional historical row deltas may be absent; they cannot prove full
        # classification, but do not erase independently metered payload floors.
        row_keys_present = any("prepared_stage_rows_" + name in counters for name in ("reserved", "claimed", "burned"))
        row_conserved = (all(inventory[name] is not None for name in ("reserved", "claimed", "burned"))
                         and inventory["reserved"] == inventory["claimed"] + inventory["burned"])
        if (not unsupported and preparation is not None and preparation == _count(counters.get("preparation_attributed_body_bytes"))
                and rows is not None and rows == _count(counters.get("prepared_stage_rows_issued"))
                and prefill_material is not None and decode_material is not None
                and prefill_material + decode_material <= preparation
                and (not row_keys_present or row_conserved)):
            components.update(prefill_material=prefill_material, decode_material=decode_material)
        else:
            reasons.append("prepared row payload attribution unavailable or row totals mismatch")
    else:
        reasons.append("versioned phase/material counters not recorded")
    result = body_decomposition(total,
        base_body_bytes=sum(components.get(key, 0) for key in _BASE),
        output_dependent_body_bytes=sum(components.get(key, 0) for key in _OUTPUT),
        attribution_present=present, headline_eligible=row_conserved)
    result.update(source=SOURCE, components_body_bytes=components,
        # Startup has no run record. Retain bounded raw evidence so readers can
        # reconcile its partition too; missing/invalid counters stay unknown.
        audit_counters={key: _count(counters[key]) for key in sorted(_ATTRIBUTION_COUNTERS)
                        if key in counters},
        residual_scope="mixed preparation envelopes, padding, shared packed phase-boundary bytes, correction acknowledgements, lifecycle controls and unclassified phases",
        unknown_reasons=reasons,
        prepared_rows={name: _count(counters.get("preparation_" + name + "_rows_issued"))
                       for name in ("prefill", "decode", "padding")},
        inventory_rows={name: _count(counters.get("prepared_stage_rows_" + name))
                         for name in ("issued", "reserved", "claimed", "burned", "discarded")},
        reserved_equals_claimed_plus_burned=row_conserved,
        executed_decode_steps=_count(counters.get("generation_executed_decode_steps")),
        generated_output_tokens=_count(counters.get("generation_output_tokens")),
        output_allowance=_count(counters.get("generation_requested_output_tokens")))
    return result


def benchmark_traffic_decomposition(report: Mapping[str, Any]) -> dict[str, Any]:
    """Cold first request includes startup once; repeats retain their own windows.

    Discarded warmups are never silently charged to repeat-request base. Their
    windows remain separately visible. No unused-byte estimate from row ratios.
    """
    if report.get("schema_version") != "pllm.loopback_benchmark.v1":
        raise ValueError("traffic decomposition requires a loopback benchmark report")
    accounting = report.get("topology_accounting") or {}
    if not isinstance(accounting, Mapping):
        raise ValueError("invalid traffic accounting")
    records = report.get("runs", [])
    warmups = report.get("warmup_runs", [])
    ledgers = accounting.get("runs", [])
    warm_ledgers = accounting.get("warmups", [])
    if any(type(rows) is not list for rows in (records, warmups, ledgers, warm_ledgers)):
        raise ValueError("invalid traffic windows")
    if len(records) + len(warmups) > 1024:
        raise ValueError("traffic report exceeds bounded run count")
    configuration = report.get("configuration") or {}
    if not isinstance(configuration, Mapping) or any(
        not isinstance(row, Mapping) or not isinstance(row.get("tokens", {}), Mapping)
        for row in records + warmups
    ):
        raise ValueError("invalid traffic records")

    def window(record: Mapping[str, Any], ledger: Any) -> dict[str, Any]:
        ledger = ledger if isinstance(ledger, Mapping) else {}
        attribution = ledger.get("traffic_attribution")
        attribution = attribution if isinstance(attribution, Mapping) and attribution.get("schema") == SCHEMA else {}
        privacy = record.get("privacy")
        if not record and not isinstance(privacy, Mapping):
            privacy = attribution.get("audit_counters")
        if isinstance(privacy, Mapping):
            # Raw disjoint audit and directed ledger must still agree. Never
            # trust a stale cached partition after counters or edges changed.
            attribution = prepared_window_attribution(privacy, ledger)
        else:
            # A cached partition alone is not measurement evidence, regardless
            # of its claimed source, validity flag or exact-looking totals.
            attribution = {}
        measured = _ledger_total(ledger)
        tokens = record.get("tokens", {})
        output = (_count(tokens.get("output_tokens")) if record.get("status") == "completed"
                  and tokens.get("authoritative") is True else None)
        observed = attribution.get("generated_output_tokens")
        accepted = (attribution.get("total_all_link_body_bytes") == measured
                     and attribution.get("attribution_valid") is True
                     and (output is None or observed is None or observed == output)
                     and (not record or record.get("status") == "completed" and tokens.get("authoritative") is True))
        result = body_decomposition(measured,
            base_body_bytes=attribution.get("known_base_all_link_body_bytes", 0),
            output_dependent_body_bytes=attribution.get("known_output_dependent_all_link_body_bytes", 0),
            unused_output_material_body_bytes=attribution.get("unused_output_material_body_bytes"),
            attribution_present=accepted,
            headline_eligible=(attribution.get("headline_eligible") is True
                               and (not record or output is not None and observed == output)))
        result.update(input_tokens=_count(tokens.get("input_tokens")), generated_output_tokens=output,
            output_allowance=attribution.get("output_allowance") or _count(configuration.get("max_output_tokens")),
            cold=record.get("cold"),
            base_all_link_mb_per_request=(result["base_all_link_body_bytes"] / 1_000_000
                 if result["base_all_link_body_bytes"] is not None and output else None),
            incremental_all_link_mb_per_generated_output=(result["output_dependent_all_link_body_bytes"] / output / 1_000_000
                if result["output_dependent_all_link_body_bytes"] is not None and output else None),
            known_output_dependent_all_link_mb_per_generated_output=(result["known_output_dependent_all_link_body_bytes"] / output / 1_000_000
                if accepted and output else None),
            attribution=attribution or None)
        return result

    aligned = len(records) == len(ledgers) and len(warmups) == len(warm_ledgers)
    runs = [window(row, ledgers[i] if aligned else None) for i, row in enumerate(records)]
    warm = [window(row, warm_ledgers[i] if aligned else None) for i, row in enumerate(warmups)]
    startup = window({}, accounting.get("startup"))
    first = (warm or runs or [None])[0]
    cold = None
    from .topology_accounting import accounting_coverage_identity
    coverage = accounting_coverage_identity(accounting)
    if first is not None:
        total = _sum([startup["total_all_link_body_bytes"], first["total_all_link_body_bytes"]])
        cold = body_decomposition(total,
            base_body_bytes=startup["known_base_all_link_body_bytes"] + first["known_base_all_link_body_bytes"],
            output_dependent_body_bytes=startup["known_output_dependent_all_link_body_bytes"] + first["known_output_dependent_all_link_body_bytes"],
            attribution_present=coverage is not None and startup["attribution_valid"] and first["attribution_valid"],
            headline_eligible=startup["headline_eligible"] and first["headline_eligible"])
        output = first["generated_output_tokens"]
        cold.update(input_tokens=first["input_tokens"], generated_output_tokens=output,
            output_allowance=first["output_allowance"],
             base_all_link_mb_per_request=cold["base_all_link_body_bytes"] / 1_000_000 if cold["base_all_link_body_bytes"] is not None and output else None,
            incremental_all_link_mb_per_generated_output=cold["output_dependent_all_link_body_bytes"] / output / 1_000_000
                if cold["output_dependent_all_link_body_bytes"] is not None and output else None)
    windows = [startup, *warm, *runs]
    lifecycle_total = (_sum([row["total_all_link_body_bytes"] for row in windows])
                       if aligned and coverage is not None else None)
    lifecycle = body_decomposition(lifecycle_total,
        base_body_bytes=sum(row["known_base_all_link_body_bytes"] for row in windows),
        output_dependent_body_bytes=sum(row["known_output_dependent_all_link_body_bytes"] for row in windows),
        attribution_present=aligned and all(row["attribution_valid"] for row in windows),
        headline_eligible=all(row["headline_eligible"] for row in windows))
    run_base = _sum([row["base_all_link_body_bytes"] for row in runs])
    run_output = _sum([row["output_dependent_all_link_body_bytes"] for row in runs])
    generated = _sum([row["generated_output_tokens"] for row in runs])
    return {
        "schema": SCHEMA, "source": SOURCE,
        "scope": "measured directed all-link serialized application bodies; not TCP stream or physical wire",
        "byte_unit": "decimal MB (1000000 bytes)",
        "tcp_stream_bytes": None, "physical_wire_bytes": None,
        "body_counter_coverage": coverage,
        "law": "total = known base + known output-dependent + unattributed; unused material is included, not added",
        "rate_scope": "workload-specific output-dependent average, not constant next-token marginal cost",
        "packed_boundary_convention": "only whole-cohort serialized payload bytes classified; shared phase-boundary bytes stay residual",
        "lifecycle_scope": "startup charged once to first warmup or first measured request; repeats exclude startup",
        "cache_state": {key: configuration.get(key) for key in ("inventory_policy", "prefill_cache_mode", "prefill_cache_bound_tokens")},
        "unmeasured": ["HTTP/TLS/transport framing and retries outside audit coverage",
                       "checkpoint distribution and pre-positioned artifacts outside measured links",
                       "exact unused output material bytes unless directly measured"],
        "windows_aligned": aligned, "startup": startup, "cold_first_request": cold,
        "warmups": warm, "runs": runs,
        "repeat_requests": runs if warm else runs[1:],
        "covered_lifecycle": lifecycle,
        "summary": {
            "scope": "measured run windows only; cold startup and discarded warmups reported separately",
            "measured_requests": len(runs), "generated_output_tokens": generated,
            "base_all_link_mb_per_request": run_base / len(runs) / 1_000_000
                 if run_base is not None and runs and generated and coverage is not None else None,
            "incremental_all_link_mb_per_generated_output": run_output / generated / 1_000_000
                if run_output is not None and generated and coverage is not None else None,
        },
    }
