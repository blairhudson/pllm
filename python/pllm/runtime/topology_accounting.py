"""Bounded accounting of already-recorded topology application-body counters.

This is deliberately *not* a wire meter: control frames, HTTP/TLS framing,
transport retries, inference-to-preparation acks, and setup before the separately
captured initial inventory are not represented by the client privacy counters.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

ACCOUNTING_SCHEMA = "pllm.topology_body_accounting.v1"

# Count every serialized body once, from the sending side of one edge. Other
# audit counters (masked_online_*, upload_bytes, and server metrics) overlap.
_PREPARED_BODY_COUNTERS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("client", "preparation", "offline", (
        "session_authorization_upload_bytes", "preparation_upload_bytes",
    )),
    ("preparation", "client", "offline", (
        "session_authorization_download_bytes", "preparation_download_bytes",
    )),
    ("preparation", "inference", "offline", ("correction_push_bytes",)),
    ("inference", "client", "cold", ("bundle_network_bytes",)),
    ("client", "inference", "online", ("inference_upload_bytes",)),
    ("inference", "client", "online", ("inference_download_bytes",)),
)
_REQUIRED_COUNTERS = frozenset(
    key for _, _, _, fields in _PREPARED_BODY_COUNTERS for key in fields
)
_ROLES = ("client", "preparation", "inference")
_CLIENT_ONLY_ZERO_COUNTERS = (
    "inference_upload_bytes", "inference_download_bytes",
    "preparation_upload_bytes", "preparation_download_bytes",
    "correction_push_bytes", "plaintext_prompt_bytes_sent", "plaintext_token_ids_sent",
)


def client_owned_body_accounting(record: Mapping[str, Any]) -> dict[str, Any]:
    """Account only inference-graph links for an all-client role placement."""
    privacy = record.get("privacy")
    counters: dict[str, Any] = privacy if type(privacy) is dict else {}
    complete = all(
        type(counters.get(key)) is int and counters[key] == 0
        for key in _CLIENT_ONLY_ZERO_COUNTERS
    )
    processes = record.get("processes")
    raw_client = processes.get("client") if type(processes) is dict else None
    raw_cpu = raw_client.get("cpu_seconds") if type(raw_client) is dict else None
    cpu = (
        float(raw_cpu)
        if isinstance(raw_cpu, (int, float)) and not isinstance(raw_cpu, bool)
        and math.isfinite(raw_cpu) and raw_cpu >= 0
        else None
    )
    return {
        "schema": ACCOUNTING_SCHEMA,
        "scope": "client-owned semantic execution; no inference-provider graph links",
        "tracked_body_counter_set_present": complete,
        "body_bytes_by_edge": [] if complete else None,
        "client_serialized_body_bytes": 0 if complete else None,
        "all_link_serialized_body_bytes": 0 if complete else None,
        "online_client_serialized_body_bytes": 0 if complete else None,
        "online_all_link_serialized_body_bytes": 0 if complete else None,
        "run_window_cpu_seconds_by_role": {"client": cpu} if cpu is not None else None,
        "aggregate_run_window_cpu_seconds": cpu,
        "total_wire_bytes": None,
        "full_response_compute_cap_checked": False,
        "unmeasured": [
            "client model import and cold checkpoint distribution",
            "client peak memory, energy, and disk activity",
            "application/dashboard HTTP traffic outside the inference role graph",
        ],
    }


def prepared_body_accounting(
    record: Mapping[str, Any], *, initial_preparation: bool = False,
) -> dict[str, Any]:
    """Return the covered body/CPU ledger, or explicitly unavailable fields.

    No coarser ``upload_bytes`` or OTLP receive-side metric is added: those
    would double-count bodies already charged to the source of the edge.
    """
    privacy = record.get("privacy")
    counters: dict[str, Any] = privacy if type(privacy) is dict else {}
    complete = (
        _REQUIRED_COUNTERS <= counters.keys()
        and all(type(counters[key]) is int and counters[key] >= 0 for key in _REQUIRED_COUNTERS)
    )
    links: list[dict[str, str | int]] | None = None
    client_bytes: int | None = None
    all_bytes: int | None = None
    online_client_bytes: int | None = None
    online_all_bytes: int | None = None
    if complete:
        links = [
            {"source": source, "destination": destination, "phase": phase,
             "serialized_body_bytes": sum(counters[field] for field in fields)}
            for source, destination, phase, fields in _PREPARED_BODY_COUNTERS
        ]
        client_bytes = sum(
            int(item["serialized_body_bytes"]) for item in links
            if "client" in (item["source"], item["destination"])
        )
        all_bytes = sum(int(item["serialized_body_bytes"]) for item in links)
        online_client_bytes = sum(
            int(item["serialized_body_bytes"]) for item in links
            if item["phase"] == "online" and "client" in (item["source"], item["destination"])
        )
        online_all_bytes = sum(
            int(item["serialized_body_bytes"]) for item in links if item["phase"] == "online"
        )

    processes = record.get("processes")
    cpu: dict[str, float] | None = None
    if type(processes) is dict and all(
        type(processes.get(role)) is dict
        and type(processes[role].get("cpu_seconds")) in (int, float)
        and math.isfinite(processes[role]["cpu_seconds"])
        and processes[role]["cpu_seconds"] >= 0
        for role in _ROLES
    ):
        cpu = {role: float(processes[role]["cpu_seconds"]) for role in _ROLES}
    return {
        "schema": ACCOUNTING_SCHEMA,
        "scope": (
            "initial inventory and client-bundle serialized bodies; startup CPU unmeasured"
            if initial_preparation else
            "recorded prepared-protocol serialized bodies and run-window CPU samples only"
        ),
        "tracked_body_counter_set_present": complete,
        "body_bytes_by_edge": links,
        "client_serialized_body_bytes": client_bytes,
        "all_link_serialized_body_bytes": all_bytes,
        "online_client_serialized_body_bytes": online_client_bytes,
        "online_all_link_serialized_body_bytes": online_all_bytes,
        "run_window_cpu_seconds_by_role": cpu,
        "aggregate_run_window_cpu_seconds": sum(cpu.values()) if cpu is not None else None,
        "total_wire_bytes": None,
        "full_response_compute_cap_checked": False,
        "unmeasured": [
            "HTTP/TLS/WebSocket framing and control traffic",
            "inference-to-preparation correction acknowledgements",
            "duplicate or failed transport attempts not represented by client audit counters",
            (
                "model import and checkpoint distribution before initial inventory"
                if initial_preparation else
                "process/model startup, client bundle transfer, and offline inventory prepared before the run window"
            ),
            "GPU work, uninstrumented client work, and cold checkpoint distribution",
        ],
    }


def two_worker_body_accounting(record: Mapping[str, Any]) -> dict[str, Any]:
    """Count observed worker setup, online, teardown and bundle bodies once."""
    privacy = record.get("privacy")
    counters: dict[str, Any] = privacy if type(privacy) is dict else {}
    roles = ("worker_a", "worker_b")
    fields = (
        f"role_link.{role}.{phase}_{direction}_bytes"
        for role in roles
        for phase in ("setup", "online", "teardown")
        for direction in ("upload", "download")
    )
    required = frozenset(fields) | {"bundle_network_bytes"}
    complete = required <= counters.keys() and all(
        type(counters[key]) is int and counters[key] >= 0 for key in required
    )
    edges: list[dict[str, str | int]] | None = None
    if complete:
        edges = []
        for role in roles:
            for phase in ("setup", "online", "teardown"):
                for direction in ("upload", "download"):
                    source, destination = (
                        ("client", role) if direction == "upload" else (role, "client")
                    )
                    edges.append({
                        "source": source, "destination": destination,
                        "phase": phase,
                        "serialized_body_bytes": counters[
                            f"role_link.{role}.{phase}_{direction}_bytes"
                        ],
                    })
        edges.append({
            "source": "worker_a", "destination": "client", "phase": "bundle",
            "serialized_body_bytes": counters["bundle_network_bytes"],
        })
    processes = record.get("processes")
    cpu: dict[str, float] | None = None
    if type(processes) is dict and all(
        type(processes.get(role)) is dict
        and type(processes[role].get("cpu_seconds")) in (int, float)
        and math.isfinite(processes[role]["cpu_seconds"])
        and processes[role]["cpu_seconds"] >= 0
        for role in ("client", *roles)
    ):
        cpu = {
            role: float(processes[role]["cpu_seconds"])
            for role in ("client", *roles)
        }
    return {
        "schema": ACCOUNTING_SCHEMA,
        "scope": "authenticated two-worker HTTP application bodies and run-window CPU",
        "tracked_body_counter_set_present": complete,
        "body_bytes_by_edge": edges,
        "client_serialized_body_bytes": (
            sum(int(edge["serialized_body_bytes"]) for edge in edges) if edges is not None
            else None
        ),
        "all_link_serialized_body_bytes": (
            sum(int(edge["serialized_body_bytes"]) for edge in edges) if edges is not None
            else None
        ),
        "online_client_serialized_body_bytes": (
            sum(int(edge["serialized_body_bytes"]) for edge in edges
                if edge["phase"] == "online") if edges is not None else None
        ),
        "online_all_link_serialized_body_bytes": (
            sum(int(edge["serialized_body_bytes"]) for edge in edges
                if edge["phase"] == "online") if edges is not None else None
        ),
        "run_window_cpu_seconds_by_role": cpu,
        "aggregate_run_window_cpu_seconds": sum(cpu.values()) if cpu is not None else None,
        "total_wire_bytes": None,
        "full_response_compute_cap_checked": False,
        "unmeasured": [
            "HTTP headers, TLS and transport framing",
            "model descriptor and authenticated monitoring request bodies",
            "worker model import and cold checkpoint distribution",
            "peak client memory, energy, and disk activity",
        ],
    }


def cold_process_cpu_accounting(
    readings: Mapping[str, Any] | None,
    *,
    roles: tuple[str, ...],
    first_measurement_is_cold: bool,
) -> dict[str, Any]:
    """Account cumulative CPU from role birth/dashboard start through first response.

    A missing role or a replaced PID invalidates the aggregate. No sampled GPU
    time, upstream source distribution, or later warm-run CPU is inferred here.
    """
    expected = frozenset(roles)

    def checked(key: str) -> dict[str, float] | None:
        value = readings.get(key) if readings is not None else None
        if type(value) is not dict or set(value) != expected:
            return None
        result: dict[str, float] = {}
        for role in roles:
            sample = value[role]
            if type(sample) not in (int, float) or not math.isfinite(sample) or sample < 0:
                return None
            result[role] = float(sample)
        return result

    startup = checked("startup")
    complete = checked("first_response") if first_measurement_is_cold else None
    if startup is not None and complete is not None and any(
        complete[role] < startup[role] for role in roles
    ):
        complete = None
    return {
        "schema": "pllm.topology_process_cpu.v1",
        "scope": (
            "dashboard/client CPU from benchmark startup and role CPU from process birth; "
            "through the first cold measured response"
        ),
        "startup_cpu_seconds_by_role": startup,
        "aggregate_startup_cpu_seconds": sum(startup.values()) if startup is not None else None,
        "cold_first_response_cpu_seconds_by_role": complete,
        "aggregate_cold_first_response_cpu_seconds": (
            sum(complete.values()) if complete is not None else None
        ),
        "full_response_compute_cap_checked": False,
        "unmeasured": [
            "source resolution before dashboard startup, if any",
            "accelerator compute, energy, and CPU after the first response",
            "HTTP/TLS payload and control bytes outside the serialized-body ledger",
        ],
    }


__all__ = [
    "ACCOUNTING_SCHEMA", "client_owned_body_accounting", "cold_process_cpu_accounting",
    "prepared_body_accounting",
    "two_worker_body_accounting",
]
