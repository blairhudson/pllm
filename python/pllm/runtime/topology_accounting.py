"""Bounded accounting of already-recorded prepared-protocol body counters.

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


__all__ = ["ACCOUNTING_SCHEMA", "client_owned_body_accounting", "prepared_body_accounting"]
