"""Reclassify locked archived prepared bodies by locality; never start providers.

Default reproduction consumes the compact directed ledger retained in the screen
artifact. --archive-dir additionally verifies original raw benchmark reports.
All counts are covered serialized application bodies, never complete wire bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/evidence/preparation-placement-screen-2026-10-01.json"
LOCKS = {
    "prepared-stage-attribution-qwen25-2026-09-29.json": "037e969be7c240a6ee61828d39c6627226cfece0efa23db6ae7623a8c92acf26",
    "incremental-network-qwen25-2026-10-01.json": "fad309506f2cc6d537b0778ca7d29c3ef4fb7a9a70ab8df66b99d114da586082",
}
# Full raw report hashes bind extracted directed edges to their original samples.
RAW_LOCKS = {
    "qwen-incremental-32-warm.json": "871d6d7c2aa3d5cc0f337e2aa73e73ebff10e7b82510816b59e99d2ae82665d8",
    "qwen-incremental-32-cold.json": "aa1c05efa40628c3b458ff39a36d35a641639dcd4e54b04d8edfb006960b8011",
}
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
FINGERPRINT = "5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974"
SOURCE_LOCK = "880f80ec61274d3c80e2a0c1336394b9e955d53ff98436a426a680a460a592e7"
RETAINED_SHA256 = "fdfd268636005b210ea8718a6d6e88837e03a71e0d0b721520bc24725d573a2e"
NAMES = ("prepared-request-sized", "client-attention")
PLACEMENTS = (
    "remote-preparation",
    "trusted-lan-appliance",
    "client-host-loopback",
    "client-host-ipc-counterfactual",
)
EDGES = {
    ("client", "preparation", "offline"),
    ("preparation", "client", "offline"),
    ("preparation", "inference", "offline"),
    ("inference", "client", "cold"),
    ("client", "inference", "online"),
    ("inference", "client", "online"),
}
ROLE_SUFFIXES = {
    "self_attn.qkv_proj": "qkv_projection",
    "self_attn.o_proj": "attention_output",
    "mlp.gate_up_proj": "mlp_gate_up",
    "mlp.down_proj": "mlp_down",
}
# Public Qwen geometry; products independently reconcile archived declared MACs.
GEOMETRY = {
    "qkv_projection": (896, 1152),
    "attention_output": (896, 896),
    "mlp_gate_up": (896, 9728),
    "mlp_down": (4864, 896),
}


def locked_json(path: Path, expected: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"source lock mismatch: {path.name}")
    return json.loads(raw)


def checked_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    for edge in edges:
        key = (edge["source"], edge["destination"], edge["phase"])
        amount = edge["serialized_body_bytes"]
        if key not in EDGES or key in seen or type(amount) is not int or amount < 0:
            raise ValueError("invalid, duplicate or negative directed body edge")
        seen.add(key)
    if seen != EDGES:
        raise ValueError("missing directed body edge")
    return edges


def place(edges: list[dict[str, Any]], placement: str) -> dict[str, Any]:
    """Preserve every logical body; optionally separate deliberately absent IPC."""
    if placement not in PLACEMENTS:
        raise ValueError("unknown preparation placement")
    totals: dict[str, int] = dict.fromkeys(("lan", "wan", "loopback", "ipc_logical"), 0)
    online_wan = 0
    links = []
    for edge in checked_edges(edges):
        local = {edge["source"], edge["destination"]} == {"client", "preparation"}
        locality = "wan"
        if local:
            locality = {
                "remote-preparation": "wan",
                "trusted-lan-appliance": "lan",
                "client-host-loopback": "loopback",
                "client-host-ipc-counterfactual": "ipc_logical",
            }[placement]
        amount = edge["serialized_body_bytes"]
        totals[locality] += amount
        if edge["phase"] == "online" and locality == "wan":
            online_wan += amount
        links.append(
            {
                **edge,
                "locality": locality,
                "trust_boundary": "within-user-trust"
                if local and placement != "remote-preparation"
                else "remote-boundary",
            }
        )
    return {
        "placement": placement,
        "lan_body_bytes": totals["lan"],
        "wan_body_bytes": totals["wan"],
        "loopback_body_bytes": totals["loopback"],
        "absent_network_ipc_logical_body_bytes": totals["ipc_logical"],
        "all_network_link_body_bytes": totals["lan"] + totals["wan"] + totals["loopback"],
        "all_logical_link_body_bytes": sum(totals.values()),
        "online_wan_body_bytes": online_wan,
        "directed_links": links,
    }


def ring_reference(
    weights: list[list[int]], x: list[int], r: list[int], s: list[int], bits: int
) -> dict[str, list[int]]:
    """Exact integer reference, not a cryptographic implementation or proof."""
    if bits not in (16, 24, 32) or not weights or len(x) != len(r):
        raise ValueError("invalid ring or input dimensions")
    if len(weights) != len(s) or any(len(row) != len(x) for row in weights):
        raise ValueError("invalid matrix/output dimensions")
    modulus = 1 << bits
    masked = [(a - b) % modulus for a, b in zip(x, r, strict=True)]
    correction = [
        (sum(w * mask for w, mask in zip(row, r, strict=True)) - out) % modulus
        for row, out in zip(weights, s, strict=True)
    ]
    response = [
        (sum(w * value for w, value in zip(row, masked, strict=True)) + corr) % modulus
        for row, corr in zip(weights, correction, strict=True)
    ]
    recovered = [(value + mask) % modulus for value, mask in zip(response, s, strict=True)]
    return {
        "masked_input": masked,
        "correction": correction,
        "masked_response": response,
        "recovered": recovered,
    }


def extract(archive_dir: Path) -> dict[str, Any]:
    retained: dict[str, Any] = {}
    for phase in ("warm", "cold"):
        filename = f"qwen-incremental-32-{phase}.json"
        raw = locked_json(archive_dir / filename, RAW_LOCKS[filename])
        selected = [item for item in raw["candidates"] if item["name"] in NAMES]
        if {item["name"] for item in selected} != set(NAMES):
            raise ValueError("archive candidates unavailable")
        retained[phase] = {}
        for item in selected:
            report = item["report"]
            configuration = report["configuration"]
            record = report["runs"][0]
            if (
                configuration["source_lock_digest"] != SOURCE_LOCK
                or record["model_fingerprint"] != FINGERPRINT
                or (
                    record["tokens"]["input_tokens"],
                    record["tokens"]["output_tokens"],
                    record["cold"],
                )
                != (39, 32, phase == "cold")
                or not all(report["checks"].values())
            ):
                raise ValueError("raw archive source/workload/privacy mismatch")
            stage = report["topology_accounting"]["stages"]["runs"][0]
            role_edges: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            stage_counts: dict[str, int] = defaultdict(int)
            if not stage["reconciled_with_protocol_bodies"]:
                raise ValueError("unreconciled archived stages")
            for stage_id, links in stage["body_bytes_by_stage_and_edge"].items():
                role = next(
                    (role for suffix, role in ROLE_SUFFIXES.items() if stage_id.endswith(suffix)),
                    None,
                )
                if role is None:
                    raise ValueError("unrecognized compiled stage role")
                stage_counts[role] += 1
                for edge, amount in links.items():
                    role_edges[role][edge] += amount
            ledger = report["topology_accounting"]["runs"][0]
            retained[phase][item["name"]] = {
                "source_lock_digest": configuration["source_lock_digest"],
                "prompt_digest": configuration["prompt_digest"],
                "inventory": record["inventory"],
                "edges": checked_edges(ledger["body_bytes_by_edge"]),
                "all_link_body_bytes": ledger["all_link_serialized_body_bytes"],
                "online_body_bytes": ledger["online_all_link_serialized_body_bytes"],
                "stage_counts_by_role": dict(stage_counts),
                "stage_body_bytes_by_role_and_edge": dict(role_edges),
                "stage_expected_edges": stage["expected_stage_body_bytes_by_edge"],
                "stage_measured_edges": stage["measured_body_bytes_by_edge"],
                "run_cpu_seconds_by_role": ledger["run_window_cpu_seconds_by_role"],
            }
    return retained


def validate(
    retained: dict[str, Any], stage_source: dict[str, Any], incremental: dict[str, Any]
) -> None:
    if (
        incremental["source"]
        != {
            "model_id": MODEL,
            "revision": REVISION,
            "weight_bits": 8,
            "activation_bits": 8,
        }
        or stage_source["source"]["body_fingerprint"] != FINGERPRINT
    ):
        raise ValueError("public model source differs")
    for phase in ("warm", "cold"):
        cohort = next(
            c for c in incremental["cohorts"] if c["file"] == f"qwen-incremental-32-{phase}.json"
        )
        for name in NAMES:
            item = retained[phase][name]
            source = next(c for c in cohort["candidates"] if c["name"] == name)
            summary = source["summary"]
            edges = checked_edges(item["edges"])
            if (
                item["source_lock_digest"] != SOURCE_LOCK
                or item["prompt_digest"] != cohort["comparison_key"]["prompt_digest"]
                or not all(source["checks"].values())
                or sum(e["serialized_body_bytes"] for e in edges) != item["all_link_body_bytes"]
                or item["all_link_body_bytes"]
                != summary["median_covered_all_link_serialized_body_bytes"]
                or item["online_body_bytes"]
                != summary["median_online_all_link_serialized_body_bytes"]
                or sum(e["serialized_body_bytes"] for e in edges if e["phase"] == "online")
                != item["online_body_bytes"]
                or item["inventory"]
                != {"burned": 0, "consumed": 70, "generated": 70, "required": 70, "reused": 0}
            ):
                raise ValueError("retained ledger differs from locked cohort/source")
            groups = item["stage_body_bytes_by_role_and_edge"]
            expected_roles = set(GEOMETRY) if name == NAMES[0] else {"mlp_gate_up", "mlp_down"}
            if set(groups) != expected_roles or item["stage_counts_by_role"] != {
                role: 24 for role in expected_roles
            }:
                raise ValueError("compiled stage role coverage differs")
            measured = {
                edge: sum(group.get(edge, 0) for group in groups.values())
                for edge in item["stage_expected_edges"]
            }
            if measured != item["stage_expected_edges"] or measured != item["stage_measured_edges"]:
                raise ValueError("directed stage reconciliation differs")
            for edge in edges:
                key = f"{edge['source']}->{edge['destination']}"
                if edge["phase"] != "cold" and measured[key] > edge["serialized_body_bytes"]:
                    raise ValueError("stage bodies exceed directed ledger")
            if phase == "warm" and name == NAMES[0]:
                control = stage_source["cohorts"]["32"]
                if {role: sum(group.values()) for role, group in groups.items()} != control[
                    "stage_body_bytes_by_semantic_role"
                ]:
                    raise ValueError("compiled role bytes differ from stage control")


def resources(item: dict[str, Any], body_placement: dict[str, Any]) -> dict[str, Any]:
    roles = item["stage_counts_by_role"]
    rows = item["inventory"]["required"]
    weights = sum(GEOMETRY[role][0] * GEOMETRY[role][1] * count for role, count in roles.items())
    inputs = sum(GEOMETRY[role][0] * count for role, count in roles.items())
    outputs = sum(GEOMETRY[role][1] * count for role, count in roles.items())
    if weights != body_placement["declared_body_linear_macs_per_row_remote"]:
        raise ValueError("Qwen geometry differs from archived declared MACs")
    return {
        "fresh_rows_per_stage": rows,
        "stage_row_instances": rows * sum(roles.values()),
        "reused_rows": 0,
        "active_preparation_int8_matrix_bytes_lower_bound": weights,
        "active_matrix_fp32_scale_bytes": outputs * 4,
        "preparation_linear_macs_per_inventory": weights * rows,
        "client_expanded_uint32_r_and_s_bytes": (inputs + outputs) * rows * 4,
        "remote_uint32_correction_array_bytes": outputs * rows * 4,
        "correction_tensor_minimum_u16_body_bytes": outputs * rows * 2,
        "minimum_seed_bytes_with_client_and_preparation": sum(roles.values()) * 32,
        "seed_bytes_to_inference": 0,
        "full_public_body_int8_snapshot_bytes_lower_bound": 357826560,
        "full_public_body_fp32_scale_bytes": 1216512,
        "checkpoint_distribution_bytes": None,
        "checkpoint_disk_bytes": None,
        "actual_preparation_snapshot_disk_bytes": None,
        "peak_preparation_ram_bytes": None,
        "peak_client_ram_bytes": None,
        "appliance_cpu_seconds": None,
        "resource_admission": "unknown; matrix/mask floors only, peak RAM/CPU/disk unmeasured",
    }


def build(retained: dict[str, Any]) -> dict[str, Any]:
    stage_source = locked_json(ROOT / "docs/evidence" / next(iter(LOCKS)), LOCKS[next(iter(LOCKS))])
    incremental = locked_json(
        ROOT / "docs/evidence/incremental-network-qwen25-2026-10-01.json",
        LOCKS["incremental-network-qwen25-2026-10-01.json"],
    )
    validate(retained, stage_source, incremental)
    digest = hashlib.sha256(
        json.dumps(retained, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != RETAINED_SHA256:
        raise ValueError("retained directed ledger source lock mismatch")
    cold_cohort = next(
        c for c in incremental["cohorts"] if c["file"] == "qwen-incremental-32-cold.json"
    )
    source_candidates = {c["name"]: c for c in cold_cohort["candidates"]}
    rows = []
    for phase in ("warm", "cold"):
        for name in NAMES:
            item = retained[phase][name]
            for placement in PLACEMENTS:
                counted = place(item["edges"], placement)
                counted.pop("directed_links")
                rows.append({"window": phase, "candidate": name, **counted})
    return {
        "schema": "pllm.trusted_local_preparation_placement_screen.v1",
        "date": "2026-10-01",
        "status": "archived-body exact reclassification; placement itself unmeasured",
        "source": {
            "model": MODEL,
            "revision": REVISION,
            "body_fingerprint": FINGERPRINT,
            "source_lock_digest": SOURCE_LOCK,
            "weight_bits": 8,
            "activation_bits": 8,
        },
        "workload": {"input_tokens": 39, "output_tokens": 32, "fresh_rows_per_stage": 70},
        "source_sha256": LOCKS,
        "raw_archive_sha256": RAW_LOCKS,
        "retained_directed_ledger_sha256": digest,
        "archived_ledgers": retained,
        "placement_table": rows,
        "resource_floors": {
            name: resources(
                retained["warm"][name], source_candidates[name]["client_body_placement"]
            )
            for name in NAMES
        },
        "archived_cold_cpu": {name: source_candidates[name]["cold_cpu"] for name in NAMES},
        "algebra": {
            "input": "u = x - r (mod 2^k)",
            "correction": "c = W r - s (mod 2^k)",
            "response": "v = W u + c = W x - s (mod 2^k)",
            "recovery": "v + s = W x (mod 2^k); signed decode then existing scales/bias",
            "freshness": "independent per-stage roots; purpose/context-bound r,s; rows one-use",
            "proof_claim": False,
        },
        "availability": {
            "refill_during_active_response": "prohibited by existing client activity guards",
            "ready_before_online": "matching correction acks plus sealed inference inventory required",
            "prep_can_disconnect_after_seal": "conditional: enough reserved rows, surviving remote inventory",
            "inference_offline": "no remote online response or correction acknowledgement",
            "exhausted_or_expired_inventory": "wait for fresh preparation before online; no reuse",
        },
        "trust": {
            "remote_preparation": "privacy depends on preparation/inference not colluding",
            "local_preparation": "user trusts local OS/appliance; remote inference never gets roots/r/s",
            "external_noncollusion_dependency": "removed only if preparation is genuinely user-controlled",
            "appliance_compromise": "mask roots plus intercepted online bodies can reveal activations",
            "seed_path": "client to Prep only: WAN remotely, LAN/loopback locally, no inference seed transfer",
            "local_plaintext_execution": "unchanged client nonlinear work; no gateway decrypt shortcut",
        },
        "unmeasured": [
            "Actual LAN/WAN deployment, endpoint ownership and operator independence",
            "Cold model/checkpoint distribution, full wire, control/auth forwarding and correction acks",
            "Peak RAM, complete model/snapshot/cache disk, energy, appliance CPU/latency and concurrency",
            "Representative availability, complete adversarial security proof and generation quality",
            "Exact archived per-stage ring profiles; u16 resource floor is deliberately conservative",
        ],
        "recommendation": "conditional WAN-locality candidate; not a 10x all-link method; admission unknown",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument(
        "--section",
        choices=(
            "archived_ledgers",
            "placement_table",
            "resource_floors",
            "retained_directed_ledger_sha256",
        ),
    )
    args = parser.parse_args()
    retained = (
        extract(args.archive_dir)
        if args.archive_dir
        else json.loads(ARTIFACT.read_text())["archived_ledgers"]
    )
    result = build(retained)
    print(json.dumps(result[args.section] if args.section else result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
