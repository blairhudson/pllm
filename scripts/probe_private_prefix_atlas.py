"""Deterministic, source-locked public-prefix atlas cost/algebra screen.

No model/native imports, benchmarks, private inputs, or production admission.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE_BYTES = 65536
LOW_BYTES = 256 << 20
SOURCES = (
    "scripts/probe_private_prefix_atlas.py",
    "tests/test_private_prefix_atlas_probe.py",
    "docs/research/private-prefix-atlas-reference.md",
    "python/pllm/runtime/public_prefix.py",
    "python/pllm/state/public_prefix.py",
    "python/pllm/runtime/shared_lookup.py",
    "python/pllm/metrics/private_pages.py",
    "crates/pllm-garble/src/private_pages.rs",
    "crates/pllm-python/src/private_pages.rs",
    "docs/evidence/incremental-sdk-publisher.json",
    "docs/research/incremental-sdk.md",
    "docs/research/context-graph-reference.md",
    "docs/plans/low-tier-all-link-hypotheses.md",
)


def positive(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("expected positive integer")
    return value


def layout(catalog: int, capsule_bytes: int, page_bytes: int = PAGE_BYTES) -> dict:
    """Each stripe is a distinct native table with N equal-width records."""
    positive(catalog)
    positive(capsule_bytes)
    positive(page_bytes)
    if catalog > 262144 or page_bytes > PAGE_BYTES or catalog * page_bytes > 512 << 20:
        raise ValueError("stripe exceeds native descriptor policy")
    if capsule_bytes > 64 << 20:
        raise ValueError("capsule exceeds existing selection bound")
    pages = (capsule_bytes + page_bytes - 1) // page_bytes
    depth = (catalog - 1).bit_length()
    return {
        "catalog": catalog,
        "capsule_bytes": capsule_bytes,
        "pages": pages,
        "page_bytes": page_bytes,
        "padded_capsule_bytes": pages * page_bytes,
        "dpf_depth": depth,
        "query_bytes_each": 65 + 16 + 17 * depth + 1,
        "reply_bytes_each": 65 + page_bytes,
        "ledger_imports_per_epoch": 4096,
    }


def cost(catalog: int, capsule_bytes: int, clients: int, requests: int,
         misses: int, *, page_bytes: int = PAGE_BYTES, frame_bytes: int = 64) -> dict:
    """Application-body model; 64-byte authenticated framing is a design budget.

    Public epoch descriptor = 256 + N*128 + stripes*48 bytes. Entries bind
    lengths/digests/identity; stripe entries bind 32-byte native table digest and
    dimensions. Public index resolution stays local. No per-index metadata fetch.
    """
    geometry = layout(catalog, capsule_bytes, page_bytes)
    positive(clients)
    positive(requests)
    if type(misses) is not int or not 0 <= misses <= clients * requests:
        raise ValueError("invalid aggregate misses")
    if type(frame_bytes) is not int or frame_bytes < 0:
        raise ValueError("invalid frame budget")
    pages = geometry["pages"]
    padded = geometry["padded_capsule_bytes"]
    metadata = 256 + catalog * 128 + pages * 48
    replica_delivery = 2 * (catalog * padded + metadata + frame_bytes)
    client_metadata = clients * (metadata + frame_bytes)
    queries = 2 * pages * geometry["query_bytes_each"]
    replies = 2 * pages * geometry["reply_bytes_each"]
    frames = 4 * pages * frame_bytes
    retrieval = queries + replies + frames
    local_catalog = clients * (catalog * padded + metadata + frame_bytes)
    one_capsule_cold = clients * (padded + metadata + 2 * frame_bytes)
    return {
        "layout": geometry,
        "clients": clients,
        "requests_per_client": requests,
        "aggregate_misses": misses,
        "metadata_bytes": metadata,
        "frame_budget_bytes_each": frame_bytes,
        "links": {
            "publisher_to_preparation_and_inference": replica_delivery,
            "publisher_to_clients_metadata": client_metadata,
            "client_to_two_servers_queries": misses * queries,
            "two_servers_to_client_replies": misses * replies,
            "query_reply_auth_frames": misses * frames,
        },
        "atlas_known_incremental_bytes": replica_delivery + client_metadata + misses * retrieval,
        "per_miss_bytes": retrieval,
        "dense_uniform_xor_alternative_per_miss_bytes": (
            2 * pages * (65 + (catalog + 7) // 8) + replies + frames
        ),
        "dpf_material_per_miss_bytes": 2 * pages * (16 + 17 * geometry["dpf_depth"] + 1),
        "local_full_catalog_cold_bytes": local_catalog,
        "download_all_each_request_bytes": clients * requests * (catalog * padded + metadata + frame_bytes),
        "prepositioned_one_capsule_cold_bytes": one_capsule_cold,
        "existing_local_one_capsule_hit_incremental_bytes": 0,
        "existing_local_full_catalog_hit_incremental_bytes": 0,
        "publisher_capsule_output_bytes": catalog * capsule_bytes,
        "publisher_exact_public_prefills": catalog,
        "server_stripe_owners": 2 * pages,
        "one_use_dpf_queries_issued": 2 * pages * misses,
        "client_capsule_authentications_and_imports": misses,
        "two_server_catalog_owned_bytes": 2 * catalog * padded,
        "two_server_logical_scan_bytes": misses * 2 * catalog * padded,
        "client_reconstructed_authenticated_bytes": misses * capsule_bytes,
        "client_query_reply_page_payload_bound": (
            2 * geometry["query_bytes_each"] + 2 * geometry["reply_bytes_each"] + page_bytes
        ),
        "client_payload_plus_metadata_plus_file_bytes": padded + metadata,
        "unknown": {
            "publisher_source_distribution_bytes": None,
            "additional_public_template_mapping_bytes": None,
            "publisher_compilation_cpu_seconds": None,
            "server_admission_and_dpf_cpu_seconds": None,
            "client_rng_authentication_import_cpu_seconds": None,
            "ordinary_prefill_all_link_bytes_per_template": None,
            "common_base_suffix_decode_bytes": None,
            "transport_handshake_retry_physical_wire_bytes": None,
            "whole_client_peak_bytes": None,
            "remote_body_mac_fraction_prefill_decode": None,
        },
        "beats_local_full_catalog_known_bytes": replica_delivery + client_metadata + misses * retrieval < local_catalog,
    }


def cold_client_threshold(catalog: int, capsule_bytes: int) -> int | None:
    """Strict break-even vs one cold full-catalog delivery per client, one miss."""
    row = cost(catalog, capsule_bytes, 1, 1, 1)
    denominator = row["local_full_catalog_cold_bytes"] - (
        row["links"]["publisher_to_clients_metadata"] + row["per_miss_bytes"]
    )
    if denominator <= 0:
        return None
    return row["links"]["publisher_to_preparation_and_inference"] // denominator + 1


def all_miss_horizon_limit(catalog: int, capsule_bytes: int, clients: int) -> int:
    """Maximum integer H giving a strict known-byte win vs retained catalog."""
    row = cost(catalog, capsule_bytes, clients, 1, clients)
    allowance = row["local_full_catalog_cold_bytes"] - (
        row["links"]["publisher_to_preparation_and_inference"]
        + row["links"]["publisher_to_clients_metadata"]
    )
    return max(0, (allowance - 1) // (clients * row["per_miss_bytes"]))


def expected_misses(probabilities: list[float], horizon: int, capacity: int) -> float:
    """IID finite trace: optimal static cache vs full catalog, not claimed LRU oracle.

    Static top-C templates begin cold. Each selected cached template fetched at
    first use only; remaining templates fetched each time. Distribution is public
    workload assumption, never provider-observed private prompt metadata.
    """
    positive(horizon)
    if (not probabilities or any(p < 0 or p > 1 for p in probabilities)
            or abs(sum(probabilities) - 1) > 1e-12
            or type(capacity) is not int or not 0 <= capacity <= len(probabilities)):
        raise ValueError("invalid distribution/cache capacity")
    ranked = sorted(probabilities, reverse=True)
    return sum(1 - (1 - p) ** horizon for p in ranked[:capacity]) + horizon * sum(ranked[capacity:])


def xor_reply(table: tuple[bytes, ...], mask: int) -> bytes:
    if not table or len(set(map(len, table))) != 1 or not 0 <= mask < 1 << len(table):
        raise ValueError("invalid dense reference table/query")
    result = bytearray(len(table[0]))
    for index, row in enumerate(table):
        # Reference arithmetic only. Not constant-time executable server code.
        if mask >> index & 1:
            for lane, value in enumerate(row):
                result[lane] ^= value
    return bytes(result)


def toy_checks(catalog: int = 4) -> dict:
    """Exhaust uniform dense XOR queries; NOT AES/DPF security validation."""
    if type(catalog) is not int or not 1 <= catalog <= 8:
        raise ValueError("toy catalog must be 1..8")
    table = tuple(bytes((index, index ^ 0xA5, 255 - index)) for index in range(catalog))
    checked = 0
    views = [[], []]
    for target in range(catalog):
        histograms = [Counter(), Counter()]
        for random_mask in range(1 << catalog):
            masks = (random_mask, random_mask ^ (1 << target))
            replies = tuple(xor_reply(table, mask) for mask in masks)
            recovered = bytes(a ^ b for a, b in zip(*replies, strict=True))
            if recovered != table[target]:
                raise AssertionError("dense XOR identity failed")
            for party in (0, 1):
                histograms[party][(masks[party], replies[party])] += 1
            checked += 1
        for party in (0, 1):
            views[party].append(histograms[party])
    identical = all(all(view == party_views[0] for view in party_views) for party_views in views)
    if not identical:
        raise AssertionError("single-party toy views differ")
    return {
        "catalog": catalog,
        "exact_reconstructions": checked,
        "single_party_query_and_reply_view_histograms_equal": identical,
        "collusion_recovers_target": True,
        "scope": "exhaustive dense uniform XOR, public synthetic bytes; not DPF/AES, transport, timing, or decoder privacy",
    }


def build_report() -> dict:
    publisher = json.loads((ROOT / "docs/evidence/incremental-sdk-publisher.json").read_text())
    size = publisher["capsule"]["bytes"]
    cases = [cost(n, size, k, h, k * h) for n, k, h in
             ((1, 16, 1), (2, 16, 1), (4, 16, 1), (32, 1, 1), (32, 3, 1),
              (32, 16, 1), (32, 16, 16), (128, 16, 1))]
    hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCES}
    implementation_digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {
        "schema": "pllm.private_prefix_atlas_screen.v1",
        "verdict": "conditional catalog-distribution saving only; no complete-cost or Low/decoder/privacy admission",
        "source_lock_digest": publisher["source_lock_digest"],
        "body_fingerprint": publisher["body_fingerprint"],
        "reference_capsule_digest": publisher["capsule"]["digest"],
        "source_sha256": hashes,
        "implementation_digest": implementation_digest,
        "toy": toy_checks(),
        "cases": cases,
        "strict_cold_client_thresholds": {str(n): cold_client_threshold(n, size) for n in (1, 2, 4, 32, 128)},
        "strict_all_miss_horizon_limit_n32_k16": all_miss_horizon_limit(32, size, 16),
        "iid_expected_misses_per_client": {
            "uniform_32_h16_cache1": expected_misses([1 / 32] * 32, 16, 1),
            "uniform_32_h16_cache32": expected_misses([1 / 32] * 32, 16, 32),
            "hot_90pct_h16_cache1": expected_misses([0.9] + [0.1 / 31] * 31, 16, 1),
        },
        "historical_not_low_matched": {
            "ordinary_covered_bytes": 481915095,
            "capsule_covered_bytes": 266912708,
            "covered_delta_bytes": 215002387,
            "publisher_combined_cpu_seconds_not_per_template": publisher["offline_publisher_cpu_seconds"],
        },
        "complete_cost_break_even": None,
        "native_full_width_sdk_run": "not launched; parent-only fresh_workers protocol requires new atlas integration first",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(build_report(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
