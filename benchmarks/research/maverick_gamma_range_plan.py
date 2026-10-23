"""Metadata-only Gamma range plan. Never opens a source/cache or issues HTTP.

Reconstruct the retained tensor directory and require its observed header digest
before trusting offsets. This is metadata identity, not payload authentication.
The coalesced plan is a proposal with explicit previously read scale overlap;
neither plan authorizes source I/O or certifies any Gamma value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct

from benchmarks.research import maverick_scale_source as owner


NORMS = {"model.norm.weight": [2560]}
for _layer in range(36):
    for _role, _n in (("input_layernorm", 2560), ("post_attention_layernorm", 2560),
                      ("self_attn.k_norm", 128), ("self_attn.q_norm", 128)):
        NORMS[f"model.layers.{_layer}.{_role}.weight"] = [_n]


def retained_directory():
    """No assumption about writer order survives a mismatch to the retained hash."""
    specs = {name: ("BF16", shape) for name, shape in NORMS.items()}
    specs["model.embed_tokens.weight"] = ("BF16", [151936, 2560])
    for layer in range(36):
        for role, grid in owner.SHAPES.items():
            prefix = f"model.layers.{layer}.{role}"
            specs[prefix + ".weight"] = ("F8_E4M3", [128*n for n in grid])
            specs[prefix + ".weight_scale_inv"] = ("BF16", grid)
    directory = {"__metadata__": {"format": "pt"}}
    cursor = 0
    for name, (dtype, shape) in sorted(specs.items(), key=lambda item: (item[1][0] != "BF16", item[0])):
        end = cursor + math.prod(shape) * (2 if dtype == "BF16" else 1)
        directory[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, end]}
        cursor = end
    raw = json.dumps(directory, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    if (len(raw) != owner.HEADER_BYTES or hashlib.sha256(raw).hexdigest() != owner.HEADER_SHA256
            or cursor + 8 + len(raw) != owner.FILE_SIZE):
        raise ValueError("metadata reconstruction differs from retained header; offsets unproved")
    prefix = struct.pack("<Q", len(raw)) + raw

    class RetainedHeader:
        size = owner.FILE_SIZE

        def read(self, offset, count, purpose):
            if purpose != "headers" or not 0 <= offset <= offset + count <= len(prefix):
                raise ValueError("metadata-only source cannot read payload")
            return prefix[offset:offset+count]

    rows, observed = owner.parse_header(RetainedHeader())
    if len(rows) != 650 or observed["header_sha256"] != owner.HEADER_SHA256:
        raise ValueError("retained header parse mismatch")
    return rows


def merge_norms(intervals, bridges=()):
    groups = []
    for name, row in sorted(intervals.items(), key=lambda item: item[1]["absolute_offset"]):
        start, length = row["absolute_offset"], row["elements"] * 2
        if groups:
            previous = groups[-1]
            end = previous["start"] + previous["length"]
            if (start == end or (end, start-end) in bridges) and start + length - previous["start"] <= owner.MAX_RANGE:
                previous["length"] = start + length - previous["start"]
                previous["gamma_names"].append(name)
                continue
        groups.append({"start": start, "length": length, "gamma_names": [name]})
    return groups


def plan():
    rows = retained_directory()
    norms = {name: rows[name] for name in NORMS}
    if any(row["dtype"] != "BF16" or row["shape"] != NORMS[name] for name, row in norms.items()):
        raise ValueError("Gamma inventory differs from selected map")
    bridges = {(start, length) for start, length, members in owner.scale_groups(rows)
               if all(".self_attn." in name for _, _, name in members)}
    strict = merge_norms(norms)
    coalesced = merge_norms(norms, bridges)
    permitted = set(norms) | {name for name in owner.EXPECTED if ".self_attn." in name}
    for group in coalesced:
        start, end = group["start"], group["start"] + group["length"]
        for name, row in rows.items():
            left = row["absolute_offset"]
            right = left + row["data_offsets"][1] - row["data_offsets"][0]
            if max(start, left) < min(end, right) and name not in permitted:
                raise ValueError("coalesced span touches non-approved tensor family")
    gamma_bytes = sum(row["elements"] * 2 for row in norms.values())
    plans = {}
    for label, groups in (("strict_gamma_only", strict), ("proposed_known_scale_overlap", coalesced)):
        payload = sum(group["length"] for group in groups)
        plans[label] = {
            "groups": groups, "body_bearing_requests": len(groups), "payload_bytes": payload,
            "previously_inspected_scale_overlap_bytes": payload - gamma_bytes,
            "within_proposed_512_KiB_payload_cap": payload <= 512 << 10,
            "cumulative_inspection_bytes": 845976 + payload,
            "within_original_4_MiB_cap": 845976 + payload <= owner.BODY_LIMIT,
            "minimum_cumulative_request_reservation": 217 + len(groups),
            "redirect_headroom_under_300": owner.REQUEST_LIMIT - 217 - len(groups),
            "metadata_refetch_requests_assumed": 0,
            "source_reads_authorized_or_executed": False,
        }
    return {
        "repository": "Qwen/Qwen3-4B-FP8", "revision": owner.PIN, "asset": owner.ASSET,
        "metadata_identity": "reconstructed_bytes_match_retained_header_sha256",
        "header_sha256": owner.HEADER_SHA256,
        "gamma_tensors": len(norms), "gamma_coefficients": gamma_bytes // 2, "gamma_payload_bytes": gamma_bytes,
        "norms": [{"name": name, "dtype": row["dtype"], "shape": row["shape"],
                   "start": row["absolute_offset"], "length": row["elements"] * 2}
                  for name, row in sorted(norms.items(), key=lambda item: item[1]["absolute_offset"])],
        "historical_byte_debit": 845976, "historical_request_reservation": 217,
        "plans": plans, "actual_gamma_values": "unread", "gamma_premise": "unproved",
        "embedding_head_and_source_no_wrap": "unproved", "whole_shard_integrity": "unverified",
    }


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(plan(), indent=2))


if __name__ == "__main__":
    main()
