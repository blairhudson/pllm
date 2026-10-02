"""Compiler-bound native correlation costs and explicit incomplete placements."""

from __future__ import annotations

import json
from typing import Any

from pllm import _native
from pllm.configuration import Pipeline
from pllm.modeling import ModelPlan

from .region_contract_cost import compiler_region_contract_cost
from .semantic_stages import scheduled_stage_specs
from .shared_resources import SharedResourceError


def project_polynomial_cost(
    plan: ModelPlan, composition: Pipeline, *, mode: str, ring_bits: int, response_new_tokens: int
) -> dict[str, Any]:
    # Reuse full semantic coverage, common-source, W8A8 and phase-budget gates.
    base = compiler_region_contract_cost(
        plan,
        composition,
        response_new_tokens=response_new_tokens,
        maximum_online_all_link_body_bytes=1 << 40,
        maximum_total_all_link_body_bytes=1 << 40,
    )
    stages = scheduled_stage_specs(plan, composition)
    groups = {(stage.layer_index, stage.role): stage for stage in stages}
    phase_rows = [
        min(16, base["input_tokens"] - start) for start in range(0, base["input_tokens"], 16)
    ]
    phase_rows += [1] * (response_new_tokens - 1)
    costs = []
    totals = {
        "material_a": 0,
        "material_b": 0,
        "peer_a": 0,
        "peer_b": 0,
        "client_input_one_worker": 0,
        "client_output_one_worker": 0,
        "client_mask_seed_channel": 0,
        "dealer_macs": 0,
        "online_macs": 0,
        "offset_mlp_macs": 0,
    }
    for lineage in base["layer_provenance"]:
        index = lineage["semantic_layer"]
        gate = groups[(index, "mlp_gate_up")]
        down = groups[(index, "mlp_down")]
        h, m, o = gate.in_features, down.in_features, down.out_features
        if gate.out_features != 2 * m or o != h:
            raise SharedResourceError(
                "polynomial cost requires complete common-source gate/up/down roles"
            )
        layer = {key: 0 for key in totals}
        for rows in phase_rows:
            measured = json.loads(
                _native.projected_polynomial_estimate(mode, ring_bits, rows, h, m, o)
            )
            # Exact current key/opening frames; output/input share frame assumption
            # uses the same 41-byte commitment envelope, not HTTP/TLS framing.
            layer["material_a"] += measured["material_bytes"][0]
            layer["material_b"] += measured["material_bytes"][1]
            layer["peer_a"] += measured["opening_bytes"][0]
            layer["peer_b"] += measured["opening_bytes"][1]
            layer["client_input_one_worker"] += 41 + rows * h * (ring_bits // 8)
            layer["client_output_one_worker"] += 41 + rows * o * (ring_bits // 8)
            # Hypothesis: trusted client receives both independent mask seeds,
            # opens z=x+r directly to both workers, eliminating peer openings.
            layer["client_mask_seed_channel"] += 41 + 64
            layer["dealer_macs"] += measured["dealer_matrix_macs"]
            layer["online_macs"] += measured["online_both_parties_matrix_macs"]
            layer["offset_mlp_macs"] += measured["two_offset_mlp_matrix_macs"]
        for key in totals:
            totals[key] += layer[key]
        costs.append(
            {
                "semantic_layer": index,
                "hidden": h,
                "channels": m,
                "outputs": o,
                "executed_rows": sum(phase_rows),
                "issuances": len(phase_rows),
                **layer,
            }
        )
    offline = totals["material_a"] + totals["material_b"]
    peer = totals["peer_a"] + totals["peer_b"]
    boundary = 2 * (totals["client_input_one_worker"] + totals["client_output_one_worker"])
    common_missing = [
        "checkpoint scale/range certificate and Qwen SiLU fidelity",
        "private dynamic quantization, ties-to-even rescaling and overflow rejection",
        "independent trusted dealer transport, authenticated roles and failure handling",
        "source distribution, cold storage, full wire, peak memory and aggregate CPU",
    ]
    scenarios = {}
    for name, online, material, missing in (
        (
            "resident_mlp_only",
            peer,
            offline,
            ["attention, RMSNorm, KV state, token ingress and feedback"],
        ),
        (
            "client_attention_mlp_cut",
            boundary + peer,
            offline,
            ["client attention/norm/KV/head work"],
        ),
        (
            "client_masked_mlp_cut_hypothesis",
            boundary,
            offline + totals["client_mask_seed_channel"],
            [
                "client mask-seed delivery and direct masked-input evaluator contract",
                "client attention/norm/KV/head work",
            ],
        ),
    ):
        edges = [
            {
                "phase": "offline",
                "source": "dealer",
                "destination": worker,
                "body_bytes": totals[key],
            }
            for worker, key in (("worker_a", "material_a"), ("worker_b", "material_b"))
        ]
        if name != "client_masked_mlp_cut_hypothesis":
            edges.extend(
                {
                    "phase": "online",
                    "source": worker,
                    "destination": other,
                    "body_bytes": totals[key],
                }
                for worker, other, key in (
                    ("worker_a", "worker_b", "peer_a"),
                    ("worker_b", "worker_a", "peer_b"),
                )
            )
        else:
            edges.append(
                {
                    "phase": "offline",
                    "source": "dealer",
                    "destination": "client",
                    "body_bytes": totals["client_mask_seed_channel"],
                }
            )
        if name != "resident_mlp_only":
            for worker in ("worker_a", "worker_b"):
                edges.extend(
                    (
                        {
                            "phase": "online",
                            "source": "client",
                            "destination": worker,
                            "body_bytes": totals["client_input_one_worker"],
                        },
                        {
                            "phase": "online",
                            "source": worker,
                            "destination": "client",
                            "body_bytes": totals["client_output_one_worker"],
                        },
                    )
                )
        scenarios[name] = {
            "body_bytes_by_edge": edges,
            "known_online_body_bytes": online,
            "known_offline_body_bytes": material,
            "known_all_link_body_bytes": online + material,
            "unknown_required_work": common_missing + missing,
            "complete_total_body_bytes": None,
            "byte_admitted": False,
            "whole_decoder_executable": False,
        }
    attn = base["client_attention_remote_mlp"]["client_attention_projection_integer_macs"]
    candidate_macs = totals["dealer_macs"] + totals["online_macs"] + attn
    offset_macs = totals["offset_mlp_macs"] + 2 * attn
    return {
        "schema": "pllm.projected_polynomial_compiler_cost.v1",
        "scope": "native structural costs projected onto compiler shapes; no decoder material issued",
        "mode": mode,
        "ring_bits": ring_bits,
        "plan_digest": plan.digest,
        "composition_digest": base["composition_digest"],
        "schedule_digest": base["schedule_digest"],
        "input_tokens": base["input_tokens"],
        "response_new_tokens": response_new_tokens,
        "layers": costs,
        "totals": totals,
        "placements": scenarios,
        "client_attention_candidate_known_body_matrix_macs": candidate_macs,
        "two_offset_body_matrix_macs": offset_macs,
        "known_matrix_mac_ratio_to_offset": candidate_macs / offset_macs,
        "matrix_mac_scope": "structural body projections, excludes head and nonlinear/crypto work; not CPU equivalence",
        "numeric_scale_and_range_validated": False,
        "whole_decoder_executable": False,
    }
