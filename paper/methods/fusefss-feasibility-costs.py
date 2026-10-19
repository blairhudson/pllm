"""Static design scenarios only; no native execution, checkpoint or benchmark."""
import json


def report():
    n, p, m = 50, 2, 512
    # u64 storage per arithmetic coordinate; seeds/control encoding retained.
    scalar_dcf = 16 + 8 + n * (16 + 8 + 2)
    vector_dcf = 16 + 8 * p + n * (16 + 8 * p + 2)
    # One input-mask share, p constant shares, one existing ring64 triple.
    extras = 8 + 8 * p + 24
    scenarios = {
        "independent_scalar_dcf_per_threshold_coordinate": {
            "party_scalar_payload_bytes": (m - 1) * p * scalar_dcf + extras,
            "party_dcf_path_levels_per_scalar": (m - 1) * p * n,
        },
        "product_group_vector_dcf_per_threshold": {
            "party_scalar_payload_bytes": (m - 1) * vector_dcf + extras,
            "party_dcf_path_levels_per_scalar": (m - 1) * n,
        },
        "public_descriptor_universal_boundary_dcf": {
            "party_scalar_payload_bytes": scalar_dcf + extras,
            # 510 original interior boundaries plus eval(0).
            "party_dcf_path_levels_per_scalar": (511 - 1 + 1) * n,
        },
    }
    workloads = []
    for selections in (1, 8):
        prefix = 128
        rows = sum(prefix + j for j in range(selections))
        causal_pairs = sum((prefix + j) * (prefix + j + 1) // 2
                           for j in range(selections))
        activations = 12 * 3072 * rows
        workloads.append({
            "geometry": "illustrative GPT-2-base: 12 layers, hidden768, FFN3072, heads12, vocab50257",
            "checkpoint_admitted": False,
            "prefix": prefix,
            "selections": selections,
            "full_prefix_rows": rows,
            "gelu_scalars": activations,
            "attention_nexp_values_all_layers_heads": 12 * 12 * causal_pairs,
            "embedding_token_equalities": rows * 50257,
            "embedding_public_weight_multiply_adds": rows * 50257 * 768,
            "last_row_full_head_public_weight_multiply_adds": selections * 50257 * 768,
            "argmax_pairwise_comparisons": selections * (50257 - 1),
            "activation_only_scenarios": {
                name: {
                    "both_parties_cumulative_payload_bytes": 2 * activations * cost["party_scalar_payload_bytes"],
                    "party_cumulative_dcf_path_levels": activations * cost["party_dcf_path_levels_per_scalar"],
                    # Assumes vectorized 78-byte-header openings, full 8-byte words.
                    "both_directions_peer_frame_bytes_batched_256":
                        48 * activations + 312 * ((activations + 255) // 256),
                } for name, cost in scenarios.items()
            },
        })
    return {
        "schema": "pllm.fusefss.feasibility-static-scenarios.v1",
        "scope": "Calculated candidate layouts; not implemented-resource claims, cryptographic review, measurements or full-model totals.",
        "excluded": ["allocator/padding/owner overhead", "dealer scratch and copies", "codecs/authentication/relay legs", "non-GELU gates", "all model tensor/source owners", "all other communication"],
        "scalar_dcf_candidate_bytes": scalar_dcf,
        "vector_dcf_candidate_bytes": vector_dcf,
        "scenarios": scenarios,
        "workloads": workloads,
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, sort_keys=True))
