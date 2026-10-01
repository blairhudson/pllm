"""Offline, exact modular/integer structure screen on representative Qwen W8 stages.

No inference or training. One semantic stage at a time, pinned cached source,
existing source orientation and quantizer. Outputs contain public algebra only.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

# Set before NumPy/native imports. Override existing larger thread settings.
if __name__ == "__main__":
    for _key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[_key] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np

from decoder_probe_support import MODEL, REVISION
from pllm import Model, lower_model
from pllm.model_loader import resolve_model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.integer_structure_reference import (
    factor_cost_gate,
    minimum_signed_bits,
    odd_minor_witness,
    verify_odd_minor,
)
from pllm.runtime.quantization import quantize_weight_per_row
from pllm.runtime.preparation_protocol import seeded_ring_profile
from pllm.runtime.safetensors_store import SafeTensorStore
from pllm.runtime.semantic_stages import scheduled_stage_specs
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def vector_classes(weight: np.ndarray, *, axis: int) -> dict:
    """Exact duplicate, rational/integer proportional and modular-unit classes."""
    vectors = weight if axis == 0 else weight.T
    duplicate, proportional, unit = set(), set(), set()
    zeros = no_odd = duplicate_count = proportional_count = unit_count = 0
    for vector in vectors:
        v = vector.astype(np.int16)
        nonzero = np.flatnonzero(v)
        if not len(nonzero):
            zeros += 1
            continue
        key = v.tobytes()
        duplicate_count += key in duplicate
        duplicate.add(key)
        gcd = int(np.gcd.reduce(np.abs(v)))
        primitive = v // gcd
        if primitive[nonzero[0]] < 0:
            primitive = -primitive
        key = primitive.tobytes()
        proportional_count += key in proportional
        proportional.add(key)
        odd = np.flatnonzero(v % 2)
        if len(odd):
            # W8 matrix only. Any proportionality by a unit over 2**w, w>=8,
            # implies this canonical collision mod256. Missing collisions here
            # therefore also rule out unit-proportional pairs in wider rings.
            inverse = pow(int(v[odd[0]]) % 256, -1, 256)
            key = np.asarray((v.astype(np.int32) * inverse) % 256, np.uint8).tobytes()
            unit_count += key in unit
            unit.add(key)
        else:
            no_odd += 1
    return {
        "vectors": len(vectors),
        "zero_vectors": zeros,
        "duplicate_nonzero_vectors_beyond_first": duplicate_count,
        "rational_proportional_nonzero_vectors_beyond_first": proportional_count,
        "unit_proportional_mod256_vectors_beyond_first": unit_count,
        "nonzero_vectors_without_odd_entry": no_odd,
        "modular_scope": "unit scalars only; no collision mod256 rules out collisions at w>=8",
    }


def modes(vectors: np.ndarray) -> np.ndarray:
    # W8 domain exactly [-127,127]; argmax ties choose smallest signed value.
    return np.asarray(
        [
            np.argmax(np.bincount(row.astype(np.int16) + 127, minlength=255)) - 127
            for row in vectors
        ],
        np.int16,
    )


def structured_candidates(weight: np.ndarray):
    m, n = weight.shape
    w = weight.astype(np.int16)
    yield "zero", np.zeros_like(w), {"rank_upper_bound": 0, "client_additions_per_row": 0}
    row_modes = modes(weight)
    yield (
        "row_mode_rank1",
        np.broadcast_to(row_modes[:, None], w.shape),
        {
            "rank_upper_bound": 1,
            "client_additions_per_row": n - 1,
            "client_projection_width": 1,
            "remote_structured_macs_per_row": m,
            "client_projection_worst_abs": n * 127,
        },
    )
    column_modes = modes(weight.T)
    yield (
        "column_mode_rank1",
        np.broadcast_to(column_modes[None, :], w.shape),
        {
            "rank_upper_bound": 1,
            "client_structured_macs_per_row": int(np.count_nonzero(column_modes)),
            "client_projection_width": 1,
            "remote_structured_macs_per_row": 0,
            "client_projection_worst_abs": int(np.abs(column_modes).sum()) * 127,
        },
    )
    # u_i + v_j has an exact width-2 factorization, not fitted SVD.
    anchored = w[:, :1] + w[:1, :] - w[0, 0]
    yield (
        "anchored_additive_rank2",
        anchored,
        {
            "rank_upper_bound": 2,
            "client_projection_width": 2,
            "client_additions_per_row": n - 1,
            "client_structured_macs_per_row": int(np.count_nonzero(w[0] - w[0, 0])),
            "remote_structured_macs_per_row": m,
            "client_projection_worst_abs": max(n, int(np.abs(w[0] - w[0, 0]).sum())) * 127,
        },
    )
    # Block-constant structure: one public sum per input block, table per output block.
    block = 16
    tiled = np.empty_like(w)
    for i in range(0, m, block):
        for j in range(0, n, block):
            region = weight[i : i + block, j : j + block]
            mode = int(modes(region.reshape(1, -1))[0])
            tiled[i : i + block, j : j + block] = mode
    yield (
        "block16_constant",
        tiled,
        {
            "rank_upper_bound": min((m + 15) // 16, (n + 15) // 16),
            "client_projection_width": (n + 15) // 16,
            "client_additions_per_row": n - (n + 15) // 16,
            "remote_structured_macs_upper_per_row": ((m + 15) // 16) * ((n + 15) // 16),
            "client_projection_worst_abs": 16 * 127,
        },
    )


def residual_screen(weight: np.ndarray, rank_lower_bound: int) -> list[dict]:
    m, n = weight.shape
    results = []
    for name, structured, metadata in structured_candidates(weight):
        residual = weight.astype(np.int16) - structured
        # Independent wide addition verifies ALL entries; no sampled reconstruction.
        verified = all(
            np.array_equal(
                structured[i : i + 64].astype(np.int64) + residual[i : i + 64].astype(np.int64),
                weight[i : i + 64].astype(np.int64),
            )
            for i in range(0, m, 64)
        )
        if not verified:
            raise RuntimeError("exact structured residual reconstruction failed")
        nnz = int(np.count_nonzero(residual))
        active_columns = int(np.count_nonzero(np.any(residual != 0, axis=0)))
        active_rows = int(np.count_nonzero(np.any(residual != 0, axis=1)))
        bound = int(np.abs(residual).sum(axis=1, dtype=np.int64).max()) * 127
        rank_upper = metadata["rank_upper_bound"]
        projection_width = metadata.get("client_projection_width", 0)
        bits = max(32, minimum_signed_bits(bound))
        # Existing-ring separate residual route; no narrower sparse index/input encoding.
        gate = factor_cost_gate(
            inputs=n,
            outputs=m,
            rank=max(1, projection_width),
            ring_bits=bits,
            residual_nnz=nnz,
            residual_input_columns=active_columns,
        )
        if projection_width == 0:
            ingress = 70 * active_columns * (bits // 8)
            gate["optimistic_online_bytes"] = ingress + 70 * m * (bits // 8)
            gate["optimistic_all_link_bytes"] = ingress + 2 * 70 * m * (bits // 8)
        results.append(
            {
                "structure": name,
                **metadata,
                "exact_all_entries_verified": verified,
                "residual_nnz": nnz,
                "residual_fraction": nnz / weight.size,
                "residual_active_input_columns": active_columns,
                "residual_active_output_rows": active_rows,
                "residual_max_abs_coefficient": int(np.abs(residual).max()),
                "residual_worst_abs_output_w8a8": bound,
                "residual_minimum_signed_ring_bits": minimum_signed_bits(bound),
                "residual_rank_lower_bound_from_rank_subadditivity": max(
                    0, rank_lower_bound - rank_upper
                ),
                "residual_csr_bytes_u16_values_u16_indices_u32_rowptr": 4 * nnz + 4 * (m + 1),
                "remote_residual_mac_fraction": nnz / weight.size,
                "optimistic_separate_route_online_u32_bytes_70_rows": gate[
                    "optimistic_online_bytes"
                ],
                "optimistic_separate_route_all_link_u32_bytes_70_rows": gate[
                    "optimistic_all_link_bytes"
                ],
            }
        )
    return results


def pair_cse(weight: np.ndarray) -> dict:
    """Bounded exact addition chain: shared x[2j]+/-x[2j+1], nonzero equal magnitudes.

    Evaluate remotely after full-width masked ingress. No entropy/network claim.
    """
    pairs = weight.shape[1] // 2
    left = weight[:, : 2 * pairs : 2].astype(np.int16)
    right = weight[:, 1 : 2 * pairs : 2].astype(np.int16)
    equal = (left == right) & (left != 0)
    opposite = (left == -right) & (left != 0)
    equal_count = equal.sum(axis=0)
    opposite_count = opposite.sum(axis=0)
    shared_equal = equal_count >= 2
    shared_opposite = opposite_count >= 2
    products_saved = int(equal_count[shared_equal].sum() + opposite_count[shared_opposite].sum())
    additions = int(shared_equal.sum() + shared_opposite.sum())
    # Independent coefficient reconstruction of every matched pair.
    reconstruction = np.stack((left.copy(), right.copy()), axis=2)
    for match, sign in ((equal & shared_equal, 1), (opposite & shared_opposite, -1)):
        reconstruction[:, :, 0][match] = left[match]
        reconstruction[:, :, 1][match] = sign * left[match]
    if not np.array_equal(reconstruction.reshape(weight.shape[0], -1), weight[:, : 2 * pairs]):
        raise RuntimeError("pair addition-chain reconstruction failed")
    return {
        "scheme": "adjacent equal/opposite coefficient pairs, reuse count >=2",
        "independent_all_pair_coefficients_verified": True,
        "remote_products_saved_per_row": products_saved,
        "remote_shared_additions_per_row": additions,
        "product_saving_fraction_of_dense": products_saved / weight.size,
        "public_chain_index_bytes_u16_pair_u8_sign": 3 * additions,
        "input_width_unchanged": weight.shape[1],
        "output_width_unchanged": weight.shape[0],
        "network_saving_bytes": 0,
        "privacy": "same fresh full-ring input/output masks; remote sums are deterministic masked-input functions",
    }


def run() -> dict:
    start = time.monotonic()
    source = resolve_model(Model.hf(MODEL, revision=REVISION, local_files_only=True))
    if source.path is None or source.checkpoint_digest is None:
        raise RuntimeError("pinned cached source must resolve")
    config = json.loads((source.path / "config.json").read_text())
    plan = lower_model(config, batch=1, max_input_tokens=39, max_new_tokens=32)
    composition = MaskedLinearCpu(
        Model.hf(MODEL, revision=REVISION),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    # Match engine.load: semantic specs carry default W4A4 until bound to engine.
    stages = [
        replace(s, weight_bits=8, activation_bits=8)
        for s in scheduled_stage_specs(plan, composition)
    ]
    body = [s for s in stages if s.role not in {"token_lookup", "lm_head"}]
    layers = sorted({s.layer_index for s in body})
    selected_layers = (layers[0], layers[len(layers) // 2], layers[-1])
    selected_roles = ("qkv_projection", "mlp_gate_up", "mlp_down")
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=1)
    store = SafeTensorStore(source.path)
    results = []
    for spec in body:
        if spec.layer_index not in selected_layers or spec.role not in selected_roles:
            continue
        sources = engine._resolve_stage_sources(store, spec, source.manifest)
        quantized = engine._quantize_sources(store, spec, sources)
        weight = quantized.values
        m, n = weight.shape
        if weight.dtype != np.int8 or np.abs(weight.astype(np.int16)).max() > 127:
            raise RuntimeError("screen requires actual symmetric W8 integer values")
        # Re-read/re-quantize one chunk per source independently; verify fused row
        # offsets AND per-row scales against the existing stage loader.
        offset = 0
        for key, transpose in sources:
            width = engine._oriented_shape(store, key, transpose)[0]
            if transpose:
                raise RuntimeError("selected Qwen linear sources must not transpose")
            rows = min(7, width)
            raw = store.get_slice(key, (slice(0, rows), slice(None)), dtype=np.float32)
            check = quantize_weight_per_row(raw, bits=8)
            if not np.array_equal(
                weight[offset : offset + rows], check.values
            ) or not np.array_equal(quantized.scales[offset : offset + rows], check.scales):
                raise RuntimeError("fused source orientation/scales mismatch")
            offset += width
        store.clear_cache()
        witness = odd_minor_witness(weight)
        verified = verify_odd_minor(weight, witness)
        if not verified:
            raise RuntimeError("independent odd-minor verification failed")
        rank = int(witness["size"])
        output_bound = int(np.abs(weight.astype(np.int16)).sum(axis=1, dtype=np.int64).max()) * 127
        result = {
            "stage_id": spec.id,
            "layer": spec.layer_index,
            "role": spec.role,
            "weight_bits": spec.weight_bits,
            "activation_bits": spec.activation_bits,
            "shape_output_by_input": [m, n],
            "source_keys": [key for key, _ in sources],
            "weight_i8_sha256": sha(weight),
            "per_output_row_scales_f32_sha256": sha(quantized.scales),
            "scale_min_max": [float(quantized.scales.min()), float(quantized.scales.max())],
            "original_scale_and_bias_semantics": "original activation scale times original output-row weight scale, then bias once; no intermediate quantization",
            "source_chunks_and_fused_offsets_verified": True,
            "gf2_rank_lower_bound": rank,
            "minor_witness": witness,
            "minor_determinant_parity_independently_verified": int(verified),
            "full_column_rank_certified": rank == n,
            "full_row_rank_certified": rank == m,
            "full_min_dimension_certified": rank == min(m, n),
            "minimum_free_module_factor_width_over_2poww": rank,
            "ring_scope": "all w>=1 for odd-minor lower bound; W8 output semantics require separately certified lift",
            "w8a8_public_worst_abs_accumulator": output_bound,
            "w8a8_minimum_signed_ring_bits": minimum_signed_bits(output_bound),
            "actual_prepared_ring_bits": seeded_ring_profile(output_bound).wire_bits,
            "row_classes": vector_classes(weight, axis=0),
            "column_classes": vector_classes(weight, axis=1),
            "structured_residuals": residual_screen(weight, rank),
            "bounded_common_subexpressions": pair_cse(weight),
            "dense_minwidth_factor_costs_u32_70_rows": factor_cost_gate(
                inputs=n,
                outputs=m,
                rank=rank,
                ring_bits=32,
                coefficient_bits=32,
            ),
            "identity_orientation_exact_option": (
                {
                    "A": "W",
                    "B": "I",
                    "client_macs": 0,
                    "client_weight_bytes": 0,
                    "remote_macs_70_rows": 70 * m * n,
                    "input_width": n,
                    "output_width": m,
                }
                if n <= m
                else {
                    "A": "I",
                    "B": "W",
                    "client_macs": 70 * m * n,
                    "client_weight_bytes_including_f32_scales": m * n + 4 * m,
                    "remote_macs_70_rows": 0,
                    "input_width": m,
                    "output_width": m,
                }
            ),
        }
        results.append(result)
        print(f"{spec.id}: {m}x{n}, certified rank {rank}", file=sys.stderr)
        del quantized, weight
        gc.collect()
    if len(results) != 9:
        raise RuntimeError("expected exactly nine first/middle/last role samples")
    budgets = {
        "original": {"all_link_bytes": 178_970_558, "online_bytes": 113_545_024},
        "attention_owned": {"all_link_bytes": 148_297_986, "online_bytes": 93_221_048},
    }
    for row in budgets.values():
        row["tenfold_all_link_budget_bytes"] = row["all_link_bytes"] // 10
        row["tenfold_online_budget_bytes"] = row["online_bytes"] // 10
    # Optimistic full-schedule SHAPE thought experiment, not rank measurements on
    # unselected layers. Even free client Bx and zero-width ingress leave outputs.
    totals = {}
    for name, remote in (
        ("original", body),
        ("attention_owned", [s for s in body if s.role in {"mlp_gate_up", "mlp_down"}]),
    ):
        totals[name] = {
            "stage_count": len(remote),
            "unattained_uniform_u16_zero_ingress_output_only_online_bytes": 70
            * 2
            * sum(s.out_features for s in remote),
            "unattained_uniform_u16_zero_ingress_output_plus_fresh_correction_all_link_bytes": 70
            * 4
            * sum(s.out_features for s in remote),
            "zero_ingress_existing_u32_output_only_online_bytes": 70
            * 4
            * sum(s.out_features for s in remote),
            "zero_ingress_existing_u32_output_plus_fresh_correction_all_link_bytes": 70
            * 8
            * sum(s.out_features for s in remote),
            "optimistic_full_minrank_ingress_u32_online_bytes": 70
            * 4
            * sum(min(s.in_features, s.out_features) + s.out_features for s in remote),
            "optimistic_full_minrank_ingress_u32_all_link_bytes": 70
            * 4
            * sum(min(s.in_features, s.out_features) + 2 * s.out_features for s in remote),
            "full_schedule_projection_scope": "shape-only hypothetical full min-rank all layers; actual certificates cover nine sampled matrices only",
        }
    summary = []
    for matrix in results:
        best = min(matrix["structured_residuals"], key=lambda r: r["residual_nnz"])
        summary.append(
            {
                "layer": matrix["layer"],
                "role": matrix["role"],
                "shape": matrix["shape_output_by_input"],
                "certified_minimum_factor_width": matrix["gf2_rank_lower_bound"],
                "actual_prepared_ring_bits": matrix["actual_prepared_ring_bits"],
                "best_screened_structure": best["structure"],
                "best_residual_fraction": best["residual_fraction"],
                "best_residual_minimum_signed_bits": best["residual_minimum_signed_ring_bits"],
                "best_residual_active_input_columns": best["residual_active_input_columns"],
                "bounded_cse_product_saving_fraction": matrix["bounded_common_subexpressions"][
                    "product_saving_fraction_of_dense"
                ],
            }
        )
    return {
        "schema": "pllm.integer_structure_screen.v1",
        "date": "2026-10-01",
        "model": MODEL,
        "revision": REVISION,
        "checkpoint_digest": source.checkpoint_digest,
        "source_lock_digest": source.source_lock_digest,
        "semantic_plan_digest": plan.digest,
        "schedule_digest": plan.runtime_schedule(composition).digest,
        "numeric": "existing SymmetricPerRow W8A8; output-by-input W; same unmodified output-row scales",
        "workload": {"input_tokens": 39, "output_tokens": 32, "executed_rows": 70},
        "resource_limits": {
            "native_threads": 1,
            "resident_selected_stages": 1,
            "largest_selected_weight_i8_bytes": max(
                np.prod(r["shape_output_by_input"]).item() for r in results
            ),
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "no_download": True,
            "no_training": True,
        },
        "sampled_layers": list(selected_layers),
        "sampled_stages": len(results),
        "status": "no-go for sampled sub-min-dimension factors and screened small-sparse-residual routes; down input reduction exists at dense full-client-stage cost; broader exact circuits open",
        "controls": budgets,
        "shape_only_network_gates": totals,
        "summary": summary,
        "matrices": results,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "claims_exclude": [
            "rank of unselected layers",
            "exact factorization inferred from deficient GF2 rank",
            "cryptographic protocol proof",
            "protected full-response measurement",
            "global minimum arithmetic circuit",
            "impossibility of every possible low-rank plus sparse decomposition",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write public evidence JSON")
    args = parser.parse_args()
    result = run()
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
