"""Bounded two-worker MLP contract/cost gate, not a cryptographic implementation.

Cached Qwen config and six public W8 matrices only. No model forward, downloads,
training, new dependencies, material issuance, or executable-profile activation.
The complete function is specified; missing executable coverage fails closed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform

if __name__ == "__main__":
    for _key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[_key] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/complete-mlp-contract-screen-2026-10-01.json"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
CONTROL_LOCK = "ccfbb820e00be5c9e0ba95cee675bda82c9625321d6cd9a90659586a858f8242"
H, M, L, W = 896, 4864, 24, 32
MISSING = (
    "compiled exact float32 int cast/multiply/add/divide and exceptional paths",
    "compiled installed NumPy float32 exp implementation, not ideal real exp",
    "compiled whole-4864-row abs/max, zero/underflow checks and private scale",
    "compiled float32 division, ties-even rounding, clipping and signed i8 encode",
    "secure A2B/B2A, output masks, OT and 256-bit half-gates backend",
    "transport/framing/authentication, durable burn ledger and cancellation",
    "checkpoint-bit parity, next-layer/client continuation and two decode steps",
    "CPU/peak-memory/cold-distribution admission",
)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Bits:
    """Tiny explicit XOR/AND netlist; clear evaluator, never private execution."""

    def __init__(self, width: int):
        if type(width) is not int or not 2 <= width <= 32:
            raise ValueError("bounded width 2..32 required")
        self.width = width
        self.input_count = 2 * width
        self.gates: list[tuple[str, int, int]] = []

    def gate(self, op: str, a: int, b: int) -> int:
        self.gates.append((op, a, b))
        return self.input_count + len(self.gates) - 1

    def add(self, *, subtract: bool = False) -> list[int]:
        # -1 is public 1. NOT is XOR 1; XOR/NOT are free in table bytes.
        carry = -1 if subtract else None
        outputs = []
        for bit in range(self.width):
            a, b = bit, self.width + bit
            if subtract:
                b = self.gate("xor", b, -1)
            p = self.gate("xor", a, b)
            outputs.append(p if carry is None else self.gate("xor", p, carry))
            if bit == self.width - 1:
                break  # modulo 2**w: discard top carry, no gates to generate it
            generate = self.gate("and", a, b)
            if carry is None:
                carry = generate
            elif carry == -1:
                carry = self.gate("xor", p, generate)  # OR(a,b)
            else:
                propagated = self.gate("and", p, carry)
                carry = self.gate("xor", generate, propagated)
        return outputs

    @property
    def ands(self) -> int:
        return sum(op == "and" for op, _, _ in self.gates)

    def evaluate(self, a: int, b: int, outputs: list[int]) -> int:
        if not 0 <= a < 1 << self.width or not 0 <= b < 1 << self.width:
            raise ValueError("word outside circuit domain")
        values = [(word >> bit) & 1 for word in (a, b) for bit in range(self.width)]
        for op, left, right in self.gates:
            x, y = (1 if i == -1 else values[i] for i in (left, right))
            values.append(x ^ y if op == "xor" else x & y)
        return sum(values[wire] << bit for bit, wire in enumerate(outputs))


def signed(word: int, width: int) -> int:
    return word - (1 << width) if word & (1 << (width - 1)) else word


def int_f32_word(value: int) -> int:
    """Integer-only independent round-to-nearest-even int -> float32 control."""
    if value == 0:
        return 0
    negative, n = value < 0, abs(value)
    exponent = n.bit_length() - 1
    if exponent <= 23:
        significand = n << (23 - exponent)
    else:
        shift = exponent - 23
        significand, remainder = divmod(n, 1 << shift)
        halfway = 1 << (shift - 1)
        if remainder > halfway or (remainder == halfway and significand & 1):
            significand += 1
        if significand == 1 << 24:
            significand >>= 1
            exponent += 1
    return (int(negative) << 31) | ((exponent + 127) << 23) | (significand & ((1 << 23) - 1))


def arithmetic_controls() -> dict:
    import numpy as np

    cases = 0
    for width in range(2, 7):
        add, sub = Bits(width), Bits(width)
        add_out, sub_out = add.add(), sub.add(subtract=True)
        assert add.ands == sub.ands == 2 * width - 3
        for a in range(1 << width):
            for b in range(1 << width):
                assert add.evaluate(a, b, add_out) == (a + b) % (1 << width)
                assert sub.evaluate(a, b, sub_out) == (a - b) % (1 << width)
                cases += 2
    # Sign extension must precede B2A in the *target* accumulator ring.
    # Exhaustive signed i8 and all R8 masks, independently reconstructing R32.
    b2a = 0
    for q in range(-128, 128):
        for mask in range(256):
            encoded = q & 0xFFFFFFFF
            other = (encoded - mask) % (1 << 32)
            assert signed((mask + other) % (1 << 32), 32) == q
            b2a += 1
    cast_values = sorted(
        {
            sign * (base + offset)
            for sign in (-1, 1)
            for base in (0, 127, 1 << 23, 1 << 24, 1 << 25, 1 << 26)
            for offset in range(-5, 6)
        }
    )
    for n in cast_values:
        actual = int(np.asarray(np.float32(n)).view(np.uint32))
        assert actual == int_f32_word(n)
    return {
        "exhaustive_modular_add_sub_cases": cases,
        "signed_i8_target_ring_b2a_cases": b2a,
        "independent_integer_to_float32_cases": len(cast_values),
        "32bit_add_and_sub_ANDs_each": 61,
        "wrong_local_lift_witness": {
            "R8_shares": [5, 124],
            "correct_signed_sum": -127,
            "sum_of_separate_signed_lifts": 129,
        },
        "full_float_or_exp_circuit_checked": False,
    }


def remove_bodies(total: int, groups: dict, removal: list[tuple[str, str]]) -> dict:
    """Exact disjoint scope subtraction; no payload/body or link double credit."""
    if type(total) is not int or total < 0 or len(removal) != len(set(removal)):
        raise ValueError("invalid total or duplicate removal")
    if any(type(n) is not int or n < 0 for edges in groups.values() for n in edges.values()):
        raise ValueError("negative/noninteger body")
    stage_total = sum(sum(edges.values()) for edges in groups.values())
    if stage_total > total:
        raise ValueError("stage bodies exceed control")
    removed = 0
    for role, edge in removal:
        if role not in groups or edge not in groups[role]:
            raise ValueError("unknown removal scope")
        removed += groups[role][edge]
    fixed = total - removed
    return {
        "removed_body_bytes": removed,
        "fixed_other_body_bytes": fixed,
        "replacement_budget_25pct_bytes": total * 3 // 4 - fixed,
        "replacement_budget_50pct_bytes": total // 2 - fixed,
    }


def known_cost(rows: int, *, label_bytes: int = 32, width: int = 32) -> dict:
    """Only explicit conversion netlist and non-OT application payloads.

    Every unknown remains None in the complete cost, not an additive zero.
    16-byte comparator is deliberately optimistic (127 hidden label bits).
    """
    if rows not in (46, 70) or label_bytes not in (16, 32) or width not in (24, 32):
        raise ValueError("only pinned cohorts and explicit width comparators")
    instances = L * rows
    carry_ands = 2 * width - 3
    # GC1: 2M reconstructed accumulators + M target-ring output subtractions.
    # GC2: H reconstructed down accumulators. Private scales use XOR sharing.
    a2b_ands = (2 * M + H) * carry_ands * instances
    b2a_ands = M * carry_ands * instances
    # Garbler-owned shares + q masks + scale mask + final output XOR masks.
    a_bits = (3 * M * width + H * width + 2 * H * 32 + 96) * instances
    b_bits = (2 * M * width + H * width + H * 32 + 64) * instances
    output_bits = (M * width + H * 32 + 32) * instances
    # Client initially shares q into Rw and scale/residual words into XOR shares.
    one_ingress = (H * width // 8 + H * 4 + 4) * instances
    one_egress = H * 4 * instances
    links = {
        "client->A:input_code_scale_residual_shares": one_ingress,
        "client->B:input_code_scale_residual_shares": one_ingress,
        "A->B:A2B_half_gate_tables": 2 * label_bytes * a2b_ands,
        "A->B:B2A_half_gate_tables": 2 * label_bytes * b2a_ands,
        "A->B:garbler_selected_input_labels": a_bits * label_bytes,
        "A->B:masked_output_translation_bits": (output_bits + 7) // 8,
        "A->client:final_residual_output_XOR_share": one_egress,
        "B->client:final_residual_output_XOR_share": one_egress,
    }
    known = sum(links.values())
    return {
        "label_bytes": label_bytes,
        "accumulator_ring_bits": width,
        "MLP_row_instances": instances,
        "A2B_ANDs": a2b_ands,
        "B2A_ANDs": b2a_ands,
        "conversion_ANDs": a2b_ands + b2a_ands,
        "garbler_input_bits": a_bits,
        "evaluator_input_OTs": b_bits,
        "known_payload_bytes_by_link_and_kind": links,
        "known_payload_subtotal_bytes": known,
        "garbler_hashes_for_conversion": 4 * (a2b_ands + b2a_ands),
        "evaluator_hashes_for_conversion": 2 * (a2b_ands + b2a_ands),
        "remote_integer_MACs_per_worker": instances * 3 * H * M,
        "conversion_tables_can_be_offline": True,
        "fresh_garbler_labels_online_bytes": links["A->B:garbler_selected_input_labels"],
        "OT_transcript_bytes": None,
        "numeric_function_tables_bytes": None,
        "complete_recurring_body_bytes": None,
        "complete_wire_bytes": None,
        "unknown_required_coverage": list(MISSING),
        "admitted": False,
    }


def cost_gate(cost: dict, budget: int) -> str:
    if cost["known_payload_subtotal_bytes"] > budget:
        return "veto_known_payload_alone"
    if cost["complete_recurring_body_bytes"] is None or cost["unknown_required_coverage"]:
        return "reject_missing_executable_coverage"
    return "eligible_for_measurement_only"


def plans_and_weights() -> tuple[dict, dict, dict]:
    import numpy as np

    from pllm import Model, lower_model
    from pllm.model_loader import resolve_model
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.preparation_protocol import seeded_ring_profile
    from pllm.runtime.safetensors_store import SafeTensorStore
    from pllm.runtime.semantic_stages import scheduled_stage_specs
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    source = resolve_model(Model.hf(MODEL, revision=REVISION, local_files_only=True))
    if source.path is None or source.checkpoint_digest is None:
        raise ValueError("cached checkpoint unavailable")
    config_path = source.path / "config.json"
    config = json.loads(config_path.read_text())
    if (
        config["hidden_size"],
        config["intermediate_size"],
        config["num_hidden_layers"],
        config["hidden_act"],
    ) != (H, M, L, "silu"):
        raise ValueError("pinned geometry differs")
    composition = MaskedLinearCpu(
        Model.hf(MODEL, revision=REVISION),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    controls = json.loads(
        (ROOT / "docs/evidence/prepared-stage-attribution-qwen25-2026-09-29.json").read_text()
    )
    inventories, role_maps = {}, {}
    for new_tokens in (8, 32):
        plan = lower_model(config, batch=1, max_input_tokens=39, max_new_tokens=new_tokens)
        schedule = plan.runtime_schedule(composition)
        control = controls["cohorts"][str(new_tokens)]
        if plan.digest != control["plan_digest"] or schedule.digest != control["schedule_digest"]:
            raise ValueError("plan/schedule differs from locked cohort")
        graph, sched = plan.to_dict(), schedule.to_dict()
        shapes = {}
        for phase in ("prefill", "decode"):
            operations = graph[phase]["operations"]
            if Counter(o["id"] for o in operations) != Counter(
                i for step in sched[phase]["steps"] for i in step["operation_ids"]
            ):
                raise ValueError("schedule lacks exact operation coverage")
            silu = [o for o in operations if o["operator"] == "silu"]
            if len(silu) != L or {o["output_shape"][-1] for o in silu} != {M}:
                raise ValueError("whole-MLP shapes differ")
            shapes[phase] = silu[0]["output_shape"]
            repetitions = 1 if phase == "prefill" else new_tokens - 1
            count = sum(math.prod(o["output_shape"]) for o in silu) * repetitions
            shapes[f"{phase}_executed_silu_elements"] = count
        specs = [
            replace(s, weight_bits=8, activation_bits=8)
            for s in scheduled_stage_specs(plan, composition)
        ]
        role_maps[str(new_tokens)] = {s.id: s.role for s in specs}
        mlp = [s for s in specs if s.role in {"mlp_gate_up", "mlp_down"}]
        if len(mlp) != 48 or {(s.in_features, s.out_features) for s in mlp} != {(H, 2 * M), (M, H)}:
            raise ValueError("scheduled MLP stage geometry differs")
        inventories[str(new_tokens)] = {
            "executed_rows": 39 + new_tokens - 1,
            "plan_digest": plan.digest,
            "schedule_digest": schedule.digest,
            "composition_digest": composition.digest(),
            "native_semantic_silu_shapes": shapes,
            "MLP_stage_count": len(mlp),
            "public_W8A8_accumulator_bounds": {"gate_up": H * 127 * 127, "down": M * 127 * 127},
        }
    store = SafeTensorStore(source.path)
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=1)
    samples = []
    for spec in mlp:
        if spec.layer_index not in (0, 12, 23):
            continue
        sources = engine._resolve_stage_sources(store, spec, source.manifest)
        q = engine._quantize_sources(store, spec, sources)
        bound = int(np.abs(q.values.astype(np.int16)).sum(axis=1, dtype=np.int64).max()) * 127
        shape = list(q.values.shape)
        if shape != [spec.out_features, spec.in_features] or any(
            store.contains(f"model.layers.{spec.layer_index}.mlp.{name}_proj.bias")
            for name in ("gate", "up", "down")
        ):
            raise ValueError("cached MLP source shape/bias differs")
        samples.append(
            {
                "layer": spec.layer_index,
                "role": spec.role,
                "shape": shape,
                "sources": [key for key, _ in sources],
                "weight_i8_sha256": hashlib.sha256(q.values.tobytes()).hexdigest(),
                "scale_f32_sha256": hashlib.sha256(q.scales.tobytes()).hexdigest(),
                "certified_abs_accumulator_bound": bound,
                "minimum_signed_bits": bound.bit_length() + 1,
                "actual_prepared_ring_bits": seeded_ring_profile(bound).wire_bits,
                "chosen_resident_ring_bits": W,
            }
        )
        store.clear_cache()
        del q
    if len(samples) != 6:
        raise ValueError("expected six representative MLP matrices")
    source_info = {
        "model": MODEL,
        "revision": REVISION,
        "checkpoint_digest": source.checkpoint_digest,
        "config_sha256": file_sha(config_path),
        "weight_samples": samples,
        "width_scope": "actual prepared widths sampled at layers 0/12/23; R32 proposal certified for all layers by public geometry, no silent per-share lift",
    }
    return inventories, role_maps, source_info


def archived_ledgers(role_maps: dict, archive_dir: Path | None) -> dict:
    if archive_dir is None:
        result = json.loads(EVIDENCE.read_text())["archived_controls"]
        return result
    result = {}
    for new_tokens in (8, 32):
        path = archive_dir / f"qwen-incremental-{new_tokens}-warm.json"
        raw = json.loads(path.read_text())
        item = next(c for c in raw["candidates"] if c["name"] == "client-attention")
        report = item["report"]
        if not all(report["checks"].values()):
            raise ValueError("archive checks failed")
        stage = report["topology_accounting"]["stages"]["runs"][0]
        if not stage["reconciled_with_protocol_bodies"]:
            raise ValueError("archive stage reconciliation failed")
        groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for stage_id, edges in stage["body_bytes_by_stage_and_edge"].items():
            role = role_maps[str(new_tokens)][stage_id]
            for edge, amount in edges.items():
                groups[role][edge] += amount
        ledger = report["topology_accounting"]["runs"][0]
        result[str(new_tokens)] = {
            "archive_sha256": file_sha(path),
            "prompt_digest": report["configuration"]["prompt_digest"],
            "source_lock_digest": report["configuration"]["source_lock_digest"],
            "body_fingerprint": report["runs"][0]["model_fingerprint"],
            "covered_all_link_body_bytes": ledger["all_link_serialized_body_bytes"],
            "covered_online_body_bytes": ledger["online_all_link_serialized_body_bytes"],
            "directed_stage_body_bytes": dict(groups),
            "stage_edges": stage["measured_body_bytes_by_edge"],
        }
    return result


def validate_controls(controls: dict) -> None:
    if digest(controls) != CONTROL_LOCK:
        raise ValueError("locked directed control changed")
    prior = json.loads(
        (ROOT / "docs/evidence/prepared-stage-attribution-qwen25-2026-09-29.json").read_text()
    )
    incremental = json.loads(
        (ROOT / "docs/evidence/incremental-network-qwen25-2026-10-01.json").read_text()
    )
    for count in (8, 32):
        c = controls[str(count)]
        groups = c["directed_stage_body_bytes"]
        cohort = next(
            x for x in incremental["cohorts"] if x["file"] == f"qwen-incremental-{count}-warm.json"
        )
        summary = next(x for x in cohort["candidates"] if x["name"] == "client-attention")[
            "summary"
        ]
        if (
            set(groups) != {"mlp_gate_up", "mlp_down"}
            or c["covered_all_link_body_bytes"]
            != summary["median_covered_all_link_serialized_body_bytes"]
            or c["prompt_digest"] != cohort["comparison_key"]["prompt_digest"]
            or c["body_fingerprint"] != prior["source"]["body_fingerprint"]
        ):
            raise ValueError("control cohort/source/scope differs")
        if (
            c["covered_online_body_bytes"]
            != summary["median_online_all_link_serialized_body_bytes"]
        ):
            raise ValueError("online control differs")
        for role, edges in groups.items():
            if (
                sum(edges.values())
                != prior["cohorts"][str(count)]["stage_body_bytes_by_semantic_role"][role]
            ):
                raise ValueError("role bodies differ from independent attribution")
        calculated = {
            edge: sum(g.get(edge, 0) for g in groups.values()) for edge in c["stage_edges"]
        }
        if calculated != c["stage_edges"]:
            raise ValueError("directed edge conservation failed")
        remove_bodies(c["covered_all_link_body_bytes"], groups, [])
    if controls["32"]["covered_all_link_body_bytes"] != 148_297_986:
        raise ValueError("39+32 warm anchor differs")


def run(archive_dir: Path | None = None) -> dict:
    import numpy as np

    from pllm.runtime._native_support import extension

    inventories, role_maps, source = plans_and_weights()
    controls = archived_ledgers(role_maps, archive_dir)
    validate_controls(controls)
    cohorts = {}
    for count in (8, 32):
        c = controls[str(count)]
        groups, total = c["directed_stage_body_bytes"], c["covered_all_link_body_bytes"]
        narrow = remove_bodies(
            total,
            groups,
            [("mlp_gate_up", e) for e in ("preparation->inference", "inference->client")],
        )
        full = remove_bodies(
            total, groups, [(role, edge) for role, edges in groups.items() for edge in edges]
        )
        costs = {
            "chosen_R32_label32": known_cost(39 + count - 1),
            "optimistic_R32_label16": known_cost(39 + count - 1, label_bytes=16),
            "unadmitted_uniform_R24_label16": known_cost(39 + count - 1, label_bytes=16, width=24),
        }
        for cost in costs.values():
            cost["full_region_25pct_gate"] = cost_gate(cost, full["replacement_budget_25pct_bytes"])
            cost["full_region_50pct_gate"] = cost_gate(cost, full["replacement_budget_50pct_bytes"])
        cohorts[str(count)] = {
            "inventory": inventories[str(count)],
            "gate_up_output_plus_correction_only_scope": narrow,
            "specified_full_MLP_region_retirement_scope": full,
            "costs": costs,
            "full_GC1_semantic_input_bits_before_masks": 2 * M * 32 + 32,
            "complete_truth_table_log2_assignments_GC1": 2 * M * 32 + 32,
            "full_truth_table_feasible": False,
        }
    hashes = {
        name: file_sha(ROOT / name)
        for name in (
            "python/pllm/runtime/quantization.py",
            "python/pllm/runtime/semantic_executor.py",
            "python/pllm/runtime/transformer_client.py",
            "python/pllm/runtime/transformer_engine.py",
            "python/pllm/runtime/semantic_stages.py",
            "python/pllm/runtime/preparation_protocol.py",
            "docs/evidence/prepared-stage-attribution-qwen25-2026-09-29.json",
            "docs/evidence/incremental-network-qwen25-2026-10-01.json",
            "docs/data/research/papers.json",
        )
    }
    return {
        "schema": "pllm.complete_mlp_contract_cost_gate.v1",
        "date": "2026-10-01",
        "decision": "specified conventional two-worker construction vetoed by known conversions; executable coverage incomplete",
        "source": source,
        "source_sha256": hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "native_quantizer_available": extension() is not None,
            "machine": platform.machine(),
            "offline": True,
            "threads": 1,
        },
        "archived_controls": controls,
        "archived_controls_sha256": digest(controls),
        "arithmetic_controls": arithmetic_controls(),
        "cohorts": cohorts,
        "protocol": {
            "workers": "non-colluding semi-honest A garbler / B evaluator; public fixed W8 weights",
            "region": "client-owned post-attention residual -> existing client RMSNorm/input quantization -> shared gate/up -> exact Boolean SiLU/product/whole-row quantization -> shared down -> Boolean rescale/residual -> client",
            "linear_ring": "Z/(2^32), signed i8 extended before sharing; each worker evaluates public W locally",
            "private_scale": "XOR-shared exact float32 word, never public numeric scale",
            "GC1": "A2B gate/up accumulators; XOR-reconstruct input scale; rounded dequant -> stable float32 SiLU -> multiply -> 4864-way dynamic quantize; B2A signextended q minus fresh A pad in R32; XOR-mask scale",
            "GC2": "A2B down accumulator; XOR-reconstruct scale/residual; rounded dequant then residual add; XOR-mask output word",
            "freshness": "fresh per invocation labels/delta/tables/OTs/q pads/scale and output pads; reserve before send, consume or burn on cancel; no replay/rollback",
            "client_boundary": "two word shares in, two XOR output shares back per layer-row; existing client attention/KV, norms, final head/selection and fresh next-token feedback retained",
            "half_gates_source": "https://eprint.iacr.org/2014/756; 2 ciphertexts/AND, free XOR, 4/2 hashes per garbler/evaluator",
            "label_parameter": "256-bit labels, exposed selection bit leaves 255; parameterized correlation-robust-hash construction, no reviewed backend here",
            "optimistic_label_comparator": "16 bytes; 128 serialized bits is only 127 hidden bits with exposed selection; not a claimed 128-hidden-bit instantiation",
        },
        "complete_function_specified": True,
        "complete_executable_coverage": False,
        "complete_protocol_cost_bytes": None,
        "feasible": False,
        "claims_exclude": [
            "universal protocol/FSS lower bound",
            "exact exp circuit implementation",
            "secure execution or model parity",
            "full-wire/latency/CPU measurement",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="re-extract matched raw warm ledgers; otherwise replay saved compact controls",
    )
    parser.add_argument(
        "--section", choices=("cohorts", "source", "arithmetic_controls", "archived_controls")
    )
    args = parser.parse_args()
    result = run(args.archive_dir)
    print(
        json.dumps(
            result if args.section is None else result[args.section], indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
