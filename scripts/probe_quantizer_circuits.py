"""Bounded clear Boolean synthesis of encoded nonlinear outputs; no cryptography.

Only public finite tables are compiled. Secret-dependent Python/table access is
an oracle, never a protocol. No weights, material issuance, or runtime activation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pllm.runtime.quantization import quantize_activation_per_row
from pllm.runtime.semantic_executor import SemanticDecoderRuntime

ROOT = Path(__file__).resolve().parents[1]
MAX_INPUT_BITS = 16
MAX_WIRES = 2_000_000
MAX_ANDS = 1_000_000
CONFIG_SHA256 = "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"


class Circuit:
    """XOR/AND straight-line program with free complemented edges and constants.

    Literal 0/1 denotes constant; other literals are 2*wire_id + complement.
    Hash-consing and algebraic simplification change structure, not functionality.
    Evaluation executes every gate, with no branch on any input bit.
    """

    def __init__(self, input_bits: int, *, shared: bool):
        if type(input_bits) is not int or not 1 <= input_bits <= MAX_INPUT_BITS:
            raise ValueError("input width exceeds the exhaustive reference bound")
        self.input_bits = input_bits
        self.shared = shared
        self.gates: list[tuple[str, int, int]] = []
        self.outputs: list[int] = []
        self.cache: dict[tuple[str, int, int], int] = {}
        self.ands = 0
        self.depth = [0] * (input_bits + 1)
        self.logic_depth = self.depth.copy()

    def input(self, bit: int) -> int:
        return 2 * (bit + 1)

    def gate(self, kind: str, left: int, right: int) -> int:
        invert = 0
        if kind == "XOR":
            invert = (left ^ right) & 1
            left &= ~1
            right &= ~1
            if left == right:
                return invert
            if left == 0 or right == 0:
                return (left | right) ^ invert
        elif kind == "AND":
            if left == 0 or right == 0 or left == (right ^ 1):
                return 0
            if left == 1:
                return right
            if right == 1 or left == right:
                return left
        else:
            raise ValueError("unknown Boolean gate")
        left, right = sorted((left, right))
        key = (kind, left, right)
        if self.shared and key in self.cache:
            return self.cache[key] ^ invert
        wire = len(self.depth)
        if wire >= MAX_WIRES or (kind == "AND" and self.ands >= MAX_ANDS):
            raise ValueError("circuit exceeds bounded synthesis limits")
        self.gates.append(key)
        self.ands += kind == "AND"
        self.depth.append(max(self.depth[left >> 1], self.depth[right >> 1]) + (kind == "AND"))
        self.logic_depth.append(max(self.logic_depth[left >> 1], self.logic_depth[right >> 1]) + 1)
        literal = 2 * wire
        if self.shared:
            self.cache[key] = literal
        return literal ^ invert

    def mux(self, selector: int, low: int, high: int) -> int:
        if low == high:
            return low
        difference = self.gate("XOR", low, high)
        return self.gate("XOR", low, self.gate("AND", selector, difference))

    def evaluate_all(self, *, chunk_bits: int = 8192) -> np.ndarray:
        """Bit-sliced clear evaluator independent of table lookup/synthesis.

        Each Python integer carries a chunk of truth assignments. Last-use release
        bounds live values; this measures correctness, not secure backend speed.
        """
        if chunk_bits < 1:
            raise ValueError("positive evaluation chunk required")
        domain = 1 << self.input_bits
        result = np.zeros(domain, np.uint64)
        uses = Counter(literal >> 1 for _, a, b in self.gates for literal in (a, b))
        uses.update(output >> 1 for output in self.outputs)
        for start in range(0, domain, chunk_bits):
            end = min(domain, start + chunk_bits)
            mask = (1 << (end - start)) - 1
            assignments = np.arange(start, end, dtype=np.uint64)
            live = {0: 0}
            for bit in range(self.input_bits):
                packed = np.packbits(((assignments >> bit) & 1).astype(np.uint8), bitorder="little")
                live[bit + 1] = int.from_bytes(packed.tobytes(), "little")
            remaining = uses.copy()
            for wire, (kind, a, b) in enumerate(self.gates, self.input_bits + 1):
                left = live[a >> 1] ^ (mask if a & 1 else 0)
                right = live[b >> 1] ^ (mask if b & 1 else 0)
                live[wire] = left ^ right if kind == "XOR" else left & right
                for literal in (a, b):
                    source = literal >> 1
                    remaining[source] -= 1
                    if remaining[source] == 0:
                        del live[source]
            for bit, output in enumerate(self.outputs):
                packed = live[output >> 1] ^ (mask if output & 1 else 0)
                byte_body = packed.to_bytes((end - start + 7) // 8, "little")
                values = np.unpackbits(np.frombuffer(byte_body, np.uint8), bitorder="little")
                result[start:end] |= values[: end - start].astype(np.uint64) << bit
        return result

    def summary(self) -> dict:
        body = json.dumps(
            [self.input_bits, self.gates, self.outputs], separators=(",", ":")
        ).encode()
        return {
            "input_bits": self.input_bits,
            "output_bits": len(self.outputs),
            "and_non_xor_gates": self.ands,
            "xor_gates": len(self.gates) - self.ands,
            "wire_count_including_constant": len(self.depth),
            "and_depth": max(self.depth[o >> 1] for o in self.outputs),
            "logic_depth": max(self.logic_depth[o >> 1] for o in self.outputs),
            "public_circuit_json_bytes": len(body),
            "public_circuit_sha256": hashlib.sha256(body).hexdigest(),
        }


def synthesize(table: np.ndarray, output_bits: int, order: list[int], *, shared: bool) -> Circuit:
    """Shannon expansion; identical order/constant folding, optional global CSE."""
    table = np.asarray(table, dtype=np.uint64).reshape(-1)
    input_bits = len(order)
    if len(table) != 1 << input_bits or sorted(order) != list(range(input_bits)):
        raise ValueError("table/order does not define a complete finite domain")
    if type(output_bits) is not int or not 1 <= output_bits <= 64:
        raise ValueError("output width must be in [1, 64]")
    if output_bits < 64 and np.any(table >= 1 << output_bits):
        raise ValueError("table output exceeds encoded width")
    circuit = Circuit(input_bits, shared=shared)
    # Put root variable in the most significant index bit. Pairing adjacent
    # elements then eliminates the last/root-farthest variable at each level.
    indices = np.arange(len(table), dtype=np.uint64)
    permutation = np.zeros(len(table), np.uint64)
    for position, bit in enumerate(order):
        permutation |= ((indices >> (input_bits - position - 1)) & 1) << bit
    ordered = table[permutation]
    for output_bit in range(output_bits):
        literals = ((ordered >> output_bit) & 1).tolist()
        for bit in reversed(order):
            selector = circuit.input(bit)
            literals = [
                circuit.mux(selector, literals[i], literals[i + 1])
                for i in range(0, len(literals), 2)
            ]
        circuit.outputs.append(literals[0])
    return circuit


def signed(raw: int, bits: int) -> int:
    return raw - (1 << bits) if raw & (1 << (bits - 1)) else raw


def scalar_product(gate: int, up: int, bits: int) -> np.float32:
    """Independent scalar arithmetic oracle, preserving each float32 edge."""
    scale = np.float32(1 / (1 << (bits - 1)))
    x = np.float32(gate) * scale
    y = np.float32(up) * scale
    exponential = np.exp(np.float32(-abs(x)), dtype=np.float32)
    denominator = np.float32(np.float32(1) + exponential)
    numerator = x if x >= 0 else np.float32(x * exponential)
    nonlinear = np.float32(numerator / denominator)
    return np.float32(nonlinear * y)


def scalar_quantize(value: np.float32, scale: np.float32) -> int:
    # Python round(float) is an independent ties-even implementation after the
    # required float32 division. No np.rint or compiled runtime quantizer here.
    return max(-127, min(127, round(float(np.float32(value / scale)))))


def runtime_products(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    runtime = SimpleNamespace(nonlinear_evaluator=None)
    operation = {"operator": "silu", "inputs": ["gate"], "attributes": {}, "layer": 0}
    nonlinear = SemanticDecoderRuntime._local(runtime, operation, {"gate": gate}, {}, {})
    operation = {"operator": "multiply", "inputs": ["gate", "up"], "attributes": {}}
    return SemanticDecoderRuntime._local(runtime, operation, {"gate": nonlinear, "up": up}, {}, {})


def function_table(*, bits: int, dynamic_vector: bool = False) -> tuple[np.ndarray, dict]:
    """Runtime table plus separately evaluated scalar oracle, exhaustively checked.

    Fixed case: two signed 8-bit inputs, input/output scales 1/128.
    Dynamic case: two elements, four signed 4-bit inputs, input scales 1/8;
    output is two signed i8 codes AND the actual float32 max/127 row scale.
    """
    if (bits, dynamic_vector) not in ((8, False), (4, False), (4, True)):
        raise ValueError("unsupported bounded numeric contract")
    elements = 2 if dynamic_vector else 1
    inputs = 2 * bits * elements
    indices = np.arange(1 << inputs, dtype=np.uint64)
    domain = 1 << bits
    raw = np.column_stack([(indices >> (bits * i)) & (domain - 1) for i in range(2 * elements)])
    values = raw.astype(np.int32)
    values[values >= domain // 2] -= domain
    scale = np.float32(1 / (domain // 2))
    gates = values[:, 1::2].astype(np.float32) * scale
    ups = values[:, ::2].astype(np.float32) * scale
    products = runtime_products(gates, ups)
    quantized = quantize_activation_per_row(
        products, bits=8, scales=None if dynamic_vector else scale
    )
    codes = quantized.values.view(np.uint8).astype(np.uint64)
    encoded = sum(codes[:, i] << (8 * i) for i in range(elements))
    if dynamic_vector:
        encoded |= quantized.scales.view(np.uint32).astype(np.uint64) << (8 * elements)
    oracle = np.zeros(len(encoded), np.uint64)
    product_mismatches = 0
    for index, row in enumerate(values):
        scalar = [
            scalar_product(int(row[2 * i + 1]), int(row[2 * i]), bits) for i in range(elements)
        ]
        product_mismatches += sum(
            struct.pack("<f", value) != struct.pack("<f", products[index, i])
            for i, value in enumerate(scalar)
        )
        maximum = max(abs(value) for value in scalar)
        row_scale = (
            np.float32(maximum / np.float32(127))
            if dynamic_vector and maximum > 0
            else (np.float32(1) if dynamic_vector else scale)
        )
        word = sum(
            (scalar_quantize(value, row_scale) & 255) << (8 * i) for i, value in enumerate(scalar)
        )
        if dynamic_vector:
            word |= struct.unpack("<I", struct.pack("<f", row_scale))[0] << (8 * elements)
        oracle[index] = word
    mismatches = int(np.count_nonzero(oracle != encoded))
    if mismatches or product_mismatches:
        raise AssertionError(
            f"independent scalar oracle mismatch: products={product_mismatches}, codes/scales={mismatches}"
        )
    return encoded, {
        "signed_input_bits_each": bits,
        "elements": elements,
        "truth_assignments": len(encoded),
        "input_scale": float(scale),
        "output_scale": "float32(max(abs(products))/127), zero row => 1"
        if dynamic_vector
        else float(scale),
        "output_bits": 8 * elements + (32 if dynamic_vector else 0),
        "includes_minimum_twos_complement_input_extra_to_symmetric_quantizer": True,
        "scalar_float32_product_comparisons": len(encoded) * elements,
        "independent_product_bit_mismatches": product_mismatches,
        "independent_code_and_scale_mismatches": mismatches,
        "truth_table_sha256": hashlib.sha256(encoded.astype("<u8").tobytes()).hexdigest(),
        "truth_table_storage_bytes": encoded.nbytes,
        "distinct_outputs": len(np.unique(encoded)),
        "distinct_scale_words": len(np.unique(encoded >> 16)) if dynamic_vector else 1,
        "exact_qwen_numeric_domain": False,
    }


def material_cost(summary: dict) -> dict:
    """Specified fresh half-gate payload projection, not measured transport.

    An exposed selection bit leaves only 127 hidden bits in a 128-bit label.
    Minimum byte-packed label with >=128 hidden bits is 17 bytes (129 bits).
    A reviewed backend for that width is missing. Also show 256-bit alternative.
    """
    inputs, outputs, ands = (summary[k] for k in ("input_bits", "output_bits", "and_non_xor_gates"))
    return {
        "security_implementation": False,
        "required_hidden_label_entropy_bits": 128,
        "minimum_byte_packed_label_bytes_with_select_bit": 17,
        "half_gate_fresh_ciphertext_body_bytes": 34 * ands,
        "standard_128_total_bit_label_127_hidden_bit_body_comparator_bytes": 32 * ands,
        "256_total_bit_label_body_comparator_bytes": 64 * ands,
        "selected_input_label_payload_bytes": 17 * inputs,
        "input_wire_encoding_and_global_delta_storage_bytes": 17 * (inputs + 1),
        "output_labels_to_authorized_recipient_payload_bytes": 17 * outputs,
        "plain_output_decode_bits_only_to_authorized_recipient": outputs,
        "output_decode_packed_bytes": (outputs + 7) // 8,
        "fresh_garbler_hash_calls": 4 * ands,
        "fresh_evaluator_hash_calls": 2 * ands,
        "ot_transfers_if_all_function_inputs_evaluator_owned": inputs,
        "ot_transfers_if_half_function_inputs_evaluator_owned": inputs // 2,
        "ot_extension_and_base_ot_body_bytes": None,
        "resident_share_reconstruction_and_masked_output_conversion_bytes": None,
        "constant_label_setup_bytes": None,
        "backend_execution_seconds": None,
        "fresh_circuit_required_each_invocation": True,
        "public_structure_reusable_but_ciphertexts_labels_delta_not_reused": True,
    }


def screen_case(bits: int, *, dynamic_vector: bool = False) -> dict:
    started = time.perf_counter()
    table, contract = function_table(bits=bits, dynamic_vector=dynamic_vector)
    oracle_seconds = time.perf_counter() - started
    inputs = 2 * bits * contract["elements"]
    # Index order stores up then gate for each element. Grouped order visits each
    # gate MSB..LSB, then its up MSB..LSB, then the next element.
    grouped = [
        bit
        for element in range(contract["elements"])
        for base in (bits, 0)
        for bit in range(2 * bits * element + base + bits - 1, 2 * bits * element + base - 1, -1)
    ]
    interleaved = [
        2 * bits * element + base + bit
        for element in range(contract["elements"])
        for bit in reversed(range(bits))
        for base in (bits, 0)
    ]
    results = []
    for name, shared, order in (
        ("generic_unshared_grouped", False, grouped),
        ("shared_grouped", True, grouped),
        ("shared_interleaved", True, interleaved),
    ):
        tracemalloc.start()
        started = time.perf_counter()
        circuit = synthesize(table, contract["output_bits"], order, shared=shared)
        synth_seconds = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        started = time.perf_counter()
        evaluated = circuit.evaluate_all()
        evaluation_seconds = time.perf_counter() - started
        mismatches = int(np.count_nonzero(evaluated != table))
        if mismatches:
            raise AssertionError(f"{name} exhaustive circuit mismatch: {mismatches}")
        summary = circuit.summary()
        results.append(
            {
                "variant": name,
                "variable_order_root_first": order,
                **summary,
                "synthesis_seconds": synth_seconds,
                "synthesis_tracemalloc_peak_bytes": peak,
                "clear_exhaustive_evaluation_seconds": evaluation_seconds,
                "exhaustive_bit_evaluator_mismatches": mismatches,
                "material_projection": material_cost(summary),
            }
        )
    assert inputs == results[0]["input_bits"]
    return {
        "contract": contract,
        "runtime_table_and_scalar_oracle_seconds": oracle_seconds,
        "circuits": results,
        "same_order_shared_and_ratio": results[0]["and_non_xor_gates"]
        / results[1]["and_non_xor_gates"],
    }


def rounding_edges() -> dict:
    """Independent hand-derived ties, adjacent floats, saturation, and scale zero."""
    count = 0
    for lower in range(-130, 130):
        tie = np.float32((lower + 0.5) / 128)
        even = lower if lower % 2 == 0 else lower + 1
        for value, expected in (
            (np.nextafter(tie, np.float32(-np.inf)), lower),
            (tie, even),
            (np.nextafter(tie, np.float32(np.inf)), lower + 1),
        ):
            expected = max(-127, min(127, expected))
            actual = int(
                quantize_activation_per_row(np.asarray([value]), bits=8, scales=1 / 128).values[
                    0, 0
                ]
            )
            assert scalar_quantize(value, np.float32(1 / 128)) == actual == expected
            count += 1
    zero = quantize_activation_per_row(np.zeros((1, 2), np.float32), bits=8)
    assert zero.scales[0] == 1 and np.all(zero.values == 0)
    # Actual float32 dynamic rounding invalidates ideal real max cancellation.
    row = np.asarray([[1, np.float32(4.5 / 127)]], np.float32)
    original = quantize_activation_per_row(row, bits=8)
    divided = quantize_activation_per_row(row / np.float32(3), bits=8)
    assert original.values.tolist() == [[127, 4]]
    assert divided.values.tolist() == [[127, 5]]
    return {
        "fixed_scale_tie_neighbor_saturation_cases": count,
        "dynamic_zero_row_scale": 1.0,
        "dynamic_rounding_counterexample": {
            "row": row[0].tolist(),
            "divisor": 3,
            "before": original.values[0].tolist(),
            "after": divided.values[0].tolist(),
        },
    }


def pinned_region_prices(fixed: dict) -> list[dict]:
    # Reuse inspected config-only semantic inventory and its digest checks. No
    # checkpoint weights or upstream implementation code are imported/downloaded.
    from probe_projective_feasibility import pinned_inventory
    from probe_region_contract_cost import run as pinned_region

    control = json.loads(
        (ROOT / "docs/evidence/latent-response-network-qwen25-2026-09-28.json").read_text()
    )
    reports = []
    for outputs in (8, 32):
        inventory = pinned_inventory(outputs)
        region = pinned_region(outputs)
        assert region["plan_digest"] == inventory["plan_digest"]
        if inventory["config_sha256"] != CONFIG_SHA256:
            raise ValueError("official config hash changed")
        count = inventory["response_tensor_counts"]["silu_elements"]
        prepared = next(row for row in control["cohorts"] if row["output_tokens"] == outputs)[
            "prepared_control"
        ]
        prices = []
        for circuit in fixed["circuits"]:
            cost = circuit["material_projection"]
            body = count * cost["half_gate_fresh_ciphertext_body_bytes"]
            prices.append(
                {
                    "variant": circuit["variant"],
                    "hypothetical_fixed_domain_element_invocations": count,
                    "non_xor_gates": count * circuit["and_non_xor_gates"],
                    "fresh_garbler_to_evaluator_ciphertext_body_bytes": body,
                    "selected_input_label_payload_bytes_if_all_reencoded": count
                    * cost["selected_input_label_payload_bytes"],
                    "output_label_payload_bytes_if_all_returned": count
                    * cost["output_labels_to_authorized_recipient_payload_bytes"],
                    "ot_count_if_all_function_inputs_evaluator_owned": count
                    * cost["ot_transfers_if_all_function_inputs_evaluator_owned"],
                    "ot_body_bytes": None,
                    "garbler_hash_calls": count * cost["fresh_garbler_hash_calls"],
                    "evaluator_hash_calls": count * cost["fresh_evaluator_hash_calls"],
                    "ciphertext_only_over_full_prepared_covered_body_ratio": body
                    / prepared["covered_all_link_body_bytes"],
                    "ciphertext_only_over_full_prepared_online_body_ratio": body
                    / prepared["online_all_link_body_bytes"],
                    "complete_protected_region_cost_bytes": None,
                    "actual_qwen_equivalent": False,
                }
            )
        reports.append(
            {
                "output_tokens": outputs,
                "input_tokens": 39,
                "executed_rows": inventory["executed_rows"],
                "semantic_layers": inventory["semantic_layers"],
                "hidden": inventory["hidden"],
                "intermediate": inventory["intermediate"],
                "checkpoint": inventory["checkpoint"],
                "config_sha256": inventory["config_sha256"],
                "body_fingerprint_from_control": inventory["body_fingerprint_from_control"],
                "plan_digest": inventory["plan_digest"],
                "schedule_digest": inventory["schedule_digest"],
                "composition_digest": inventory["composition_digest"],
                "whole_mlp_semantics": [
                    "rms_norm",
                    "gate/up linear",
                    "silu",
                    "multiply",
                    "dynamic per-row quantization",
                    "down linear",
                    "residual_add",
                ],
                "mlp_silu_and_product_elements_each": count,
                "mlp_rows_requiring_dynamic_scale": count // inventory["intermediate"],
                "mlp_projection_integer_macs_per_worker": region["client_attention_remote_mlp"][
                    "remote_mlp_projection_integer_macs_per_worker"
                ],
                "one_fresh_and_per_element_ciphertext_only_comparator_bytes": 34 * count,
                "prepared_control": prepared,
                "quantizer_circuit_prices": prices,
                "all_link_tenfold_budget_bytes_per_gate_product_element": prepared[
                    "tenfold_covered_budget_bytes"
                ]
                / count,
                "current_prepared_nonlinear_and_quantizer_location": "client-local; no garbled gates to eliminate",
                "demonstrated_stage_crossings_removed": 0,
                "missing_required_operations": [
                    "actual float32 gate/up projection outputs are not fixed-public-scale i8 inputs",
                    "full width-4864 private maximum and float32 max/127 scale",
                    "SiLU float32 exponential portability/certificates beyond measured public domain",
                    "rounded rescaling, linear accumulators, learned weight scales and bias",
                    "arithmetic-share to Boolean-label conversions and output resharing for down projection",
                    "OT extension/base OT, reviewed >=128 hidden-bit label/hash backend and one-use issuance",
                    "RMSNorm, residual, attention, rotary and protected KV outside this fused function",
                    "client final selection/head and token feedback; private selection only if placement changes",
                    "dealer-to-each-worker/setup/cold checkpoint distribution and complete directed-link framing",
                    "local garbler/evaluator compute, worker linear compute and full wire",
                ],
                "complete_encoded_qwen_region_available": False,
                "whole_response_saving_demonstrated": False,
            }
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-pinned-config", action="store_true")
    parser.add_argument(
        "--source-pdf", type=Path, help="optional local primary half-gates PDF hash"
    )
    args = parser.parse_args()
    fixed = screen_case(8)
    result = {
        "schema": "pllm.quantizer_circuit_screen.v1",
        "status": "clear finite-domain exact synthesis; no security implementation or runtime admission",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "limits": {
            "max_input_bits": MAX_INPUT_BITS,
            "max_wires": MAX_WIRES,
            "max_and_gates": MAX_ANDS,
        },
        "primary_source": {
            "url": "https://eprint.iacr.org/2014/756.pdf",
            "sections": "Table 1, Figure 2, sections 3.2 and 4; select-bit caveat page 19",
            "pdf_sha256": hashlib.sha256(args.source_pdf.read_bytes()).hexdigest()
            if args.source_pdf
            else None,
        },
        "local_oracle_source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                ROOT / "python/pllm/runtime/semantic_executor.py",
                ROOT / "python/pllm/runtime/quantization.py",
            )
        },
        "fixed_public_scale": fixed,
        "dynamic_scale_two_element_vector": screen_case(4, dynamic_vector=True),
        "independent_rounding_edges": rounding_edges(),
        "security_implementation": False,
        "runtime_activated": False,
    }
    if args.with_pinned_config:
        result["pinned_semantic_qwen_cohorts"] = pinned_region_prices(fixed)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
