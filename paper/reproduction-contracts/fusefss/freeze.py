"""Emit reference vectors to stdout; never silently update frozen artifacts.

Semantic expected outputs come from the independent direct mathematical oracle.
Compiler transcript hashes are explicitly regression identities, not independent
correctness evidence. Exhaustive and negative-control tests supply that evidence.
"""

from dataclasses import asdict
from fractions import Fraction
from hashlib import sha256
import json

import math_oracle as oracle
import reference as ref


def vectors():
    table = oracle.gelu_table()
    descriptor = ref.gelu_descriptor(table)
    modulus = 1 << 50
    samples = [-(1 << 49), -(1 << 37), -16385, -16384, -16321, -16320, -16257, -16256,
               -4097, -4096, -65, -64, -63, -1, 0, 1, 31, 63, 64, 65, 4096,
               16255, 16256, 16319, 16320, 16384, (1 << 37) - 1, (1 << 49) - 1]
    masks = [0, 1, 63, 64, (1 << 49), modulus - 64, modulus - 1]
    fixtures = []
    for numerator in [-7, -5, -3, -1, 1, 3, 5, 7]:
        value = Fraction(numerator, 8192)
        fixtures.append({"numerator": numerator, "denominator": 8192,
                         "expected_signed_q12": round(value * 4096)})
    return {
        "schema": "pllm.fusefss.bounded-reference-vectors.v1",
        "scope": "clear scalar specification only; no native, FSS, model, SDK, security or performance evidence",
        "profile": ref.PROFILE,
        "source_lock_sha256": descriptor.source_lock_sha256,
        "table_sha256_u64le": ref.table_digest(table),
        "canonical_descriptor_sha256": descriptor.digest(),
        "canonical_descriptor_bytes": len(descriptor.canonical_bytes()),
        "gelu_shape": asdict(ref.admission(descriptor)),
        "external_encoding_ties": fixtures,
        "gelu_inputs_signed_q12": samples,
        "gelu_expected_signed_q12": [oracle.gelu_signed(x, table) for x in samples],
        "full_ring_tail_vectors": "Mathematical extension only outside signed38; not evidence for admitting a narrowed source CPU input",
        "masks": masks,
        "compiled_trace_sha256_regression_only": [sha256(ref.compile_descriptor(descriptor, r).canonical_bytes()).hexdigest() for r in masks],
        "partition_0_3_8": [
            {"mask": 0, "boundaries": [0, 1, 3, 8], "ordinals": [0, 0, 1]},
            {"mask": 2, "boundaries": [0, 2, 5, 8], "ordinals": [1, 0, 1]},
            {"mask": 5, "boundaries": [0, 1, 5, 8], "ordinals": [1, 1, 0]},
        ],
        "negative_controls": {
            "early_floor_before_abs": {"X": -1, "correct": -31, "abs_before_floor": 0},
            "q0_slope_needs_no_final_rescale": {"X": 64, "correct": 33, "extra_ars12": 0},
            "original_x_horner": {"bits": 3, "x": 2, "mask": 4, "constant": 1, "linear": 3, "correct": 7, "using_xhat": 3},
            "xor_not_arithmetic_add": {"bits": 3, "xhat": 0, "mask": 3, "beta": 2, "correct": 0, "arithmetic_sum": 2},
            "reduce_low_mask": {"bits": 4, "low_bits": 2, "x": 0, "mask": 6, "gamma": 3, "correct": 1, "unreduced_mask": 0},
            "wrap_before_rescale": {"bits": 50, "a": 1 << 25, "b": 1 << 25, "drop": 12, "ring_then_ars": 0, "unwrapped_floor": 1 << 38},
            "cannot_commute_sum_and_floor": {"a": 63, "b": 1, "drop": 6, "floor_sum": 1, "sum_floors": 0},
            "tr_is_not_same_width_ars": {"bits": 50, "word": modulus - 1, "drop": 6, "tr_A44": (1 << 44) - 1, "ars_A50": modulus - 1},
            "source_effective_width_needs_provenance": {"X": (1 << 37) + 64, "full_ring_map": (1 << 37) + 64, "forced_m38_selector": 0},
        },
    }


if __name__ == "__main__":
    print(json.dumps(vectors(), indent=2, sort_keys=True))
