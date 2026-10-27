"""Tiny synthetic research quote; refuses unadmitted HE, never fabricates timing.

Run later, when execution is authorized:
    uv run python benchmarks/research/encrypted_phase_index.py
No model/checkpoint/backend reads, trained inference, or encrypted claims.
The printed quote is static only, not backend qualification. Policy ceilings
are unchanged; hard backend execution limits are still unimplemented. Neither
dictionary-entry liveness nor unproved IR byte estimates bound backend memory.
"""
from dataclasses import asdict
import json

from pllm.runtime.encrypted_phase_index import (
    PhaseIndexRefused,
    PhaseIndexSpec,
    compile_phase_index,
)


def main():
    circuit = compile_phase_index(PhaseIndexSpec(q=17, n=2), 1, (1, 3))
    result = {
        "experiment": "rosetta-extra-bfv-key-boolean-phase-index-candidate",
        "synthetic": True,
        "q": 17,
        "n": 2,
        "quote": asdict(circuit.quote),
        "caps": {"bytes": 8 << 30, "threads": 1, "seconds": 30},
        "hard_backend_limits_implemented": False,
        "live_bit_planes_meaning": "interpreter dictionary entries only, not backend peak",
        "ir_heap_estimate_proven_bound": False,
        "independent_binary_source_assumption": True,
        "bfv_profile_and_public_owner_available": False,
        "security_claim": False,
        "backend_executed": False,
    }
    try:
        circuit.quote.require_admission()
    except PhaseIndexRefused as exc:
        result.update(status="refused", reason=str(exc))
    else:
        raise RuntimeError("admission changed; encrypted benchmark harness still required")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
