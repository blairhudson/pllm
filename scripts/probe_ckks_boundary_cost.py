"""Cost-gate CKKS-to-protected-nonlinear boundaries on a compiled Qwen plan.

This measures real serialized TenSEAL ciphertexts on public synthetic values.
It does not implement BLB's fused MatMul, secure CKKS/MPC conversion, or a
protected decoder. Counts are an optimistic extrapolation for this backend:
only one *outbound* ciphertext for each SiLU gate is charged.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from pllm import Model, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow


_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
_SLOTS = 4096
_POLY_DEGREE = 8192
_CONTEXT_COEFF_BITS = [60, 40, 40, 60]
_BASELINES = {
    8: {"online": 73_831_744, "covered": 116_843_966},
    32: {"online": 113_545_024, "covered": 178_970_558},
}


def _gate_shapes(plan: Any, schedule: Any) -> list[tuple[int, int]]:
    graph = plan.to_dict()
    steps = schedule.to_dict()
    result: list[tuple[int, int]] = []
    for phase in ("prefill", "decode"):
        operators = {row["id"]: row for row in graph[phase]["operations"]}
        placed = {
            operation_id: step
            for step in steps[phase]["steps"]
            for operation_id in step["operation_ids"]
        }
        gates = [row for row in operators.values() if row["operator"] == "silu"]
        layers = {row["layer"] for row in gates}
        if len(gates) != len(layers) or len(gates) != 24:
            raise ValueError("expected one semantic SiLU gate per Qwen layer")
        for gate in gates:
            sources = gate["inputs"]
            if len(sources) != 1 or operators[sources[0]]["operator"] != "linear":
                raise ValueError("SiLU gate is not fed by one linear projection")
            source_step = placed[sources[0]]
            if (
                source_step["executor"] != "remote_stage"
                or placed[gate["id"]]["executor"] != "client_local"
            ):
                raise ValueError("compiled SiLU placement changed")
            shape = gate["output_shape"]
            if len(shape) != 3 or shape[0] != 1 or shape[1] < 1 or shape[2] < 1:
                raise ValueError("unsupported SiLU gate shape")
            result.append((shape[1], shape[2]))
    return result


def _ckks_samples(widths: set[int]) -> dict[str, Any]:
    import tenseal as ts

    started = time.process_time()
    client = ts.context(
        ts.SCHEME_TYPE.CKKS,
        _POLY_DEGREE,
        coeff_mod_bit_sizes=_CONTEXT_COEFF_BITS,
        n_threads=1,
    )
    client.global_scale = 2**40
    public_context = client.serialize(
        save_public_key=True,
        save_secret_key=False,
        save_galois_keys=False,
        save_relin_keys=False,
    )
    provider = ts.context_from(public_context, n_threads=1)
    if provider.has_secret_key():
        raise RuntimeError("provider CKKS context contains a secret key")
    setup_cpu_seconds = time.process_time() - started
    samples = {}
    rng = np.random.default_rng(3509)
    for width in sorted(widths):
        if width < 1 or width > _SLOTS:
            raise ValueError("ciphertext sample exceeds checked CKKS slot bound")
        values = rng.uniform(-0.5, 0.5, size=width)
        started = time.process_time()
        request = ts.ckks_vector(client, values.tolist()).serialize()
        encrypted = ts.ckks_vector_from(provider, request)
        response = (encrypted * 0.75 + 0.125).serialize()
        actual = np.asarray(ts.ckks_vector_from(client, response).decrypt())
        cpu_seconds = time.process_time() - started
        error = float(np.max(np.abs(actual - (0.75 * values + 0.125))))
        if error > 1e-5:
            raise RuntimeError("bounded public affine CKKS oracle exceeds numeric tolerance")
        samples[str(width)] = {
            "input_ciphertext_bytes": len(request),
            "outbound_ciphertext_bytes": len(response),
            "affine_max_absolute_error": error,
            "in_process_cpu_seconds": cpu_seconds,
        }
    return {
        "slots_per_ciphertext": _SLOTS,
        "poly_modulus_degree": _POLY_DEGREE,
        "coeff_mod_bit_sizes": _CONTEXT_COEFF_BITS,
        "public_context_bytes": len(public_context),
        "provider_has_secret_key": provider.has_secret_key(),
        "setup_cpu_seconds": setup_cpu_seconds,
        "samples": samples,
    }


def _project(
    gates: list[tuple[int, int]], decode_steps: int, bytes_per_ciphertext: int
) -> dict[str, int]:
    prefill = sum(math.ceil(rows * width / _SLOTS) for rows, width in gates[:24])
    decode = sum(math.ceil(rows * width / _SLOTS) for rows, width in gates[24:]) * decode_steps
    return {
        "prefill_gate_ciphertexts": prefill,
        "decode_gate_ciphertexts": decode,
        "one_way_gate_body_bytes": (prefill + decode) * bytes_per_ciphertext,
    }


def run() -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    source = Path(
        hf_hub_download(
            _MODEL,
            "config.json",
            revision=_REVISION,
            local_files_only=True,
        )
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    composition = MaskedLinearCpu(
        Model.hf(_MODEL, revision=_REVISION),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    plans = {
        output: lower_model(config, batch=1, max_input_tokens=39, max_new_tokens=output)
        for output in _BASELINES
    }
    schedules = {output: plan.runtime_schedule(composition) for output, plan in plans.items()}
    gates = {output: _gate_shapes(plans[output], schedules[output]) for output in plans}
    if {shape for rows in gates.values() for shape in rows[24:]} != {(1, 4864)}:
        raise ValueError("pinned decode SiLU contract changed")
    widths = {_SLOTS, 4864 % _SLOTS, (39 * 4864) % _SLOTS}
    ckks = _ckks_samples(widths)
    # Give the hypothetical converter the smallest observed ciphertext, even
    # for larger gate chunks. This favors fusion and is not a lower bound for
    # other CKKS implementations or ciphertext compression schemes.
    favorable_bytes = min(row["outbound_ciphertext_bytes"] for row in ckks["samples"].values())
    cohorts = {}
    for output, plan in plans.items():
        schedule = schedules[output]
        projected = _project(gates[output], output - 1, favorable_bytes)
        baseline = _BASELINES[output]
        cohorts[str(output)] = {
            "plan_digest": plan.digest,
            "schedule_digest": schedule.digest,
            "composition_digest": schedule.composition_digest,
            "consumed_input_and_feedback_rows": 39 + output - 1,
            "remote_gate_to_client_silu_boundaries_per_phase": {"prefill": 24, "decode": 24},
            **projected,
            "measured_prepared_online_body_bytes": baseline["online"],
            "measured_prepared_covered_body_bytes": baseline["covered"],
            "gate_only_exceeds_prepared_covered": projected["one_way_gate_body_bytes"]
            > baseline["covered"],
            "gate_only_exceeds_tenfold_budget": projected["one_way_gate_body_bytes"]
            > baseline["covered"] // 10,
        }
    return {
        "schema": "pllm.he_fusion_boundary_gate.v1",
        "scope": "isolated CKKS public-affine sample and optimistic one-way SiLU output ciphertext count; no BLB fusion or secure CKKS/MPC conversion",
        "source": {"model": _MODEL, "revision": _REVISION, "input_tokens": 39, "precision": "W8A8"},
        "ckks": ckks,
        "optimistic_ciphertext_bytes_per_gate_chunk": favorable_bytes,
        "cohorts": cohorts,
        "decoder_executable": False,
        "measured_full_wire": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
