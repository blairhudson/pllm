"""Cost-gate public sparse integer predictors on a pinned compiled decoder.

The oracle computes normal clear W8A8 stage products, then tries a client-side
predictor retaining 10/50/90% of public weight columns. Per-stage modulus sizes
are derived both from sampled residuals and from the independent, worst-case
W8A8 exact-lift certificate. Only the latter can authorize a private request.
No prompt, token ID, input, logit, stage weight or prediction is written out.
This is an offline diagnostic, not a public protocol or live Experiment.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from pllm import Model, lower_model
from pllm.model_loader import resolve_model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.predictive_modular_reference import (
    minimum_certified_ring_bits,
    sparse_predictor_options,
)
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine

_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
_PROMPT = "Explain why neither server can see the prompt."
_BODY = "5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974"
_CONTROL = {8: (116_843_966, 73_831_744), 32: (178_970_558, 113_545_024)}
_FRACTIONS = (10, 50, 90)


def _packed(elements: int, bits: int) -> int:
    return (elements * bits + 7) // 8


def run(*, output_tokens: int = 32) -> dict[str, Any]:
    if type(output_tokens) is not int or output_tokens not in _CONTROL:
        raise ValueError("predictor probe only admits locked 39+8 and 39+32 cohorts")
    source = resolve_model(Model.hf(_MODEL, revision=_REVISION))
    if source.path is None or source.checkpoint_digest is None:
        raise RuntimeError("the pinned cached public checkpoint must resolve")
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=4)
    asyncio.run(engine.load(source.manifest))
    model = engine.models[source.manifest.id]
    if model.manifest.metadata.get("body_fingerprint") != _BODY:
        raise RuntimeError("the cached W8A8 body differs from the matched control")
    config = (source.path / "config.json").read_bytes()
    plan = lower_model(config, batch=1, max_input_tokens=39, max_new_tokens=output_tokens)
    composition = MaskedLinearCpu(
        Model.hf(_MODEL, revision=_REVISION),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    schedule = plan.runtime_schedule(composition)
    bundle = ClientBundle.unpack(engine.client_bundle(source.manifest.id))
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    if (
        not schedule.complete
        or schedule.protected_execution
        or schedule.digest != compiled.runtime_schedule_digest
    ):
        raise RuntimeError("real checkpoint must bind a complete compiled baseline")

    options = {
        stage_id: sparse_predictor_options(stage.weight.values)
        for stage_id, stage in model.stages.items()
        if stage.spec.role not in {"token_lookup", "lm_head"}
    }
    stage_rows: dict[str, dict[str, Any]] = {}

    def remote(stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = model.stages[stage_id]
        qinput = quantize_activation_per_row(activation, bits=stage.spec.activation_bits)
        clear = np.asarray(
            engine._public_stage_matrix(source.manifest.id, stage, qinput.rows).clear(
                qinput.values
            ),
            dtype=np.int64,
        ).reshape(qinput.rows, stage.spec.out_features)
        current = stage_rows.setdefault(
            stage_id,
            {
                "semantic_role": stage.spec.role,
                "layer": stage.spec.layer_index,
                "input_width": stage.spec.in_features,
                "output_width": stage.spec.out_features,
                "original_exact_wire_bits": stage.seeded_profile.wire_bits,
                "executed_rows": 0,
            },
        )
        current["executed_rows"] += qinput.rows
        signed = qinput.values.astype(np.int64)
        for fraction, option in options[stage_id].items():
            selected = option["selected"]
            predictor = signed[:, selected] @ stage.weight.values[:, selected].astype(np.int64).T
            option["observed_worst_error"] = max(
                option["observed_worst_error"], int(np.max(np.abs(clear - predictor)))
            )
            option["client_integer_macs"] += qinput.rows * len(selected) * stage.spec.out_features
        out = dequantize_matmul(
            clear,
            qinput.scales,
            stage.weight.scales,
            output_shape=qinput.original_shape[:-1] + (stage.spec.out_features,),
        )
        if stage.bias is not None:
            out = out + stage.bias
        return np.ascontiguousarray(out, dtype=np.float32)

    runtime = compiled.runtime(remote)
    rendered = bundle.render_prompt([{"role": "user", "content": _PROMPT}])
    tokens = tuple(runtime.encode_prompt(rendered))
    if len(tokens) != 39:
        raise RuntimeError("the pinned prompt changed its tokenizer workload")
    session = compiled.session(remote)
    session.prefill_ids(tokens)
    selected: list[int] = []
    for index in range(output_tokens):
        selected.append(session.select_next())
        if index + 1 < output_tokens:
            session.decode_selected()
    if not session.complete or len(stage_rows) != 96:
        raise RuntimeError("predictor probe did not execute the complete decoder")
    token_digest = hashlib.sha256(
        b"pllm.predictive_modular_probe.tokens.v1\0"
        + np.asarray((*tokens, *selected), dtype="<u4").tobytes()
    ).hexdigest()

    aggregates: dict[int, dict[str, Any]] = {}
    stage_summaries: dict[str, Any] = {}
    uniform_two_bit_arithmetic_bytes = 0
    for fraction in _FRACTIONS:
        aggregates[fraction] = {
            "fraction_of_columns": fraction,
            "public_i8_predictor_bytes_including_u16_indices": 0,
            "client_integer_macs": 0,
            "remote_integer_macs": 0,
            "optimistic_certified_all_link_arithmetic_bytes": 0,
            "optimistic_certified_online_arithmetic_bytes": 0,
            "optimistic_observed_all_link_arithmetic_bytes": 0,
            "certified_minimum_ring_bits": 32,
            "certified_maximum_ring_bits": 0,
            "observed_maximum_ring_bits": 0,
            "stages_certified_at_two_bits": 0,
            "stages_certified_at_eight_bits_or_less": 0,
        }
    for stage_id, row in sorted(stage_rows.items()):
        stages = options[stage_id]
        rows = row["executed_rows"]
        if rows != 39 + output_tokens - 1:
            raise RuntimeError("remote stage row count differs from the compiled workload")
        inp, out = row["input_width"], row["output_width"]
        uniform_two_bit_arithmetic_bytes += _packed(rows * inp, 2) + 2 * _packed(rows * out, 2)
        row["predictors"] = {}
        for fraction, option in stages.items():
            cert_bits = option["certified_ring_bits"]
            obs_bits = minimum_certified_ring_bits(option["observed_worst_error"])
            if cert_bits > 32:
                raise RuntimeError("whole-input certificate needs more than a 32-bit ring")
            online = _packed(rows * inp, cert_bits) + _packed(rows * out, cert_bits)
            offline = _packed(rows * out, cert_bits)
            aggregate = aggregates[fraction]
            aggregate["public_i8_predictor_bytes_including_u16_indices"] += option[
                "public_i8_predictor_bytes"
            ]
            aggregate["client_integer_macs"] += option["client_integer_macs"]
            aggregate["remote_integer_macs"] += rows * inp * out
            aggregate["optimistic_certified_all_link_arithmetic_bytes"] += online + offline
            aggregate["optimistic_certified_online_arithmetic_bytes"] += online
            aggregate["optimistic_observed_all_link_arithmetic_bytes"] += _packed(
                rows * inp, obs_bits
            ) + 2 * _packed(rows * out, obs_bits)
            aggregate["certified_minimum_ring_bits"] = min(
                aggregate["certified_minimum_ring_bits"], cert_bits
            )
            aggregate["certified_maximum_ring_bits"] = max(
                aggregate["certified_maximum_ring_bits"], cert_bits
            )
            aggregate["observed_maximum_ring_bits"] = max(
                aggregate["observed_maximum_ring_bits"], obs_bits
            )
            aggregate["stages_certified_at_two_bits"] += cert_bits == 2
            aggregate["stages_certified_at_eight_bits_or_less"] += cert_bits <= 8
            row["predictors"][str(fraction)] = {
                "selected_columns": len(option["selected"]),
                "public_worst_error_accumulator_units": option["public_worst_error"],
                "observed_worst_error_accumulator_units": option["observed_worst_error"],
                "certified_ring_bits": cert_bits,
                "observed_ring_bits_not_an_admission": obs_bits,
            }
        stage_summaries[stage_id] = row
    by_role: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in stage_summaries.values():
        role = row["semantic_role"]
        for fraction in _FRACTIONS:
            by_role[role][f"{fraction}_max_certificate_bits"] = max(
                by_role[role][f"{fraction}_max_certificate_bits"],
                row["predictors"][str(fraction)]["certified_ring_bits"],
            )
    return {
        "schema": "pllm.predictive_modular_cost_gate.v1",
        "checkpoint_digest": source.checkpoint_digest,
        "source_lock_digest": source.source_lock_digest,
        "model_body_fingerprint": _BODY,
        "semantic_plan_digest": plan.digest,
        "runtime_schedule_digest": schedule.digest,
        "input_tokens": 39,
        "output_tokens": output_tokens,
        "token_cohort_digest": token_digest,
        "remote_stage_count": len(stage_rows),
        "covered_prepared_all_link_body_bytes": _CONTROL[output_tokens][0],
        "covered_prepared_online_body_bytes": _CONTROL[output_tokens][1],
        "tenfold_all_link_budget_bytes": _CONTROL[output_tokens][0] // 10,
        "tenfold_online_budget_bytes": _CONTROL[output_tokens][1] // 10,
        "unattained_uniform_two_bit_arithmetic_floor_bytes": uniform_two_bit_arithmetic_bytes,
        "optimistic_projections_exclude": [
            "tickets and all framing/control",
            "source and predictor distribution and persistent disk",
            "private-input error-dependent fallback",
            "full wire, client predictor runtime and cryptographic review",
        ],
        "by_role": {role: dict(values) for role, values in sorted(by_role.items())},
        "predictors": {str(k): v for k, v in sorted(aggregates.items())},
        "stage_summaries_sha256": hashlib.sha256(
            json.dumps(stage_summaries, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "stages": stage_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-output-tokens", type=int, choices=_CONTROL, default=32)
    parser.add_argument("--summary", action="store_true", help="omit 96 per-stage diagnostics")
    args = parser.parse_args()
    result = run(output_tokens=args.max_output_tokens)
    if args.summary:
        del result["stages"]
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
