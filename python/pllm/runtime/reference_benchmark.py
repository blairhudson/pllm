"""Bounded local compiled-decoder quality comparison against a pinned FP32 reference."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import platform
from collections.abc import Sequence
from typing import Any

import numpy as np

from pllm.configuration import Experiment
from pllm.metrics import ReferenceAgreement, measure_reference_agreement
from pllm.model_loader import resolve_model
from pllm.profiles import resolve_runtime_composition
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.public_equalization import equalize_activation
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine


SCHEMA = "pllm.reference_quality_benchmark.v1"
MAX_CHECKPOINT_BYTES = 12 * 1024**3
MAX_LOGIT_WORKING_SET_BYTES = 512 * 1024**2


class ReferenceBenchmarkError(ValueError):
    """The reference workload or candidate is outside the bounded quality lane."""


def _digest(domain: bytes, value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(domain + canonical.encode("utf-8")).hexdigest()


def _candidate_remote(engine: MaskedTransformerEngine, model_id: str):
    stages = engine.models[model_id].stages

    def remote(stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = stages[stage_id]
        if stage.input_equalization is not None:
            activation = equalize_activation(activation, stage.input_equalization)
        q_activation = quantize_activation_per_row(activation, bits=stage.spec.activation_bits)
        integer = engine._public_stage_matrix(model_id, stage, q_activation.rows).clear(q_activation.values)
        output = dequantize_matmul(
            integer,
            q_activation.scales,
            stage.weight.scales,
            output_shape=q_activation.original_shape[:-1] + (stage.spec.out_features,),
        )
        if stage.bias is not None:
            output = output + stage.bias
        return np.ascontiguousarray(output, dtype=np.float32)

    return remote


def _prompts(value: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not 1 <= len(value) <= 32:
        raise ReferenceBenchmarkError("quality benchmark needs 1 to 32 prompts")
    if any(
        type(prompt) is not str or not prompt.strip() or len(prompt.encode("utf-8")) > 4096
        for prompt in value
    ):
        raise ReferenceBenchmarkError(
            "quality prompts must be nonempty and at most 4096 bytes each"
        )
    return tuple(value)


def run_reference_benchmark(
    experiments: Sequence[Experiment],
    prompts: Sequence[str],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Measure prefill logits locally; never put prompts, tokens or logits in the report.

    This clear-kernel diagnostic shares the compiled numeric path with public
    masked-linear execution, but neither starts provider roles nor measures wire
    traffic or private online latency. Reference setup cost is excluded.
    """
    prompt_set = _prompts(prompts)
    if not isinstance(experiments, (tuple, list)) or not 1 <= len(experiments) <= 8:
        raise ReferenceBenchmarkError("quality benchmark needs 1 to 8 Experiments")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ReferenceBenchmarkError("top_k must be an integer from 1 to 100")
    configurations = []
    source_lock = None
    resolved_source = None
    for experiment in experiments:
        if not isinstance(experiment, Experiment):
            raise ReferenceBenchmarkError("quality candidates must be Experiments")
        experiment.resolve()
        runtime_options = resolve_runtime_composition(experiment.pipeline)
        if (
            runtime_options is None
            or runtime_options.client_runtime != "masked_transformer_v1"
            or runtime_options.privacy_mode != "public"
            or runtime_options.verification_component is not None
        ):
            raise ReferenceBenchmarkError(
                "quality diagnostic requires baseline masked-linear composition"
            )
        if not 1 <= experiment.budget.max_input_tokens <= 128:
            raise ReferenceBenchmarkError("quality diagnostic requires at most 128 input tokens")
        if len(prompt_set) > experiment.budget.requests:
            raise ReferenceBenchmarkError("prompt cohort exceeds the Experiment request budget")
        resolved = resolve_model(experiment.pipeline.model)
        if resolved.path is None or resolved.manifest.metadata.get("config_only"):
            raise ReferenceBenchmarkError(
                "quality diagnostic requires an imported weight checkpoint"
            )
        weight_files = [
            row
            for row in resolved.manifest.metadata["source_lock"]["files"]
            if row["path"].endswith(".safetensors")
        ]
        if not weight_files or sum(row["size"] for row in weight_files) > MAX_CHECKPOINT_BYTES:
            raise ReferenceBenchmarkError(
                "quality diagnostic requires at most 12 GiB of safetensors"
            )
        if source_lock is not None and resolved.source_lock_digest != source_lock:
            raise ReferenceBenchmarkError("quality Experiments must bind the same source lock")
        source_lock = resolved.source_lock_digest
        resolved_source = resolved
        configurations.append((experiment, runtime_options, resolved))
    assert resolved_source is not None and resolved_source.path is not None
    names = [experiment.name for experiment, _, _ in configurations]
    digests = [experiment.configuration_digest() for experiment, _, _ in configurations]
    if len(set(names)) != len(names) or len(set(digests)) != len(digests):
        raise ReferenceBenchmarkError(
            "quality candidates must have unique names and configurations"
        )
    # Hold only bounded logits from clear candidates, release their stage kernels,
    # then load the float32 reference. This avoids keeping two large model bodies
    # resident at once and does not put logits or token IDs in the report.
    logit_bytes = (len(experiments) + 1) * len(prompt_set) * resolved_source.manifest.vocab_size * 4
    if logit_bytes > MAX_LOGIT_WORKING_SET_BYTES:
        raise ReferenceBenchmarkError("quality logit working set exceeds 512 MiB")

    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ReferenceBenchmarkError("quality diagnostic requires torch and transformers") from exc
    reference_threads = torch.get_num_threads()
    checkpoint_digest = resolved_source.checkpoint_digest
    if source_lock is None or checkpoint_digest is None:
        raise ReferenceBenchmarkError("reference checkpoint has no source lock")
    dataset_digest = _digest(b"pllm.reference_prompts.v1\0", prompt_set)
    metric = ReferenceAgreement(
        dataset_digest=dataset_digest,
        reference_checkpoint_digest=checkpoint_digest,
        top_k=top_k,
    )
    token_cohort: tuple[tuple[int, ...], ...] | None = None
    pending: list[tuple[dict[str, Any], list[np.ndarray]]] = []
    for experiment, options, resolved in configurations:
        assert resolved.path is not None
        model_id = resolved.manifest.id
        kernel_choice = experiment.pipeline.components["kernels"]
        threads = kernel_choice.params.get("threads")
        metal_min_rows = kernel_choice.params.get("min_rows")
        engine = MaskedTransformerEngine(
            weight_bits=options.weight_bits,
            activation_bits=options.activation_bits,
            threads=threads,
            metal_min_rows=metal_min_rows,
            public_equalization_digest=options.public_equalization_digest,
        )
        asyncio.run(engine.load(resolved.manifest))
        bundle = ClientBundle.unpack(engine.client_bundle(model_id))
        config_bytes = (resolved.path / "config.json").read_bytes()
        from pllm.modeling import lower_model

        plan = lower_model(
            config_bytes,
            batch=1,
            max_input_tokens=experiment.budget.max_input_tokens,
            max_new_tokens=experiment.budget.max_new_tokens,
        )
        compiled = compile_runtime_model(plan, bundle, composition=experiment.pipeline)
        remote = _candidate_remote(engine, model_id)
        ids_this_candidate: list[tuple[int, ...]] = []
        actual_rows: list[np.ndarray] = []
        for prompt in prompt_set:
            ids = tuple(compiled.runtime(remote).encode_prompt(prompt))
            if not 1 <= len(ids) <= experiment.budget.max_input_tokens:
                raise ReferenceBenchmarkError(
                    "tokenized prompt exceeds the Experiment input budget"
                )
            ids_this_candidate.append(ids)
            session = compiled.session(remote)
            actual = np.array(session.prefill_ids(ids), dtype=np.float32, copy=True)
            session.finish()
            actual_rows.append(actual)
            del session
        current_cohort = tuple(ids_this_candidate)
        if token_cohort is not None and token_cohort != current_cohort:
            raise ReferenceBenchmarkError("quality candidates tokenized the cohort differently")
        token_cohort = current_cohort
        pending.append((
            {
                "name": experiment.name,
                "configuration_digest": experiment.configuration_digest(),
                "pipeline_digest": experiment.pipeline.digest(),
                "weight_bits": options.weight_bits,
                "activation_bits": options.activation_bits,
                "kernel_threads": threads,
                "kernel_backend": kernel_choice.component,
                **({
                    "public_equalization_profile_digest": options.public_equalization_digest,
                    "public_calibration_digest": engine._public_equalization_profile.calibration_digest,
                    "public_profile_bytes": len(engine._public_equalization_profile.pack()),
                } if engine._public_equalization_profile is not None else {}),
            }, actual_rows,
        ))
        del remote, compiled, bundle, engine
        gc.collect()
    assert token_cohort is not None
    try:
        reference = transformers.AutoModelForCausalLM.from_pretrained(
            resolved_source.path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="eager",
        ).eval()
    except Exception as exc:
        raise ReferenceBenchmarkError("the locked local FP32 reference cannot be loaded") from exc
    with torch.inference_mode():
        expected_rows = [
            reference(input_ids=torch.tensor([ids])).logits[0, -1].float().numpy().copy()
            for ids in token_cohort
        ]
    del reference
    gc.collect()
    candidates: list[dict[str, Any]] = []
    for metadata, actual_rows in pending:
        samples = [
            measure_reference_agreement(actual, expected, top_k=top_k)
            for actual, expected in zip(actual_rows, expected_rows, strict=True)
        ]
        candidates.append({
            **metadata,
            "sample_count": len(samples),
            "top1_agreement": sum(row["top1_agreement"] for row in samples) / len(samples),
            "top_k_recall": sum(row["top_k_recall"] for row in samples) / len(samples),
            "max_abs_logit_error": max(row["max_abs_logit_error"] for row in samples),
        })
    environment = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python_reference": f"transformers/{transformers.__version__}",
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "reference_dtype": "float32",
        "reference_attention": "eager",
        "reference_threads": reference_threads,
    }
    return {
        "schema_version": SCHEMA,
        "scope": "local-compiled-clear-kernel-prefill-reference",
        "model": {
            "id": resolved_source.manifest.id,
            "checkpoint_digest": checkpoint_digest,
            "source_lock_digest": source_lock,
        },
        "cohort": {
            "dataset_digest": dataset_digest,
            "token_cohort_digest": _digest(b"pllm.reference_tokens.v1\0", token_cohort),
            "prompt_count": len(prompt_set),
            "input_token_counts": [len(ids) for ids in token_cohort],
            "phase": "prefill",
            "metric": metric.to_spec(),
        },
        "environment": environment,
        "environment_digest": _digest(b"pllm.reference_environment.v1\0", environment),
        "candidates": candidates,
        "limitations": [
            "local clear integer kernels, not a provider transport or privacy measurement",
            "prefill next-token reference agreement, not full-generation or task accuracy",
            "float32 reference loading and evaluation excluded from provider latency and cost",
        ],
    }


__all__ = ["ReferenceBenchmarkError", "run_reference_benchmark"]
