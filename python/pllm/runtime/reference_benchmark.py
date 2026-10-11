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
        integer = engine._public_stage_matrix(model_id, stage, q_activation.rows).clear(
            q_activation.values
        )
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
    reference_storage: str = "resident",
    decode_steps: int = 0,
) -> dict[str, Any]:
    """Measure prefill and optional same-token decode without recording private payloads.

    This clear-kernel diagnostic shares the compiled numeric path with public
    masked-linear execution, but neither starts provider roles nor measures wire
    traffic or private online latency. Reference setup cost is excluded.
    Optional decode uses the first candidate's greedy tokens for every candidate
    and the reference; it reports those comparisons separately from prefill.
    """
    prompt_set = _prompts(prompts)
    if reference_storage not in {"resident", "streamed"}:
        raise ReferenceBenchmarkError("reference_storage must be resident or streamed")
    if type(decode_steps) is not int or not 0 <= decode_steps <= 8:
        raise ReferenceBenchmarkError("decode_steps must be an integer from 0 to 8")
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
        if decode_steps >= experiment.budget.max_new_tokens:
            raise ReferenceBenchmarkError("decode steps exceed the Experiment output budget")
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
    logit_bytes = ((len(experiments) + 1) * len(prompt_set)
                   * (decode_steps + 1) * resolved_source.manifest.vocab_size * 4)
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
    decode_cohort: list[tuple[int, ...]] = []
    pending_decode: list[list[np.ndarray]] = []
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
            client_prefix_layers=options.client_prefix_layers,
            client_linear_roles=options.client_linear_roles,
            client_linear_layer_roles=options.client_linear_layer_roles,
            remote_output_head=options.remote_output_head,
            prepared_output_encoding=options.prepared_output_encoding,
        )
        asyncio.run(engine.load(resolved.manifest))
        # Streamed diagnostics reuse the same immutable provider document and
        # client validator without materializing a second contiguous weight body.
        bundle = (ClientBundle._from_document(engine.client_bundle_document(model_id)._value)
                  if reference_storage == "streamed"
                  else ClientBundle.unpack(engine.client_bundle(model_id)))
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
        decode_rows: list[np.ndarray] = []
        for prompt_index, prompt in enumerate(prompt_set):
            ids = tuple(compiled.runtime(remote).encode_prompt(prompt))
            if not 1 <= len(ids) <= experiment.budget.max_input_tokens:
                raise ReferenceBenchmarkError(
                    "tokenized prompt exceeds the Experiment input budget"
                )
            ids_this_candidate.append(ids)
            session = compiled.session(remote) if not decode_steps else None
            runtime = compiled.runtime(remote) if decode_steps else None
            try:
                compiled.validate()
                prefill = (session.prefill_ids(ids) if session is not None
                           else runtime.prepare_ids(ids)[1])
                actual = np.array(prefill, dtype=np.float32, copy=True)
                actual_rows.append(actual)
                selected = []
                for step in range(decode_steps):
                    # The first candidate freezes the public diagnostic trajectory;
                    # every candidate and the reference consume the same tokens.
                    token = (int(np.argmax(actual)) if token_cohort is None
                             else decode_cohort[prompt_index][step])
                    selected.append(token)
                    compiled.validate()
                    actual = np.array(runtime.decode_step(token, runtime.caches, len(ids) + step)[0],
                                      dtype=np.float32, copy=True)
                    decode_rows.append(actual)
                if token_cohort is None:
                    decode_cohort.append(tuple(selected))
            finally:
                if session is not None:
                    session.close()
                if runtime is not None:
                    runtime.reset()
                del session, runtime
        current_cohort = tuple(ids_this_candidate)
        if token_cohort is not None and token_cohort != current_cohort:
            raise ReferenceBenchmarkError("quality candidates tokenized the cohort differently")
        token_cohort = current_cohort
        pending_decode.append(decode_rows)
        pending.append(
            (
                {
                    "name": experiment.name,
                    "configuration_digest": experiment.configuration_digest(),
                    "pipeline_digest": experiment.pipeline.digest(),
                    "weight_bits": options.weight_bits,
                    "activation_bits": options.activation_bits,
                    "kernel_threads": threads,
                    "kernel_backend": kernel_choice.component,
                    **(
                        {
                            "public_equalization_profile_digest": options.public_equalization_digest,
                            "public_calibration_digest": engine._public_equalization_profile.calibration_digest,
                            "public_profile_bytes": len(engine._public_equalization_profile.pack()),
                        }
                        if engine._public_equalization_profile is not None
                        else {}
                    ),
                },
                actual_rows,
            )
        )
        del remote, compiled, bundle, engine
        gc.collect()
    assert token_cohort is not None
    loading = None
    if reference_storage == "streamed":
        from pllm.runtime.streamed_reference import StreamedFloat32Reference

        with StreamedFloat32Reference(
            resolved_source.path,
            max_input_tokens=max(item.budget.max_input_tokens for item, _, _ in configurations),
            max_decode_steps=decode_steps,
        ) as reference:
            trajectories = [reference.trajectory(ids, continuation)
                            for ids, continuation in zip(token_cohort, decode_cohort, strict=True)]
            loading = {"storage": "module-streamed", "active_weight_bound_bytes": reference.weight_bound,
                       "allocation_estimate_bytes": reference.allocation_estimate_bytes,
                       "maximum_swap_growth_bytes": reference._guard.maximum_swap_growth_bytes,
                       "kv_payload_bound_bytes": reference.kv_bound_bytes,
                       "buffer_bytes": reference.buffer_bytes, **reference.stats}
        del reference
    else:
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
            trajectories = []
            for ids, continuation in zip(token_cohort, decode_cohort, strict=True):
                output = reference(input_ids=torch.tensor([ids]), use_cache=bool(continuation))
                rows = [output.logits[0, -1].float().numpy().copy()]
                cache = output.past_key_values if continuation else None
                del output
                for token in continuation:
                    output = reference(input_ids=torch.tensor([[token]]), past_key_values=cache,
                                       use_cache=True)
                    rows.append(output.logits[0, -1].float().numpy().copy())
                    cache = output.past_key_values
                    del output
                del cache
                trajectories.append(rows)
        del reference
    gc.collect()
    expected_rows = [rows[0] for rows in trajectories]
    expected_decode = [row for rows in trajectories for row in rows[1:]]
    candidates: list[dict[str, Any]] = []
    for (metadata, actual_rows), decode_rows in zip(pending, pending_decode, strict=True):
        samples = [
            measure_reference_agreement(actual, expected, top_k=top_k)
            for actual, expected in zip(actual_rows, expected_rows, strict=True)
        ]
        candidates.append(
            {
                **metadata,
                "sample_count": len(samples),
                "top1_agreement": sum(row["top1_agreement"] for row in samples) / len(samples),
                "top_k_recall": sum(row["top_k_recall"] for row in samples) / len(samples),
                "max_abs_logit_error": max(row["max_abs_logit_error"] for row in samples),
            }
        )
        if decode_steps:
            samples = [measure_reference_agreement(actual, expected, top_k=top_k)
                       for actual, expected in zip(decode_rows, expected_decode, strict=True)]
            candidates[-1]["same_token_decode"] = {
                "sample_count": len(samples),
                "top1_agreement": sum(row["top1_agreement"] for row in samples) / len(samples),
                "top_k_recall": sum(row["top_k_recall"] for row in samples) / len(samples),
                "max_abs_logit_error": max(row["max_abs_logit_error"] for row in samples),
            }
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
    if loading is not None:
        environment["reference_weight_storage"] = "module-streamed"
    return {
        "schema_version": SCHEMA,
        "scope": ("local-compiled-clear-kernel-prefill-and-same-token-decode-reference"
                  if decode_steps else "local-compiled-clear-kernel-prefill-reference"),
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
            **({"same_token_decode": {
                "steps_per_prompt": decode_steps,
                "teacher_forcing": "first-candidate-greedy",
                "teacher_configuration_digest": configurations[0][0].configuration_digest(),
                "token_cohort_digest": _digest(b"pllm.reference_decode_tokens.v1\0", decode_cohort),
            }} if decode_steps else {}),
        },
        "environment": environment,
        "environment_digest": _digest(b"pllm.reference_environment.v1\0", environment),
        "candidates": candidates,
        **({"reference_loading": loading} if loading is not None else {}),
        "limitations": [
            "local clear integer kernels, not a provider transport or privacy measurement",
            "prefill and optional same-token decode agreement, not full-generation or task accuracy",
            "float32 reference loading and evaluation excluded from provider latency and cost",
        ],
    }


__all__ = ["ReferenceBenchmarkError", "run_reference_benchmark"]
