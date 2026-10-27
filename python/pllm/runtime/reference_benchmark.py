"""Compiled-decoder quality comparison against a shared pinned FP32 reference."""

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
from pllm.protocols import FuseFssSourceDecoder
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


def _final_token_runtime(runtime):
    return runtime in {"compiled_replicated_v1", "compiled_fss_v1"}


def _source_ids(source, prompts, *, runtime=None):
    from .responses import _owned_prompt_source, _prepare_prompt

    if runtime == "compiled_fss_v1":
        from tokenizers import Tokenizer
        from .transformer_client import TokenizersJSONTokenizer

        descriptor = source.tokenizer_descriptor
        tokenizer = TokenizersJSONTokenizer(Tokenizer.from_buffer(source.tokenizer_bytes),
            descriptor["bos_token_id"], descriptor["eos_token_id"])
    else:
        tokenizer = source.tokenizer()
    try:
        identity = _owned_prompt_source(source)
        values = []
        for prompt in prompts:
            body = {"model": str(source.model.source), "input": prompt,
                    "max_output_tokens": source.contract["decode_steps"] + 1}
            prepared = _prepare_prompt(body, descriptor=source.tokenizer_descriptor,
                                       tokenizer=tokenizer, source=identity)
            values.append(prepared._payload.ids)
        return tuple(values)
    finally:
        close = getattr(tokenizer, "close", None)
        if close is not None:
            close()


def _fss_numeric(experiment, source):
    """Check native source identity, never substitute replicated/Q12 oracles."""
    contract = source.contract
    profile = experiment.pipeline.components["linear"].params.get("numeric_profile")
    expected = {"schema_version": "pllm.fss.sdk-contract.v1", "weights": "public",
        "configuration": experiment.configuration_digest(), "composition": experiment.pipeline.digest(),
        "input_rows": experiment.budget.max_input_tokens, "output_tokens": experiment.budget.max_new_tokens,
        "source_lock": source.source_lock_digest, "tokenizer": source.tokenizer_digest}
    if (profile not in FuseFssSourceDecoder.NUMERIC_PROFILES
            or any(type(contract.get(key)) is not type(value) or contract.get(key) != value
                   for key, value in expected.items())
            or contract.get("numeric_profile") != profile
            or contract.get("capacity", "capsule") != experiment.pipeline.components["linear"].params.get("capacity", "capsule")
            or contract.get("runtime_kind") != "fss" or contract.get("client_runtime") != "compiled_fss_v1"
            or type(contract.get("ring_bits")) is not int or contract["ring_bits"] != 50
            or type(contract.get("fractional_bits")) is not int or contract["fractional_bits"] != 12):
        from .quality_outcomes import QualityIntegrityError
        raise QualityIntegrityError("quality FSS native numeric/source contract differs")
    return contract


def _fss_reference_source(experiment, source):
    """Bind reference metadata only after FSS native source admission, with exact pins."""
    _fss_numeric(experiment, source)
    resolved = resolve_model(experiment.pipeline.model)
    if (resolved.path != source.path or resolved.source_lock_digest != source.source_lock_digest
            or resolved.checkpoint_digest != source.checkpoint_digest
            or resolved.manifest.vocab_size != source.contract["vocabulary"]):
        from .quality_outcomes import QualityIntegrityError
        raise QualityIntegrityError("quality FSS reference source binding failed")
    return resolved


def _fss_quality_identity(experiment, source, decode_steps, prompts):
    contract = _fss_numeric(experiment, source)
    return {"numeric_profile": contract["numeric_profile"], "fraction_bits": 12, "ring_bits": 50,
        "candidate_logits_available": False, "execution": "three-role-offline-tape-final-token-full-prefix",
        "diagnostic_requests": len(prompts)*(decode_steps+1), "output_selections_per_request": 1}


def _memory_watchdog_note(guard):
    """Sanitize cached guard data only; at most 256 ASCII bytes, no new sample.

    Unknown exception text is never copied: it may contain private paths or
    caller data. Record mode can retain this sanitized note in existing report
    limitations; it is not a new structured guard field or scored outcome.
    """
    try:
        reason = getattr(guard, "error", None)
        observation = getattr(guard, "failure_observation", None)
        parts = []
        if type(reason) is str:
            public_reasons = (
                "benchmark stopped: macOS memory pressure is warning",
                "benchmark stopped: macOS memory pressure is critical",
                "benchmark stopped: host memory fell below runtime reserve",
                "benchmark stopped: host swap grew by more than 256 MiB",
            )
            parts.append(reason if len(reason) <= 80 and reason in public_reasons
                         else "unclassified failure (text omitted)")
        if (type(observation) is dict and len(observation) == 3
                and all(type(key) is str and key in ("pressure", "available_bytes", "swap_used_bytes")
                        for key in observation)):
            pressure = observation["pressure"]
            if type(pressure) is str and pressure in ("normal", "warning", "critical"):
                parts.append("pressure=" + pressure)
            for key in ("available_bytes", "swap_used_bytes"):
                value = observation[key]
                if type(value) is int and 0 <= value <= (1 << 63) - 1:
                    parts.append(f"{key}={value}")
        if parts:
            return "memory watchdog observed: " + "; ".join(parts)
    except BaseException:
        pass  # Diagnostics must not replace an execution/cleanup failure.
    return None


def _memory_admission_note(report):
    """At most 256 ASCII bytes from cached integers; no sampling or GiB inversion."""
    try:
        if (type(report) is not dict or report.get("admitted") is not False
                or type(report.get("requested_backend")) is not str
                or report["requested_backend"] != "native"):
            return None
        host, candidates = report.get("host"), report.get("candidates")
        if type(host) is not dict or type(candidates) is not dict:
            return None
        native = candidates.get("native")
        if type(native) is not dict or native.get("admitted") is not False:
            return None
        required, budget = native.get("required_host_bytes"), host.get("admission_budget_bytes")
        if any(type(value) is not int or not 0 <= value <= (1 << 63) - 1
               for value in (required, budget)):
            return None
        parts = ["backend=native", f"estimated_peak_bytes={required}", f"safe_headroom_bytes={budget}"]
        available = host.get("available")
        if type(available) is int and 0 <= available <= (1 << 63) - 1:
            parts.append(f"available_bytes={available}")
        estimate = report.get("estimate")
        margin = estimate.get("allocation_slack_percent") if type(estimate) is dict else None
        if type(margin) is int and 0 <= margin <= 100:
            parts.append(f"margin_percent={margin}")
        return "quality memory admission: " + "; ".join(parts)
    except BaseException:
        pass  # Never replace the original admission error with diagnostics.
    return None


def _delegated_candidate(experiment, prompts, source, decode_steps, continuations, memory_safety_margin_percent=25, *, outcomes=None, expected_cohort=None):
    """Run the installed two-worker path; return only finally accepted logits.

    Cached admission bytes and guard triggers annotate exceptions. Record mode carries that
    same bounded note through its outcomes owner into existing limitations,
    without retaining the exception, private text or mutable guard observation.
    """
    from threading import Event

    from .benchmark_memory import BenchmarkMemoryError, MemoryWatchdog, benchmark_memory, require_admission
    from .delegated_transport import execute_request
    from .servers import build_roles

    ids = _source_ids(source, prompts)
    if expected_cohort is not None and ids != expected_cohort:
        from .quality_outcomes import QualityIntegrityError
        raise QualityIntegrityError("quality source/cohort binding failed")
    if any(not 1 <= len(row) <= experiment.budget.max_input_tokens for row in ids):
        raise ReferenceBenchmarkError("tokenized prompt exceeds the Experiment input budget")
    report = benchmark_memory(experiment, backend="native",
                              memory_safety_margin_percent=memory_safety_margin_percent)
    try:
        require_admission(report)
    except BenchmarkMemoryError as error:
        try:
            note = _memory_admission_note(error.report) if error.report is report else None
            if note is not None:
                if outcomes is not None and getattr(outcomes, "_memory_guard_note", None) is None:
                    outcomes._memory_guard_note = note  # Existing record-mode limitations owner.
                notes = getattr(error, "__notes__", None)
                if not (type(notes) is list and notes and type(notes[-1]) is str and notes[-1] == note):
                    error.add_note(note)
        except BaseException:
            pass
        raise
    cancelled = Event()
    guard_note = None

    def remember_guard():
        nonlocal guard_note
        if guard_note is None:
            note = _memory_watchdog_note(guard)
            if guard_note is None:
                guard_note = note

    def abort(_reason):
        try:
            remember_guard()  # Freeze the first trigger before signalling cancel.
        except BaseException:
            pass
        finally:
            cancelled.set()

    def annotate(error):
        try:
            remember_guard()
            if guard_note is not None:
                if outcomes is not None:
                    outcomes._memory_guard_note = guard_note
                notes = getattr(error, "__notes__", None)
                if (type(notes) is list and notes and type(notes[-1]) is str
                        and notes[-1] == guard_note):
                    return
                error.add_note(guard_note)
        except BaseException:
            pass

    guard = MemoryWatchdog(report, abort)
    actual, decoded, selected = [], [], []
    result = None
    try:
        try:
            guard.start()
            with build_roles(experiment) as topology:
                try:
                    for index, row in enumerate(ids):
                        if outcomes is not None:
                            outcomes.current, outcomes.attempted = index * (decode_steps + 1), False
                        guard.check()
                        if outcomes is not None:
                            outcomes.attempted = True
                        try:
                            result = execute_request(
                                experiment, source, row, decode_steps + 1,
                                topology.worker_connections(), 3600.0, cancelled.is_set,
                                decode_tokens=None if continuations is None else continuations[index],
                            )
                        except Exception as error:
                            from .quality_outcomes import numeric_rejection, Outcome
                            if outcomes is None or not numeric_rejection(error):
                                raise
                            for step in range(decode_steps + 1):
                                outcomes.record(index * (decode_steps + 1) + step, Outcome("rejected", "numeric_domain_rejected"))
                            selected.append(tuple(continuations[index]) if continuations is not None else ())
                            continue
                        guard.check()
                        vectors = [np.frombuffer(value, dtype="<f4").copy() for value in result.logits]
                        if (len(vectors) != decode_steps + 1 or any(vector.shape != (source.contract["vocabulary"],)
                                or not np.isfinite(vector).all() for vector in vectors)):
                            raise ReferenceBenchmarkError("quality output malformed")
                        if outcomes is None:
                            actual.append(vectors[0])
                            decoded.extend(vectors[1:])
                        else:
                            from .quality_outcomes import Outcome
                            for step, vector in enumerate(vectors):
                                outcomes.record(index * (decode_steps + 1) + step, Outcome("accepted"), vector)
                        selected.append(result.executed[len(row):])
                        result = vectors = None  # Complete KV retires before next request.
                except BaseException as error:
                    annotate(error)  # Before topology cleanup can replace the error.
                    raise
        finally:
            guard.close()
            result = None
    except BaseException as error:
        annotate(error)  # Include startup/cleanup errors, with Python precedence intact.
        raise
    finally:
        guard = None  # Break guard -> abort -> remember_guard -> guard ownership.
    kernel = experiment.pipeline.components["kernels"]
    contract = source.contract
    metadata = {
        "name": experiment.name,
        "configuration_digest": experiment.configuration_digest(),
        "pipeline_digest": experiment.pipeline.digest(),
        "weight_bits": 16, "activation_bits": None,
        "numeric_profile": contract["numeric_profile"],
        "local_numeric_profile": contract["local_numeric_profile"],
        "kernel_threads": kernel.params.get("threads"), "kernel_backend": kernel.component,
        "execution": "two-worker-request-end-verified",
        "memory_admission": report,
    }
    return metadata, actual, decoded, ids, selected


def _replicated_candidate(experiment, prompts, source, decode_steps, continuations, memory_safety_margin_percent=25, *, outcomes=None, expected_cohort=None):
    return _final_token_candidate(experiment, prompts, source, decode_steps, continuations,
        memory_safety_margin_percent, outcomes=outcomes, expected_cohort=expected_cohort,
        runtime="compiled_replicated_v1")


def _fss_candidate(experiment, prompts, source, decode_steps, continuations, memory_safety_margin_percent=25, *, outcomes=None, expected_cohort=None):
    _fss_numeric(experiment, source)
    return _final_token_candidate(experiment, prompts, source, decode_steps, continuations,
        memory_safety_margin_percent, outcomes=outcomes, expected_cohort=expected_cohort,
        runtime="compiled_fss_v1")


def _final_token_candidate(experiment, prompts, source, decode_steps, continuations, memory_safety_margin_percent=25, *, outcomes=None, expected_cohort=None, runtime):
    """Score final IDs through ordinary protected requests on caller-owned prefixes.

    One fixed-count selection completes before each diagnostic prefix extension.
    Each prefix has a fresh protected request, setup and material. FSS imports
    role-local public weights, not secret source shares. This permits a common
    teacher-forced cohort without opening hidden logits or modifying feedback.
    """
    from threading import Event

    from .benchmark_memory import (
        BenchmarkMemoryError, MemoryWatchdog, benchmark_memory, execution_memory_admission, require_admission)
    if runtime == "compiled_fss_v1":
        from .fss_transport import execute_request
    else:
        from .replicated_transport import execute_request
    from .servers import build_roles

    requests = len(prompts) * (decode_steps + 1)
    if requests > (min(64, experiment.budget.requests) if runtime == "compiled_fss_v1" else experiment.budget.requests):
        raise ReferenceBenchmarkError("final-token quality requests exceed the Experiment request budget")
    ids = _source_ids(source, prompts, runtime=runtime)
    if expected_cohort is not None and ids != expected_cohort:
        from .quality_outcomes import QualityIntegrityError
        raise QualityIntegrityError("quality source/cohort binding failed")
    if any(not 1 <= len(row) or len(row) + decode_steps > experiment.budget.max_input_tokens for row in ids):
        raise ReferenceBenchmarkError("quality full-prefix requests exceed the Experiment input budget")
    if continuations is not None and (
        len(continuations) != len(ids)
        or any(len(row) != decode_steps or any(type(token) is not int or not 0 <= token < source.contract["vocabulary"]
                                               for token in row) for row in continuations)
    ):
        raise ReferenceBenchmarkError("quality continuation does not match the bounded token cohort")
    decoder = (runtime == "compiled_fss_v1"
               and experiment.pipeline.components["linear"].params.get("capacity") == "decoder")
    inspect_memory = execution_memory_admission if decoder else benchmark_memory
    report = inspect_memory(experiment, backend="native",
                            memory_safety_margin_percent=memory_safety_margin_percent)
    require_admission(report)
    cancelled = Event()
    guard = MemoryWatchdog(report, lambda _: cancelled.set())
    actual, decoded, selected = [], [], []
    result = None
    policy = {"memory_safety_margin_percent": memory_safety_margin_percent} if decoder else {}
    try:
        guard.start()
        with build_roles(experiment, **policy) as topology:
            if decoder:
                # The initial gate admits only source qualification on disk. Keep
                # its conservative memory watchdog; consume the topology's final
                # native-derived quote before sending any protected request.
                report = getattr(topology, "memory_admission", None)
                if type(report) is not dict or report.get("full_topology_admitted") is not True:
                    raise BenchmarkMemoryError("quality FSS requires complete native topology admission")
                require_admission(report)
            for index, row in enumerate(ids):
                prefix, continuation = list(row), []
                for step in range(decode_steps + 1):
                    if outcomes is not None:
                        outcomes.current, outcomes.attempted = index * (decode_steps + 1) + step, False
                    guard.check()
                    if outcomes is not None:
                        outcomes.attempted = True
                    try:
                        result = execute_request(experiment, source, tuple(prefix), 1,
                            topology.worker_connections(), 600.0 if runtime == "compiled_fss_v1" else 3600.0,
                            cancelled.is_set)
                    except Exception as error:
                        from .quality_outcomes import numeric_rejection, Outcome
                        if outcomes is None or not numeric_rejection(error):
                            raise
                        outcomes.record(index * (decode_steps + 1) + step, Outcome("rejected", "numeric_domain_rejected"))
                        if step < decode_steps:
                            prefix.append(continuations[index][step])
                            continuation.append(continuations[index][step])
                        continue
                    guard.check()
                    if (len(result.selected) != 1 or type(result.selected[0]) is not int
                            or not 0 <= result.selected[0] < source.contract["vocabulary"]):
                        raise ReferenceBenchmarkError("final-token quality response has an invalid output count")
                    token = result.selected[0]
                    if outcomes is None:
                        (actual if step == 0 else decoded).append(token)
                    else:
                        from .quality_outcomes import Outcome
                        outcomes.record(index * (decode_steps + 1) + step, Outcome("accepted"), token)
                    result = None
                    if step < decode_steps:
                        feedback = token if continuations is None else continuations[index][step]
                        continuation.append(feedback)
                        prefix.append(feedback)
                selected.append(tuple(continuation))
    finally:
        guard.close()
        result = None
    kernel = experiment.pipeline.components["kernels"]
    metadata = {
        "name": experiment.name, "configuration_digest": experiment.configuration_digest(),
        "pipeline_digest": experiment.pipeline.digest(), "weight_bits": None, "activation_bits": None,
        "numeric_profile": source.contract["numeric_profile"],
        "fraction_bits": 12 if runtime == "compiled_fss_v1" else 18,
        "ring_bits": 50 if runtime == "compiled_fss_v1" else 64,
        "kernel_threads": kernel.params.get("threads"), "kernel_backend": kernel.component,
        "execution": ("three-role-offline-tape-final-token-full-prefix" if runtime == "compiled_fss_v1"
                      else "three-worker-final-token-full-prefix"), "candidate_logits_available": False,
        "diagnostic_requests": requests, "output_selections_per_request": 1,
        "memory_admission": report,
    }
    return metadata, actual, decoded, ids, selected


def _agreement_summary(actual, expected, *, top_k, final_tokens=False):
    samples = [measure_reference_agreement(None if final_tokens else candidate, reference,
                top_k=top_k, selected_token=candidate if final_tokens else None)
               for candidate, reference in zip(actual, expected, strict=True)]
    result = {"sample_count": len(samples)}
    for key in ("top1_agreement", "top_k_recall", "max_abs_logit_error"):
        values = [row[key] for row in samples]
        result[key] = (None if any(value is None for value in values) else
                       max(values) if key == "max_abs_logit_error" else sum(values) / len(values))
    return result


def _check_reference_source(resolved_source):
    """Rebind freshly read reference files to the already admitted source pins.

    A shared path is not a shared checkpoint: candidates retire before reference
    loading. Reuse the ordinary source-lock reader, not cached resolution. This
    detects observed changes; immutable local source remains an assumption, not
    an atomic snapshot, publisher authentication, or malicious-source proof.
    """
    from pllm.model_loader import _source_lock
    from .quality_outcomes import QualityIntegrityError

    try:
        current = _source_lock(resolved_source.path, resolved_source.model)
    except (OSError, ValueError) as error:
        raise QualityIntegrityError("quality reference source binding failed") from error
    expected = (resolved_source.checkpoint_digest, resolved_source.source_lock_digest,
                resolved_source.manifest.metadata["source_lock"])
    if current != expected:
        raise QualityIntegrityError("quality reference source binding failed")


def _reference_trajectories(resolved_source, configurations, token_cohort, decode_cohort,
                            reference_storage, decode_steps, memory_safety_margin_percent):
    _check_reference_source(resolved_source)
    try:
        return _load_reference_trajectories(resolved_source, configurations, token_cohort,
            decode_cohort, reference_storage, decode_steps, memory_safety_margin_percent)
    finally:
        _check_reference_source(resolved_source)


def _reference_dependencies(*, initialize):
    """Check optional presence without initializing the reference's frameworks.

    Streamed reference has its own post-setup admission before weight loaders;
    its imports can wait until candidate owners retire. Resident reference has
    no separate post-framework host gate, so retains its original eager path.
    Presence is not binary/import compatibility: those failures still need the
    actual import at use. No thread settings, seeds or module cache are changed.
    """
    try:
        if not initialize:
            from importlib.util import find_spec
            import sys

            for name in ("torch", "transformers"):
                if sys.modules.get(name) is None and find_spec(name) is None:
                    raise ImportError("reference dependency not installed")
            return None, None
        import torch
        import transformers
        return torch, transformers
    except ImportError as exc:
        raise ReferenceBenchmarkError("quality diagnostic requires torch and transformers") from exc


def _load_reference_trajectories(resolved_source, configurations, token_cohort, decode_cohort,
                                reference_storage, decode_steps, memory_safety_margin_percent):
    import torch
    import transformers
    loading = None
    if reference_storage == "streamed":
        from .streamed_reference import StreamedFloat32Reference
        with StreamedFloat32Reference(resolved_source.path,
                max_input_tokens=max(item.budget.max_input_tokens for item, _, _, _ in configurations),
                max_decode_steps=decode_steps, memory_safety_margin_percent=memory_safety_margin_percent) as reference:
            trajectories = [reference.trajectory(ids, continuation)
                            for ids, continuation in zip(token_cohort, decode_cohort, strict=True)]
            loading = {"storage": "module-streamed", "active_weight_bound_bytes": reference.weight_bound,
                "allocation_estimate_bytes": reference.allocation_estimate_bytes,
                "maximum_swap_growth_bytes": reference._guard.maximum_swap_growth_bytes,
                "kv_payload_bound_bytes": reference.kv_bound_bytes,
                "buffer_bytes": reference.buffer_bytes, **reference.stats}
    else:
        try:
            reference = transformers.AutoModelForCausalLM.from_pretrained(resolved_source.path,
                local_files_only=True, trust_remote_code=False, dtype=torch.float32, attn_implementation="eager").eval()
        except Exception as exc:
            raise ReferenceBenchmarkError("the locked local FP32 reference cannot be loaded") from exc
        try:
            with torch.inference_mode():
                trajectories = []
                for ids, continuation in zip(token_cohort, decode_cohort, strict=True):
                    output = reference(input_ids=torch.tensor([ids]), use_cache=bool(continuation))
                    rows = [output.logits[0, -1].float().numpy().copy()]
                    cache = output.past_key_values if continuation else None
                    del output
                    for token in continuation:
                        output = reference(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True)
                        rows.append(output.logits[0, -1].float().numpy().copy())
                        cache = output.past_key_values
                        del output
                    del cache
                    trajectories.append(rows)
        finally:
            reference = None
    return trajectories, loading


def run_reference_benchmark(
    experiments: Sequence[Experiment],
    prompts: Sequence[str],
    *,
    top_k: int = 5,
    reference_storage: str = "resident",
    decode_steps: int = 0,
    memory_safety_margin_percent: int = 25,
    outcomes: str = "abort",
    cohort_key: bytes | None = None,
    cohort_key_id: str | None = None,
    continuations: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Measure prefill and optional same-token decode without recording private payloads.

    Masked-linear candidates use their compiled clear kernels. Delegated candidates
    use their installed worker topology and final verification barrier. Serving
    costs belong to benchmark run; this report measures numeric agreement only.
    Final-token-only candidates use fresh protected one-selection requests on each
    common prefix; top-k and logit errors remain unknown. Optional decode uses the
    first candidate's greedy tokens for every candidate and the reference.

    Explicit ``outcomes='record'`` produces v2 with a caller-owned private
    32-byte campaign key/public label. Rejected logical positions are retained;
    decode requires frozen caller continuations, never a replacement teacher.
    Operational failure fences unattempted work and disables further reference
    execution. Cancellation raises QualityRunCancelled with a sanitized report.
    Invalid global/source/cohort preflight still aborts without a score/report.
    """
    from .benchmark_memory import _memory_margin
    from .quality_outcomes import CandidateOutcomes, CaptureFailure, Outcome, commitment, numeric_rejection_available

    _memory_margin(memory_safety_margin_percent)
    if outcomes not in {"abort", "record"}:
        raise ReferenceBenchmarkError("outcomes must be abort or record")
    record = outcomes == "record"
    if record:
        if (type(cohort_key) is not bytes or len(cohort_key) != 32
                or type(cohort_key_id) is not str or not 1 <= len(cohort_key_id) <= 64
                or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in cohort_key_id)):
            raise ReferenceBenchmarkError("record mode requires a 32-byte private cohort key and public campaign label")
    elif any(value is not None for value in (cohort_key, cohort_key_id, continuations)):
        raise ReferenceBenchmarkError("cohort key and frozen continuations require record mode")
    prompt_set = _prompts(prompts)
    if reference_storage not in {"resident", "streamed"}:
        raise ReferenceBenchmarkError("reference_storage must be resident or streamed")
    if type(decode_steps) is not int or not 0 <= decode_steps <= 8:
        raise ReferenceBenchmarkError("decode_steps must be an integer from 0 to 8")
    if not isinstance(experiments, (tuple, list)) or not 1 <= len(experiments) <= 8:
        raise ReferenceBenchmarkError("quality benchmark needs 1 to 8 Experiments")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ReferenceBenchmarkError("top_k must be an integer from 1 to 100")
    if record and decode_steps and (not isinstance(continuations, (tuple, list))
            or len(continuations) != len(prompt_set)
            or any(not isinstance(row, (tuple, list)) or len(row) != decode_steps
                   or any(type(token) is not int or token < 0 for token in row) for row in continuations)):
        raise ReferenceBenchmarkError("record decode requires frozen caller continuations")
    if record and not decode_steps and continuations is not None:
        raise ReferenceBenchmarkError("prefill record mode does not consume continuations")
    configurations = []
    source_lock = None
    resolved_source = None
    for experiment in experiments:
        if not isinstance(experiment, Experiment):
            raise ReferenceBenchmarkError("quality candidates must be Experiments")
        if record and (type(experiment.name) is not str or len(experiment.name) > 1024):
            raise ReferenceBenchmarkError("quality public metadata bound")
        experiment.resolve()
        runtime_options = resolve_runtime_composition(experiment.pipeline)
        delegated = runtime_options is not None and runtime_options.client_runtime == "compiled_gkm_v1"
        replicated = runtime_options is not None and runtime_options.client_runtime == "compiled_replicated_v1"
        fss = runtime_options is not None and runtime_options.client_runtime == "compiled_fss_v1"
        final_tokens = replicated or fss
        if not (delegated or final_tokens) and (
            runtime_options is None
            or runtime_options.client_runtime != "masked_transformer_v1"
            or runtime_options.privacy_mode != "public"
            or runtime_options.verification_component is not None
        ):
            raise ReferenceBenchmarkError(
                "quality diagnostic requires a supported compiled numeric composition"
            )
        if not 1 <= experiment.budget.max_input_tokens <= 128:
            raise ReferenceBenchmarkError("quality diagnostic requires at most 128 input tokens")
        if decode_steps >= experiment.budget.max_new_tokens:
            raise ReferenceBenchmarkError("decode steps exceed the Experiment output budget")
        if len(prompt_set) > experiment.budget.requests:
            raise ReferenceBenchmarkError("prompt cohort exceeds the Experiment request budget")
        if final_tokens and len(prompt_set) * (decode_steps + 1) > (min(64, experiment.budget.requests) if fss else experiment.budget.requests):
            raise ReferenceBenchmarkError("final-token quality requests exceed the Experiment request budget")
        if (record and (fss or replicated and experiment.pipeline.components["linear"].params.get("range_policy") == "checked")
                and not numeric_rejection_available()):
            raise ReferenceBenchmarkError("record mode requires typed native final acceptance")
        admitted = None
        if delegated or final_tokens:
            if fss:
                from .fss_source import resolve_client_source as admit_source
            elif replicated:
                from .replicated_source import admit_source
            else:
                from .delegated_source import admit_source

            admitted = admit_source(experiment)
            resolved = _fss_reference_source(experiment, admitted) if fss else admitted.source
        else:
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
        configurations.append((experiment, runtime_options, resolved, admitted))
    assert resolved_source is not None and resolved_source.path is not None
    names = [experiment.name for experiment, _, _, _ in configurations]
    digests = [experiment.configuration_digest() for experiment, _, _, _ in configurations]
    if len(set(names)) != len(names) or len(set(digests)) != len(digests):
        raise ReferenceBenchmarkError(
            "quality candidates must have unique names and configurations"
        )
    # Hold only bounded logits from candidates, release their execution owners,
    # then load the float32 reference. This avoids keeping two large model bodies
    # resident at once and does not put logits or token IDs in the report.
    logit_bytes = ((len(experiments) + 1) * len(prompt_set)
                   * (decode_steps + 1) * resolved_source.manifest.vocab_size * 4)
    if logit_bytes > MAX_LOGIT_WORKING_SET_BYTES:
        raise ReferenceBenchmarkError("quality logit working set exceeds 512 MiB")

    torch, transformers = _reference_dependencies(initialize=reference_storage == "resident")
    reference_threads = torch.get_num_threads() if torch is not None else None
    checkpoint_digest = resolved_source.checkpoint_digest
    if source_lock is None or checkpoint_digest is None:
        raise ReferenceBenchmarkError("reference checkpoint has no source lock")
    if record and len(resolved_source.manifest.id) > 1024:
        raise ReferenceBenchmarkError("quality public model label bound")
    if record and continuations is not None and any(token >= resolved_source.manifest.vocab_size for row in continuations for token in row):
        raise ReferenceBenchmarkError("frozen continuation outside vocabulary")
    dataset_digest = (commitment(cohort_key, "quality-prompts-v2", prompt_set) if record
                      else _digest(b"pllm.reference_prompts.v1\0", prompt_set))
    metric = ReferenceAgreement(
        dataset_digest=dataset_digest,
        reference_checkpoint_digest=checkpoint_digest,
        top_k=top_k,
    )
    token_cohort: tuple[tuple[int, ...], ...] | None = None
    if record:
        for candidate, candidate_options, _, admitted_source in configurations:
            if admitted_source is not None:
                current = _source_ids(admitted_source, prompt_set, runtime=candidate_options.client_runtime)
                if token_cohort is not None and current != token_cohort:
                    raise ReferenceBenchmarkError("shared source/cohort preflight binding failed")
                token_cohort = current
        if token_cohort is not None:
            for candidate, candidate_options, _, _ in configurations:
                extra_prefix = decode_steps if _final_token_runtime(candidate_options.client_runtime) else 0
                if any(not 1 <= len(row) or len(row) + extra_prefix > candidate.budget.max_input_tokens for row in token_cohort):
                    raise ReferenceBenchmarkError("tokenized cohort exceeds the Experiment input budget")
    decode_cohort: list[tuple[int, ...]] = ([tuple(row) for row in continuations] if record and decode_steps
                                          else [()] * len(prompt_set) if record else [])
    pending_decode: list[list[np.ndarray | int]] = []
    pending: list[tuple[dict[str, Any], list[np.ndarray | int]]] = []
    batches = []
    stopped = cancelled_run = False
    for experiment, options, resolved, admitted in configurations:
        assert resolved.path is not None
        batch = CandidateOutcomes(len(prompt_set) * (decode_steps + 1)) if record else None
        batches.append(batch)
        if record and stopped:
            batch.fence("candidate_stopped")
            metadata = {"name": experiment.name, "configuration_digest": experiment.configuration_digest(),
                "pipeline_digest": experiment.pipeline.digest()}
            if options.client_runtime == "compiled_fss_v1":
                metadata.update(_fss_quality_identity(experiment, admitted, decode_steps, prompt_set))
            pending.append((metadata, []))
            pending_decode.append([])
            continue
        if admitted is not None:
            runner = (_fss_candidate if options.client_runtime == "compiled_fss_v1"
                      else _replicated_candidate if options.client_runtime == "compiled_replicated_v1"
                      else _delegated_candidate)
            result_rows = None
            teacher = None if token_cohort is None else decode_cohort
            if record:
                teacher = decode_cohort if decode_steps else None
            with CaptureFailure(record) as phase:
                if not stopped:
                    kwargs = {"outcomes": batch, "expected_cohort": token_cohort} if record else {}
                    try:
                        result_rows = runner(experiment, prompt_set, admitted, decode_steps,
                            teacher, memory_safety_margin_percent, **kwargs)
                    finally:
                        kwargs.clear()
            if result_rows is None:
                if not record:
                    raise ReferenceBenchmarkError("quality candidate failed")
                current_cohort = token_cohort
                if current_cohort is None:
                    current_cohort = _source_ids(admitted, prompt_set, runtime=options.client_runtime)
                extra_prefix = decode_steps if _final_token_runtime(options.client_runtime) else 0
                if any(not 1 <= len(row) or len(row) + extra_prefix > experiment.budget.max_input_tokens for row in current_cohort):
                    raise ReferenceBenchmarkError("quality prompt exceeds the Experiment input budget")
                batch.fence(phase.code)
                stopped, cancelled_run = True, cancelled_run or phase.cancelled
                metadata = {"name": experiment.name, "configuration_digest": experiment.configuration_digest(),
                    "pipeline_digest": experiment.pipeline.digest(), "numeric_profile": admitted.contract["numeric_profile"],
                    "candidate_logits_available": not _final_token_runtime(options.client_runtime)}
                if options.client_runtime == "compiled_fss_v1":
                    metadata.update(_fss_quality_identity(experiment, admitted, decode_steps, prompt_set))
                actual_rows, decode_rows, selected = [], [], decode_cohort
            else:
                metadata, actual_rows, decode_rows, current_cohort, selected = result_rows
                result_rows = None
            if token_cohort is not None and token_cohort != current_cohort:
                raise ReferenceBenchmarkError("quality candidates tokenized the cohort differently")
            if token_cohort is None and not (record and decode_steps):
                decode_cohort = selected
            token_cohort = current_cohort
            pending.append((metadata, actual_rows))
            pending_decode.append(decode_rows)
            continue
        model_id = resolved.manifest.id
        kernel_choice = experiment.pipeline.components["kernels"]
        threads = kernel_choice.params.get("threads")
        metal_min_rows = kernel_choice.params.get("min_rows")
        engine = compiled = bundle = remote = None
        with CaptureFailure(record) as setup_phase:
            engine = MaskedTransformerEngine(
                weight_bits=options.weight_bits, activation_bits=options.activation_bits,
                threads=threads, metal_min_rows=metal_min_rows,
                public_equalization_digest=options.public_equalization_digest,
                client_prefix_layers=options.client_prefix_layers,
                client_linear_roles=options.client_linear_roles,
                client_linear_layer_roles=options.client_linear_layer_roles,
                remote_output_head=options.remote_output_head,
                prepared_output_encoding=options.prepared_output_encoding,
            )
            asyncio.run(engine.load(resolved.manifest))
            # Reuse the immutable document without a second weight body.
            bundle = (ClientBundle._from_document(engine.client_bundle_document(model_id)._value)
                      if reference_storage == "streamed"
                      else ClientBundle.unpack(engine.client_bundle(model_id)))
            config_bytes = (resolved.path / "config.json").read_bytes()
            from pllm.modeling import lower_model
            plan = lower_model(config_bytes, batch=1,
                max_input_tokens=experiment.budget.max_input_tokens,
                max_new_tokens=experiment.budget.max_new_tokens)
            compiled = compile_runtime_model(plan, bundle, composition=experiment.pipeline)
            remote = _candidate_remote(engine, model_id)
            encoded = [tuple(compiled.runtime(remote).encode_prompt(prompt)) for prompt in prompt_set]
            if record and token_cohort is not None and tuple(encoded) != token_cohort:
                from .quality_outcomes import QualityIntegrityError
                raise QualityIntegrityError("quality source/cohort binding failed")
        if setup_phase.failed:
            engine = compiled = bundle = remote = None
            if token_cohort is None:
                raise ReferenceBenchmarkError("quality source/cohort preflight failed") from None
            batch.fence(setup_phase.code)
            stopped, cancelled_run = True, cancelled_run or setup_phase.cancelled
            pending.append(({"name": experiment.name, "configuration_digest": experiment.configuration_digest(),
                "pipeline_digest": experiment.pipeline.digest()}, []))
            pending_decode.append([])
            continue
        ids_this_candidate: list[tuple[int, ...]] = []
        actual_rows: list[np.ndarray] = []
        decode_rows: list[np.ndarray] = []
        if record and token_cohort is None:
            for candidate, candidate_options, _, _ in configurations:
                extra_prefix = decode_steps if _final_token_runtime(candidate_options.client_runtime) else 0
                if any(not 1 <= len(row) or len(row) + extra_prefix > candidate.budget.max_input_tokens for row in encoded):
                    raise ReferenceBenchmarkError("tokenized cohort exceeds the Experiment input budget")
            token_cohort = tuple(encoded)
        for prompt_index, ids in enumerate(encoded):
            if not 1 <= len(ids) <= experiment.budget.max_input_tokens:
                raise ReferenceBenchmarkError(
                    "tokenized prompt exceeds the Experiment input budget"
                )
            ids_this_candidate.append(ids)
            session = runtime = None
            try:
                if record:
                    batch.current, batch.attempted = prompt_index * (decode_steps + 1), False
                session = compiled.session(remote) if not decode_steps else None
                runtime = compiled.runtime(remote) if decode_steps else None
                compiled.validate()
                if record:
                    batch.attempted = True
                prefill = (session.prefill_ids(ids) if session is not None
                           else runtime.prepare_ids(ids)[1])
                actual = np.array(prefill, dtype=np.float32, copy=True)
                if record:
                    if actual.shape != (resolved.manifest.vocab_size,) or not np.isfinite(actual).all():
                        raise ReferenceBenchmarkError("quality output malformed")
                    batch.record(prompt_index * (decode_steps + 1), Outcome("accepted"), actual)
                else:
                    actual_rows.append(actual)
                selected = []
                for step in range(decode_steps):
                    # The first candidate freezes the public diagnostic trajectory;
                    # every candidate and the reference consume the same tokens.
                    token = (int(np.argmax(actual)) if token_cohort is None and not record
                             else decode_cohort[prompt_index][step])
                    selected.append(token)
                    compiled.validate()
                    if record:
                        batch.current = prompt_index * (decode_steps + 1) + step + 1
                    actual = np.array(runtime.decode_step(token, runtime.caches, len(ids) + step)[0],
                                       dtype=np.float32, copy=True)
                    if record:
                        if actual.shape != (resolved.manifest.vocab_size,) or not np.isfinite(actual).all():
                            raise ReferenceBenchmarkError("quality output malformed")
                        batch.record(prompt_index * (decode_steps + 1) + step + 1, Outcome("accepted"), actual)
                    else:
                        decode_rows.append(actual)
                if token_cohort is None and not record:
                    decode_cohort.append(tuple(selected))
            except (Exception, KeyboardInterrupt, asyncio.CancelledError) as error:
                if not record:
                    raise
                from .delegated_protocol import clear_failure_frames
                clear_failure_frames(error)
                cancelled_run = isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
                stopped = True
                batch.fence("cancelled" if cancelled_run else "execution_failed")
            finally:
                with CaptureFailure(record) as cleanup_phase:
                    if session is not None:
                        session.close()
                    if runtime is not None:
                        runtime.reset()
                if cleanup_phase.failed:
                    batch.fence(cleanup_phase.code)
                    stopped, cancelled_run = True, cancelled_run or cleanup_phase.cancelled
                del session, runtime
            if record and stopped:
                break
        current_cohort = tuple(encoded)
        if token_cohort is not None and token_cohort != tuple(encoded):
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
    trajectories = []
    with CaptureFailure(record) as reference_phase:
        if not stopped:
            if torch is None:
                torch, transformers = _reference_dependencies(initialize=True)
                reference_threads = torch.get_num_threads()
            trajectories, loading = _reference_trajectories(resolved_source, configurations, token_cohort,
                decode_cohort, reference_storage, decode_steps, memory_safety_margin_percent)
    reference_complete = not stopped and not reference_phase.failed
    cancelled_run = cancelled_run or reference_phase.cancelled
    if not reference_complete:
        trajectories = []
    gc.collect()
    expected_rows = [rows[0] for rows in trajectories]
    expected_decode = [row for rows in trajectories for row in rows[1:]]
    expected = {index: row for index, row in enumerate(row for trajectory in trajectories for row in trajectory)}
    scored = []
    with CaptureFailure(record) as scoring_phase:
        if record and reference_complete:
            if (len(trajectories) != len(prompt_set) or any(len(rows) != decode_steps + 1 for rows in trajectories)
                    or any(row.shape != (resolved_source.manifest.vocab_size,) or not np.isfinite(row).all()
                           for row in expected.values())):
                raise ReferenceBenchmarkError("reference output malformed")
            for index, (metadata, _) in enumerate(pending):
                final = metadata.get("candidate_logits_available") is False
                scores = {}
                for position, value in batches[index].values.items():
                    score = measure_reference_agreement(None if final else value, expected[position], top_k=top_k,
                        selected_token=value if final else None)
                    if any(item is not None and not np.isfinite(item) for item in score.values()):
                        raise ReferenceBenchmarkError("reference agreement unavailable")
                    scores[(id(value), id(expected[position]))] = score
                scored.append(scores)
    if scoring_phase.failed:
        reference_complete, reference_phase.failed = False, True
        cancelled_run = cancelled_run or scoring_phase.cancelled
        scored.clear()
    candidates: list[dict[str, Any]] = []
    for candidate_index, ((metadata, actual_rows), decode_rows) in enumerate(zip(pending, pending_decode, strict=True)):
        final_tokens = metadata.get("candidate_logits_available") is False
        if record:
            from .quality_outcomes import summary
            batch = batches[candidate_index]
            batch.fence("candidate_stopped")
            metric_call = lambda actual, reference: scored[candidate_index][(id(actual), id(reference))]
            public_metadata = {key: value for key, value in metadata.items() if key in {
                "name", "configuration_digest", "pipeline_digest", "weight_bits", "activation_bits", "numeric_profile",
                "local_numeric_profile", "fraction_bits", "ring_bits", "kernel_threads", "kernel_backend",
                "execution", "candidate_logits_available", "diagnostic_requests", "output_selections_per_request",
                "public_equalization_profile_digest", "public_calibration_digest", "public_profile_bytes"}}
            try:
                prefill_indices = list(range(0, batch.total, decode_steps + 1))
                row = {**public_metadata, **summary(batch, prefill_indices, expected, metric_call,
                                                  reference_complete=reference_complete)}
                row["all_positions"] = summary(batch, list(range(batch.total)), expected, metric_call,
                                               reference_complete=reference_complete)
                if decode_steps:
                    indices = [index for index in range(batch.total) if index % (decode_steps + 1)]
                    row["same_token_decode"] = summary(batch, indices, expected, metric_call,
                                                       reference_complete=reference_complete)
                candidates.append(row)
            finally:
                batch.clear()
            continue
        candidates.append({**metadata, **_agreement_summary(actual_rows, expected_rows,
                           top_k=top_k, final_tokens=final_tokens)})
        if decode_steps:
            candidates[-1]["same_token_decode"] = _agreement_summary(decode_rows, expected_decode,
                top_k=top_k, final_tokens=final_tokens)
    environment = {
        "system": platform.system(),
        "machine": platform.machine(),
        # Skipped reference in record mode must not import frameworks merely to
        # report versions/threads. None means reference initialization not done.
        "python_reference": f"transformers/{transformers.__version__}" if transformers is not None else None,
        "torch": str(torch.__version__) if torch is not None else None,
        "numpy": np.__version__,
        "reference_dtype": "float32",
        "reference_attention": "eager",
        "reference_threads": reference_threads,
    }
    if loading is not None:
        environment["reference_weight_storage"] = "module-streamed"
    delegated = any(admitted is not None for _, _, _, admitted in configurations)
    scope = "local-compiled-decoder" if delegated else "local-compiled-clear-kernel"
    report = {
        "schema_version": "pllm.reference_quality_benchmark.v2" if record else SCHEMA,
        "scope": scope + ("-prefill-and-same-token-decode-reference" if decode_steps else "-prefill-reference"),
        "model": {
            "id": resolved_source.manifest.id,
            "checkpoint_digest": checkpoint_digest,
            "source_lock_digest": source_lock,
        },
        "cohort": {
            "dataset_digest": dataset_digest,
            "token_cohort_digest": (commitment(cohort_key, "quality-tokens-v2", token_cohort) if record
                                    else _digest(b"pllm.reference_tokens.v1\0", token_cohort)),
            "prompt_count": len(prompt_set),
            "input_token_counts": [len(ids) for ids in token_cohort],
            "phase": "prefill",
            "metric": metric.to_spec(),
            **({"same_token_decode": {
                "steps_per_prompt": decode_steps,
                "teacher_forcing": "caller-frozen" if record else "first-candidate-greedy",
                **({} if record else {"teacher_configuration_digest": configurations[0][0].configuration_digest()}),
                "token_cohort_digest": (commitment(cohort_key, "quality-teacher-v2", decode_cohort) if record
                                        else _digest(b"pllm.reference_decode_tokens.v1\0", decode_cohort)),
            }} if decode_steps else {}),
        },
        "environment": environment,
        "environment_digest": _digest(b"pllm.reference_environment.v1\0", environment),
        "candidates": candidates,
        **({"reference_loading": loading} if loading is not None else {}),
        "limitations": [
            *([note for batch in batches
               if (note := getattr(batch, "_memory_guard_note", None)) is not None] if record else []),
            ("local worker execution; serving costs require benchmark run; no independent-operator privacy measurement"
             if any(admitted is not None for _, _, _, admitted in configurations)
             else "local clear integer kernels, not a provider transport or privacy measurement"),
            "prefill and optional same-token decode agreement, not full-generation or task accuracy",
            "float32 reference loading and evaluation excluded from provider latency and cost",
            *(["final-token protocols expose top-1 only; fresh one-selection prefix requests are numeric diagnostics, not one serving response"]
               if any(row.get("candidate_logits_available") is False for row in candidates) else []),
            *(["FSS source-model agreement requires this pinned FP32 comparison; independent integer/Q12 correctness is not source-model reference quality"]
               if any(options.client_runtime == "compiled_fss_v1" for _, options, _, _ in configurations) else []),
        ],
    }
    if record:
        report["cohort"].update(position_count=len(prompt_set) * (decode_steps + 1),
            commitment_scheme="hmac-sha256-campaign-v1", cohort_key_id=cohort_key_id)
        report["reference_status"] = "complete" if reference_complete else "unavailable"
        report["terminal_state"] = ("cancelled" if cancelled_run else "execution_failed" if stopped or reference_phase.failed
                                    else "incomplete" if any(row["all_positions"]["coverage"]["rejected"] for row in candidates)
                                    else "complete")
        # Release owned aliases before public cancellation; not secure erasure.
        # Caller-owned objects are never modified.
        for batch in batches:
            batch.clear()
        pending.clear()
        pending_decode.clear()
        expected_rows.clear()
        expected_decode.clear()
        expected.clear()
        trajectories.clear()
        scored.clear()
        prompt_set = prompts = token_cohort = decode_cohort = continuations = cohort_key = None
        current = current_cohort = encoded = ids = ids_this_candidate = selected = teacher = actual = prefill = value = token = None
        result_rows = kwargs = None
        if cancelled_run:
            from .quality_outcomes import QualityRunCancelled
            raise QualityRunCancelled(report)
    return report


__all__ = ["ReferenceBenchmarkError", "run_reference_benchmark"]
