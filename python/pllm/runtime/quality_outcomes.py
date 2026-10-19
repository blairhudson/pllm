"""Shared quality v2: bounded aligned outcomes, never persisted model values.

V1 remains abort-only and is not automatically migrated: historical sample counts
cannot reconstruct acceptance masks. V2 replay needs the original private campaign
key plus immutable inputs/source to reproduce commitments; reports never store it.
Conditional scores describe accepted positions only and never qualify a partial
cohort for ranking. Existing ranking execution-scope/metric requirements still
apply (complete final-token-only reports do not invent top-k/logit evidence).
"""
from dataclasses import dataclass
import asyncio
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import tempfile

from pllm import _native

SCHEMA = "pllm.reference_quality_benchmark.v2"
CODES = frozenset({"numeric_domain_rejected", "execution_failed", "admission_failed", "transport_failed",
                  "malformed_output", "source_integrity_failed", "cancelled", "candidate_stopped", "reference_unavailable"})


class QualityIntegrityError(ValueError):
    """An immutable shared source/cohort binding failed; never numeric rejection."""


def numeric_rejection(error):
    kind = getattr(_native, "NumericDomainRejected", None)
    return numeric_rejection_available() and type(error) is kind


def numeric_rejection_available():
    kind = getattr(_native, "NumericDomainRejected", None)
    return isinstance(kind, type) and issubclass(kind, ValueError)


def commitment(key, domain, value):
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hmac.new(key, domain.encode() + b"\0" + encoded, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class Outcome:
    status: str
    code: str | None = None
    attempted: bool = True

    def __post_init__(self):
        if (type(self.status) is not str or type(self.code) not in (str, type(None))
                or self.status not in {"accepted", "rejected", "failed"} or type(self.attempted) is not bool
                or self.status == "accepted" and (self.code is not None or not self.attempted)
                or self.status == "rejected" and (self.code != "numeric_domain_rejected" or not self.attempted)
                or self.status == "failed" and (self.code not in CODES or self.code == "numeric_domain_rejected")):
            raise ValueError("invalid quality outcome")


def coverage(rows):
    return {"total": len(rows), **{name: sum(row.status == name for row in rows)
                                  for name in ("accepted", "rejected", "failed")}}


class CandidateOutcomes:
    """Ephemeral accepted values owned by shared benchmark; serialized rows omit them."""
    def __init__(self, total):
        if type(total) is not int or not 1 <= total <= 288:
            raise ValueError("quality position bound")
        self.total, self.rows, self.values = total, {}, {}
        self.current = 0
        self.attempted = False

    def record(self, index, outcome, value=None):
        if type(outcome) is not Outcome or type(index) is not int or not 0 <= index < self.total or index in self.rows:
            raise ValueError("quality position already terminal or outside cohort")
        self.rows[index] = outcome
        if outcome.status == "accepted":
            self.values[index] = value

    def fence(self, code="execution_failed"):
        for index in range(self.total):
            if index not in self.rows:
                self.record(index, Outcome("failed", code if index == self.current else "candidate_stopped",
                                           self.attempted if index == self.current else False))

    def clear(self):
        self.values.clear()


class CaptureFailure:
    """Record mode only: retain fixed failure state, never exception objects/text."""
    def __init__(self, enabled):
        self.enabled, self.failed, self.cancelled = enabled, False, False
        self.code = "execution_failed"

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is None or not self.enabled:
            return False
        cancelled = isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
        if not isinstance(error, Exception) and not cancelled:
            return False
        from .delegated_protocol import clear_failure_frames
        clear_failure_frames(error)
        self.failed, self.cancelled = True, cancelled
        from .benchmark_memory import BenchmarkMemoryError
        from .client import ProtocolError
        self.code = ("cancelled" if cancelled else "admission_failed" if isinstance(error, BenchmarkMemoryError)
                     else "source_integrity_failed" if isinstance(error, QualityIntegrityError)
                     else "transport_failed" if isinstance(error, (ProtocolError, TimeoutError, ConnectionError))
                     else "execution_failed")
        return True


class QualityRunCancelled(KeyboardInterrupt):
    def __init__(self, report):
        super().__init__("quality run cancelled")
        self.report = report


def summary(batch, indices, expected, metric, *, reference_complete=True):
    if any(index not in batch.rows for index in indices):
        raise ValueError("quality outcome coverage incomplete")
    rows = [batch.rows[index] for index in indices]
    accepted = [index for index in indices if batch.rows[index].status == "accepted"]
    samples = [metric(batch.values[index], expected[index]) for index in accepted] if reference_complete else []
    scores = {}
    for name in ("top1_agreement", "top_k_recall", "max_abs_logit_error"):
        values = [sample[name] for sample in samples]
        if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value)
               or value < 0 or name != "max_abs_logit_error" and value > 1) for value in values):
            raise ValueError("invalid quality metric")
        scores[name] = (None if not values or any(value is None for value in values)
                        else max(values) if name == "max_abs_logit_error" else sum(values) / len(values))
    complete = reference_complete and len(accepted) == len(indices)
    return {"coverage": coverage(rows), "planned_sample_count": len(indices), "sample_count": len(samples),
            "score_status": "complete" if complete else "incomplete" if reference_complete else "reference_unavailable",
            "outcomes": [{"position": index, "status": batch.rows[index].status,
                          "code": batch.rows[index].code, "attempted": batch.rows[index].attempted} for index in indices],
            "accepted_agreement": {"sample_count": len(samples), **scores},
            **{name: value if complete else None for name, value in scores.items()}}


def require_complete_for_ranking(report):
    validate_report(report)
    total = report.get("cohort", {}).get("position_count")
    if (report.get("schema_version") != SCHEMA or type(total) is not int or not 1 <= total <= 288
            or report.get("reference_status") != "complete"):
        raise ValueError("quality reference/cohort unavailable")
    if report.get("terminal_state") != "complete":
        raise ValueError("partial or failed quality coverage cannot be ranked")
    rows = report.get("candidates")
    if type(rows) is not list or not 1 <= len(rows) <= 8:
        raise ValueError("invalid quality candidates")
    for row in rows:
        all_rows = row.get("all_positions", row)
        counts = all_rows.get("coverage", {})
        if (any(type(v) is not int for v in counts.values())
                or counts != {"total": total, "accepted": total, "rejected": 0, "failed": 0}
                or all_rows.get("score_status") != "complete"):
            raise ValueError("partial quality coverage cannot be ranked")
        outcomes = all_rows.get("outcomes", [])
        if len(outcomes) != total or any(type(item.get("position")) is not int
                or type(item.get("attempted")) is not bool for item in outcomes):
            raise ValueError("quality position coverage invalid")
        if outcomes != [{"position": i, "status": "accepted", "code": None, "attempted": True} for i in range(total)]:
            raise ValueError("quality position masks differ")


def validate_report(report):
    """Validate v2 logical coverage and metric availability before replay/ranking."""
    remaining = 32768
    def bounded(value, depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 12:
            raise ValueError("quality report structure bound")
        kind = type(value)
        if kind is dict:
            if any(type(key) is not str or len(key) > 128 for key in value):
                raise ValueError("quality report field bound")
            for item in value.values():
                bounded(item, depth + 1)
        elif kind is list:
            for item in value:
                bounded(item, depth + 1)
        elif kind is str:
            if len(value) > 1024:
                raise ValueError("quality metadata bound")
        elif kind is int:
            if value.bit_length() > 128:
                raise ValueError("quality integer bound")
        elif kind is float:
            if not math.isfinite(value):
                raise ValueError("quality metric must be finite")
        elif kind not in (bool, type(None)):
            raise ValueError("quality reports contain JSON primitives only")
    bounded(report)
    if type(report) is not dict or report.get("schema_version") != SCHEMA:
        raise ValueError("expected quality report v2")
    def fields(value, allowed):
        if type(value) is not dict or set(value) - set(allowed.split()):
            raise ValueError("unexpected quality report fields")
    fields(report, "schema_version scope model cohort candidates environment environment_digest reference_loading limitations reference_status terminal_state")
    fields(report.get("model"), "id checkpoint_digest source_lock_digest")
    def digest(value):
        return type(value) is str and re.fullmatch("[a-f0-9]{64}", value) is not None
    if (type(report["model"].get("id")) is not str or not report["model"]["id"]
            or not digest(report["model"].get("checkpoint_digest"))
            or not digest(report["model"].get("source_lock_digest"))
            or not digest(report.get("environment_digest"))):
        raise ValueError("quality source/environment locks invalid")
    fields(report.get("environment"), "system machine python_reference torch numpy reference_dtype reference_attention reference_threads reference_weight_storage")
    if "reference_loading" in report:
        fields(report["reference_loading"], "storage active_weight_bound_bytes allocation_estimate_bytes maximum_swap_growth_bytes kv_payload_bound_bytes buffer_bytes parameter_loads loaded_float32_bytes load_seconds peak_active_weight_bytes")
    cohort = report.get("cohort", {})
    fields(cohort, "position_count prompt_count commitment_scheme cohort_key_id dataset_digest token_cohort_digest input_token_counts phase metric same_token_decode")
    fields(cohort.get("metric"), "component params")
    fields(cohort["metric"].get("params"), "dataset_digest reference_checkpoint_digest top_k")
    if "same_token_decode" in cohort:
        fields(cohort["same_token_decode"], "steps_per_prompt teacher_forcing token_cohort_digest")
    total, prompts = cohort.get("position_count"), cohort.get("prompt_count")
    if (type(total) is not int or type(prompts) is not int or not 1 <= prompts <= 32
            or total % prompts or not 1 <= total // prompts <= 9
            or cohort.get("commitment_scheme") != "hmac-sha256-campaign-v1"
            or type(cohort.get("cohort_key_id")) is not str):
        raise ValueError("quality cohort invalid")
    if (not digest(cohort.get("dataset_digest")) or not digest(cohort.get("token_cohort_digest"))
            or re.fullmatch("[A-Za-z0-9_-]{1,64}", cohort["cohort_key_id"]) is None
            or type(cohort.get("input_token_counts")) is not list
            or len(cohort["input_token_counts"]) != prompts
            or any(type(count) is not int or count < 1 for count in cohort["input_token_counts"])
            or cohort.get("phase") != "prefill"):
        raise ValueError("quality cohort commitments/counts invalid")
    metric = cohort["metric"]
    if (metric.get("component") != "pllm/reference-agreement/v1"
            or metric["params"].get("dataset_digest") != cohort["dataset_digest"]
            or metric["params"].get("reference_checkpoint_digest") != report["model"]["checkpoint_digest"]
            or type(metric["params"].get("top_k")) is not int or not 1 <= metric["params"]["top_k"] <= 100):
        raise ValueError("quality metric binding invalid")
    rows = report.get("candidates")
    if type(rows) is not list or not 1 <= len(rows) <= 8:
        raise ValueError("quality candidate bound")
    if report.get("reference_status") not in {"complete", "unavailable"}:
        raise ValueError("quality reference state invalid")
    if report.get("terminal_state") not in {"complete", "incomplete", "execution_failed", "cancelled"}:
        raise ValueError("quality terminal state invalid")
    width = total // prompts
    if width > 1:
        teacher = cohort.get("same_token_decode", {})
        if (type(teacher.get("steps_per_prompt")) is not int or teacher["steps_per_prompt"] != width - 1
                or teacher.get("teacher_forcing") != "caller-frozen" or not digest(teacher.get("token_cohort_digest"))):
            raise ValueError("quality frozen teacher binding invalid")
    elif "same_token_decode" in cohort:
        raise ValueError("prefill report cannot carry decode cohort")
    def section(row, indices):
        outcomes = row.get("outcomes")
        if type(outcomes) is not list or len(outcomes) != len(indices):
            raise ValueError("quality outcome coverage invalid")
        parsed = []
        for item, index in zip(outcomes, indices, strict=True):
            if (type(item) is not dict or set(item) != {"position", "status", "code", "attempted"}
                    or type(item["position"]) is not int or item["position"] != index):
                raise ValueError("quality outcome position invalid")
            parsed.append(Outcome(item["status"], item["code"], item["attempted"]))
        counts = row.get("coverage", {})
        if (type(counts) is not dict or any(type(v) is not int for v in counts.values())
                or counts != coverage(parsed) or type(row.get("planned_sample_count")) is not int
                or row["planned_sample_count"] != len(indices)):
            raise ValueError("quality coverage totals invalid")
        complete = counts["accepted"] == len(indices) and report["reference_status"] == "complete"
        if row.get("score_status") != ("complete" if complete else "incomplete"
                if report["reference_status"] == "complete" else "reference_unavailable"):
            raise ValueError("quality score availability invalid")
        accepted = row.get("accepted_agreement", {})
        fields(accepted, "sample_count top1_agreement top_k_recall max_abs_logit_error")
        sample_count = counts["accepted"] if report["reference_status"] == "complete" else 0
        if (type(row.get("sample_count")) is not int or row["sample_count"] != sample_count
                or type(accepted.get("sample_count")) is not int or accepted["sample_count"] != sample_count):
            raise ValueError("quality scored coverage invalid")
        for name in ("top1_agreement", "top_k_recall", "max_abs_logit_error"):
            value = accepted.get(name)
            if name == "top1_agreement" and sample_count and value is None:
                raise ValueError("accepted quality score missing")
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)
                    or value < 0 or name != "max_abs_logit_error" and value > 1):
                raise ValueError("quality metric invalid")
            if sample_count == 0 and value is not None or row.get(name) != (value if complete else None):
                raise ValueError("partial quality metrics cannot be unconditional")
        return outcomes
    for row in rows:
        section_names = "coverage planned_sample_count sample_count score_status outcomes accepted_agreement top1_agreement top_k_recall max_abs_logit_error"
        fields(row, section_names + " all_positions same_token_decode name configuration_digest pipeline_digest weight_bits activation_bits numeric_profile local_numeric_profile fraction_bits ring_bits kernel_threads kernel_backend execution candidate_logits_available diagnostic_requests output_selections_per_request public_equalization_profile_digest public_calibration_digest public_profile_bytes")
        if (type(row.get("name")) is not str or not row["name"]
                or not digest(row.get("configuration_digest")) or not digest(row.get("pipeline_digest"))):
            raise ValueError("quality candidate identity invalid")
        fields(row.get("all_positions"), section_names)
        all_rows = section(row["all_positions"], list(range(total)))
        prefill = section(row, list(range(0, total, width)))
        if prefill != [item for item in all_rows if item["position"] % width == 0]:
            raise ValueError("quality phase outcomes disagree")
        if width > 1:
            fields(row.get("same_token_decode"), section_names)
            decode = section(row["same_token_decode"], [i for i in range(total) if i % width])
            if decode != [item for item in all_rows if item["position"] % width]:
                raise ValueError("quality decode outcomes disagree")
        elif "same_token_decode" in row:
            raise ValueError("prefill report cannot carry decode metrics")
    return report


def read_report(raw):
    if type(raw) not in (str, bytes) or len(raw) > 1 << 20:
        raise ValueError("quality report size bound")
    if type(raw) is str:
        try:
            raw = raw.encode("utf-8")
        except UnicodeError:
            raise ValueError("invalid quality report") from None
    if len(raw) > 1 << 20:
        raise ValueError("quality report size bound")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate quality field")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError("nonfinite quality JSON")
    try:
        return validate_report(json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant))
    except (RecursionError, OverflowError, UnicodeError, KeyError, TypeError, AttributeError):
        raise ValueError("invalid quality report") from None


def write_report(report, path, *, force=False):
    validate_report(report)
    raw = json.dumps(report, allow_nan=False, sort_keys=True, indent=2) + "\n"
    if len(raw.encode()) > 1 << 20:
        raise ValueError("quality report size bound")
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as destination:
            temporary = Path(destination.name)
            destination.write(raw)
        if force:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
