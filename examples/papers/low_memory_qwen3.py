"""Inert Qwen3-4B 64+8 memory experiment; no downloads or execution on import.

Reuse the source pin from examples/benchmarks/qwen3_4b.py and the paged storage
mechanism from examples/benchmarks/client_memory.py. Both arms retain all body
linear work on providers, full client KV, and identical explicit W8A8/prefix_f32
arithmetic. The candidate uses existing paged public bundle storage, not demand
preparation; it is not a new method or a promised tenfold reduction. Both arms
prewarm 71 uniform one-use rows (64 prefill + 7 decode) before online execution.
Idle refill stays idle-only; reading committed pages is not material generation.

One ordinary invocation (only after authorizing the expensive measurement):

    from pathlib import Path
    from pllm.metrics import BuildRef, benchmark_suite
    from examples.papers.low_memory_qwen3 import matched_suite, private_inputs

    # Public candidate: tokenizer-only 64, not an accepted 64/8 response.
    prompt = Path("examples/papers/low_memory_qwen3_prompt.txt").read_text()
    suite = matched_suite(build=BuildRef.capture(), memory_bytes=host_budget_bytes)
    run = benchmark_suite(suite, inputs=private_inputs(prompt),
                          output="target/low-memory-qwen3.json")

The ordinary runner supplies one private cohort salt and a fresh isolated worker
per arm/replicate. It retains blocked, unsupported, failed and timed-out attempts.
Do not execute arms separately or merge salts. No admission/host guard is changed.
``memory_bytes`` is a caller-chosen whole-topology allowance within the existing
16 GiB API ceiling, not a client RSS target. Admission can block either arm.

BenchmarkRun.complete only means every attempt has a record, not acceptance.
Require completed, admitted, guard-safe attempts with authoritative counts and
matching source/body/numeric/private workload/output-state identities. Keep failed
attempts and unavailable metrics. Ordinary expected_input_tokens/expected_output_tokens
checks now enforce actual 64/8 equality in both targets; the execution budget and
output cap remain ceilings, not forced length. Early EOS fails. Verify prompt by
authoritative usage, not an external tokenizer guess. The existing state audit
commits every selected full-logit/active-KV vector, positions, selections and RNG;
its greedy sampler uses seed 42 and fresh cryptographic randomness. Same-cohort
BenchmarkRun.compare checks matching source/body/workload/output-state contracts.
It does not compare independently salted runs or prove BF16 fidelity/task quality,
stochastic parity, intermediate-tensor equality, history or pending-token equality.

process.peak_rss and client.peak_rss are individual-process diagnostics, not a
complete client/helper group. CLIENT_MEMORY_METRICS selects ordinary guarded
owner-aware comparisons: sampled sums are descriptive estimates; individual
observations are lower bounds; only complete isolated lifetime intervals can
establish a scoped reduction bound. Unknown helper membership, source setup,
per-member HWM or retirement must remain unranked, never root-only success.
Provider resources, external filesystem cache and disk snapshots stay separate.
The historical 8.25 GiB estimate and rounded 0.83 GiB budget are not observations
or thresholds here. Only matched measured peaks can establish a tenfold ratio;
no ratio or qualification is produced by importing these factories.
"""
from __future__ import annotations

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.deployment import WanConditions
from pllm.kernels import Cpu
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, BuildRef, MetricSpec, Sampling, Scenario, TargetBudget
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport, MaskedLinear
from pllm.quantization import SymmetricPerRow


MODEL_ID = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
SOURCE = Model.hf(MODEL_ID, revision=REVISION, local_files_only=True)
BUDGET = ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=8)
PREPARED_ROWS = BUDGET.max_input_tokens + BUDGET.max_new_tokens - 1
NAMES = ("maintained", "paged-prepared")
CLIENT_MEMORY_METRICS = tuple(MetricSpec(name) for name in (
    "client.owned_sampled_peak_rss_sum", "client.owned_peak_rss_lower_bound",
    "client.owned_lifetime_peak_rss_sum_interval"))

# Measurement requirements, not invented telemetry or observed results.
MEASUREMENTS = {
    "process.peak_rss": ("process_birth_through_target", "process_lifetime_high_water"),
    "client.peak_rss": ("process_birth_through_scenario", "process_lifetime_high_water"),
    "client.owned_sampled_peak_rss_sum": (
        "benchmark_sampler_start_through_responses_before_teardown", "non_atomic_same_pass_owner_declared_RSS_sum"),
    "client.owned_peak_rss_lower_bound": (
        "benchmark_sampler_start_through_responses_before_teardown", "maximum_individual_RSS_observation_not_group_peak"),
    "client.cold_cpu": ("benchmark_start_through_first_response", "process_cpu"),
    "preparation.cold_cpu": ("role_birth_through_first_response", "process_cpu"),
    "preparation.peak_rss": ("request_windows", "sampled_process_rss"),
    "response.ttft": ("online_to_first_token", "target"),
    "response.decode_tps": ("after_first_output", "target"),
    "network.all_link_stream_bytes": ("relay_start_through_scenario", "tcp_stream_payload"),
    "request.0.full": ("request_setup_through_completion", "ordinary_response_lifecycle"),
}


def _pipeline(*, paged: bool) -> MaskedLinearCpu:
    return MaskedLinearCpu(
        SOURCE,
        kernels=Cpu(threads=1),
        quantization=SymmetricPerRow(
            weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
        linear=MaskedLinear(),
        inventory=PreparedInventory("prewarm", rows=PREPARED_ROWS, refill="idle"),
        delivery=(ClientBundleTransport("artifacts", storage="paged", compression="zlib",
                                        batch_objects=64) if paged else None),
    )


def validate_pair(maintained: Experiment, candidate: Experiment) -> None:
    """Reject out-of-scope public configurations, not substitute source admission.

    No request-sized/demand issuance, decode epochs, cache/state substitutions or
    relaxed numerics are eligible. The ordinary source/compiler/native/privacy
    gates still decide actual support; metadata cannot qualify checkpoint tensors.
    """
    for name, value, paged in zip(NAMES, (maintained, candidate), (False, True), strict=True):
        if not isinstance(value, Experiment):
            raise TypeError(f"{name}: an Experiment is required")
        if value.pipeline.model != SOURCE:
            raise ValueError(f"{name}: pinned offline source required")
        if value.budget != BUDGET or value.deployment.kind != "local":
            raise ValueError(f"{name}: local 64/8 execution budget required")
        inventory = value.pipeline.components.get("inventory")
        if inventory is None or inventory != _pipeline(paged=False).components["inventory"]:
            raise ValueError(f"{name}: committed prewarm inventory of {PREPARED_ROWS} rows with idle refill required")
        if value.pipeline.canonical_bytes() != _pipeline(paged=paged).canonical_bytes():
            raise ValueError(f"{name}: pinned numeric/body/state/storage composition required")


def recipes():
    """Configure existing paths only; source/tensor/host admission is deferred.

    Both arms are maintained prepared with the same prewarmed rows. Candidate
    changes public-bundle delivery/storage, not inventory lifetime, kernels, weights,
    activation quantization, attention, body placement, or response-cache policy.
    Artifact framing/compression differ too; this is not a storage-only ablation.
    dashboard.py startup prewarms before readiness; client.py:_begin_preparation
    rejects online activity and _open_transformer_session rejects unavailable
    reservations. The full row bound avoids ordinary request-setup issuance.
    The shared foreground gate fails on missing inventory, retaining ownership
    through cleanup; explicit idle preparation remains supported. No recipe
    authorizes active generation. Demand/epoch alternatives stay blocked below.
    """
    return {
        name: Experiment(
            f"low-memory-qwen3-{name}",
            _pipeline(paged=name == "paged-prepared"),
            Deployment.local(root=f".pllm/papers/low-memory-qwen3/{name}"),
            BUDGET,
        )
        for name in NAMES
    }


def matched_suite(*, build: BuildRef, memory_bytes: int, processes: int = 3,
                  timeout_seconds: float = 3600) -> BenchmarkSuite:
    """One ordinary suite with exact-count gates; matched trajectory is post-run.

    Require a caller-captured build; never inspect/build/install an interpreter.
    Missing peaks remain unknown through ordinary metric-budget assessments.
    The nonnegative RSS budget is an availability check, not Low/10x eligibility.
    """
    if not isinstance(build, BuildRef) or build.expected_digest is None:
        raise ValueError("a caller-captured BuildRef with expected_digest is required")
    if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 3600:
        raise ValueError("prepared response timeout_seconds must be in (0, 3600]")
    configured = recipes()
    validate_pair(configured[NAMES[0]], configured[NAMES[1]])
    cases = tuple(
        BenchmarkCase(
            name,
            BenchmarkTarget.response(experiment, backend="native", wan=WanConditions(),
                                      scenario=Scenario.cold(max_output_tokens=8), state_audit=True).with_params(
                                          expected_input_tokens=64, expected_output_tokens=8),
            build,
            budgets=tuple(TargetBudget(metric, minimum=0) for metric in MEASUREMENTS),
        )
        for name, experiment in configured.items()
    )
    return BenchmarkSuite(cases, Sampling(processes=processes, repetitions=1, warmups=0,
        order="balanced", threads=1, timeout_seconds=timeout_seconds, memory_bytes=memory_bytes),
        "low-memory-qwen3-64x8-v2")


def cli_suite() -> BenchmarkSuite:
    """Zero-argument trusted CLI factory; all build/resource choices are explicit.

    No capture/build, model resolution or execution occurs here. Environment values
    are public configuration, serialized into BuildRef/Sampling in the suite.
    """
    import os

    required = ("PLLM_LOW_MEMORY_BUILD_DIGEST", "PLLM_LOW_MEMORY_BYTES",
                "PLLM_LOW_MEMORY_PROCESSES", "PLLM_LOW_MEMORY_TIMEOUT_SECONDS")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError("explicit memory campaign settings required: " + ", ".join(missing))
    return matched_suite(
        build=BuildRef(expected_digest=os.environ[required[0]]),
        memory_bytes=int(os.environ[required[1]]),
        processes=int(os.environ[required[2]]),
        timeout_seconds=float(os.environ[required[3]]),
    )


def private_inputs(prompt: str) -> dict[str, dict[str, str]]:
    """Same caller-owned plain-text input for both arms; no token-count claim."""
    if type(prompt) is not str or not prompt.strip() or len(prompt.encode()) > 16_384:
        raise ValueError("prompt must contain 1 to 16384 UTF-8 bytes")
    return {name: {"prompt": prompt} for name in NAMES}


def blocked_cells() -> tuple[dict[str, str], ...]:
    """Unqualified alternative stays explicit, without inventing source facts.

    PrivateLookupAndFullHead is deliberately NOT instantiated: this factory has
    not established its tied, bias-free source/head contract for the pinned model.
    Ordinary paged prepared execution does not require tied embeddings.
    """
    return ({
        "name": "paged-demand",
        "status": "blocked",
        "reason": "foreground_missing_inventory_preparation_not_eligible",
        "required_contract": (
            "Explicit idle preparation is supported; the foreground gate now rejects "
            "implicit generation. Demand/request-sized allocation remains excluded. Require "
            "committed one-use inventory before response admission, fail-fast exhaustion, "
            "and no foreground/online generation or decode-epoch issuance. The configured "
            "prewarm alternative reserves the full 71-row bound before online; ordinary "
            "privacy, source, compiler and host admission remain mandatory."
        ),
    }, {
        "name": "coupled-boundary",
        "status": "blocked",
        "reason": "source_boundary_contract_not_qualified",
        "required_contract": (
            "Locked source must pass decoder_boundary.rs and private_boundary_loader.py: "
            "tied bias-free W8A8 lookup/full head, matching prefill/decode source and scale "
            "identities, native table/ring/frame/workload bounds, PreparedBoundaryRoles, "
            "unchanged prepared body, and fresh complete-client audit/host admission. "
            "Its changed four-provider privacy/measurement contract needs separate reviewed "
            "comparison; do not force BenchmarkRun.compare across privacy digests."
        ),
    },)
