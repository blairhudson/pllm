"""Inert accepted-response campaign: two methods, three unmeasured cells.

Run only in an explicitly authorized matching installation with the pinned source
already cached. The ordinary suite engine supports SDK and trusted-factory CLI
dispatch, not a separate campaign runner:

    from pllm.metrics import BuildRef, benchmark_suite
    from examples.papers.accepted_response_comparison import suite, inputs

    selected = suite(build=BuildRef.capture(), memory_bytes=host_budget_bytes)
    run = benchmark_suite(selected, inputs=inputs(), output="target/responses.json")

The ordinary runner supplies one private cohort salt, balanced fresh processes,
and retained blocked/unsupported/failed/timeout attempts. No admission, disk,
swap, numeric-range, final-acceptance or authoritative-count gate is bypassed.
The existing `hi` workload renders as `user: hi`; three input tokens and one
fixed-count greedy output are REQUIREMENTS checked after execution, not results
inferred from the prompt. This new common cell is not the historical GKM 4+2 run
or a successful FuseFSS 3+1 run. No execution occurs on import or construction.

All cells remain unmeasured and publication-unqualified. GKM has bounded trained
functionality elsewhere. FuseFSS has an ordinary decoder factory, but its last
trained attempt timed out without an accepted response; admission is not serving
success. PUMA's bounded factory does not admit the shared complete checkpoint;
SIGMA's native implementation is not an ordinary registered full-checkpoint SDK
composition. Neither is substituted with a component, tiny model or clear kernel.

`compatibility_cells()` describes exclusions, not benchmark observations. GKM
full/seeded share arithmetic but NOT privacy assumptions. FuseFSS uses distinct
ring50/Q12 arithmetic, dealer ownership, private prefix state and final-ID-only
delivery. No pair is a matched numeric/privacy/measurement control. Keep ordinary
BenchmarkRun.compare rejection; never rewrite comparison digests or merge salts.
Independent FP32/source-quality comparisons are separate from this cost suite.

Costs use request.0.full, role CPU/RSS and covered application-body bytes, with
the ordinary report's windows/exclusions. Minimum-zero metric budgets require
availability, not a performance threshold; unknown metrics stay unknown. A
completed suite means attempts were recorded, not all cells accepted. Publication
requires completed trials, exact-count/runtime checks, met_in_scope budgets,
source identity and any separately required quality qualification. No streaming
decode, physical-wire, WAN, GPU, price, energy or independently operated privacy
claim follows. No client full-logit/KV audit is invented for final-ID-only FSS.

For the ordinary CLI, save this zero-argument configuration wrapper as
``target/accepted_response_suite.py``. Choose the host budget explicitly; 8 GiB
is a declared allocation limit, not a claim that this host/workload is admitted.
The build capture and all memory/count/acceptance gates remain ordinary SDK code.

```python
from pllm.metrics import BuildRef
from examples.papers.accepted_response_comparison import suite

def campaign():
    return suite(build=BuildRef.capture(), memory_bytes=8 << 30,
                 processes=3, timeout_seconds=1800)
```

From the repository root, in the matching installed environment:

```sh
python -m pllm benchmark run \
  --experiment target/accepted_response_suite.py:campaign --factory --trust-python \
  --prompt hi --max-output-tokens 1 --backend native --timeout 1800 \
  --warmups 0 --repetitions 1 --no-input --format json \
  --output target/accepted-response-comparison.json
```

One CLI call dispatches the complete BenchmarkSuite, with one private salt and
fresh isolated trials; exact 3+1 guards live in its response targets, not invented
CLI count options. Defaults do not override the factory's resource declarations.
Cancellation drains the active worker and retains an incomplete cancelled cohort.
CLI dispatch does not change the declared method/pair exclusions below.
This command is unrun with trained models here. No ranking or success follows
from resolving the factory or recording all declared attempts.
"""

from pllm import ExecutionBudget
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, BuildRef, Sampling, Scenario, TargetBudget
from pllm.profiles import resolve_runtime_composition
from pllm.protocols import OpenWeightVerifiableOutsourcing
from pllm.roles.topology import graph_for_runtime

from examples.papers.common import MODEL
from examples.papers.fusefss import qwen_checked_tensor_minimal_smoke
from examples.papers.gkm import experiment as _gkm_full
from examples.papers.gkm import seeded as _gkm_seeded


CASE_NAMES = ("gkm-full", "gkm-seeded", "fusefss-checked-tensor")
INPUT_TOKENS = 3
OUTPUT_TOKENS = 1
PROMPT = "hi"


def recipes():
    """Existing approved complete-decoder factories; source admission is deferred."""
    budget = ExecutionBudget(requests=1, max_input_tokens=INPUT_TOKENS,
                             max_new_tokens=OUTPUT_TOKENS)
    return {
        "gkm-full": _gkm_full(budget=budget),
        "gkm-seeded": _gkm_seeded(budget=budget),
        "fusefss-checked-tensor": qwen_checked_tensor_minimal_smoke(),
    }


def inputs():
    """The same existing public workload for every cell; no tokenizer estimate."""
    return {name: {"prompt": PROMPT} for name in CASE_NAMES}


def suite(*, build: BuildRef, memory_bytes: int, processes: int = 3,
          timeout_seconds: float = 1800) -> BenchmarkSuite:
    """Declare all candidate attempts, including unqualified FuseFSS; never run them."""
    if not isinstance(build, BuildRef) or build.expected_digest is None:
        raise ValueError("a caller-captured BuildRef with expected_digest is required")
    cases = []
    for name, experiment in recipes().items():
        options = resolve_runtime_composition(experiment.pipeline)
        roles = graph_for_runtime(options).roles
        required = ("request.0.full", "process.peak_rss", "network.covered_body_bytes") + tuple(
            f"{role.id}.{metric}" for role in roles for metric in ("cold_cpu", "peak_rss")
        )
        cases.append(BenchmarkCase(name, BenchmarkTarget.response(
            experiment, backend="native", scenario=Scenario.cold(max_output_tokens=OUTPUT_TOKENS),
            expected_input_tokens=INPUT_TOKENS, expected_output_tokens=OUTPUT_TOKENS,
        ), build, tuple(TargetBudget(metric, minimum=0) for metric in required)))
    return BenchmarkSuite(tuple(cases), Sampling(
        processes=processes, repetitions=1, warmups=0, order="balanced", threads=1,
        memory_bytes=memory_bytes, timeout_seconds=timeout_seconds,
    ), name="accepted-response-qwen-3x1")


def compatibility_cells():
    """JSON-serializable scope/eligibility declarations, never measured outcomes."""
    cells = []
    for name, experiment in recipes().items():
        options = resolve_runtime_composition(experiment.pipeline)
        fss = name == "fusefss-checked-tensor"
        cells.append({
            "name": name, "method": "FuseFSS" if fss else "Gupta-Katz-Miers",
            "status": "planned_unmeasured", "publication_qualified": False,
            "functionality": ("no_accepted_trained_response" if fss
                              else "bounded_trained_response_exists_outside_this_cell"),
            "source": MODEL.to_spec(), "configuration_digest": experiment.configuration_digest(),
            "numeric_profile": options.numeric_profile,
            "local_numeric_profile": None if fss else OpenWeightVerifiableOutsourcing.LOCAL_NUMERIC_PROFILE,
            "privacy_protocol": options.privacy_protocol,
            "sharing": ("two_holders_with_offline_dealer" if fss
                        else experiment.pipeline.components["linear"].params["sharing"]),
            "role_graph": graph_for_runtime(options).to_spec(),
            "reference": "ordinary_source_bound_runtime_checks_not_independent_fp32_quality",
            "acceptance": ("checked_numeric_domain_then_final_ids" if fss
                           else "request_end_verification_then_full_logits_and_text"),
            "state": "private_complete_prefix_recomputation" if fss else "client_full_kv",
            "workload": {"prompt": PROMPT, "expected_input_tokens": INPUT_TOKENS,
                         "expected_output_tokens": OUTPUT_TOKENS, "requests": 1,
                         "generation": "fixed_length_greedy", "backend": "native", "wan": None},
            "matched_control_cell": name,
        })
    blocked = [
        {"method": "PUMA", "status": "blocked",
         "reason": "shared_complete_checkpoint_outside_admitted_sdk_envelope"},
        {"method": "SIGMA", "status": "unsupported",
         "reason": "native_decoder_not_registered_as_complete_checkpoint_sdk_composition"},
    ] + [
        {"method": method, "status": "unsupported",
         "reason": "no_approved_complete_checkpoint_sdk_factory_in_this_study"}
        for method in ("Maverick", "BumbleBee", "NEXUS", "THOR", "CacheMir", "ROSETTA")
    ]
    return {
        "cells": cells, "excluded_methods": blocked, "rankable_pairs": [],
        "pair_exclusions": [
            {"cases": ["gkm-full", "gkm-seeded"], "status": "blocked",
             "reason": "same_numeric_contract_but_different_sharing_privacy_assumptions"},
        ] + [
            {"cases": [gkm, "fusefss-checked-tensor"], "status": "blocked",
             "reason": "different_numeric_privacy_state_and_output_delivery_contracts"}
            for gkm in ("gkm-full", "gkm-seeded")
        ],
        "cli": {"status": "supported", "target": "trusted_zero_argument_BenchmarkSuite_factory",
                "report_schema": "pllm.benchmark_run.v1",
                "exact_count_guards": "response_target_expected_input_tokens_and_expected_output_tokens"},
    }
