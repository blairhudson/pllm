#!/usr/bin/env python3
"""Generate documentation inventories from PLLM's public Python and CLI surfaces."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import cache
from html import escape as html_escape
import importlib
import inspect
import json
from pathlib import Path
import re
import sys
import textwrap
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
from pllm.components._planned import PendingComponent, planned_components
from pllm.components._model_capabilities import PendingModelCapability, model_capabilities
from pllm.configuration import ComponentRef

CLI_REFERENCE_ROOT = ROOT / "docs/content/docs/reference/cli"
SDK_GUIDES_ROOT = ROOT / "docs/content/docs/sdk"
MODEL_COMPATIBILITY = ROOT / "docs/data/model-compatibility.json"
MODEL_CAPABILITY_DOC_ROOT = SDK_GUIDES_ROOT / "models/capabilities"
CLI_ROOT_ORDER = (
    "gateway", "serve", "network", "plan", "config", "components", "topology", "benchmark", "dev"
)


@dataclass(frozen=True)
class CliExample:
    title: str
    description: str
    command: str
    validate_resolution: bool = False


CLI_EXAMPLES: dict[str, tuple[CliExample, ...]] = {
    "pllm gateway": (
        CliExample(
            "Serve a Python experiment locally",
            "Import the checked-in `Experiment` object and start both provider roles behind the "
            "trusted loopback gateway.",
            "pllm gateway --local --experiment examples/composition.py:experiment --trust-python",
            validate_resolution=True,
        ),
        CliExample(
            "Build an experiment with a Python factory",
            "Use `--factory` when the selected Python object is a zero-argument callable rather "
            "than an existing `Experiment`.",
            "pllm gateway --local --experiment examples/composition.py:build_experiment "
            "--factory --trust-python",
            validate_resolution=True,
        ),
        CliExample(
            "Serve a declarative experiment",
            "JSON and YAML targets are data, so they do not require `--trust-python`.",
            "pllm gateway --local --experiment examples/pllm.yaml",
            validate_resolution=True,
        ),
        CliExample(
            "Connect to separately operated providers",
            "Set `PLLM_INFERENCE_API_KEY` and `PLLM_PREPARATION_API_KEY` in the environment; "
            "credentials should not appear in shell arguments.",
            "pllm gateway --inference-url https://inference.example.com "
            "--preparation-url https://preparation.example.com",
            validate_resolution=True,
        ),
        CliExample(
            "Serve a fixed network decision",
            "Use `plan.json` and `network.json` exported by the [network planning guide]"
            "(/learn/topologies/network-planning/). Each response revalidates the selection and "
            "opens a fresh execution lease; operator declarations are not independence proof.",
            "pllm gateway --plan plan.json --network network.json",
        ),
        CliExample(
            "Plan before each response",
            "Use the guide's public `request.json` and matching network declaration. "
            "Discovery and bounded planning run before each ordinary SDK response attempt.",
            "pllm gateway --request request.json --network network.json",
        ),
    ),
    "pllm serve inference": (
        CliExample(
            "Serve the inference role from Python configuration",
            "Set `PLLM_API_KEY` and `PLLM_PROVIDER_PUSH_API_KEY` before starting this public-model "
            "role.",
            "pllm serve inference --experiment examples/composition.py:experiment --trust-python",
            validate_resolution=True,
        ),
        CliExample(
            "Build the inference role configuration with a factory",
            "Add `--factory` when the trusted Python target constructs the `Experiment` on demand.",
            "pllm serve inference --experiment examples/composition.py:build_experiment "
            "--factory --trust-python",
            validate_resolution=True,
        ),
        CliExample(
            "Serve the inference role from YAML",
            "The same declarative experiment used by gateway and benchmark can configure a "
            "standalone provider role.",
            "pllm serve inference --experiment examples/pllm.yaml",
            validate_resolution=True,
        ),
    ),
    "pllm serve preparation": (
        CliExample(
            "Serve trusted preparation from Python configuration",
            "Set `PLLM_API_KEY`, `PLLM_PROVIDER_PUSH_API_KEY`, and `PLLM_INFERENCE_URL` before "
            "starting the preparation role.",
            "pllm serve preparation --experiment examples/composition.py:experiment --trust-python",
            validate_resolution=True,
        ),
        CliExample(
            "Build preparation configuration with a factory",
            "The preparation role accepts the same zero-argument experiment factory as inference.",
            "pllm serve preparation --experiment examples/composition.py:build_experiment "
            "--factory --trust-python",
            validate_resolution=True,
        ),
        CliExample(
            "Serve trusted preparation from YAML",
            "Use the identical experiment document at both roles so their model and pipeline "
            "commitments agree.",
            "pllm serve preparation --experiment examples/pllm.yaml",
            validate_resolution=True,
        ),
    ),
    "pllm serve party": (
        CliExample(
            "Host installed CPU worker roles",
            "Generate `network.json` and `party-a.json` with the [two-party fixture]"
            "(https://github.com/blairhudson/pllm/blob/main/examples/networks/README.md). "
            "Set the configured credential environment references before startup. "
            "Separate loopback processes remain co-located functionality evidence.",
            "pllm serve party --network network.json --party party-a.json --port 8101",
        ),
    ),
    "pllm serve directory": (
        CliExample(
            "Host approved expiring membership",
            "Create `directory.json` from a live network with an approved directory origin "
            "and `directory_credential_env`, as described in the [network guide]"
            "(/learn/topologies/network-planning/). Membership does not reserve capacity "
            "or introduce new trust roots.",
            "pllm serve directory --network directory.json --port 8103",
        ),
    ),
    "pllm network inspect": (
        CliExample(
            "Inspect the shipped loopback declaration",
            "Read public trust roots and credential reference names without contacting hosts.",
            "pllm network inspect examples/networks/loopback.json --format json",
            validate_resolution=True,
        ),
    ),
    "pllm network parties": (
        CliExample(
            "Inspect embedded offers without discovery",
            "The shipped declaration embeds local client authority. Remove `--dry-run` only "
            "after starting the approved hosts to obtain authenticated current offers.",
            "pllm network parties examples/networks/loopback.json --dry-run --format json",
            validate_resolution=True,
        ),
    ),
    "pllm network snapshot": (
        CliExample(
            "Export an observed network snapshot",
            "Use `network.json` from the [planning guide](/learn/topologies/network-planning/). "
            "Local networks export static declarations; HTTP networks discover fresh installed "
            "offers. Existing output files require `--force`.",
            "pllm network snapshot network.json --output snapshot.json --format json",
        ),
    ),
    "pllm network drain": (
        CliExample(
            "Drain an approved live party",
            "Set the configured credential reference and start party `a` from the "
            "[two-party fixture](https://github.com/blairhudson/pllm/blob/main/examples/"
            "networks/README.md). New reservations stop while admitted work finishes.",
            "pllm network drain examples/networks/loopback.json --party a --format json",
        ),
    ),
    "pllm network leave": (
        CliExample(
            "Leave after admitted work drains",
            "Use the same approved HTTP party and credential environment as `drain`. "
            "Departure reports `draining` until admitted leases release or expire, then `left`.",
            "pllm network leave examples/networks/loopback.json --party a --format json",
        ),
    ),
    "pllm plan create": (
        CliExample(
            "Create an immutable offline decision",
            "Export `request.json` and `snapshot.json` from the [pure planning example]"
            "(/sdk/deployment/), then rank bounded assignments without loading weights "
            "or reserving hosts. Existing output files require `--force`.",
            "pllm plan create request.json --snapshot snapshot.json --output plan.json --format json",
        ),
    ),
    "pllm plan inspect": (
        CliExample(
            "Replay and inspect a saved decision",
            "Read `plan.json` created by `plan create`; import replays native legality and scoring.",
            "pllm plan inspect plan.json --format json",
        ),
    ),
    "pllm plan explain": (
        CliExample(
            "Explain selection and coverage",
            "Inspect the saved winner, ordered policy, alternatives, rejections and search "
            "coverage. A feasible truncated search is not exhaustive.",
            "pllm plan explain plan.json --format json",
        ),
    ),
    "pllm plan validate": (
        CliExample(
            "Revalidate against current observations",
            "Use a fresh `snapshot.json` to check expiry, selected host epoch and native "
            "capacity. Validation does not reserve execution authority.",
            "pllm plan validate plan.json --snapshot snapshot.json --format json",
        ),
    ),
    "pllm config show": (
        CliExample(
            "Inspect declarative configuration",
            "Validate a checked-in experiment and print its canonical public form.",
            "pllm config show examples/pllm.yaml",
        ),
        CliExample(
            "Inspect Python configuration",
            "Python objects use the same explicit target syntax as gateway and benchmark.",
            "pllm config show examples/composition.py:experiment --trust-python",
        ),
        CliExample(
            "Inspect an importable module target",
            "Use `module:object` instead of `file.py:object` when the target is importable.",
            "pllm config show examples.composition:experiment --trust-python",
            validate_resolution=True,
        ),
    ),
    "pllm config export": (
        CliExample(
            "Export canonical JSON",
            "Write a validated declarative experiment as strict canonical JSON.",
            "pllm config export examples/pllm.yaml --output experiment.json",
        ),
    ),
    "pllm components list": (
        CliExample(
            "Browse components",
            "List component identity, category, version, and lifecycle in a terminal-friendly form.",
            "pllm components list",
        ),
        CliExample(
            "Produce machine-readable inventory",
            "Use JSON when another tool will filter or compare component metadata.",
            "pllm components list --format json",
        ),
    ),
    "pllm components show": (
        CliExample(
            "Inspect one component",
            "Resolve a component by its stable identity and print its complete descriptor.",
            "pllm components show pllm/cpu --format json",
        ),
    ),
    "pllm topology inspect": (
        CliExample(
            "Inspect installed role channels",
            "Resolve an Experiment without starting roles; local co-location fails the declared "
            "Preparation/Inference operator-separation requirement.",
            "pllm topology inspect examples/pllm.yaml --format json",
            validate_resolution=True,
        ),
    ),
    "pllm benchmark run": (
        CliExample(
            "Run a fast transport smoke test",
            "Generated tiny weights exercise the real local role topology without downloading a "
            "checkpoint.",
            "pllm benchmark run --tiny --max-output-tokens 1 --repetitions 1",
        ),
        CliExample(
            "Measure a Python experiment",
            "Resolve a typed `Experiment`, then run its real client, preparation, and inference "
            "roles.",
            "pllm benchmark run --experiment examples/benchmarks/qwen_prepared.py:cpu_4 "
            "--trust-python --max-output-tokens 8 --repetitions 1",
            validate_resolution=True,
        ),
        CliExample(
            "Build a benchmark experiment with a factory",
            "Factory targets construct the typed experiment before the benchmark driver resolves "
            "its profile and role topology.",
            "pllm benchmark run --experiment examples/composition.py:build_experiment "
            "--factory --trust-python --max-output-tokens 8 --repetitions 1",
            validate_resolution=True,
        ),
        CliExample(
            "Compare two matched experiments",
            "Repeated `--experiment` options run candidates sequentially; `--save-best` exports a "
            "winner only when their measured cohorts are comparable.",
            "pllm benchmark run --experiment examples/benchmarks/qwen_prepared.py:cpu_1 "
            "--experiment examples/benchmarks/qwen_prepared.py:cpu_4 --trust-python "
            "--max-output-tokens 8 --repetitions 3 --save-best best-experiment.json",
            validate_resolution=True,
        ),
        CliExample(
            "Measure declarative configuration",
            "A YAML experiment follows the same resolution and execution path as its Python form.",
            "pllm benchmark run --experiment examples/benchmarks/qwen-prepared-cpu-4.yaml "
            "--max-output-tokens 8 --repetitions 1 --output benchmark.json",
            validate_resolution=True,
        ),
        CliExample(
            "Benchmark a fixed network decision",
            "Use the [network guide's exported files](/learn/topologies/network-planning/). "
            "Each measured response opens a fresh lease through the ordinary SDK route. "
            "The selected experiment owns execution settings.",
            "pllm benchmark run --plan plan.json --network network.json "
            "--max-output-tokens 2 --warmups 0 --repetitions 1",
        ),
        CliExample(
            "Compare bounded feasible network controls",
            "Discover and plan from `request.json`, then compare matched feasible compositions "
            "or live host assignments under the same source, numeric and workload contracts. "
            "Covered bodies and available CPU samples do not establish full-wire cost "
            "or independently operated privacy.",
            "pllm benchmark run --request request.json --network network.json "
            "--compare-feasible --max-output-tokens 2 --warmups 0 --repetitions 1",
        ),
        CliExample(
            "Forecast consumer WAN capacity",
            "Use decimal Mbps and one shared upload/download budget per party. "
            "These are application-body bandwidth floors, not measured WAN latency.",
            "pllm benchmark run --tiny --max-output-tokens 2 --warmups 0 "
            "--wan-download-mbps 100 --wan-upload-mbps 40 "
            "--wan-party preparation:50:10 --output wan-benchmark.json",
        ),
        CliExample(
            "Select exact prepared residue coding",
            "The ordinary compiler and role path bind public output widths, compressed "
            "artifacts and exact prefix reuse. Raw input masks remain full width.",
            "pllm benchmark run --experiment examples/benchmarks/prepared_residues.py:compact "
            "--trust-python --max-output-tokens 8 --warmups 0 --output prepared-residues.json",
            validate_resolution=True,
        ),
    ),
    "pllm benchmark quality": (
        CliExample(
            "Inspect two numeric candidates",
            "Resolve the W4A4 and W8A8 Experiments and bounded prompt cohort without loading the model.",
            "pllm benchmark quality --experiment examples/benchmarks/qwen3_reference_quality.py:w4a4 "
            "--experiment examples/benchmarks/qwen3_reference_quality.py:w8a8 "
            "--trust-python --prompts-file examples/benchmarks/reference_prompts.json --dry-run",
            validate_resolution=True,
        ),
        CliExample(
            "Measure same-token reference agreement",
            "Load the pinned Qwen3-0.6B checkpoint and a local float32 reference; report only "
            "digests and prefill aggregate scores. Requires the optional quality dependencies.",
            "pllm benchmark quality --experiment examples/benchmarks/qwen3_reference_quality.py:w4a4 "
            "--experiment examples/benchmarks/qwen3_reference_quality.py:w8a8 "
            "--trust-python --prompts-file examples/benchmarks/reference_prompts.json --top-k 5 --format json",
            validate_resolution=True,
        ),
    ),
    "pllm dev dashboard": (
        CliExample(
            "Open the real-model development dashboard",
            "The dashboard starts the loopback benchmark topology and defaults to the documented "
            "Qwen checkpoint.",
            "pllm dev dashboard",
        ),
        CliExample(
            "Run a non-interactive transport smoke dashboard",
            "Use tiny random weights and suppress browser launch for local automation.",
            "pllm dev dashboard --tiny --no-open",
        ),
    ),
}

CLI_VIRTUAL_CHILDREN: dict[tuple[str, ...], tuple[str, ...]] = {
    ("gateway",): ("local-experiments", "provider-connections"),
}
PUBLIC_MODULES = (
    "pllm",
    "pllm.client",
    "pllm.config",
    "pllm.models",
    "pllm.native",
    "pllm.plan",
    "pllm.components",
    "pllm.assurance",
    "pllm.correlation",
    "pllm.kernels",
    "pllm.metrics",
    "pllm.nonlinear",
    "pllm.official",
    "pllm.passes",
    "pllm.profiles",
    "pllm.providers",
    "pllm.research",
    "pllm.protocols",
    "pllm.protocols.masked_linear",
    "pllm.preparation",
    "pllm.quantization",
    "pllm.roles",
    "pllm.schedulers",
    "pllm.search",
    "pllm.server",
    "pllm.sources",
    "pllm.state",
    "pllm.verification",
    "pllm.pipeline",
    "pllm.deployment",
    "pllm.runtime",
    "pllm.compiler",
    "pllm.evidence",
)
PYTHON_REFERENCE_ROOT = ROOT / "docs/content/docs/reference/python/pllm"
CLI_REFERENCE_ROOT = ROOT / "docs/content/docs/reference/cli"


def _example(source: str) -> str:
    return textwrap.dedent(source).strip()


def _masked_experiment(
    *,
    import_line: str,
    setup: str,
    pipeline_argument: str,
    assertion: str,
) -> str:
    return _example(
        f"""
        import pllm
        {import_line}
        from pllm.profiles import MaskedLinearCpu
        from pllm.sources import TinyModel

        {setup}
        experiment = pllm.Experiment(
            name="focused-inference",
            pipeline=MaskedLinearCpu(
                TinyModel("qwen2", model_id="transport-smoke"),
                {pipeline_argument}
            ),
            deployment=pllm.Deployment.local(root="local://focused-inference"),
            budget=pllm.ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=2),
        )
        experiment.resolve()
        {assertion}
        """
    )


MODEL_EXAMPLE = _example(
    """
    import pllm
    from pllm.kernels import Cpu
    from pllm.profiles import MaskedLinearCpu

    experiment = pllm.Experiment(
        name="qwen-local",
        pipeline=MaskedLinearCpu(
            pllm.Model("Qwen/Qwen2.5-0.5B-Instruct"),
            kernels=Cpu(threads=2),
        ),
        deployment=pllm.Deployment.local(root=".pllm/qwen-local"),
        budget=pllm.ExecutionBudget(requests=1, max_input_tokens=32, max_new_tokens=8),
    )
    assert len(pllm.configuration_digest(experiment)) == 64
    """
)

CONFIG_EXAMPLE = MODEL_EXAMPLE.replace("import pllm\n", "import pllm.config as pllm\n", 1)

CLIENT_EXAMPLE = _example(
    """
    import pllm.client as client

    closed = []
    stream = client.ResponseStream(iter(["first", "second"]), lambda: closed.append(True))
    assert list(stream) == ["first", "second"]
    stream.close()
    assert closed == [True]
    """
)

NATIVE_EXAMPLE = _example(
    """
    import pllm.native as native

    details = native.capabilities()
    assert details["implementation"] == "rust"
    assert details["matrix_storage"] == "owned-int8"
    """
)

OFFICIAL_EXAMPLE = _example(
    """
    import pllm.official as official

    client = official.create_openai_client()
    try:
        base_url = str(client.base_url)
    finally:
        client.close()
    assert base_url == "https://pllm.local/v1/"
    """
)

SERVER_EXAMPLE = _example(
    """
    import pllm.server as server

    app = server.create_app()
    routes = {route.path for route in app.routes}
    assert {"/healthz", "/v1/responses"} <= routes
    """
)

PLAN_EXAMPLE = _example(
    """
    import pllm

    config = {
        "model_type": "qwen2",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 64,
        "max_position_embeddings": 128,
        "hidden_act": "silu",
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "attention_bias": True,
    }
    plan = pllm.lower_model(config, batch=1, max_input_tokens=8, max_new_tokens=2)
    assert plan.to_dict()["model_family"] == "qwen2"
    """
)

COMPONENT_EXAMPLE = _masked_experiment(
    import_line="import pllm.components as components",
    setup='backend = components.create_component("pllm/cpu", {"threads": 2})',
    pipeline_argument="kernels=backend,",
    assertion='assert experiment.pipeline.kernels.component == "pllm/cpu"',
)

CORRELATION_EXAMPLE = _example(
    """
    import pllm.correlation as correlation

    source = correlation.SeededExpansion()
    assert source.to_spec() == {"component": "pllm/seeded-expansion", "params": {}}
    """
)

KERNEL_EXAMPLE = _masked_experiment(
    import_line="import pllm.kernels as kernels",
    setup="backend = kernels.Cpu(threads=2)",
    pipeline_argument="kernels=backend,",
    assertion='assert experiment.pipeline.kernels.get_params() == {"threads": 2}',
)

QUANTIZATION_EXAMPLE = _masked_experiment(
    import_line="import pllm.quantization as quantization",
    setup="numeric = quantization.SymmetricPerRow(weight_bits=4, activation_bits=4)",
    pipeline_argument="quantization=numeric,",
    assertion='assert experiment.pipeline.quantization.component == "pllm/symmetric-per-row-quantization/v1"',
)

NONLINEAR_EXAMPLE = _example(
    """
    import pllm.nonlinear as nonlinear

    methods = [nonlinear.BinaryTableGatedMultiplyQ7(), nonlinear.R03CrtGatedMultiplyQ7()]
    assert [method.component for method in methods] == ["pllm/binary-table/v1", "pllm/r03-crt/v1"]
    """
)

SCHEDULER_EXAMPLE = _example(
    """
    import pllm.schedulers as schedulers

    scheduler = schedulers.IndependentLanesProtectedTensorSchedule(max_elements=4)
    assert scheduler.get_params() == {"max_elements": 4}
    """
)

STATE_EXAMPLE = _example(
    """
    import pllm.state as state

    protocol = state.ClientLocalKv()
    assert protocol.component == "pllm/client-local-kv"
    """
)

METRIC_EXAMPLE = _example(
    """
    import pllm.metrics as metrics

    latency = metrics.Latency(statistic="p95", phase="online")
    assert latency.to_spec()["params"] == {"phase": "online", "statistic": "p95"}
    agreement = metrics.ReferenceAgreement(dataset_digest="a" * 64, reference_checkpoint_digest="b" * 64)
    assert agreement.component == "pllm/reference-agreement/v1"
    assert metrics.measure_reference_agreement([1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0])["top1_agreement"] == 1.0
    """
)

PASS_EXAMPLE = _example(
    """
    import pllm.passes as passes
    cache = passes.KvCacheEviction(cluster_sizes=(32, 16), share_adjacent_layers=True)
    assert cache.get_params()["implementation"] == "pllm/importance-kv-cache-eviction/v1"
    """
)

PROVIDER_EXAMPLE = _example(
    """
    import pllm.providers as providers

    # An explicit empty entry-point set disables external package discovery.
    assert providers.discover_providers(entry_points=()) == ()
    """
)

RESEARCH_EXAMPLE = _example(
    """
    import pllm.research as research

    lock = research.ArtifactLock(
        id="slalom-upstream",
        source_record_id="R07",
        revision="a" * 40,
        artifact_digest="b" * 64,
        license_review="external_reproduction_only",
        isolation="external_process",
    )
    assert lock.verify(evidence_digest="c" * 64).status == "reproduction_verified"
    """
)

ASSURANCE_EXAMPLE = _example(
    """
    from pllm.assurance import PublicSubspaceMaskRegression
    from pllm.assurance import SubspaceLeakageRegression
    from pllm.components import NotYetImplementedError

    witness = PublicSubspaceMaskRegression(ring_bits=16).evaluate(
        ((1, 1, 0), (0, 1, 1)), (20, 28, 38)
    )
    assert witness is not None and witness.indices == (0, 1, 2)
    assert witness.leaked_parity == 0

    try:
        SubspaceLeakageRegression()
    except NotYetImplementedError as error:
        assert error.paper_id == "breaking-euston"
        assert error.kind == "security_control"
    else:
        raise AssertionError("unimplemented assurance must fail closed")
    """
)

PROTOCOL_EXAMPLE = _masked_experiment(
    import_line="import pllm.protocols as protocols",
    setup="method = protocols.MaskedLinear()",
    pipeline_argument="linear=method,",
    assertion='assert experiment.pipeline.linear.component == "pllm/masked-linear"',
)

PROFILE_EXAMPLE = _example(
    """
    import pllm
    import pllm.profiles as profiles

    from pllm.sources import TinyModel

    experiment = pllm.Experiment(
        name="profile-inference",
        pipeline=profiles.MaskedLinearCpu(TinyModel("qwen2", model_id="transport-smoke")),
        deployment=pllm.Deployment.local(root="local://profile-inference"),
        budget=pllm.ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=2),
    )
    assert experiment.resolve().client_runtime == "masked_transformer_v1"
    """
)

SOURCE_EXAMPLE = _example(
    """
    import pllm
    from pllm.profiles import MaskedLinearCpu
    import pllm.sources as sources

    model = sources.TinyModel("qwen2", model_id="transport-smoke")
    experiment = pllm.Experiment(
        name="source-inference",
        pipeline=MaskedLinearCpu(model),
        deployment=pllm.Deployment.local(root="local://source-inference"),
        budget=pllm.ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=2),
    )
    assert experiment.resolve().model == "transport-smoke"
    """
)

SEARCH_EXAMPLE = _example(
    """
    import pllm
    import pllm.search as search
    from pllm.kernels import Cpu
    from pllm.profiles import MaskedLinearCpu

    base = pllm.Experiment(
        name="search-base",
        pipeline=MaskedLinearCpu(pllm.Model("org/model"), kernels=Cpu(threads=1)),
        deployment=pllm.Deployment.local(root="local://search"),
        budget=pllm.ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=2),
    )
    space = search.SearchSpace(base, {"pipeline__kernels__threads": [1, 2]})
    assert [candidate.parameters for candidate in search.GridSearch("cpu", space).candidates()] == [
        {"pipeline__kernels__threads": 1},
        {"pipeline__kernels__threads": 2},
    ]
    """
)

RUNTIME_EXAMPLE = _example(
    """
    import pllm.runtime as runtime

    app = runtime.create_app(runtime.GatewayConfig(api_keys=("local-test",)))
    routes = {route.path for route in app.routes}
    assert {"/health", "/v1/responses"} <= routes
    """
)

COMPILER_EXAMPLE = _example(
    """
    import pllm.compiler as compiler

    rejected = False
    try:
        compiler.compile(b"{}")
    except compiler.CompilationError as error:
        rejected = "E_INVALID_DOCUMENT" in str(error)
    assert rejected
    """
)

PLAN_RECORD_EXAMPLE = _example(
    """
    import pllm.plan as plan

    try:
        plan.CompiledPlan(object())
    except TypeError as error:
        rejected = "handle must come from pllm.compile" in str(error)
    else:
        rejected = False
    assert rejected
    """
)

EVIDENCE_EXAMPLE = _example(
    """
    import pllm.evidence as evidence

    digest = evidence.environment_digest({"python": "3.13", "machine": "local"})
    assert len(digest) == 64 and int(digest, 16) >= 0
    """
)

MASKED_LINEAR_EXAMPLE = _masked_experiment(
    import_line="import pllm.protocols.masked_linear as masked_linear",
    setup="method = masked_linear.MaskedLinear()",
    pipeline_argument="linear=method,",
    assertion='assert experiment.pipeline.linear.component == "pllm/masked-linear"',
)

PREPARATION_EXAMPLE = _masked_experiment(
    import_line="import pllm.preparation as preparation",
    setup="provider = preparation.ModelAwareCorrections()",
    pipeline_argument="preparation=provider,",
    assertion='assert experiment.pipeline.preparation.component == "pllm/model-aware-corrections"',
)

ROLE_EXAMPLE = _masked_experiment(
    import_line="import pllm.roles as roles",
    setup="role = roles.Inference()",
    pipeline_argument="inference=role,",
    assertion='assert experiment.pipeline.inference.component == "pllm/inference"',
)

VERIFICATION_EXAMPLE = _example(
    """
    import pllm
    import pllm.verification as verification
    from pllm.profiles import VerifiedMaskedLinearCpu
    from pllm.sources import TinyModel

    verifier = verification.FreivaldsVerify(target_failure_bits=48)
    experiment = pllm.Experiment(
        name="verified-inference",
        pipeline=VerifiedMaskedLinearCpu(
            TinyModel("qwen2", model_id="transport-smoke"),
            verification=verifier,
        ),
        deployment=pllm.Deployment.local(root="local://verified-inference"),
        budget=pllm.ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=2),
    )
    assert experiment.resolve().verification_target_failure_bits == 48
    """
)

PIPELINE_EXAMPLE = _example(
    """
    import pllm
    import pllm.pipeline as pipeline
    from pllm.sources import TinyModel

    configured = pipeline.VerifiedMaskedLinearCpu(TinyModel("qwen2"))
    experiment = pllm.Experiment(
        name="pipeline-inference",
        pipeline=configured,
        deployment=pllm.Deployment.local(root="local://pipeline-inference"),
        budget=pllm.ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=2),
    )
    assert experiment.resolve().verification_component == "pllm/freivalds-verify/v1"
    """
)

DEPLOYMENT_EXAMPLE = _example(
    """
    import pllm
    import pllm.deployment as deployment
    from pllm.profiles import MaskedLinearCpu
    from pllm.sources import TinyModel

    local = deployment.Deployment(kind="local", root="/tmp/pllm-example")
    experiment = pllm.Experiment(
        name="local-inference",
        pipeline=MaskedLinearCpu(TinyModel("qwen2", model_id="transport-smoke")),
        deployment=local,
        budget=pllm.ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=2),
    )
    assert experiment.resolve().model == "transport-smoke"
    graph = experiment.resolve().role_graph
    roles = deployment.RoleDeployment(
        graph_digest=graph.digest(),
        roles=(
            deployment.RolePlacement("client", "customer", None),
            deployment.RolePlacement("preparation", "customer", "https://prep.example.invalid"),
            deployment.RolePlacement("inference", "provider", "https://infer.example.invalid"),
        ),
    )
    assert roles.assess(graph).declared_separation_violations == ()
    assert not roles.assess(graph).runtime_admission_supported
    """
)

DASH_CITATION = "[DASH](/research/papers/dash/)"
REDASH_CITATION = "[ReDASH](/research/papers/redash/)"
SLALOM_CITATION = "[Slalom](/research/papers/slalom/)"
MPCACHE_CITATION = "[MPCache](/research/papers/mpcache/)"
COMPACT_CITATION = "[Compact](/research/papers/compact/)"
R03_CITATION = "[Garbling Gadgets](/research/papers/garbling-gadgets/)"
HYCC_CITATION = "[HyCC](/research/papers/hycc/)"

MODULE_GUIDES: dict[str, dict[str, object]] = {
    "pllm": {
        "purpose": "The root package is the task-oriented SDK surface for describing a model, bounding a workload, selecting a pipeline, lowering semantic plans, and constructing clients without depending on internal module paths.",
        "citations": (),
        "example": MODEL_EXAMPLE,
    },
    "pllm.client": {
        "purpose": "Client facades expose synchronous and asynchronous private-inference clients plus bounded response streams without requiring callers to import runtime internals.",
        "citations": (),
        "example": CLIENT_EXAMPLE,
    },
    "pllm.config": {
        "purpose": "Immutable configuration records describe experiments before any checkpoint, network connection, credential, or one-use material is opened.",
        "citations": (),
        "example": CONFIG_EXAMPLE,
    },
    "pllm.models": {
        "purpose": "Model APIs separate semantic architecture lowering from checkpoint import so a plan can be reviewed before weights or runtime state exist.",
        "citations": (),
        "example": PLAN_EXAMPLE.replace("import pllm", "import pllm.models as pllm"),
    },
    "pllm.native": {
        "purpose": "Native APIs inspect the installed Rust backend and compile reusable integer matrix stages with explicit owned storage and conversion boundaries.",
        "citations": (),
        "example": NATIVE_EXAMPLE,
    },
    "pllm.plan": {
        "purpose": "Plan records and locks bind semantic, numeric, privacy, and deployment decisions into immutable digest-addressed documents. PlanningResult preserves bounded network search coverage, costs and rejections, and replays native legality and scoring on import; it is not an execution reservation.",
        "citations": (),
        "example": PLAN_RECORD_EXAMPLE,
    },
    "pllm.components": {
        "purpose": "The component registry discovers typed built-in capabilities by stable identity and creates concrete implementations only when explicitly requested.",
        "citations": (),
        "example": COMPONENT_EXAMPLE,
    },
    "pllm.assurance": {
        "purpose": "Assurance controls run bounded paper-derived attack regressions and specify pending threat-model checks separately from inference components.",
        "citations": ("[Slalom at the Carnival](/research/papers/carnival/)", "[Maverick](/research/papers/maverick/)", "[Breaking Euston](/research/papers/breaking-euston/)"),
        "example": ASSURANCE_EXAMPLE,
    },
    "pllm.correlation": {
        "purpose": "Correlation components describe offline cryptographic material sources independently from online protocol scheduling.",
        "citations": (DASH_CITATION, REDASH_CITATION),
        "example": CORRELATION_EXAMPLE,
    },
    "pllm.kernels": {
        "purpose": "Kernel components select bounded matrix implementations while preserving the compiler's numeric and placement contracts.",
        "citations": (),
        "example": KERNEL_EXAMPLE,
    },
    "pllm.metrics": {
        "purpose": "Metric definitions attach units and optimization direction to measurements; reference agreement also scores bounded same-token logits without retaining their payloads.",
        "citations": (),
        "example": METRIC_EXAMPLE,
    },
    "pllm.nonlinear": {
        "purpose": "Nonlinear components identify approximation or protected-evaluation methods separately from their scheduling strategy and executable coverage.",
        "citations": (R03_CITATION, COMPACT_CITATION),
        "example": NONLINEAR_EXAMPLE,
    },
    "pllm.official": {
        "purpose": "Official integration factories construct OpenAI SDK clients that route requests through PLLM's trusted local client transport.",
        "citations": (),
        "example": OFFICIAL_EXAMPLE,
    },
    "pllm.passes": {
        "purpose": "Graph passes transform semantic plans under explicit state and lifecycle contracts rather than rewriting adapter-specific operation names.",
        "citations": (MPCACHE_CITATION,),
        "example": PASS_EXAMPLE,
    },
    "pllm.profiles": {
        "purpose": "Profiles provide reviewed slot-compatible pipeline presets while retaining the exact component identities used to compile an experiment.",
        "citations": (SLALOM_CITATION,),
        "example": PROFILE_EXAMPLE,
    },
    "pllm.providers": {
        "purpose": "Provider discovery reads package-confined static manifests without importing provider code; factory loading is a separate approved operation.",
        "citations": (),
        "example": PROVIDER_EXAMPLE,
    },
    "pllm.research": {
        "purpose": "The research registry exposes locked paper sources, clean-room method records, evidence requirements, and promotion assessments without executing quarantined upstream artifacts.",
        "citations": (),
        "example": RESEARCH_EXAMPLE,
    },
    "pllm.protocols": {
        "purpose": "Protocol components define which parties exchange which protected values and keep baseline masking separate from optional verification.",
        "citations": (SLALOM_CITATION, DASH_CITATION),
        "example": PROTOCOL_EXAMPLE,
    },
    "pllm.protocols.masked_linear": {
        "purpose": "Masked-linear records configure the public-weight prepared protocol and its optional trusted-client Freivalds verification contract.",
        "citations": (SLALOM_CITATION,),
        "example": MASKED_LINEAR_EXAMPLE,
    },
    "pllm.preparation": {
        "purpose": "Preparation components define offline inventory production and trusted preparation roles without placing preparation work on the online inference path.",
        "citations": (DASH_CITATION, REDASH_CITATION),
        "example": PREPARATION_EXAMPLE,
    },
    "pllm.quantization": {
        "purpose": "Numeric components select symmetric per-row stage bit widths and bind them to the immutable pipeline before offline preparation or online inference.",
        "citations": (),
        "example": QUANTIZATION_EXAMPLE,
    },
    "pllm.roles": {
        "purpose": "Role components make client, preparation, inference, and single-evaluator placement explicit in pipeline configuration.",
        "citations": (DASH_CITATION,),
        "example": ROLE_EXAMPLE,
    },
    "pllm.schedulers": {
        "purpose": "Scheduler components state how independent protected lanes or scalar operations are ordered without changing the underlying method identity.",
        "citations": (),
        "example": SCHEDULER_EXAMPLE,
    },
    "pllm.search": {
        "purpose": "Search APIs generate immutable experiment candidates and compare only cohort-compatible evidence with explicit metric directions. optimization_space proposes bounded model-neutral choices while retaining numeric/privacy contracts. PlanningRequest and PlanningPolicy bound offline network placement over exact source and workload identities; unknown required costs fail closed.",
        "citations": (),
        "example": SEARCH_EXAMPLE,
    },
    "pllm.server": {
        "purpose": "The server facade constructs the trusted client-boundary ASGI gateway with health, Responses API, and bounded runtime routes.",
        "citations": (),
        "example": SERVER_EXAMPLE,
    },
    "pllm.sources": {
        "purpose": "Source records identify model origins and bounded workloads without resolving files, downloading checkpoints, or opening runtime state.",
        "citations": (),
        "example": SOURCE_EXAMPLE,
    },
    "pllm.state": {
        "purpose": "State components identify where persistent decoder state lives and keep mutable KV lifecycle out of immutable configuration records.",
        "citations": (MPCACHE_CITATION,),
        "example": STATE_EXAMPLE,
    },
    "pllm.verification": {
        "purpose": "Verification components and checks bind optional result verification to explicit soundness and one-use material policies.",
        "citations": (SLALOM_CITATION,),
        "example": VERIFICATION_EXAMPLE,
    },
    "pllm.pipeline": {
        "purpose": "Pipeline records compose component identities into a digestible configuration while leaving live sessions, masks, and credentials outside the object graph.",
        "citations": (),
        "example": PIPELINE_EXAMPLE,
    },
    "pllm.deployment": {
        "purpose": "Deployment APIs separate immutable static/local or authenticated HTTP network declarations, read-only discovery, and selected execution leases. Live CPU two-online-offset admission checks installed source, host epoch, numerics, schedule, capacity and TTL; imported offers alone grant no live authority. Operator separation is declared, not verified physical independence, and public records contain credential environment references rather than values.",
        "citations": (),
        "example": DEPLOYMENT_EXAMPLE,
    },
    "pllm.runtime": {
        "purpose": "Runtime APIs execute validated native kernels, bind compiled models, construct local role topologies, and expose trusted client-boundary services.",
        "citations": (SLALOM_CITATION, DASH_CITATION),
        "example": RUNTIME_EXAMPLE,
    },
    "pllm.compiler": {
        "purpose": "The compiler accepts canonical request bytes, verifies complete capability coverage, and returns opaque native plans or fails closed. plan performs bounded pure network placement against an explicit snapshot. plan_on_load explicitly resolves a checkpoint and verifies its locked configuration before bounded candidate selection; neither reserves hosts, imports tensor values, nor benchmarks candidates. Native code owns legality.",
        "citations": (HYCC_CITATION,),
        "example": COMPILER_EXAMPLE,
    },
    "pllm.evidence": {
        "purpose": "Evidence records preserve exact benchmark cohorts, environment identity, and assurance results without silently ranking incomparable runs.",
        "citations": (),
        "example": EVIDENCE_EXAMPLE,
    },
}

RUNNABLE_EXPERIMENT_MODULES = {
    "pllm",
    "pllm.components",
    "pllm.config",
    "pllm.deployment",
    "pllm.kernels",
    "pllm.pipeline",
    "pllm.preparation",
    "pllm.quantization",
    "pllm.profiles",
    "pllm.protocols",
    "pllm.protocols.masked_linear",
    "pllm.roles",
    "pllm.sources",
    "pllm.verification",
}

UNCOMPOSED_RESEARCH_MODULES = {
    "pllm.correlation",
    "pllm.nonlinear",
    "pllm.passes",
    "pllm.schedulers",
    "pllm.state",
}

INFERENCE_CLIENT_MODULES = {
    "pllm.client",
    "pllm.official",
    "pllm.runtime",
    "pllm.server",
}


def _example_context(module: str) -> str:
    if module == "pllm.assurance":
        return (
            "This example produces a concrete bounded public-mask leakage witness, then verifies "
            "that a separate pending assurance control still fails closed. Neither is an inference slot."
        )
    if module in RUNNABLE_EXPERIMENT_MODULES:
        return (
            "This example builds a complete supported `Experiment` with the module's object in "
            "focus. Save it as `experiment.py`; after the assertions pass, serve the same object "
            "with the command below. The gateway requires the optional `he` dependency from the "
            "[installation guide](/learn/installation/)."
        )
    if module in UNCOMPOSED_RESEARCH_MODULES:
        return (
            "This capability is not a supported whole-model inference slot. The example validates "
            "its bounded configuration only; use it for research inspection, not as a claim that "
            "the component can serve a model."
        )
    if module in INFERENCE_CLIENT_MODULES:
        return (
            "This example exercises the module's inference-facing surface without opening a live "
            "provider connection. Start a supported local experiment from the "
            "[gateway guide](/cli/reference/gateway/local-experiments/) before sending a request."
        )
    return (
        "This focused example checks the module's role around inference without downloading a "
        "checkpoint or contacting a provider. Follow the linked SDK guide for the complete "
        "runnable workflow."
    )


def _frontmatter(title: str, description: str) -> str:
    return (
        "---\n"
        f"title: {json.dumps(title)}\n"
        f"description: {json.dumps(description)}\n"
        "---\n\n"
    )


def _cell(value: object) -> str:
    if value is None or value == "" or value == [] or value == () or value == {}:
        return "not recorded"
    if isinstance(value, str):
        return value.replace("|", "\\|").replace("\n", " ")
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "`" + rendered.replace("|", "\\|") + "`"


def _code(value: str) -> str:
    return "`" + value.replace("|", "\\|") + "`"


def _walk_parsers(
    parser: argparse.ArgumentParser,
) -> list[tuple[tuple[str, ...], argparse.ArgumentParser]]:
    found: list[tuple[tuple[str, ...], argparse.ArgumentParser]] = [((), parser)]
    pending: list[tuple[tuple[str, ...], argparse.ArgumentParser]] = [((), parser)]
    while pending:
        prefix, current = pending.pop(0)
        for action in current._actions:
            choices = getattr(action, "choices", None)
            if not isinstance(choices, dict):
                continue
            for name, child in choices.items():
                item = (prefix + (name,), child)
                found.append(item)
                pending.append(item)
    return found


def cli_help_sections() -> tuple[tuple[str, str], ...]:
    from pllm._cli.app import build_parser

    sections = []
    for path, parser in _walk_parsers(build_parser()):
        command = "pllm" + (f" {' '.join(path)}" if path else "")
        sections.append((command, parser.format_help()))
    return tuple(sections)


def cli_leaf_commands(
    sections: tuple[tuple[str, str], ...] | None = None,
) -> tuple[str, ...]:
    sections = sections or cli_help_sections()
    parent_commands = {command.rsplit(" ", 1)[0] for command, _ in sections if " " in command}
    return tuple(
        command for command, _ in sections if command != "pllm" and command not in parent_commands
    )


def validate_cli_examples(sections: tuple[tuple[str, str], ...] | None = None) -> None:
    leaves = set(cli_leaf_commands(sections))
    examples = set(CLI_EXAMPLES)
    if leaves == examples:
        return
    missing = ", ".join(sorted(leaves - examples)) or "none"
    extras = ", ".join(sorted(examples - leaves)) or "none"
    raise ValueError(
        f"CLI examples must match parser leaves exactly; missing: {missing}; extras: {extras}"
    )


def render_cli_help() -> str:
    sections: list[str] = []
    for command, help_text in cli_help_sections():
        sections.extend((f"\n$ {command} --help\n", help_text))
    return "".join(sections)


def render_cli_reference(
    sections: tuple[tuple[str, str], ...] | None = None,
) -> str:
    sections = sections or cli_help_sections()
    leaves = cli_leaf_commands(sections)
    body = [
        _frontmatter(
            "CLI reference",
            "Exact commands, arguments, and help text from the installed PLLM CLI.",
        ),
        "Use the CLI to run private inference, operate provider roles, benchmark the real "
        "transport, inspect configuration, and compare components. Open "
        "[Private inference](/cli/private-inference/), [Provider roles](/cli/provider-roles/), "
        "[Benchmarking](/cli/benchmarking/), [Topologies](/cli/topologies/), or "
        "[Inspect components](/cli/inspect-and-research/), or "
        "[Network planning and operation](/learn/topologies/network-planning/). "
        "Choose a command group below for "
        "exact generated arguments.\n\n",
        "## Complete grammar\n\n",
        "```text\n",
        *[f"{command} [OPTIONS]\n" for command in leaves],
        "```\n\n",
        "Global options accepted at each parser level are `--format {human,json,jsonl}`, `--quiet`, "
        "`--no-color`, `--no-input`, and `--dry-run`. Root also accepts `--version`. Argument "
        "abbreviation is disabled.\n\n",
        "[Download exact complete help](/downloads/cli-help.txt).\n\n",
        _network_cli_context(),
        "## `pllm --help`\n\n",
        "```text\n",
        sections[0][1],
        "```\n",
    ]
    return "".join(body)


def _network_cli_context() -> str:
    return (
        "## Network selection and operation\n\n"
        "`gateway` and `benchmark run` accept mutually exclusive `--plan PATH` or "
        "`--request PATH`, both requiring `--network PATH`. Plan mode revalidates a fixed "
        "selection; request mode discovers and performs bounded planning before each response. "
        "Selection owns execution settings and excludes `--experiment` and manual "
        "model/endpoint/numeric overrides. Benchmark `--compare-feasible` adds bounded "
        "matched controls. Every live response obtains fresh reserved/armed capacity; "
        "dry runs do not start services or perform discovery.\n\n"
        "`network inspect` reads declarations; `parties` and `snapshot` discover live offers "
        "unless dry-run. `drain` stops new admission; `leave` completes after admitted work "
        "drains. `plan create` writes an immutable offline result; `inspect` and `explain` "
        "replay and describe it; `validate` checks current snapshot/expiry and native placement. "
        "`serve party` requires a locked installed `PartySpec`; `serve directory` requires "
        "configured directory trust and credentials. Public credential fields name "
        "environment variables, never embedded credential values.\n\n"
        "See [checked commands and fixtures](/learn/topologies/network-planning/) and "
        "[SDK execution leases](/sdk/deployment/execution/). HTTP role hosting currently "
        "supports CPU two-online-offset; declared operator IDs are not independence proof.\n\n"
    )


def _render_cli_examples(examples: tuple[CliExample, ...]) -> str:
    body = ["## Examples\n\n"]
    for example in examples:
        body.extend(
            (
                f"### {example.title}\n\n",
                example.description,
                "\n\n```bash\n",
                example.command,
                "\n```\n\n",
            )
        )
    return "".join(body)


def render_cli_command_reference(
    command: str, help_text: str, examples: tuple[CliExample, ...] | None
) -> str:
    body = [_frontmatter(command, f"Exact {command} help from the installed PLLM CLI.")]
    if command in {"pllm gateway", "pllm benchmark run", "pllm network", "pllm plan"}:
        body.append(_network_cli_context())
    if examples is None:
        body.append(
            "Choose a subcommand below. Each executable command page includes an example.\n\n"
        )
    else:
        if command == "pllm dev dashboard":
            body.append("This dashboard is a development-only local tool.\n\n")
        body.append(_render_cli_examples(examples))
    body.extend(("## Options\n\n", "```text\n", help_text, "```\n"))
    return "".join(body)


def render_gateway_local_guide() -> str:
    examples = CLI_EXAMPLES["pllm gateway"][:3]
    return "".join(
        (
            _frontmatter(
                "Local experiments",
                "Serve declarative experiments, Python objects, and Python factories locally.",
            ),
            "`pllm gateway --local` resolves one typed `Experiment`, starts its required roles "
            "as child processes, and exposes the trusted loopback Responses and Chat Completions "
            "API. The gateway does not replace the experiment's pipeline or profile.\n\n",
            "Local gateway execution requires the optional `he` dependency from the "
            "[installation guide](/learn/installation/).\n\n",
            "## Target forms\n\n",
            "Use `file.py:object` for a Python file, `module:object` for an importable module, or "
            "`.json`/`.yaml` for declarative configuration. Python targets are imported code and "
            "therefore require `--trust-python` in unattended commands. Add `--factory` only when "
            "the selected object is a zero-argument callable returning an `Experiment`. Raw "
            "`python file.py` execution is intentionally unsupported.\n\n",
            "The repository's [composition example](https://github.com/blairhudson/pllm/blob/main/"
            "examples/composition.py) exposes both `experiment` and `build_experiment`; "
            "[`examples/pllm.yaml`](https://github.com/blairhudson/pllm/blob/main/examples/"
            "pllm.yaml) provides the equivalent declarative form.\n\n",
            _render_cli_examples(examples),
            "## Application endpoint\n\n",
            "Applications connect to `http://127.0.0.1:8080/v1` by default. Inference and "
            "preparation children remain provider-role services, not application-facing "
            "OpenAI-compatible endpoints. Stop the gateway to shut down its local role topology.\n",
        )
    )


def render_gateway_provider_guide() -> str:
    example = CLI_EXAMPLES["pllm gateway"][3:4]
    return "".join(
        (
            _frontmatter(
                "Provider connections",
                "Connect the trusted gateway to separately operated inference and preparation roles.",
            ),
            "Without `--local`, the gateway stays inside the client boundary and connects to "
            "provider-role URLs. It does not start those services and it does not make their "
            "endpoints OpenAI compatible.\n\n",
            "## Credentials\n\n",
            "Set `PLLM_INFERENCE_API_KEY` and `PLLM_PREPARATION_API_KEY` in the gateway "
            "environment. Keep those role-specific credentials distinct and out of argv, shell "
            "history, and checked-in configuration. Public prepared inference needs both URLs; a "
            "profile with no preparation role must omit the preparation URL.\n\n",
            _render_cli_examples(example),
            "Start provider roles with [`pllm serve inference`](/cli/reference/serve/inference/) "
            "and [`pllm serve preparation`](/cli/reference/serve/preparation/). Both can resolve "
            "the same Python, JSON, or YAML experiment target used for local gateway execution and "
            "benchmarking.\n",
        )
    )


def _render_meta(title: str, pages: tuple[str, ...], *, root: bool = False) -> str:
    metadata: dict[str, object] = {"title": title}
    if root:
        metadata["root"] = True
    metadata["pages"] = pages
    return json.dumps(metadata, indent=2) + "\n"


def render_cli_reference_outputs() -> dict[Path, str]:
    sections = cli_help_sections()
    validate_cli_examples(sections)
    leaf_commands = set(cli_leaf_commands(sections))
    commands = [
        (tuple(command.split()[1:]), command, help_text) for command, help_text in sections[1:]
    ]
    parent_paths = {path[:depth] for path, _, _ in commands for depth in range(1, len(path))} | set(
        CLI_VIRTUAL_CHILDREN
    )
    outputs = {
        CLI_REFERENCE_ROOT / "index.mdx": render_cli_reference(sections),
        CLI_REFERENCE_ROOT / "meta.json": _render_meta(
            "Command reference", ("index", *CLI_ROOT_ORDER), root=True
        ),
    }
    for path, command, help_text in commands:
        source = (
            CLI_REFERENCE_ROOT.joinpath(*path, "index.mdx")
            if path in parent_paths
            else CLI_REFERENCE_ROOT.joinpath(*path[:-1], f"{path[-1]}.mdx")
        )
        outputs[source] = render_cli_command_reference(
            command, help_text, CLI_EXAMPLES[command] if command in leaf_commands else None
        )
    for parent in sorted(parent_paths):
        parser_children = tuple(
            path[-1]
            for path, _, _ in commands
            if len(path) == len(parent) + 1 and path[:-1] == parent
        )
        children = (*parser_children, *CLI_VIRTUAL_CHILDREN.get(parent, ()))
        outputs[CLI_REFERENCE_ROOT.joinpath(*parent, "meta.json")] = _render_meta(
            parent[-1].replace("-", " ").title(), ("index", *children)
        )
    outputs[CLI_REFERENCE_ROOT / "gateway/local-experiments.mdx"] = render_gateway_local_guide()
    outputs[CLI_REFERENCE_ROOT / "gateway/provider-connections.mdx"] = (
        render_gateway_provider_guide()
    )
    return outputs


def public_exports() -> tuple[dict[str, Any], ...]:
    exports: list[dict[str, Any]] = []
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        names = tuple(getattr(module, "__all__", ()))
        for name in sorted(names):
            value = getattr(module, name)
            exports.append({"module": module_name, "name": name, "value": value})
    return tuple(exports)


def _signature(value: object) -> str:
    if not callable(value):
        return type(value).__name__
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return "signature not exposed"


def _private_only_parameters(value: object) -> bool:
    try:
        parameters = tuple(inspect.signature(cast(Any, value)).parameters.values())
    except (TypeError, ValueError):
        return False
    public_parameters = tuple(
        parameter
        for parameter in parameters
        if parameter.name not in {"self", "cls"}
    )
    return bool(public_parameters) and all(
        parameter.name.startswith("_") for parameter in public_parameters
    )


def _stub_path(module_name: str) -> Path | None:
    relative = Path(*module_name.split("."))
    candidates = (
        PYTHON_ROOT / relative.with_suffix(".pyi"),
        PYTHON_ROOT / relative / "__init__.pyi",
    )
    return next((path for path in candidates if path.is_file()), None)


def _stub_node(
    module_name: str, name: str, seen: frozenset[tuple[str, str]] = frozenset()
) -> ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | None:
    key = (module_name, name)
    if key in seen:
        return None
    path = _stub_path(module_name)
    if path is None:
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    return _stub_node(node.module, alias.name, seen | {key})
    return None


def _function_declaration(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    decorators = {
        decorator.id for decorator in node.decorator_list if isinstance(decorator, ast.Name)
    }
    if "property" in decorators:
        annotation = ast.unparse(node.returns) if node.returns is not None else "Any"
        return f"property {node.name}: {annotation}"
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    if "classmethod" in decorators:
        prefix += "classmethod "
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix}{node.name}({ast.unparse(node.args)}){returns}"


def _typed_details(exports: list[str], value: object) -> tuple[str, tuple[str, ...]]:
    if inspect.isclass(value) and issubclass(value, PendingComponent) and value is not PendingComponent:
        return "raises NotYetImplementedError on construction", ()
    node = None
    for export in exports:
        module_name, name = export.rsplit(".", 1)
        node = _stub_node(module_name, name)
        if node is not None:
            break
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _function_declaration(node), ()
    if isinstance(node, ast.ClassDef):
        members = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name.startswith("_") and child.name not in {"__init__", "__new__"}:
                    continue
                arguments = (*child.args.posonlyargs, *child.args.args, *child.args.kwonlyargs)
                external_arguments = tuple(
                    argument for argument in arguments if argument.arg not in {"self", "cls"}
                )
                if (
                    child.name in {"__init__", "__new__"}
                    and external_arguments
                    and all(argument.arg.startswith("_") for argument in external_arguments)
                ):
                    continue
                members.append(_function_declaration(child))
            elif (
                isinstance(child, ast.AnnAssign)
                and isinstance(child.target, ast.Name)
                and not child.target.id.startswith("_")
            ):
                members.append(f"{child.target.id}: {ast.unparse(child.annotation)}")
        constructor = next(
            (item for item in members if item.startswith("__new__") or item.startswith("__init__")),
            "constructor not exposed",
        )
        if "-> Never" in constructor:
            constructor = "not publicly constructible"
        return constructor, tuple(members)
    if inspect.isclass(value):
        members = []
        for name, member in value.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(member, property):
                annotation = (
                    inspect.Signature.empty
                    if member.fget is None
                    else inspect.signature(member.fget).return_annotation
                )
                rendered = (
                    "Any"
                    if annotation is inspect.Signature.empty
                    else inspect.formatannotation(annotation)
                )
                members.append(f"property {name}: {rendered}")
            elif isinstance(member, classmethod):
                members.append(f"classmethod {name}{_signature(member.__func__)}")
            elif callable(member):
                members.append(f"{name}{_signature(member)}")
        signature = "not publicly constructible" if _private_only_parameters(value) else _signature(value)
        return signature, tuple(sorted(members))
    return _signature(value), ()


def api_inventory() -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[str, object], dict[str, Any]] = {}
    for export in public_exports():
        value = export["value"]
        object_export = inspect.isclass(value) or callable(value)
        key = (
            ("object", id(value))
            if object_export
            else ("export", f"{export['module']}.{export['name']}")
        )
        record = grouped.setdefault(
            key,
            {
                "canonical": (
                    f"{getattr(value, '__module__', export['module'])}."
                    f"{getattr(value, '__qualname__', export['name'])}"
                    if object_export
                    else f"{export['module']}.{export['name']}"
                ),
                "kind": "class"
                if inspect.isclass(value)
                else "function"
                if callable(value)
                else type(value).__name__,
                "signature": _signature(value),
                "documentation": inspect.getdoc(value) or "",
                "exports": [],
                "value": value,
            },
        )
        record["exports"].append(f"{export['module']}.{export['name']}")
    for record in grouped.values():
        record["exports"].sort()
        record["signature"], record["members"] = _typed_details(record["exports"], record["value"])
    return tuple(sorted(grouped.values(), key=lambda item: (item["canonical"], item["exports"])))


def _module_slug(module: str) -> str:
    if module == "pllm":
        return "index"
    return module.removeprefix("pllm.").replace(".", "-").replace("_", "-")


MODULE_USER_GUIDES = {
    "pllm": "/sdk/",
    "pllm.client": "/sdk/run/clients/",
    "pllm.config": "/sdk/experiments/",
    "pllm.models": "/sdk/models/",
    "pllm.native": "/sdk/components/kernels/native-matrix/",
    "pllm.plan": "/sdk/plans/",
    "pllm.components": "/sdk/components/",
    "pllm.assurance": "/sdk/components/assurance/",
    "pllm.correlation": "/sdk/components/correlation/",
    "pllm.kernels": "/sdk/components/kernels/",
    "pllm.metrics": "/sdk/evaluate/metrics/",
    "pllm.nonlinear": "/sdk/components/nonlinear/",
    "pllm.official": "/sdk/run/clients/",
    "pllm.passes": "/sdk/components/passes/",
    "pllm.profiles": "/sdk/inference/",
    "pllm.providers": "/sdk/extend/provider/",
    "pllm.research": "/research/records/",
    "pllm.protocols": "/sdk/components/protocols/",
    "pllm.protocols.masked_linear": "/sdk/inference/prepared-protocol/",
    "pllm.preparation": "/sdk/components/preparation/",
    "pllm.quantization": "/sdk/components/quantization/",
    "pllm.roles": "/sdk/components/roles/",
    "pllm.schedulers": "/sdk/components/schedulers/",
    "pllm.search": "/sdk/evaluate/search/",
    "pllm.server": "/sdk/run/gateway/",
    "pllm.sources": "/sdk/models/sources/",
    "pllm.state": "/sdk/components/state/",
    "pllm.verification": "/sdk/components/verification/",
    "pllm.pipeline": "/sdk/experiments/",
    "pllm.deployment": "/sdk/deployment/",
    "pllm.runtime": "/sdk/run/",
    "pllm.compiler": "/sdk/plans/compile/",
    "pllm.evidence": "/sdk/evaluate/evidence/",
}

OBJECT_USER_GUIDES = {
    ("pllm", "Experiment"): "/sdk/experiments/",
    ("pllm", "Model"): "/sdk/models/",
    ("pllm", "Deployment"): "/sdk/run/deployment/",
    ("pllm", "ExecutionBudget"): "/sdk/experiments/define/",
    ("pllm", "lower_model"): "/sdk/plans/lower/",
    ("pllm", "OpenAI"): "/sdk/run/clients/",
    ("pllm.deployment", "NetworkSpec"): "/sdk/deployment/network/",
    ("pllm.deployment", "PartyTrust"): "/sdk/deployment/network/",
    ("pllm.deployment", "PartySpec"): "/sdk/deployment/network/",
    ("pllm.deployment", "PartyOffer"): "/sdk/deployment/network/",
    ("pllm.deployment", "LivePartyOffer"): "/sdk/deployment/network/",
    ("pllm.deployment", "NetworkSnapshot"): "/sdk/deployment/network/",
    ("pllm.deployment", "LinkObservation"): "/sdk/deployment/network/",
    ("pllm.deployment", "LinkConditions"): "/sdk/deployment/network/",
    ("pllm.deployment", "discover"): "/sdk/deployment/network/",
    ("pllm.deployment", "async_discover"): "/sdk/deployment/network/",
    ("pllm.deployment", "ExecutionBinding"): "/sdk/deployment/execution/",
    ("pllm.deployment", "LiveExecutionBinding"): "/sdk/deployment/execution/",
    ("pllm.deployment", "ExecutionLease"): "/sdk/deployment/execution/",
    ("pllm.deployment", "open_execution"): "/sdk/deployment/execution/",
    ("pllm.deployment", "async_open_execution"): "/sdk/deployment/execution/",
    ("pllm.search", "PlanningRequest"): "/sdk/deployment/",
    ("pllm.search", "PlanningPolicy"): "/sdk/deployment/",
    ("pllm.search", "CandidateCostEvidence"): "/sdk/deployment/",
    ("pllm.compiler", "plan"): "/sdk/deployment/",
    ("pllm.plan", "PlanningResult"): "/sdk/deployment/",
}

_REFERENCE_LINK = re.compile(
    r"\[`(?P<name>[A-Za-z_]\w*(?:\(\))?)`\]"
    r"\(/sdk/reference/python/pllm/(?:(?P<slug>[a-z0-9-]+)/)?#(?P<anchor>[a-z0-9_-]+)\)"
)


@cache
def _guide_backlinks() -> dict[tuple[str, str], tuple[tuple[str, str], ...]]:
    """Reuse authored API links rather than maintaining a second option index."""
    modules = {_module_slug(module): module for module in PUBLIC_MODULES}
    links: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for source in sorted(SDK_GUIDES_ROOT.rglob("*.mdx")):
        if source.name == "research-method-roadmap.mdx":
            continue  # Generated after this reference; pending classes use its route below.
        relative = source.relative_to(SDK_GUIDES_ROOT).with_suffix("")
        parts = relative.parts[:-1] if relative.name == "index" else relative.parts
        route = "/sdk/" + ("/".join(parts) + "/" if parts else "")
        content = source.read_text(encoding="utf-8")
        title_match = re.search(r"^title:\s*(.+)$", content, re.MULTILINE)
        title = title_match.group(1).strip().strip('"\'') if title_match else route
        for match in _REFERENCE_LINK.finditer(content):
            module = modules.get(match.group("slug") or "index")
            name = match.group("name").removesuffix("()")
            if module and name.lower() == match.group("anchor"):
                links.setdefault((module, name), set()).add((title, route))
    return {key: tuple(sorted(value, key=lambda guide: (-guide[1].count("/"), guide[1])))
            for key, value in links.items()}


@cache
def _guide_title(route: str) -> str:
    if route == "/research/records/":
        return "Research records"
    relative = route.removeprefix("/sdk/").strip("/")
    source = SDK_GUIDES_ROOT / relative
    candidates = (source / "index.mdx", source.with_suffix(".mdx"))
    for candidate in candidates:
        if candidate.is_file():
            title = re.search(r"^title:\s*(.+)$", candidate.read_text(encoding="utf-8"), re.MULTILINE)
            if title:
                return title.group(1).strip().strip('"\'')
    raise ValueError(f"user guide has no authored source: {route}")


def _object_user_guides(
    module: str, name: str, value: object, exports: list[str]
) -> tuple[tuple[str, str], ...]:
    if inspect.isclass(value) and issubclass(value, PendingModelCapability) and value is not PendingModelCapability:
        return (("Reusable model capability", value.sdk_route), ("Model support matrix", "/sdk/reference/status/"))
    if inspect.isclass(value) and issubclass(value, PendingComponent) and value is not PendingComponent:
        direct = _guide_backlinks().get((module, name), ())
        return (*direct[:2], ("Research method roadmap", "/sdk/components/research-method-roadmap/"))
    fallback = OBJECT_USER_GUIDES.get((module, name), MODULE_USER_GUIDES[module])
    if module == "pllm" and (module, name) not in OBJECT_USER_GUIDES:
        for export in exports:
            alias = export.rsplit(".", 1)[0]
            if alias != module and alias in MODULE_USER_GUIDES:
                direct = _guide_backlinks().get((alias, name), ())
                if direct:
                    return direct[:3]
                fallback = MODULE_USER_GUIDES[alias]
                break
    direct = _guide_backlinks().get((module, name), ())
    if direct:
        if not any(route == fallback or route.startswith(fallback) for _, route in direct):
            return ((_guide_title(fallback), fallback), *direct[:2])
        return direct[:3]
    return ((_guide_title(fallback), fallback),)


_PAPER_METHODS = {(plan.module, plan.name): plan for plan in planned_components()}


def _object_summary(item: dict[str, Any], public_module: str) -> str:
    value = item["value"]
    canonical = str(item["canonical"])
    name = canonical.rsplit(".", 1)[-1]
    if inspect.isclass(value) and issubclass(value, PendingModelCapability) and value is not PendingModelCapability:
        return (
            f"`{name}` reserves a reusable model capability but is not executable. "
            f"See [the capability contract]({value.sdk_route})."
        )
    if inspect.isclass(value) and issubclass(value, PendingComponent) and value is not PendingComponent:
        plan = _PAPER_METHODS[(value.__module__, value.__name__)]
        return (
            f"**{plan.title}.** {plan.summary} This planned component cannot be constructed "
            f"or selected in an experiment. [Research source]({value.paper_route})."
        )
    if not callable(value) and is_dataclass(value):
        public_values = ", ".join(
            f"`{field.name}={getattr(value, field.name)!r}`"
            for field in fields(value)
            if not field.name.startswith("_")
        )
        return (
            f"`{name}` is the predefined `{type(value).__name__}` value"
            + (f" with {public_values}." if public_values else ".")
        )
    if canonical == "pllm.PROFILES":
        return "Maps each shipped profile ID to its immutable typed pipeline preset."
    if canonical == "pllm.__version__":
        return "Reports the installed PLLM distribution version."
    if inspect.isclass(value) and hasattr(value, "describe"):
        try:
            descriptor = getattr(value, "describe")()
            capabilities = ", ".join(f"`{item}`" for item in descriptor.capabilities)
            roles = (
                f" for roles {', '.join(descriptor.role_eligibility)}"
                if descriptor.role_eligibility
                else ""
            )
            return (
                f"Selects component `{descriptor.component}` during `{descriptor.lifecycle_phase}`; "
                f"it provides {capabilities or 'its declared contract'}{roles}."
            )
        except (AttributeError, TypeError, ValueError):
            pass
    if inspect.isclass(value) and issubclass(value, Enum):
        choices = ", ".join(f"`{member.value}`" for member in value)
        return f"Defines the accepted values for `{canonical.rsplit('.', 1)[-1]}`: {choices}."
    documentation = str(item["documentation"]).strip()
    name = canonical.rsplit(".", 1)[-1]
    if inspect.isclass(value) and issubclass(value, Exception):
        condition = name.removesuffix("Error").replace("_", " ")
        return f"Raised when a {condition} condition prevents the requested operation from completing."
    inherited_builtin_docs = {
        inspect.getdoc(Exception),
        inspect.getdoc(RuntimeError),
        inspect.getdoc(TypeError),
        inspect.getdoc(ValueError),
    }
    if (
        documentation
        and documentation not in inherited_builtin_docs
        and not documentation.startswith((f"{name}(", "ComponentRef(", "dict(", "str("))
    ):
        paragraph = documentation.split("\n\n", 1)[0].replace("\n", " ")
        return paragraph.rstrip(".") + "."
    specific = {
        "NetworkSpec": "Immutable static/local or authenticated HTTP network declaration with explicit approved trust roots and credential environment references",
        "NetworkSnapshot": "Immutable bounded observation of party offers and directed-link estimates; it carries no capacity reservation",
        "PartyOffer": "Static party capability and capacity declaration with authenticated=False; it is not a live host observation",
        "PartyTrust": "Approved party/operator/origin record naming a local credential environment variable rather than its value",
        "PlanningPolicy": "Explicit ordered network-placement objectives, bounded search limits and fail-closed freshness/resource/privacy constraints",
        "PlanningRequest": "Immutable network-placement request binding a lowered model, explicit same-source/numeric/budget candidates and planning policy",
        "AsyncOpenAI": "Asynchronous OpenAI-compatible client that targets the trusted local PLLM gateway",
        "OpenAI": "Synchronous OpenAI-compatible client that targets the trusted local PLLM gateway",
        "AsyncSSETransport": "Asynchronous transport for consuming bounded server-sent response streams",
        "SSETransport": "Synchronous transport for consuming bounded server-sent response streams",
        "HttpxTransport": "HTTPX-backed transport used by the native PLLM client",
        "CompiledRuntimeModel": "Opaque validated binding between a semantic model plan and a concrete runtime bundle",
        "CompiledRuntimeSession": "Stateful execution session created from one validated compiled runtime model",
        "MaskedTransformerClientRuntime": "Client-owned decoder runtime for prepared masked-linear execution",
        "MaskedTransformerEngine": "Inference-side engine that owns quantized remote stages and prepared rows",
        "NativeMatrix": "Reusable native integer matrix executor with explicit input and output conversion",
        "RuntimeRoles": "Owned local-process topology for inference and preparation roles",
        "LinearIntegrityError": "Failure raised when a checked linear result violates its authenticated contract",
        "environment_digest": "Computes a deterministic digest for the exact benchmark environment record",
        "evaluate_search": "Evaluates candidate benchmark evidence using explicit cohort-safe metric directions",
        "load_model": "Resolves and imports a model source while recording path-independent checkpoint hashes",
        "compile_runtime_model": "Binds a semantic plan to a validated runtime bundle and fails on unresolved operations",
        "build_roles": "Starts the bounded local inference and preparation process topology",
        "close_roles": "Stops all owned local role processes and releases their resources",
        "check_linear_result": "Checks one linear output against supplied integrity material",
        "discover_providers": "Discovers static provider manifests without importing provider implementation code",
        "load_provider_factory": "Imports one explicitly selected provider factory after manifest discovery",
        "load_component": "Creates one selected component implementation from its immutable reference",
        "serve": "Runs a selected PLLM service role from validated configuration",
    }.get(name)
    if specific:
        return specific + "."
    if name.startswith("create_") and name.endswith("_app"):
        service = name.removeprefix("create_").removesuffix("_app").replace("_", " ")
        return f"Builds the {service} ASGI application from validated runtime dependencies."
    if name.startswith("get_"):
        return f"Looks up the selected {name.removeprefix('get_').replace('_', ' ')} by stable identity."
    if name.startswith("list_"):
        return f"Returns the deterministic public {name.removeprefix('list_').replace('_', ' ')} inventory."
    purpose = str(MODULE_GUIDES[public_module]["purpose"]).split(".", 1)[0]
    words = " ".join(re.findall(r"[A-Z]+(?=[A-Z][a-z]|s?$)|[A-Z]?[a-z]+|\d+", name)).lower()
    if inspect.isclass(value) and is_dataclass(value):
        field_names = ", ".join(
            f"`{field.name}`" for field in fields(value) if not field.name.startswith("_")
        )
        if field_names:
            return f"Immutable `{name}` record carrying {field_names} for the workflow where {purpose.lower()}."
        return f"Opaque immutable `{name}` record created by its public factories for the workflow where {purpose.lower()}."
    if inspect.isclass(value):
        operations = [member.split("(", 1)[0].strip("`") for member in item["members"][:3]]
        operation_text = (
            f" Public operations include {', '.join(f'`{operation}`' for operation in operations)}."
            if operations
            else ""
        )
        return f"`{name}` provides {words} behavior for the workflow where {purpose.lower()}.{operation_text}"
    if callable(value):
        return f"Performs the {words} operation for the workflow where {purpose.lower()}."
    return f"Provides the public {words} value used where {purpose.lower()}."


def _module_items(module: str) -> tuple[dict[str, Any], ...]:
    return tuple(
        item
        for item in api_inventory()
        if any(export.rsplit(".", 1)[0] == module for export in item["exports"])
    )


def _research_context(citations: tuple[str, ...]) -> str:
    if not citations:
        return (
            "This API is project infrastructure rather than an implementation of one specific "
            "paper. Relevant method pages link their primary sources separately."
        )
    return "Relevant design inputs: " + ", ".join(citations) + ". PLLM's implementation and evidence claims remain independent."


def render_python_module_reference(module: str) -> str:
    guide = MODULE_GUIDES[module]
    slug = _module_slug(module)
    items = _module_items(module)
    if not items:
        raise ValueError(f"public module {module!r} has no documented exports")
    if set(MODULE_USER_GUIDES) != set(PUBLIC_MODULES):
        raise ValueError("public module user-guide mapping is incomplete")
    inference_command = ""
    if module in RUNNABLE_EXPERIMENT_MODULES:
        inference_command = (
            "Run the focused experiment through the trusted local gateway:\n\n"
            "```bash\n"
            "pllm gateway --local --experiment experiment.py:experiment --trust-python --api-key local\n"
            "```\n\n"
        )
    body = [
        _frontmatter(
            f"{module} Python API",
            f"Public objects, practical usage, and research context for {module}.",
        ),
        str(guide["purpose"]),
         " Presence documents API identity, not executable availability. Planned classes "
         "raise `NotYetImplementedError`; check [implementation status](/sdk/reference/status/) "
         "before relying on a path.\n\n",
    ]
    if module == "pllm":
        body.extend(("## Modules\n\n", "Every public module has its own executable reference page:\n\n"))
        for public_module in PUBLIC_MODULES[1:]:
            public_slug = _module_slug(public_module)
            body.append(
                f"- [`{public_module}`](/sdk/reference/python/pllm/{public_slug}/) - "
                f"{MODULE_GUIDES[public_module]['purpose']}\n"
            )
        body.append("\n")
    body.extend(
        (
            "## Research context\n\n",
            _research_context(cast(tuple[str, ...], guide["citations"])),
            "\n\n## User guide\n\n",
            f"For practical usage, see [{_guide_title(MODULE_USER_GUIDES[module])}]({MODULE_USER_GUIDES[module]}).",
            "\n\n## Python SDK example\n\n",
            _example_context(module),
            "\n\n",
            "```python\n",
            str(guide["example"]),
            "\n```\n\n",
            inference_command,
            f"API: [`{module}`](/sdk/reference/python/pllm/{'' if slug == 'index' else slug + '/'}#objects-and-signatures)\n\n",
            "## Objects and signatures\n\n",
        )
    )
    for item in items:
        module_exports = [
            export for export in item["exports"] if export.rsplit(".", 1)[0] == module
        ]
        display_name = module_exports[0].rsplit(".", 1)[-1]
        body.extend(
            (
                f"### `{display_name}`\n\n",
                _object_summary(item, module),
                "\n\n",
                f"- Canonical object: {_code(item['canonical'])}\n",
                f"- Kind: `{item['kind']}`\n",
                f"- Signature/type: {_code(item['signature'])}\n",
                "- Public exports: "
                + ", ".join(_code(export) for export in item["exports"])
                + "\n",
            )
        )
        value = item["value"]
        body.append(
            "- User guide: "
            + ", ".join(
                f"[{title}]({route})"
                for title, route in _object_user_guides(module, display_name, value, item["exports"])
            )
            + "\n"
        )
        paper_method = _PAPER_METHODS.get((module, display_name))
        if inspect.isclass(value) and issubclass(value, PendingModelCapability) and value is not PendingModelCapability:
            body.extend((
                "- Status: **Missing model capability** (`ModelCapabilityUnavailable` on construction).\n",
                f"- Reusable contract: [capability page]({value.sdk_route}).\n",
                f"- Next gate: {value.next_gate}.\n",
                "- Search: excluded until compiler, checkpoint and runtime gates pass.\n",
            ))
        elif inspect.isclass(value) and issubclass(value, PendingComponent) and value is not PendingComponent:
            if paper_method is None:
                raise ValueError(f"Missing component plan for {module}.{display_name}")
            body.extend((
                f"- Status: **Not yet implemented** (`NotYetImplementedError` on construction).\n",
                f"- Method: {paper_method.title}. {paper_method.summary}\n",
                f"- Research source: [paper and provenance]({value.paper_route}).\n",
                f"- Next gate: {value.next_gate}.\n",
                "- Search: excluded until implementation, compatible runtime coverage, and evidence.\n",
            ))
        elif paper_method is not None:
            body.extend((
                "- Status: **Implemented Python API**; exact runtime and evidence scope depends on the registered component.\n",
                f"- Method: {paper_method.title}. {paper_method.summary}\n",
                f"- Research source: [paper and provenance]({paper_method.paper_route}).\n",
                f"- Reviewed scope: {paper_method.gate}.\n",
            ))
        if item["members"]:
            body.append(
                "- Public members: "
                + "; ".join(_code(member) for member in item["members"])
                + "\n"
            )
        body.append("\n")
    return "".join(body)


def render_python_reference() -> str:
    return render_python_module_reference("pllm")


def render_python_reference_outputs() -> dict[Path, str]:
    outputs = {
        PYTHON_REFERENCE_ROOT / f"{_module_slug(module)}.mdx": render_python_module_reference(module)
        for module in PUBLIC_MODULES
    }
    outputs[PYTHON_REFERENCE_ROOT / "meta.json"] = json.dumps(
        {
            "title": "pllm",
            "pages": [_module_slug(module) for module in PUBLIC_MODULES],
        },
        indent=2,
    ) + "\n"
    return outputs


def model_compatibility() -> dict[str, Any]:
    """Curated claims checked against the native adapter and schedule in tests."""
    document = json.loads(MODEL_COMPATIBILITY.read_text(encoding="utf-8"))
    if document.get("schema") != "pllm.model_compatibility.v1":
        raise ValueError("unsupported model compatibility schema")
    adapters = document.get("adapters")
    candidates = document.get("candidates")
    capabilities = document.get("capabilities")
    if (
        not isinstance(adapters, list) or not adapters
        or not isinstance(candidates, list)
        or not isinstance(capabilities, dict) or not capabilities
    ):
        raise ValueError("model compatibility inventory is incomplete")
    for identity, capability in capabilities.items():
        if (
            not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", identity)
            or not isinstance(capability, dict)
            or not {"name", "operator", "contract"} <= set(capability) <= {"name", "operator", "contract", "current_scope"}
            or any(type(capability[field]) is not str or not capability[field] for field in ("name", "contract"))
            or (capability["operator"] is not None and type(capability["operator"]) is not str)
            or ("current_scope" in capability and (type(capability["current_scope"]) is not str or not capability["current_scope"]))
        ):
            raise ValueError("invalid model-neutral capability contract")
    if len({item.get("name") for item in adapters}) != len(adapters):
        raise ValueError("model compatibility configurations must have unique names")
    source_keys = {
        (
            item.get("adapter"), item.get("fixture"),
            json.dumps(item.get("config"), sort_keys=True),
        )
        for item in adapters
    }
    if len(source_keys) != len(adapters):
        raise ValueError("model compatibility adapter and source must be unique")
    adapter_families: dict[str, str] = {}
    required = {"name", "adapter", "model_family", "baseline_schedule", "runtime_evidence", "remaining", "guide", "requires", "baseline_blockers", "evidence"}
    for item in adapters:
        if (
            not isinstance(item, dict)
            or set(item) - {"pinned_real_checkpoint_functionality", "scaled_fixture"}
                not in (required | {"fixture"}, required | {"config"})
            or any(type(item[field]) is not str or not item[field] for field in required - {"baseline_schedule", "requires", "baseline_blockers", "evidence"})
            or type(item["baseline_schedule"]) is not bool
            or not item["guide"].startswith("/sdk/models/families/")
            or not isinstance(item["requires"], list)
            or not item["requires"]
            or any(type(value) is not str for value in item["requires"])
            or len(item["requires"]) != len(set(item["requires"]))
            or not set(item["requires"]) <= capabilities.keys()
            or not isinstance(item["baseline_blockers"], list)
            or any(type(value) is not str for value in item["baseline_blockers"])
            or not set(item["baseline_blockers"]) <= set(item["requires"])
            or item["baseline_schedule"] == bool(item["baseline_blockers"])
            or type(item["evidence"]) is not dict
            or set(item["evidence"]) != {"checkpoint", "provider", "quality"}
            or item["evidence"]["checkpoint"] not in {"real-local", "tiny-local", "none"}
            or item["evidence"]["provider"] not in {"real-prepared-local", "real-inprocess", "tiny-prepared", "none"}
            or item["evidence"]["quality"] not in {"unmeasured", "narrow-gap", "narrow-partial-four", "tiny-parity", "narrow-improvement", "narrow-parity", "narrow-smol-prefill", "none"}
            or (item["evidence"]["checkpoint"] == "real-local") != bool(item.get("pinned_real_checkpoint_functionality"))
            or (item["evidence"]["provider"].startswith("real-") and item["evidence"]["checkpoint"] != "real-local")
            or (not item["baseline_schedule"] and item["evidence"] != {"checkpoint": "none", "provider": "none", "quality": "none"})
        ):
            raise ValueError("invalid model compatibility adapter")
        previous_family = adapter_families.setdefault(item["adapter"], item["model_family"])
        if previous_family != item["model_family"]:
            raise ValueError("one semantic adapter cannot represent different model families")
        if "fixture" in item and (
            type(item["fixture"]) is not str
            or not item["fixture"].startswith("crates/pllm-models/tests/fixtures/")
            or not (ROOT / item["fixture"]).is_file()
        ):
            raise ValueError("model compatibility fixture is missing")
        if "scaled_fixture" in item and (
            type(item["scaled_fixture"]) is not str
            or "fixture" not in item
            or not item["scaled_fixture"].startswith("crates/pllm-models/tests/fixtures/")
            or not (ROOT / item["scaled_fixture"]).is_file()
        ):
            raise ValueError("model compatibility scaled fixture is missing")
        if "pinned_real_checkpoint_functionality" in item and (
            type(item["pinned_real_checkpoint_functionality"]) is not str
            or not item["pinned_real_checkpoint_functionality"]
            or len(item["pinned_real_checkpoint_functionality"]) > 64
        ):
            raise ValueError("invalid real-checkpoint functionality label")
        if "config" in item and not isinstance(item["config"], dict):
            raise ValueError("invalid inline model compatibility config")
    labels = [
        item["pinned_real_checkpoint_functionality"]
        for item in adapters if "pinned_real_checkpoint_functionality" in item
    ]
    if len(labels) != len(set(labels)):
        raise ValueError("duplicate real-checkpoint functionality label")
    for item in candidates:
        if (
            not isinstance(item, dict)
            or set(item) - {"missing_variants"} != {"name", "model_type", "source", "priority", "gap", "requires"}
            or any(type(item[field]) is not str or not item[field] for field in ("name", "model_type", "source", "priority", "gap"))
            or not isinstance(item["requires"], list)
            or any(type(value) is not str for value in item["requires"])
            or not item["requires"] or not set(item["requires"]) <= capabilities.keys()
            or not item["source"].startswith("https://")
            or not isinstance(item.get("missing_variants", []), list)
            or not set(item.get("missing_variants", [])) <= set(item["requires"])
        ):
            raise ValueError("invalid model evaluation candidate")
    return document


def render_python_status() -> str:
    """Two linked summaries: bounded family coverage and Python API completeness."""
    compatibility = model_compatibility()
    adapters = compatibility["adapters"]
    exports = public_exports()
    declared = len(exports)
    pending_count = sum(
        inspect.isclass(item["value"])
        and issubclass(item["value"], PendingComponent)
        and item["value"] is not PendingComponent
        for item in exports
    )
    for plan in planned_components():
        cls = getattr(importlib.import_module(plan.module), plan.name)
        pending = inspect.isclass(cls) and issubclass(cls, PendingComponent)
        if pending != (plan.status == "pending"):
            raise ValueError(f"Component plan status does not match {plan.module}.{plan.name}")
        if plan.status == "implemented" and plan.kind == "candidate" and not issubclass(cls, ComponentRef):
            raise ValueError(f"Implemented candidate must have a component contract: {plan.name}")
    for stub in model_capabilities():
        cls = getattr(importlib.import_module(stub.module), stub.name)
        pending = inspect.isclass(cls) and issubclass(cls, PendingModelCapability)
        if pending != (stub.status == "pending"):
            raise ValueError(f"Model capability plan status does not match {stub.module}.{stub.name}")
    pending_paper = sum(plan.status == "pending" for plan in planned_components())
    pending_model = sum(stub.status == "pending" for stub in model_capabilities())
    if pending_count != pending_paper + pending_model:
        raise ValueError("Pending component count must match the public Python exports")

    def percent(completed: int, total: int) -> str:
        return f"{100 * completed / total:.1f}%" if total else "—"

    real_checkpoint_paths = [
        item["pinned_real_checkpoint_functionality"]
        for item in adapters if item.get("pinned_real_checkpoint_functionality")
    ]
    body = [
        _frontmatter(
            "Model and SDK status",
            "Two linked matrices: model capability coverage and public Python implementation completeness.",
        ),
        "Use the first table for **scoped decoder evidence** and the second for **public Python API "
        "completeness**. A complete baseline schedule is not protected whole-model support; "
        "an importable API symbol is not necessarily executable. Each row and capability heading "
        "links to its detailed contract or reference.\n\n",
        "## Model families × reusable capabilities\n\n",
        f"**{len(adapters)} checked decoder configurations**, "
        f"**{sum(item['baseline_schedule'] for item in adapters)} complete baseline schedule paths**, "
        f"and **{len(real_checkpoint_paths)} pinned local real-checkpoint functionality paths** "
        f"({', '.join(real_checkpoint_paths)}). "
        "**No row has full model support:** protected whole-decoder execution, representative "
        "generation quality, and independently operated provider privacy remain unvalidated. "
        "This matrix is for bounded text-decoder scopes. A config plan, compiled schedule, "
        "real local checkpoint, and prepared request are distinct evidence gates.\n\n",
        "Read across a row to see which reusable pieces the family needs, what is checked, "
        "and which deployment and quality gates have run. "
        "**✓** = required and checked in this family's complete baseline schedule; "
        "**◇** = shared executable piece exists, but this family has no complete checked schedule; "
        "**×** = required variant has no admitted shared execution; "
        "**·** = not required. Candidate rows never receive ✓ from another family's tests. "
        "Select any symbol or heading for the [individual capability page](/sdk/models/capabilities/). "
        "The first column stays visible while you scroll sideways.\n\n",
    ]
    abbreviated = {
        "gqa": "GQA", "rotary": "RoPE", "qkv-bias": "QKV+", "rmsnorm": "RMS", "gated-silu": "SiLU",
        "kv-cache": "KV", "head-width": "Heads", "qk-norm": "QK", "gated-delta": "Delta",
        "causal-convolution": "Conv", "partial-mrope": "mRoPE", "fused-qkv": "Fused", "longrope": "Long",
        "local-global": "L/G", "shared-kv": "Shared", "gelu-tanh": "GeLU", "bf16-operator-composition": "BF16",
        "checkpoint-scalar-binding": "Scalar", "branch-norm": "Branch", "ple": "PLE", "softcap": "Cap",
        "untied-head": "Untied", "sliding-window": "Slide", "rope-scaling": "RoPE+", "expert-routing": "MoE",
        "latent-attention": "Latent", "mxfp4-import": "MXFP4", "fp8-import": "FP8",
    }
    if set(abbreviated) != set(compatibility["capabilities"]):
        raise ValueError("capability matrix headings do not match the checked inventory")
    capability_ids = tuple(compatibility["capabilities"])
    checked = set().union(*(set(item["requires"]) for item in adapters if item["baseline_schedule"]))
    evidence = {
        "checkpoint": {"real-local": "Real local", "tiny-local": "Tiny only", "none": "None"},
        "provider": {"real-prepared-local": "Real: 2 local children", "real-inprocess": "Real: in process", "tiny-prepared": "Tiny prepared", "none": "None"},
        "quality": {"unmeasured": "Not scored", "narrow-gap": "1/2 W8", "narrow-partial-four": "2/4 W8", "tiny-parity": "Tiny parity", "narrow-improvement": "2/2 opt-in; 3/5 other", "narrow-parity": "2/2 narrow", "narrow-smol-prefill": "11/12 W8 prefill", "none": "None"},
    }
    columns = 6 + len(capability_ids)
    body.extend((
        '<div className="not-prose mb-6 max-w-full overflow-x-auto rounded-xl border border-fd-border" '
        'role="region" aria-label="Model family and reusable capability matrix" tabIndex={0}>\n',
        '<table className="min-w-max border-collapse text-xs">\n<thead>\n<tr>\n',
        '<th scope="col" className="sticky left-0 z-20 min-w-52 border-r border-fd-border bg-fd-background px-3 py-3 text-left">Model family</th>\n',
        '<th scope="col" className="px-2 py-3">Config</th><th scope="col" className="px-2 py-3">Schedule</th>'
        '<th scope="col" className="px-2 py-3">Checkpoint</th><th scope="col" className="px-2 py-3">Provider</th>'
        '<th scope="col" className="px-2 py-3">Quality</th>\n',
    ))
    for identity in capability_ids:
        name = html_escape(compatibility["capabilities"][identity]["name"], quote=True)
        body.append(
            f'<th scope="col" className="px-2 py-3"><a href="/sdk/models/capabilities/{identity}/" '
            f'aria-label="{name}"><abbr title="{name}">{abbreviated[identity]}</abbr></a></th>\n'
        )
    body.append('</tr>\n</thead>\n<tbody>\n')

    def matrix_row(item: dict[str, Any], *, candidate: bool) -> str:
        requires = set(item["requires"])
        blockers = set(item.get("baseline_blockers", ())) | set(item.get("missing_variants", ()))
        name = html_escape(item["name"])
        link = html_escape(item["source"] if candidate else item["guide"], quote=True)
        status = item.get("evidence", {"checkpoint": "none", "provider": "none", "quality": "none"})
        cells = [
            '<tr className="border-t border-fd-border hover:bg-fd-muted/40">',
            f'<th scope="row" className="sticky left-0 z-10 max-w-52 border-r border-fd-border bg-fd-background px-3 py-2 text-left font-medium"><a href="{link}">{name}</a></th>',
            f'<td className="px-2 py-2 text-center">{"Unverified" if candidate else "Checked"}</td>',
            f'<td className="px-2 py-2 text-center">{"Unverified" if candidate else "Complete" if item["baseline_schedule"] else "Blocked"}</td>',
            *(f'<td className="px-2 py-2 text-center">{evidence[key][status[key]]}</td>' for key in ("checkpoint", "provider", "quality")),
        ]
        for identity in capability_ids:
            symbol = (
                "·" if identity not in requires else
                "×" if identity in blockers or identity not in checked else
                "◇" if candidate or not item["baseline_schedule"] else "✓"
            )
            meaning = {"·": "not required", "×": "missing required executable variant", "◇": "shared piece available but this family unchecked", "✓": "checked in the scoped baseline schedule"}[symbol]
            cells.append(
                f'<td className="px-2 py-2 text-center" aria-label="{name}: '
                f'{html_escape(compatibility["capabilities"][identity]["name"])}: {meaning}">'
                f'<a href="/sdk/models/capabilities/{identity}/" aria-label="Read {html_escape(compatibility["capabilities"][identity]["name"])} contract for {name}">{symbol}</a></td>'
            )
        cells.append('</tr>\n')
        return "".join(cells)

    body.append(f'<tr><th colSpan={{{columns}}} className="bg-fd-muted px-3 py-2 text-left">Checked decoder configurations</th></tr>\n')
    body.extend(matrix_row(item, candidate=False) for item in adapters)
    body.append(f'<tr><th colSpan={{{columns}}} className="bg-fd-muted px-3 py-2 text-left">Candidates requiring exact adapter and runtime checks</th></tr>\n')
    body.extend(matrix_row(item, candidate=True) for item in compatibility["candidates"])
    body.extend((
        '</tbody>\n</table>\n</div>\n\n',
        "A ✓ checks a scoped baseline, not protected execution. For candidates, ◇ is a shared "
        "piece checked **elsewhere**; × is a missing variant even if a narrower implementation "
        "exists. Follow the capability or [model-family](/sdk/models/families/) links for exact "
        "operators, evidence, limitations and next gates.\n\n",
    ))
    body.extend((
        "## Python SDK completeness\n\n",
        f"**{len(PUBLIC_MODULES)} modules**, **{declared - pending_count} implemented exports**, "
        f"**{pending_paper} paper-linked stubs** and **{pending_model} reusable model-capability stubs** "
        f"({percent(declared - pending_count, declared)} of public exports implemented at the API surface). "
        "Implemented here means importable or configured, not whole-decoder, privacy or quality coverage. "
        "Stubs raise typed errors and are excluded from executable component discovery. "
        "See the [paper-method roadmap](/sdk/components/research-method-roadmap/), "
        "[model capabilities](/sdk/models/capabilities/), and "
        "[executable component catalog](/sdk/reference/components/).\n\n",
        "| Python module | Implemented API symbols | Paper stubs | Model capability stubs | Declared exports | Implemented API % |\n",
        "| --- | ---: | ---: | ---: | ---: | ---: |\n",
    ))
    for module in PUBLIC_MODULES:
        values = [item["value"] for item in exports if item["module"] == module]
        model_pending = sum(
            inspect.isclass(value)
            and issubclass(value, PendingModelCapability)
            and value is not PendingModelCapability
            for value in values
        )
        pending = sum(
            inspect.isclass(value) and issubclass(value, PendingComponent)
            and value is not PendingComponent for value in values
        )
        ready = len(values) - pending
        route = (
            "/sdk/reference/python/pllm/"
            if module == "pllm"
            else f"/sdk/reference/python/pllm/{_module_slug(module)}/"
        )
        body.append(
            f"| [`{module}`]({route}) | {ready} | {pending - model_pending} | {model_pending} | {len(values)} | "
            f"{100 * ready / len(values):.1f}% |\n"
        )
    body.append(
        "\nRows count each module's own `__all__` exports, so names re-exported across "
        "modules can appear in multiple rows. The total above counts those public "
        "module exports, not unique Python object identities.\n\n"
        "## Python SDK example\n\n"
        "```python\nfrom pllm.components import model_capability, planned_component\n\n"
        'assert model_capability("gated-delta").module == "pllm.state"\n'
        'assert planned_component("ring-pcg").module == "pllm.correlation"\n'
        "```\n\n"
        "API: [`pllm.components`](/sdk/reference/python/pllm/components/#objects-and-signatures).\n"
    )
    return "".join(body)


def render_model_capability_outputs() -> dict[Path, str]:
    """Give every matrix column one independently linkable, evidence-scoped contract."""
    inventory = model_compatibility()
    adapters = inventory["adapters"]
    candidates = inventory["candidates"]
    pages: dict[Path, str] = {
        MODEL_CAPABILITY_DOC_ROOT / "meta.json": json.dumps(
            {"title": "Reusable capabilities", "pages": ["index"]}, indent=2,
        ) + "\n",
    }
    index = [
        _frontmatter("Reusable model capabilities", "Model-neutral contracts, checked decoder scopes, and missing executable variants."),
        "The [family × capability matrix](/sdk/reference/status/) summarizes status. "
        "Each capability below has its own scoped contract, source-family links, "
        "and next gate. An isolated numeric primitive is not full model support. "
        "**No public Python API is exposed for selecting capabilities from this index**; "
        "use the linked model plan and typed pending capability APIs where available.\n\n",
    ]
    pending_by_capability: dict[str, list[Any]] = {}
    for stub in model_capabilities():
        pending_by_capability.setdefault(stub.capability, []).append(stub)
    for identity, capability in inventory["capabilities"].items():
        name = capability["name"]
        route = f"/sdk/models/capabilities/{identity}/"
        index.append(f"- [{name}]({route}) — {capability['contract']}.\n")
        scoped = [item for item in adapters if identity in item["requires"]]
        interested = [item for item in candidates if identity in item["requires"]]
        stubs = pending_by_capability.get(identity, [])
        page = [
            _frontmatter(name, f"Reusable {name} contract and family-specific execution gates."),
            f"[Model status matrix](/sdk/reference/status/) · "
            f"[all reusable capabilities](/sdk/models/capabilities/)\n\n",
            f"{capability['contract']}.\n\n",
            f"Semantic operator/representation: `{capability['operator']}`.\n\n"
            if capability["operator"] else "This variant needs a complete model-neutral IR/numeric contract.\n\n",
            "## Checked decoder configurations\n\n",
        ]
        if capability.get("current_scope"):
            page.insert(-1, f"**Current implementation scope:** {capability['current_scope']}\n\n")
        if scoped:
            for item in scoped:
                baseline = item["baseline_schedule"] and identity not in item["baseline_blockers"]
                page.append(
                    f"- [{item['name']}]({item['guide']}): "
                    f"{'part of its bounded complete baseline schedule' if baseline else 'declared; complete baseline schedule blocked'}. "
                    f"{item['runtime_evidence']} "
                    f"**Next:** {item['remaining']}.\n"
                )
        else:
            page.append("No checked source adapter currently requires this capability.\n")
        page.append("\n## Candidate families\n\n")
        if interested:
            for item in interested:
                unavailable = identity in item.get("missing_variants", ()) or all(
                    identity not in adapter["requires"] or not adapter["baseline_schedule"]
                    for adapter in adapters
                )
                page.append(
                    f"- [{item['name']}]({item['source']}): "
                    f"{'required executable variant missing' if unavailable else 'shared machinery exists, but this family is unchecked'}. "
                    f"{item['gap']}.\n"
                )
        else:
            page.append("No additional candidate family is currently mapped to this contract.\n")
        page.append("\n## Selection and next gate\n\n")
        if stubs:
            page.append(
                "The following **importable, non-executable** classes reserve missing reusable "
                "contracts. They are excluded from component discovery, Experiment selection, "
                "search and provider registration:\n\n"
            )
            for stub in stubs:
                module_route = f"/sdk/reference/python/pllm/{_module_slug(stub.module)}/#{stub.name.lower()}"
                page.append(
                    f"- [`{stub.module}.{stub.name}`]({module_route}) (`{stub.identity}`): {stub.gate}.\n"
                )
            first = stubs[0]
            numeric_example = (
                "import numpy as np\n"
                "from pllm.state import BoundedDepthwiseCausalConvolution\n\n"
                "kernel = BoundedDepthwiseCausalConvolution(np.ones((2, 4), dtype=np.float32))\n"
                "initial = kernel.initial_state()\n"
                "prefill, retained = kernel.evaluate(\n"
                "    np.ones((1, 2, 2), dtype=np.float32), initial, mode=\"prefill\"\n"
                ")\n"
                "decode, _ = kernel.evaluate(\n"
                "    np.ones((1, 1, 2), dtype=np.float32), retained, mode=\"decode\"\n"
                ")\n"
                "assert prefill.shape == (1, 2, 2) and decode.shape == (1, 1, 2)\n\n"
            ) if identity == "causal-convolution" else ""
            page.extend((
                "\n## Python SDK example\n\n",
                "```python\n",
                numeric_example,
                f"from {first.module} import {first.name}\n",
                "from pllm.components import ModelCapabilityUnavailable\n\n",
                "try:\n",
                f"    {first.name}()\n",
                "except ModelCapabilityUnavailable as exc:\n",
                f'    assert exc.identity == "{first.identity}"\n',
                "else:\n",
                '    raise AssertionError("missing capability must remain unavailable")\n',
                "```\n\n",
                "API: [`pllm.state.BoundedDepthwiseCausalConvolution`](/sdk/reference/python/pllm/state/#objects-and-signatures).\n\n"
                if numeric_example else "",
                f"API: [`{first.module}.{first.name}`](/sdk/reference/python/pllm/{_module_slug(first.module)}/#objects-and-signatures).\n\n",
                "A later implementation must bind an actual operator or numeric lifecycle "
                "before its status changes.\n",
            ))
        else:
            page.append(
                "This contract is selected through source lowering and compiler admission, not "
                "by a stand-alone component. **No public Python API is exposed for selecting this "
                "operator independently.** Use the [model workflow](/sdk/models/) to inspect a "
                "source's supported scope.\n"
            )
        pages[MODEL_CAPABILITY_DOC_ROOT / f"{identity}.mdx"] = "".join(page)
    pages[MODEL_CAPABILITY_DOC_ROOT / "index.mdx"] = "".join(index)
    if {stub.capability for stub in model_capabilities()} - set(inventory["capabilities"]):
        raise ValueError("model capability stubs must map to documented matrix columns")
    return pages


def component_catalog() -> tuple[dict[str, Any], ...]:
    from pllm.components import list_components

    return tuple(component.to_dict() for component in list_components())


def render_component_catalog() -> str:
    body = [
        _frontmatter(
            "Component catalog",
            "Built-in component records with implementation, coverage, and evidence status.",
        ),
        "Catalog reflects `pllm.components.list_components()` only. Empty descriptor evidence is shown "
        "as no applicable evidence; it is not a pass.\n\n",
        "## Python SDK example\n\n",
        "```python\n",
        "from pllm.components import get, get_component\n\n",
        'component_class = get("pllm/cpu")\n',
        'component = get_component("pllm/cpu")\n',
        'assert component_class.describe() is component\n',
        'assert "cpu" in component.capabilities\n',
        "```\n\n",
        "API: [`pllm.components.get`](/sdk/reference/python/pllm/components/#objects-and-signatures)\n\n",
        "| Identity/version | Lifecycle | Implementation maturity | Provenance | Input/output representation | Roles/topology | Privacy/assurance | Model/operator coverage | Evidence/cohort | Known limitations |\n",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
    ]
    for item in component_catalog():
        evidence = _cell(item["evidence"]) if item["evidence"] else "no applicable evidence"
        coverage = f"capabilities: {_cell(item['capabilities'])}; host features: {_cell(item['required_host_features'])}"
        body.append(
            "| "
            + " | ".join(
                (
                    f"{_code(item['component'])} / {_code(item['version'])}",
                    _cell(item["lifecycle_phase"]),
                    "not recorded",
                    f"provider {_code(item['provider'])}; distribution {_code(item['distribution'])}; source links not recorded",
                    "not recorded",
                    _cell(item["role_eligibility"]),
                    "assurance status not recorded; privacy status not recorded",
                    coverage,
                    evidence,
                    "not recorded",
                )
            )
            + " |\n"
        )
    return "".join(body)


def render_outputs() -> dict[Path, str]:
    outputs = render_cli_reference_outputs()
    outputs.update(render_python_reference_outputs())
    outputs.update(render_model_capability_outputs())
    outputs.update(
        {
            ROOT / "docs/content/docs/reference/components.mdx": render_component_catalog(),
            ROOT / "docs/content/docs/reference/status.mdx": render_python_status(),
            ROOT / "docs/public/downloads/cli-help.txt": render_cli_help(),
        }
    )
    return outputs


def generate(*, check: bool = False) -> int:
    outputs = render_outputs()
    managed_roots = (PYTHON_REFERENCE_ROOT, CLI_REFERENCE_ROOT, MODEL_CAPABILITY_DOC_ROOT)
    expected_reference_paths = {
        path for path in outputs if any(path.is_relative_to(root) for root in managed_roots)
    }
    managed_reference_paths: set[Path] = set()
    for root in managed_roots:
        if root.exists():
            managed_reference_paths.update(root.rglob("*.mdx"))
            managed_reference_paths.update(root.rglob("meta.json"))
    orphaned = sorted(managed_reference_paths - expected_reference_paths)
    stale = list(orphaned)
    if not check:
        for path in orphaned:
            path.unlink()
    for path, content in outputs.items():
        if path.exists() and path.read_text(encoding="utf-8") == content:
            continue
        stale.append(path.relative_to(ROOT))
        if not check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    if stale:
        if check:
            print("Generated developer reference is stale:", file=sys.stderr)
            for path in stale:
                print(f"  {path}", file=sys.stderr)
            return 1
        print(f"Generated {len(stale)} developer reference files.")
    else:
        print("Generated developer reference is current.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated output differs")
    return generate(check=parser.parse_args().check)


if __name__ == "__main__":
    raise SystemExit(main())
