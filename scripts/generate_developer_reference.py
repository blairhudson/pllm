#!/usr/bin/env python3
"""Generate documentation inventories from PLLM's public Python and CLI surfaces."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
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

CLI_REFERENCE_ROOT = ROOT / "docs/content/docs/reference/cli"
CLI_ROOT_ORDER = ("gateway", "serve", "config", "components", "benchmark", "dev")


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

COMPONENT_EXAMPLE = _example(
    """
    import pllm.components as components

    reference = components.ComponentRef("pllm/cpu")
    implementation = components.get(reference.component)
    assert implementation.describe().component == "pllm/cpu"
    """
)

CORRELATION_EXAMPLE = _example(
    """
    import pllm.correlation as correlation

    source = correlation.SeededExpansion()
    assert source.to_spec() == {"component": "pllm/seeded-expansion", "params": {}}
    """
)

KERNEL_EXAMPLE = _example(
    """
    import pllm.kernels as kernels

    backend = kernels.Cpu(threads=2)
    assert backend.get_params() == {"threads": 2}
    """
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

PROTOCOL_EXAMPLE = _example(
    """
    import pllm.protocols as protocols

    guarded = protocols.GuardedLinear(max_rows_per_request=128, output_dither_bound=1)
    assert guarded.get_params()["max_rows_per_request"] == 128
    """
)

PROFILE_EXAMPLE = _example(
    """
    import pllm.profiles as profiles

    from pllm.sources import TinyModel

    selected = profiles.MaskedLinearCpu(TinyModel("qwen2"))
    assert selected.to_spec()["components"]["linear"]["component"] == "pllm/masked-linear"
    """
)

SOURCE_EXAMPLE = _example(
    """
    import pllm.sources as sources

    model = sources.TinyModel("qwen2", model_id="transport-smoke")
    assert model.to_spec() == {"source": "qwen2", "kind": "tiny", "model_id": "transport-smoke"}
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

MASKED_LINEAR_EXAMPLE = _example(
    """
    import pllm.protocols.masked_linear as masked_linear

    method = masked_linear.MaskedLinear()
    assert method.to_spec() == {"component": "pllm/masked-linear", "params": {}}
    """
)

PREPARATION_EXAMPLE = _example(
    """
    import pllm.preparation as preparation

    provider = preparation.ModelAwareCorrections()
    assert provider.component == "pllm/model-aware-corrections"
    """
)

ROLE_EXAMPLE = _example(
    """
    import pllm.roles as roles

    inference = roles.Inference()
    assert inference.component == "pllm/inference"
    """
)

VERIFICATION_EXAMPLE = _example(
    """
    import pllm.verification as verification

    verifier = verification.FreivaldsVerify(target_failure_bits=48)
    assert verifier.get_params()["target_failure_bits"] == 48
    """
)

PIPELINE_EXAMPLE = _example(
    """
    import pllm.pipeline as pipeline
    from pllm.sources import TinyModel

    configured = pipeline.VerifiedMaskedLinearCpu(TinyModel("qwen2"))
    assert configured.to_spec()["components"]["verification"]["component"] == "pllm/freivalds-verify/v1"
    """
)

DEPLOYMENT_EXAMPLE = _example(
    """
    import pllm.deployment as deployment

    local = deployment.Deployment(kind="local", root="/tmp/pllm-example")
    assert local.kind == "local" and local.root.endswith("pllm-example")
    """
)

DASH_CITATION = "[DASH](/research/papers/r01-dash/)"
REDASH_CITATION = "[ReDASH](/research/papers/r02-redash/)"
SLALOM_CITATION = "[Slalom](/research/papers/r07-slalom/)"
MPCACHE_CITATION = "[MPCache](/research/papers/r23-mpcache/)"
COMPACT_CITATION = "[Compact](/research/papers/r18-compact/)"
R03_CITATION = "[Garbling Gadgets](/research/papers/r03-garbling-gadgets/)"
HYCC_CITATION = "[HyCC](/research/papers/r09-hycc/)"

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
        "purpose": "Plan records and locks bind semantic, numeric, privacy, and deployment decisions into immutable digest-addressed documents.",
        "citations": (),
        "example": PLAN_RECORD_EXAMPLE,
    },
    "pllm.components": {
        "purpose": "The component registry discovers typed built-in capabilities by stable identity and creates concrete implementations only when explicitly requested.",
        "citations": (),
        "example": COMPONENT_EXAMPLE,
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
        "purpose": "Metric definitions attach units and optimization direction to measurements so evidence and Pareto search do not guess whether larger or smaller is better.",
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
        "purpose": "Search APIs generate immutable experiment candidates and compare only cohort-compatible evidence with explicit metric directions.",
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
        "purpose": "Deployment records describe role placement, endpoints, and trust boundaries without starting services or embedding credentials.",
        "citations": (),
        "example": DEPLOYMENT_EXAMPLE,
    },
    "pllm.runtime": {
        "purpose": "Runtime APIs execute validated native kernels, bind compiled models, construct local role topologies, and expose trusted client-boundary services.",
        "citations": (SLALOM_CITATION, DASH_CITATION),
        "example": RUNTIME_EXAMPLE,
    },
    "pllm.compiler": {
        "purpose": "The compiler accepts canonical request bytes, verifies complete capability coverage, and returns opaque native plans or fails closed.",
        "citations": (HYCC_CITATION,),
        "example": COMPILER_EXAMPLE,
    },
    "pllm.evidence": {
        "purpose": "Evidence records preserve exact benchmark cohorts, environment identity, and assurance results without silently ranking incomparable runs.",
        "citations": (),
        "example": EVIDENCE_EXAMPLE,
    },
}


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
        "[Benchmarking](/cli/benchmarking/), or "
        "[Inspect components](/cli/inspect-and-research/). Choose a command group below for "
        "exact generated arguments.\n\n",
        "## Complete grammar\n\n",
        "```text\n",
        *[f"{command} [OPTIONS]\n" for command in leaves],
        "```\n\n",
        "Global options accepted at each parser level are `--format {human,json,jsonl}`, `--quiet`, "
        "`--no-color`, `--no-input`, and `--dry-run`. Root also accepts `--version`. Argument "
        "abbreviation is disabled.\n\n",
        "[Download exact complete help](/downloads/cli-help.txt).\n\n",
        "## `pllm --help`\n\n",
        "```text\n",
        sections[0][1],
        "```\n",
    ]
    return "".join(body)


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
    example = CLI_EXAMPLES["pllm gateway"][3:]
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


def _object_summary(item: dict[str, Any], public_module: str) -> str:
    value = item["value"]
    canonical = str(item["canonical"])
    name = canonical.rsplit(".", 1)[-1]
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
    body = [
        _frontmatter(
            f"{module} Python API",
            f"Public objects, practical usage, and research context for {module}.",
        ),
        str(guide["purpose"]),
        " Presence documents API availability, not complete model, security, or deployment "
        "coverage; check [implementation status](/sdk/reference/status/) before relying on a path.\n\n",
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
            "\n\n## Python SDK example\n\n",
            "The example performs a bounded offline task and checks an observable result. It does "
            "not download a checkpoint or contact a provider.\n\n",
            "```python\n",
            str(guide["example"]),
            "\n```\n\n",
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
    outputs.update(
        {
            ROOT / "docs/content/docs/reference/components.mdx": render_component_catalog(),
            ROOT / "docs/public/downloads/cli-help.txt": render_cli_help(),
        }
    )
    return outputs


def generate(*, check: bool = False) -> int:
    outputs = render_outputs()
    managed_roots = (PYTHON_REFERENCE_ROOT, CLI_REFERENCE_ROOT)
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
