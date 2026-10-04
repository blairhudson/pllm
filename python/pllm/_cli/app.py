"""Parser and command dispatch for PLLM's 0.1 CLI."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import secrets
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn

from pllm._version import __version__

from .errors import CLIError, LocalIOError, ResolutionError, RuntimeFailure, UsageError
from .output import emit_error, emit_machine


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["allow_abbrev"] = False
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> NoReturn:
        raise UsageError(message)


class _BenchmarkModelAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace.benchmark_model_override = True


def _add_globals(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=("human", "json", "jsonl"),
        default=argparse.SUPPRESS,
        help="result format (default: human)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS,
        help="suppress non-error diagnostics",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        default=argparse.SUPPRESS,
        help="disable colored output",
    )
    parser.add_argument(
        "--no-input",
        action="store_true",
        default=argparse.SUPPRESS,
        help="fail instead of prompting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=argparse.SUPPRESS,
        help="report work without persistent writes or service startup",
    )


def _command(parent: Any, name: str, **kwargs: Any) -> _Parser:
    parser = parent.add_parser(name, **kwargs)
    _add_globals(parser)
    return parser


def _target_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "TARGET", help="Experiment .json/.yaml or explicit Python path.py:object/module:object"
    )
    parser.add_argument(
        "--factory", action="store_true", help="call the Python target as a zero-argument factory"
    )
    parser.add_argument(
        "--trust-python",
        action="store_true",
        help="allow import and execution of the explicit Python target",
    )


def _add_server_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="JSON GatewayConfig file")
    parser.add_argument(
        "--experiment",
        metavar="TARGET",
        help="Experiment .json/.yaml or explicit Python path.py:object/module:object",
    )
    parser.add_argument(
        "--factory", action="store_true", help="call the Python target as a zero-argument factory"
    )
    parser.add_argument(
        "--trust-python",
        action="store_true",
        help="allow import and execution of the explicit Python experiment target",
    )
    parser.add_argument(
        "--privacy-mode",
        "--mode",
        dest="privacy_mode",
        choices=("public", "proprietary"),
        default=None,
    )
    parser.add_argument(
        "--protocol",
        "--proprietary-protocol",
        dest="proprietary_protocol",
        choices=("guarded", "blinded", "secure", "direct"),
        default=None,
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--api-key", help=argparse.SUPPRESS)
    parser.add_argument("--provider-push-api-key", help=argparse.SUPPRESS)
    parser.add_argument("--inference-url")
    parser.add_argument("--push-api-key", help=argparse.SUPPRESS)
    parser.add_argument(
        "--push-timeout",
        type=float,
        default=None,
    )
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument(
        "--model-kind",
        choices=("huggingface", "safetensors", "vllm", "mlx", "mlx-lm"),
        default=None,
    )
    parser.add_argument("--revision")
    parser.add_argument("--hf-token", help=argparse.SUPPRESS)
    parser.add_argument("--hf-cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--weight-bits", type=int, choices=(4, 8))
    parser.add_argument("--activation-bits", type=int, choices=(4, 8))
    parser.add_argument("--tenseal-path")
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--max-wait-ms", type=float)
    parser.add_argument("--fixed-batch-wait", action="store_true")
    parser.add_argument("--allow-insecure-local-correlations", action="store_true")
    parser.add_argument(
        "--engine-threads",
        type=int,
        default=None,
    )
    parser.add_argument("--metal-min-rows", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--native-library")
    parser.add_argument("--compiled-cache-dir")
    parser.add_argument("--streaming-threshold-elements", type=int)
    parser.add_argument("--quantization-chunk-rows", type=int)
    parser.add_argument(
        "--verification-component",
        choices=("none", "pllm/freivalds-verify/v1"),
    )
    parser.add_argument("--verification-target-failure-bits", type=int)
    parser.add_argument("--public-equalization-digest")
    parser.add_argument("--remote-output-head", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--client-prefix-layers", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--client-linear-roles", default="", help=argparse.SUPPRESS)
    parser.add_argument("--prepared-output-encoding", choices=("raw", "row_residues"), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--guard-max-rows-per-request", type=int)
    parser.add_argument("--guard-max-rows-per-stage", type=int)
    parser.add_argument("--guard-max-requests-per-minute", type=int)
    parser.add_argument("--guard-output-dither", type=int)
    parser.add_argument(
        "--rendezvous-timeout",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--rendezvous-capacity",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--rendezvous-max-bytes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--prepared-session-capacity",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--prepared-session-idle",
        type=float,
        default=None,
    )


def build_parser() -> _Parser:
    parser = _Parser(
        prog="pllm",
        description="Inspect PLLM configuration, run benchmarks, and operate private inference",
    )
    _add_globals(parser)
    parser.add_argument("--version", action="version", version=f"pllm {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    network = _command(commands, "network", help="inspect static local network declarations")
    network_commands = network.add_subparsers(dest="network_command", required=True)
    for name in ("inspect", "parties", "snapshot", "drain", "leave"):
        command = _command(network_commands, name)
        command.add_argument("NETWORK", help="strict static NetworkSpec JSON file")
        if name in {"drain", "leave"}:
            command.add_argument("--party", required=True)
        if name == "snapshot":
            command.add_argument("--output", required=True)
            command.add_argument("--force", action="store_true")
    planning = _command(commands, "plan", help="offline placement planning and validation")
    plan_commands = planning.add_subparsers(dest="plan_command", required=True)
    for name in ("create", "inspect", "explain", "validate"):
        command = _command(plan_commands, name)
        command.add_argument("REQUEST" if name == "create" else "PLAN")
        if name in {"create", "validate"}:
            command.add_argument("--snapshot", required=True)
        if name == "create":
            command.add_argument("--output", required=True)
            command.add_argument("--force", action="store_true")

    config = _command(commands, "config", help="inspect or export public configuration")
    config_commands = config.add_subparsers(dest="config_command", metavar="COMMAND", required=True)
    show = _command(config_commands, "show", help="show validated public configuration")
    _target_options(show)
    export = _command(config_commands, "export", help="export strict public JSON or YAML")
    _target_options(export)
    export.add_argument(
        "--output", required=True, metavar="PATH", help="new output .json/.yaml path"
    )
    export.add_argument("--force", action="store_true", help="replace an existing output file")

    components = _command(commands, "components", help="inspect built-in component descriptors")
    component_commands = components.add_subparsers(
        dest="components_command", metavar="COMMAND", required=True
    )
    _command(component_commands, "list", help="list built-in component descriptors")
    component_show = _command(component_commands, "show", help="show one component descriptor")
    component_show.add_argument("COMPONENT", help="component identity, for example pllm/cpu")

    topology = _command(commands, "topology", help="inspect admitted runtime role graphs")
    topology_commands = topology.add_subparsers(
        dest="topology_command", metavar="COMMAND", required=True
    )
    topology_inspect = _command(
        topology_commands, "inspect", help="show role, channel, and placement assumptions"
    )
    _target_options(topology_inspect)
    topology_inspect.add_argument(
        "--reference",
        choices=("client-only", "two-online-offset"),
        help="inspect a non-executable research comparator rather than the installed graph",
    )
    topology_inspect.add_argument(
        "--role-deployment",
        type=Path,
        metavar="PATH",
        help="inspect a versioned role-placement JSON declaration (never authorizes serving)",
    )

    gateway = _command(commands, "gateway", help="run the trusted local Responses API gateway")
    gateway.add_argument("--config", help="client TOML file")
    gateway.add_argument("--host", default="127.0.0.1")
    gateway.add_argument("--port", type=int, default=8080)
    gateway.add_argument("--api-key", default=os.getenv("PLLM_GATEWAY_API_KEY", "local"))
    gateway.add_argument("--inference-url")
    gateway.add_argument("--inference-key", default=os.getenv("PLLM_INFERENCE_API_KEY"))
    gateway.add_argument("--preparation-url")
    gateway.add_argument("--preparation-key", default=os.getenv("PLLM_PREPARATION_API_KEY"))
    gateway_selection = gateway.add_mutually_exclusive_group()
    gateway_selection.add_argument(
        "--experiment",
        metavar="TARGET",
        help=("local Experiment .json/.yaml or explicit Python path.py:object/module:object"),
    )
    gateway_selection.add_argument("--plan", type=Path, help="fixed native-selected plan")
    gateway_selection.add_argument("--request", type=Path, help="bounded planning before each response")
    gateway.add_argument("--network", type=Path, help="network trust/discovery configuration")
    gateway.add_argument(
        "--factory", action="store_true", help="call the Python target as a zero-argument factory"
    )
    gateway.add_argument(
        "--trust-python",
        action="store_true",
        help="allow import and execution of the explicit Python experiment target",
    )
    gateway.add_argument("--model")
    gateway.add_argument("--model-id")
    gateway.add_argument("--local", action="store_true", help="run the selected role graph locally")
    gateway.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)
    gateway.add_argument(
        "--transport", choices=("auto", "http", "websocket"), help="session transport"
    )
    gateway.add_argument("--correlation-mode", choices=("bfv", "local-test"))
    gateway.add_argument("--correlation-prefetch", type=int)
    gateway.add_argument("--prepared-inventory-rows", type=int)
    gateway.add_argument("--token-cache-size", type=int)
    gateway.add_argument(
        "--bundle-cache-mode", choices=("read-write", "read-only", "refresh", "off")
    )
    gateway.add_argument("--bundle-cache-dir")
    gateway.add_argument("--tenseal-path", default=os.getenv("PLLM_PYDEPS"))
    gateway.add_argument("--timeout", type=float)
    gateway.add_argument("--revision", default=os.getenv("PLLM_HF_REVISION"))
    gateway.add_argument("--hf-cache-dir", default=os.getenv("HF_HUB_CACHE"))
    gateway.add_argument("--local-files-only", action="store_true")
    gateway.add_argument("--weight-bits", type=int, choices=(4, 8))
    gateway.add_argument("--activation-bits", type=int, choices=(4, 8))

    serve = _command(commands, "serve", help="run an inference or preparation role")
    serve_commands = serve.add_subparsers(dest="serve_role", metavar="ROLE", required=True)
    inference = _command(
        serve_commands, "inference", help="run the remote private-inference service"
    )
    _add_server_options(inference)
    preparation = _command(
        serve_commands, "preparation", help="run the trusted preparation service"
    )
    _add_server_options(preparation)
    for name in ("party", "directory"):
        service = _command(serve_commands, name, help=f"run authenticated {name} service")
        service.add_argument("--network", required=True)
        if name == "party":
            service.add_argument("--party", required=True)
        service.add_argument("--host", default="127.0.0.1")
        service.add_argument("--port", type=int, default=8000 if name == "party" else 8001)

    benchmark = _command(commands, "benchmark", help="run reproducible local benchmarks")
    benchmark_commands = benchmark.add_subparsers(
        dest="benchmark_command", metavar="COMMAND", required=True
    )
    benchmark_run = _command(
        benchmark_commands,
        "run",
        help="run the real client, preparation, and inference roles on loopback",
    )
    benchmark_selection = benchmark_run.add_mutually_exclusive_group()
    benchmark_selection.add_argument(
        "--experiment",
        action="append",
        default=[],
        metavar="TARGET",
        help=(
            "Experiment .json/.yaml or explicit Python path.py:object/module:object; repeat to "
            "compare pipelines"
        ),
    )
    benchmark_selection.add_argument("--plan", type=Path, help="fixed selected deployment")
    benchmark_selection.add_argument("--request", type=Path, help="bounded network planning request")
    benchmark_run.add_argument("--network", type=Path)
    benchmark_run.add_argument("--compare-feasible", action="store_true", help="bounded matched feasible controls")
    benchmark_run.add_argument("--temperature", type=float, help="sampling temperature; omitted preserves SDK default 0.8")
    benchmark_run.add_argument("--capture-output-digest", action="store_true", help="opt-in public-task output fingerprint")
    benchmark_run.add_argument("--docker", action="store_true", help="run local public CPU provider roles in lightweight Linux containers")
    benchmark_run.add_argument("--docker-image", help="use an existing runtime image instead of building the checkout")
    benchmark_run.add_argument("--docker-network-profile", help="LinkConditions JSON for provider-egress latency/rate/loss")
    wan_mode = benchmark_run.add_mutually_exclusive_group()
    wan_mode.add_argument("--wan", action="store_true", help="enforce shared party UP/DOWN rates in local Docker namespaces; default 100/40 Mbps")
    wan_mode.add_argument("--wan-estimate", action="store_true", help="calculate WAN bandwidth floors without throttling execution")
    benchmark_run.add_argument("--wan-profile", help="WanConditions JSON; enables local rate enforcement unless --wan-estimate")
    benchmark_run.add_argument("--wan-download-mbps", type=float, help="shared per-party download Mbps (default: 100); implies --wan")
    benchmark_run.add_argument("--wan-upload-mbps", type=float, help="shared per-party upload Mbps (default: 40); implies --wan")
    benchmark_run.add_argument("--wan-party", action="append", default=[], metavar="PARTY:DOWN:UP",
                               help="per-party Mbps override; repeatable; implies --wan unless --wan-estimate")
    benchmark_run.add_argument(
        "--factory",
        action="store_true",
        help="call each Python target as a zero-argument factory",
    )
    benchmark_run.add_argument(
        "--trust-python",
        action="store_true",
        help="allow import and execution of explicit Python experiment targets",
    )
    benchmark_run.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        action=_BenchmarkModelAction,
        help="Hugging Face model ID or local checkpoint path",
    )
    benchmark_run.add_argument("--model-id", help="stable model identity stored in the report")
    benchmark_run.add_argument(
        "--tiny",
        action="store_true",
        help="use generated random weights for a transport smoke test",
    )
    prompt_source = benchmark_run.add_mutually_exclusive_group()
    prompt_source.add_argument(
        "--prompt",
        default="Explain why neither server can see the prompt.",
        help="prompt text (visible in shell process listings)",
    )
    prompt_source.add_argument("--prompt-file", type=Path, help="read prompt text from this file")
    benchmark_run.add_argument(
        "--warmup-prompt-file",
        type=Path,
        help="use a different private warmup prompt (for shared-prefix comparisons)",
    )
    benchmark_run.add_argument(
        "--max-output-tokens",
        type=int,
        default=24,
        help="maximum generated tokens per run (default: 24)",
    )
    benchmark_run.add_argument(
        "--warmups", type=int, default=0, help="warmup runs retained in the report (default: 0)"
    )
    benchmark_run.add_argument(
        "--prompt-sequence-file",
        type=Path,
        help="JSON array of 1-32 full conversation contexts, in order; repetitions repeat the sequence",
    )
    benchmark_run.add_argument(
        "--repetitions", type=int, default=1, help="measured runs (default: 1)"
    )
    benchmark_run.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="startup and per-run timeout in seconds (default: 900)",
    )
    benchmark_run.add_argument(
        "--show-dashboard",
        action="store_true",
        help="open the local dashboard in a browser (default: hidden)",
    )
    benchmark_run.add_argument(
        "--inventory-policy",
        choices=("prewarm", "request-sized"),
        default=None,
        help="override inventory policy (default: Experiment selection or prewarm)",
    )
    benchmark_run.add_argument(
        "--bundle-compression",
        choices=("none", "zlib", "artifacts", "artifacts-zlib"),
        default=None,
        help="override bundle encoding (default: Experiment selection or none)",
    )
    benchmark_run.add_argument(
        "--prefill-cache-mib",
        type=int,
        default=0,
        help="bound client-only exact-prompt prefill reuse in MiB (0-256; default: off)",
    )
    benchmark_run.add_argument(
        "--prefill-cache-mode",
        choices=("exact", "prefix"),
        default="exact",
        help="reuse exact prompts or cost-gated shared causal prefixes",
    )
    benchmark_run.add_argument(
        "--prefill-cache-bound-tokens",
        type=int,
        help="fixed compiler input bound required for shared-prefix reuse (2-4096)",
    )
    benchmark_run.add_argument("--output", type=Path, help="write the sanitized JSON report")
    benchmark_run.add_argument(
        "--save-best",
        type=Path,
        metavar="PATH",
        help="write the Experiment with lowest measured median full latency as canonical JSON",
    )
    benchmark_run.add_argument("--force", action="store_true", help="replace --output if it exists")

    benchmark_quality = _command(
        benchmark_commands,
        "quality",
        help="compare pinned Experiments to a local float32 reference on the same prompt cohort",
    )
    benchmark_quality.add_argument(
        "--experiment",
        action="append",
        required=True,
        metavar="TARGET",
        help="Experiment JSON/YAML or explicit trusted Python target; repeat for comparison",
    )
    benchmark_quality.add_argument("--factory", action="store_true")
    benchmark_quality.add_argument("--trust-python", action="store_true")
    benchmark_quality.add_argument("--prompts-file", type=Path, required=True, metavar="PATH")
    benchmark_quality.add_argument("--top-k", type=int, default=5)
    benchmark_quality.add_argument("--output", type=Path, help="write the prompt-free JSON report")
    benchmark_quality.add_argument("--force", action="store_true", help="replace --output")

    dev = _command(commands, "dev", help="development tools")
    dev_commands = dev.add_subparsers(dest="dev_command", metavar="COMMAND", required=True)
    dashboard = _command(
        dev_commands,
        "dashboard",
        help="launch the local benchmark dashboard",
    )
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8791)
    dashboard.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    dashboard.add_argument("--model-id")
    dashboard.add_argument("--tiny", action="store_true")
    dashboard.add_argument("--max-output-tokens", type=int, default=24)
    dashboard.add_argument("--history-db", metavar="PATH")
    dashboard.add_argument("--startup-inventory-rows", type=int, help=argparse.SUPPRESS)
    dashboard.add_argument("--experiment-config", type=Path, help=argparse.SUPPRESS)
    dashboard.add_argument("--no-open", action="store_true")
    return parser


def _globals(args: argparse.Namespace) -> tuple[str, bool, bool]:
    return (
        getattr(args, "format", "human"),
        bool(getattr(args, "no_input", False)),
        bool(getattr(args, "dry_run", False)),
    )


def _config(args: argparse.Namespace, output_format: str, no_input: bool, dry_run: bool) -> None:
    from pllm.config import canonical_bytes, configuration_digest

    from .targets import resolve_target

    target = resolve_target(
        args.TARGET,
        factory=args.factory,
        no_input=no_input,
        trust_python=args.trust_python,
        output_format=output_format,
    )
    spec = target.configuration.to_spec()
    digest = configuration_digest(target.configuration)
    if args.config_command == "show":
        if output_format == "human":
            print(json.dumps(spec, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True))
            return
        emit_machine(
            "config.show",
            {
                "configuration": spec,
                "configuration_digest": digest,
                "dry_run": dry_run,
                "python_executed": target.python_executed,
                "target_kind": target.kind,
            },
            output_format,
        )
        return

    output = Path(args.output).expanduser()
    suffix = output.suffix.lower()
    if suffix == ".json":
        content = canonical_bytes(target.configuration) + b"\n"
        serialization = "json"
    elif suffix in {".yaml", ".yml"}:
        import yaml

        content = yaml.safe_dump(spec, allow_unicode=True, sort_keys=True).encode("utf-8")
        serialization = "yaml"
    else:
        raise ResolutionError("OUTPUT_FORMAT", "output path must end in .json, .yaml, or .yml")

    if not args.force:
        try:
            output_exists = output.exists()
        except OSError as exc:
            raise LocalIOError("OUTPUT_WRITE", f"cannot inspect output: {output}") from exc
        if output_exists:
            raise LocalIOError(
                "OUTPUT_EXISTS", f"output already exists; use --force to replace it: {output}"
            )

    data = {
        "bytes": len(content),
        "configuration_digest": digest,
        "dry_run": dry_run,
        "format": serialization,
        "output": str(output),
        "python_executed": target.python_executed,
        "target_kind": target.kind,
        "written": not dry_run,
    }
    if not dry_run:
        mode = "wb" if args.force else "xb"
        try:
            with output.open(mode) as stream:
                stream.write(content)
        except FileExistsError as exc:
            raise LocalIOError(
                "OUTPUT_EXISTS", f"output already exists; use --force to replace it: {output}"
            ) from exc
        except OSError as exc:
            raise LocalIOError("OUTPUT_WRITE", f"cannot write output: {output}") from exc
    if output_format == "human":
        print(f"Would write {output}" if dry_run else f"Wrote {output}")
    else:
        emit_machine("config.export", data, output_format)


def _components(args: argparse.Namespace, output_format: str, dry_run: bool) -> None:
    from pllm.components import get_component, list_components

    if args.components_command == "show":
        try:
            descriptor = get_component(args.COMPONENT)
        except KeyError:
            raise ResolutionError("COMPONENT_NOT_FOUND", f"unknown component: {args.COMPONENT}")
        data = descriptor.to_dict()
        if output_format == "human":
            print(json.dumps(data, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            emit_machine("components.show", {"component": data, "dry_run": dry_run}, output_format)
        return

    items = [item.to_dict() for item in list_components()]
    if output_format == "human":
        for item in items:
            print(
                f"{item['component']}\t{item['category']}\t{item['version']}\t"
                f"{item['lifecycle_phase']}"
            )
    else:
        emit_machine(
            "components.list",
            {"count": len(items), "dry_run": dry_run, "items": items},
            output_format,
            items=items,
        )


def _topology(args: argparse.Namespace, output_format: str, no_input: bool, dry_run: bool) -> None:
    from .targets import resolve_target
    from pllm.configuration import ConfigurationError
    from pllm.deployment import RoleDeployment
    from pllm.roles import client_only_reference_graph, two_online_reference_graph

    target = resolve_target(
        args.TARGET,
        factory=args.factory,
        no_input=no_input,
        trust_python=args.trust_python,
        output_format=output_format,
    )
    experiment = target.configuration
    try:
        resolved = experiment.resolve()
    except (TypeError, ValueError) as exc:
        raise ResolutionError("TOPOLOGY_NOT_EXECUTABLE", str(exc)) from exc
    if args.reference is not None and resolved.privacy_mode != "public":
        raise ResolutionError(
            "TOPOLOGY_NOT_EXECUTABLE", "research baselines require a public-weight composition"
        )
    graph = (
        client_only_reference_graph()
        if args.reference == "client-only"
        else two_online_reference_graph()
        if args.reference == "two-online-offset"
        else resolved.role_graph
    )
    if graph is None:
        raise ResolutionError("TOPOLOGY_NOT_EXECUTABLE", "composition has no admitted role graph")
    # Local services share one operator. Graph validation is not evidence that
    # non-collusion, attestation, or model privacy holds in a real deployment.
    operators = {role.id: "local-operator" for role in graph.roles}
    report = {
        "configuration_digest": resolved.configuration_digest,
        "composition_digest": resolved.composition_digest,
        "topology": graph.to_spec(),
        "topology_digest": graph.digest(),
        "reference": args.reference,
        "executable_topology": args.reference is None,
        "placement": {
            "kind": experiment.deployment.kind,
            "operators": operators,
            "separation_violations": [
                list(pair) for pair in graph.separation_violations(operators)
            ],
        },
        "scope": (
            "research comparator graph; not bound to an executable plan or independently operated roles"
            if args.reference is not None
            else "installed runtime composition; deployment ownership is a declaration, not a privacy proof"
        ),
        "python_executed": target.python_executed,
        "target_kind": target.kind,
        "dry_run": dry_run,
    }
    if args.role_deployment is not None:
        try:
            declaration = RoleDeployment.from_file(args.role_deployment)
            assessment = declaration.assess(graph)
        except (OSError, ConfigurationError, ValueError) as exc:
            raise ResolutionError("TOPOLOGY_PLACEMENT", str(exc)) from exc
        report["role_deployment"] = {
            "digest": declaration.digest(),
            "declaration": declaration.to_spec(),
            "assessment": assessment.to_spec(),
            "scope": "declaration only; no operator verification, quote validation, or runtime admission",
        }
    if output_format == "human":
        print(json.dumps(report, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        emit_machine("topology.inspect", report, output_format)


def _benchmark(args: argparse.Namespace, output_format: str, no_input: bool, dry_run: bool) -> None:
    selection = _network_selection(args, gateway=False)
    from pllm.runtime.dashboard import _sampling_choice, _validate_request_temperature

    try:
        _validate_request_temperature(args.temperature)
    except ValueError as exc:
        raise ResolutionError("BENCHMARK_TEMPERATURE", str(exc)) from exc
    if selection is None:
        args.model = args.model or "Qwen/Qwen2.5-0.5B-Instruct"
    if not 1 <= args.max_output_tokens <= 512:
        raise ResolutionError(
            "BENCHMARK_OUTPUT_LIMIT", "max output tokens must be between 1 and 512"
        )
    if not 0 <= args.warmups <= 100:
        raise ResolutionError("BENCHMARK_WARMUPS", "warmups must be between 0 and 100")
    if not 1 <= args.repetitions <= 100:
        raise ResolutionError("BENCHMARK_REPETITIONS", "repetitions must be between 1 and 100")
    if not 1 <= args.timeout <= 3600:
        raise ResolutionError("BENCHMARK_TIMEOUT", "timeout must be between 1 and 3600 seconds")

    experiments = []
    if getattr(args, "experiment", None):
        from pllm.configuration import Experiment

        from .targets import resolve_target

        if args.tiny:
            raise ResolutionError(
                "BENCHMARK_EXPERIMENT_TINY", "--experiment cannot be combined with --tiny"
            )
        for target_name in args.experiment:
            target = resolve_target(
                target_name,
                factory=args.factory,
                no_input=no_input,
                trust_python=args.trust_python,
                output_format=output_format,
            )
            if not isinstance(target.configuration, Experiment):
                raise ResolutionError(
                    "BENCHMARK_EXPERIMENT_TYPE", f"target is not an Experiment: {target_name}"
                )
            try:
                target.configuration.resolve()
            except (TypeError, ValueError) as exc:
                raise ResolutionError("BENCHMARK_EXPERIMENT_INVALID", str(exc)) from exc
            if args.max_output_tokens > target.configuration.budget.max_new_tokens:
                raise ResolutionError(
                    "BENCHMARK_EXPERIMENT_BUDGET",
                    f"--max-output-tokens exceeds {target.configuration.name!r} budget",
                )
            experiments.append(target.configuration)
        digests = [experiment.configuration_digest() for experiment in experiments]
        names = [experiment.name for experiment in experiments]
        if len(set(digests)) != len(digests):
            raise ResolutionError(
                "BENCHMARK_EXPERIMENT_DUPLICATE", "Experiment configurations must be unique"
            )
        if len(set(names)) != len(names):
            raise ResolutionError("BENCHMARK_EXPERIMENT_NAME", "Experiment names must be unique")
    if args.save_best is not None and len(experiments) < 2:
        raise ResolutionError(
            "BENCHMARK_SAVE_BEST", "--save-best requires at least two --experiment targets"
        )

    output = args.output.expanduser() if args.output is not None else None
    if output is not None and not args.force:
        try:
            output_exists = output.exists()
        except OSError as exc:
            raise LocalIOError("OUTPUT_WRITE", f"cannot inspect output: {output}") from exc
        if output_exists:
            raise LocalIOError(
                "OUTPUT_EXISTS", f"output already exists; use --force to replace it: {output}"
            )

    if args.prompt_file is not None:
        try:
            prompt = args.prompt_file.expanduser().read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise LocalIOError("PROMPT_READ", "prompt file is unavailable") from exc
    else:
        prompt = args.prompt.strip()
    if not prompt or len(prompt.encode()) > 16_384:
        raise ResolutionError("BENCHMARK_PROMPT", "prompt must contain 1 to 16384 bytes")
    prompt_sequence = None
    if args.prompt_sequence_file is not None:
        if args.prompt_file is not None:
            raise ResolutionError(
                "BENCHMARK_PROMPT_SEQUENCE", "choose a prompt file or a context sequence"
            )
        try:
            sequence_path = args.prompt_sequence_file.expanduser()
            if sequence_path.stat().st_size > 524_288:
                raise ResolutionError(
                    "BENCHMARK_PROMPT_SEQUENCE", "context sequence exceeds 512 KiB"
                )
            prompt_sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ResolutionError(
                "BENCHMARK_PROMPT_SEQUENCE", "context sequence must be readable JSON"
            ) from exc
        from pllm.runtime.benchmark_context import validate_context
        try:
            if not isinstance(prompt_sequence, list) or not 1 <= len(prompt_sequence) <= 32:
                raise ValueError("context count")
            prompt_sequence = [validate_context(value) for value in prompt_sequence]
        except ValueError as exc:
            raise ResolutionError(
                "BENCHMARK_PROMPT_SEQUENCE",
                "context sequence requires 1-32 bounded text or message-array contexts",
            ) from exc
        prompt = prompt_sequence[0]
    warmup_prompt = prompt
    if args.warmup_prompt_file is not None:
        if not args.warmups:
            raise ResolutionError("BENCHMARK_WARMUP_PROMPT", "warmup prompt requires --warmups")
        try:
            warmup_prompt = args.warmup_prompt_file.expanduser().read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise LocalIOError("PROMPT_READ", "warmup prompt file is unavailable") from exc
        if not warmup_prompt or len(warmup_prompt.encode()) > 16_384:
            raise ResolutionError(
                "BENCHMARK_WARMUP_PROMPT", "warmup prompt must contain 1 to 16384 bytes"
            )
    if args.prefill_cache_mode == "prefix":
        if not (1 <= args.prefill_cache_mib <= 256) or (
            type(args.prefill_cache_bound_tokens) is not int
            or not 2 <= args.prefill_cache_bound_tokens <= 4096
        ):
            raise ResolutionError(
                "BENCHMARK_PREFIX_CACHE",
                "prefix reuse requires 1-256 MiB and a fixed bound of 2-4096 tokens",
            )
    elif args.prefill_cache_bound_tokens is not None:
        raise ResolutionError(
            "BENCHMARK_PREFIX_CACHE", "prefix bound requires --prefill-cache-mode prefix"
        )

    configuration = {
        "model": (
            experiments[0].pipeline.model.source
            if experiments and len({item.pipeline.model.source for item in experiments}) == 1
            else args.model
        ),
        "model_id": args.model_id,
        "tiny": args.tiny,
        "max_output_tokens": args.max_output_tokens,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "sequence_length": len(prompt_sequence) if prompt_sequence is not None else None,
        "timeout_seconds": args.timeout,
        "show_dashboard": args.show_dashboard,
        "inventory_policy": args.inventory_policy or "prewarm",
        "bundle_compression": args.bundle_compression or "none",
        "prefill_cache_mib": args.prefill_cache_mib,
        "prefill_cache_mode": args.prefill_cache_mode,
        "prefill_cache_bound_tokens": args.prefill_cache_bound_tokens,
        "output": str(output) if output is not None else None,
        "experiments": [
            {
                "name": experiment.name,
                "configuration_digest": experiment.configuration_digest(),
                "pipeline_digest": experiment.pipeline.digest(),
            }
            for experiment in experiments
        ],
        "save_best": str(args.save_best) if args.save_best is not None else None,
        "sampling": _sampling_choice(args.temperature),
    }
    from pllm.deployment import PartyAccess, WanConditions
    try:
        if args.wan_profile and (args.wan_download_mbps is not None or args.wan_upload_mbps is not None or args.wan_party):
            raise ValueError("choose --wan-profile or WAN rate flags")
        parties = []
        for value in args.wan_party:
            fields = value.rsplit(":", 2)
            if len(fields) != 3:
                raise ValueError("--wan-party requires PARTY:DOWN:UP in Mbps")
            parties.append(PartyAccess(fields[0], float(fields[1]), float(fields[2])))
        wan = (WanConditions.from_file(args.wan_profile) if args.wan_profile else WanConditions(
            download_mbps=100 if args.wan_download_mbps is None else args.wan_download_mbps,
            upload_mbps=40 if args.wan_upload_mbps is None else args.wan_upload_mbps,
            parties=tuple(parties)))
    except (ValueError, OSError) as exc:
        raise ResolutionError("BENCHMARK_WAN_PROFILE", str(exc)) from exc
    configuration["wan_conditions"] = wan.to_spec()
    emulate_wan = not args.wan_estimate and bool(args.wan or args.wan_profile or args.wan_party
        or args.wan_download_mbps is not None or args.wan_upload_mbps is not None)
    configuration["wan_mode"] = "kernel-enforced" if emulate_wan else "analytic-only"
    if emulate_wan and selection is not None:
        raise ResolutionError("BENCHMARK_WAN_PROFILE", "WAN emulation requires local roles; live network plans support --wan-estimate")
    if selection is not None:
        network, result, request = selection
        candidates = (result.experiment,) if result is not None else request.candidates
        if any(args.max_output_tokens > item.budget.max_new_tokens for item in candidates):
            raise ResolutionError("BENCHMARK_EXPERIMENT_BUDGET", "output cap exceeds selected network budget")
        configuration["network"] = {"backend": network.backend, "digest": network.digest,
                                    "mode": "fixed-plan" if result is not None else "per-response-request",
                                    "compare_feasible": args.compare_feasible}
    if dry_run:
        data = {"configuration": configuration, "dry_run": True}
        if output_format == "human":
            print("Would run selected network responses" if selection is not None else
                  "Would run the client, preparation, and inference roles on loopback")
        else:
            emit_machine("benchmark.run", data, output_format)
        return

    from pllm.runtime.benchmark_cli import (
        LoopbackBenchmarkError,
        build_comparison_report,
        run_loopback_benchmark,
    )

    try:
        from pllm.deployment import LinkConditions
        docker_network = (LinkConditions.from_file(args.docker_network_profile)
                          if args.docker_network_profile else None)
        if docker_network is not None and not (args.docker or emulate_wan):
            raise ValueError("--docker-network-profile requires --docker")
        candidate_reports = []
        cohort_salt = secrets.token_bytes(32)
        report: dict[str, Any] | None = None
        selected_experiments = [] if selection is not None else experiments or [None]
        if selection is not None:
            if args.docker or args.docker_image or docker_network is not None:
                raise ValueError("Docker starts local provider roles; selected network plans use their admitted hosts")
            from pllm.runtime.network_benchmark import run_network_benchmark

            report = run_network_benchmark(
                network=network, result=result, request=request, prompt=prompt,
                max_output_tokens=args.max_output_tokens, warmups=args.warmups,
                repetitions=args.repetitions, timeout_seconds=args.timeout,
                temperature=args.temperature, compare_feasible=args.compare_feasible,
                capture_output_digest=args.capture_output_digest,
                wan=wan,
            )
        for index, experiment in enumerate(selected_experiments, start=1):
            prefix = f"[{index}/{len(selected_experiments)}] " if experiments else ""
            report = run_loopback_benchmark(
                model=args.model,
                model_id=args.model_id,
                tiny=args.tiny,
                prompt=prompt,
                warmup_prompt=warmup_prompt if args.warmup_prompt_file is not None else None,
                prompt_sequence=prompt_sequence,
                max_output_tokens=args.max_output_tokens,
                warmups=args.warmups,
                repetitions=args.repetitions,
                timeout_seconds=args.timeout,
                show_dashboard=args.show_dashboard,
                inventory_policy=args.inventory_policy
                if experiment is not None
                else args.inventory_policy or "prewarm",
                bundle_compression=args.bundle_compression
                if experiment is not None
                else args.bundle_compression or "none",
                prefill_cache_mib=args.prefill_cache_mib,
                prefill_cache_mode=args.prefill_cache_mode,
                prefill_cache_bound_tokens=args.prefill_cache_bound_tokens,
                _cohort_salt=cohort_salt,
                experiment=experiment,
                temperature=args.temperature,
                capture_output_digest=args.capture_output_digest,
                progress=(
                    lambda message, prefix=prefix: (
                        print(prefix + message, file=sys.stderr, flush=True)
                        if output_format == "human"
                        else None
                    )
                ),
                docker=args.docker,
                docker_image=args.docker_image,
                docker_network=docker_network,
                wan=wan,
                emulate_wan=emulate_wan,
            )
            if experiment is not None:
                report["experiment"] = {
                    "name": experiment.name,
                    "configuration_digest": experiment.configuration_digest(),
                    "pipeline_digest": experiment.pipeline.digest(),
                }
                candidate_reports.append((experiment, report))
        if len(candidate_reports) > 1:
            report = build_comparison_report(candidate_reports)
    except (LoopbackBenchmarkError, ValueError, RuntimeError) as exc:
        raise RuntimeFailure("BENCHMARK_FAILED", str(exc)) from exc
    if report is None:
        raise RuntimeFailure("BENCHMARK_FAILED", "benchmark produced no report")

    if args.save_best is not None:
        winner_digest = report["winners"].get("full_seconds")
        if winner_digest is None:
            raise RuntimeFailure(
                "BENCHMARK_NOT_COMPARABLE",
                "winning configuration was not saved because measured workloads did not match",
            )
        winner = next(
            experiment
            for experiment in experiments
            if experiment.configuration_digest() == winner_digest
        )
        best_path = args.save_best.expanduser()
        mode = "wb" if args.force else "xb"
        try:
            best_path.parent.mkdir(parents=True, exist_ok=True)
            with best_path.open(mode) as destination:
                destination.write(winner.canonical_bytes())
                destination.write(b"\n")
        except FileExistsError as exc:
            raise LocalIOError(
                "OUTPUT_EXISTS", f"output already exists; use --force to replace it: {best_path}"
            ) from exc
        except OSError as exc:
            raise LocalIOError("OUTPUT_WRITE", f"cannot write output: {best_path}") from exc

    if output is not None:
        mode = "w" if args.force else "x"
        try:
            with output.open(mode, encoding="utf-8") as destination:
                json.dump(report, destination, allow_nan=False, indent=2, sort_keys=True)
                destination.write("\n")
        except FileExistsError as exc:
            raise LocalIOError(
                "OUTPUT_EXISTS", f"output already exists; use --force to replace it: {output}"
            ) from exc
        except OSError as exc:
            raise LocalIOError("OUTPUT_WRITE", f"cannot write output: {output}") from exc

    data = {"output": str(output) if output is not None else None, "report": report}
    if output_format == "human":
        if "candidates" in report:
            label = "selected placements" if selection is not None else "Experiment pipelines"
            print(f"Compared {len(report['candidates'])} {label}")
            for ranking in report.get("rankings", {}).get("full_seconds", []):
                print(f"{ranking['rank']}. {ranking['name']}: {ranking['value']:.3f}s median full")
            if not report["checks"]["matched_workload"]:
                print("No ranking: measured workloads did not match exactly")
            if selection is not None:
                regret = report["summary"]["selected_plan_latency_regret_seconds"]
                if regret is not None:
                    print(f"Selected placement measured median latency regret: {regret:.3f}s")
        else:
            summary = report["summary"]
            ttft = summary["median_ttft_seconds"]
            throughput = summary["median_tokens_per_second"]
            print(f"{report['configuration']['model_id']}: {summary['completed_runs']} run(s)")
            if ttft is not None:
                print(f"Median TTFT: {ttft:.3f}s")
            if throughput is not None:
                print(f"Median throughput: {throughput:.2f} token/s")
            for key, label in (
                ("online_mb_per_output_token", "Online, including prefill"),
                ("setup_inclusive_mb_per_output_token", "Setup-inclusive"),
                ("decode_online_mb_per_output_token", "Decode after first output"),
            ):
                value = summary.get(key)
                if value is not None:
                    print(f"{label}: {value:.4f} MB/output-token (application bodies)")
            for key, label in (("online", "Online"), ("setup_inclusive", "Setup-inclusive")):
                window = report.get("wan_readiness", {}).get("summary", {}).get(key)
                if window is not None:
                    print(f"{label} WAN bandwidth floor: {window['minimum_transfer_seconds']:.3f}s (analytic, shared per-party access)")
            emulation = report.get("wan_readiness", {}).get("emulation")
            if emulation is not None:
                print("WAN rates enforced: shared party upload/download kernel queues"
                      if emulation["kernel_rate_snapshots_checked"] else "WAN emulation: no inter-party links")
                for key, label in (("end_to_end", "End-to-end"), ("online", "Online"), ("decode", "Decode")):
                    value = emulation["summary"][key + "_tokens_per_second"]
                    if value is not None:
                        print(f"{label} under WAN caps: {value:.3f} token/s (measured)")
        print("Privacy/runtime checks: " + ("passed" if report["checks"]["passed"] else "failed"))
        if output is not None:
            print(f"Wrote {output}")
        if args.save_best is not None:
            print(f"Saved lowest-latency Experiment to {args.save_best.expanduser()}")
    else:
        emit_machine("benchmark.run", data, output_format)


def _benchmark_quality(
    args: argparse.Namespace, output_format: str, no_input: bool, dry_run: bool
) -> None:
    from pllm.configuration import Experiment

    from .targets import resolve_target

    if not 1 <= len(args.experiment) <= 8 or not 1 <= args.top_k <= 100:
        raise ResolutionError("QUALITY_BOUNDS", "quality requires 1-8 Experiments and top-k 1-100")
    experiments = []
    for target_name in args.experiment:
        target = resolve_target(
            target_name,
            factory=args.factory,
            no_input=no_input,
            trust_python=args.trust_python,
            output_format=output_format,
        )
        if not isinstance(target.configuration, Experiment):
            raise ResolutionError("QUALITY_EXPERIMENT_TYPE", "quality target is not an Experiment")
        try:
            target.configuration.resolve()
        except (TypeError, ValueError) as exc:
            raise ResolutionError("QUALITY_EXPERIMENT_INVALID", str(exc)) from exc
        experiments.append(target.configuration)
    try:
        prompt_file = args.prompts_file.expanduser()
        if prompt_file.stat().st_size > 131_072:
            raise ResolutionError("QUALITY_PROMPTS", "prompt cohort exceeds 131072 bytes")
        prompts = json.loads(prompt_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalIOError(
            "QUALITY_PROMPTS", "prompt cohort file is unavailable or invalid"
        ) from exc
    if (
        not isinstance(prompts, list)
        or not 1 <= len(prompts) <= 32
        or any(
            type(prompt) is not str or not prompt.strip() or len(prompt.encode("utf-8")) > 4096
            for prompt in prompts
        )
    ):
        raise ResolutionError("QUALITY_PROMPTS", "prompt cohort needs 1-32 bounded strings")
    output = args.output.expanduser() if args.output is not None else None
    if output is not None and output.exists() and not args.force:
        raise LocalIOError("OUTPUT_EXISTS", "output exists; use --force to replace it")
    if dry_run:
        data = {
            "dry_run": True,
            "candidate_digests": [item.configuration_digest() for item in experiments],
            "prompt_count": len(prompts),
            "top_k": args.top_k,
        }
        if output_format == "human":
            print("Would compare pinned local compiled decoder logits with the FP32 reference")
        else:
            emit_machine("benchmark.quality", data, output_format)
        return

    from pllm.runtime.reference_benchmark import (
        ReferenceBenchmarkError,
        run_reference_benchmark,
    )

    try:
        report = run_reference_benchmark(experiments, prompts, top_k=args.top_k)
    except (ReferenceBenchmarkError, TypeError, ValueError) as exc:
        raise RuntimeFailure("QUALITY_FAILED", str(exc)) from exc
    if output is not None:
        mode = "w" if args.force else "x"
        try:
            with output.open(mode, encoding="utf-8") as destination:
                json.dump(report, destination, allow_nan=False, indent=2, sort_keys=True)
                destination.write("\n")
        except FileExistsError as exc:
            raise LocalIOError("OUTPUT_EXISTS", "output exists; use --force to replace it") from exc
        except OSError as exc:
            raise LocalIOError("OUTPUT_WRITE", "cannot write quality report") from exc
    if output_format == "human":
        for candidate in report["candidates"]:
            print(
                f"{candidate['name']}: top-1 {candidate['top1_agreement']:.3f}, "
                f"top-{args.top_k} recall {candidate['top_k_recall']:.3f}, "
                f"max logit error {candidate['max_abs_logit_error']:.3f}"
            )
        if output is not None:
            print(f"Wrote {output}")
    else:
        emit_machine(
            "benchmark.quality",
            {"report": report, "output": str(output) if output else None},
            output_format,
        )


def _dev(args: argparse.Namespace, output_format: str, dry_run: bool) -> None:
    if output_format != "human":
        raise ResolutionError(
            "DEV_FORMAT_UNAVAILABLE", "dev dashboard supports only --format human"
        )
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ResolutionError("DASHBOARD_HOST", "dev dashboard is restricted to loopback")
    if not 1 <= args.port <= 65_535:
        raise ResolutionError("DASHBOARD_PORT", "dashboard port must be between 1 and 65535")
    if not 1 <= args.max_output_tokens <= 512:
        raise ResolutionError(
            "DASHBOARD_OUTPUT_LIMIT", "max output tokens must be between 1 and 512"
        )
    startup_inventory_rows = getattr(args, "startup_inventory_rows", None)
    if startup_inventory_rows is not None and startup_inventory_rows < 1:
        raise ResolutionError("DASHBOARD_INVENTORY_ROWS", "startup inventory rows must be positive")
    if dry_run:
        print(f"Would start local dashboard on http://{args.host}:{args.port}")
        return
    try:
        from pllm.runtime.dashboard import run_dashboard

        run_dashboard(args)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise RuntimeFailure(
            "DASHBOARD_FAILED", f"dashboard failed ({type(exc).__name__})"
        ) from exc


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _service_url(host: str, port: int) -> str:
    rendered_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{rendered_host}:{port}"


def _gateway(args: argparse.Namespace, output_format: str, no_input: bool, dry_run: bool) -> None:
    if not _is_loopback(args.host):
        raise ResolutionError("GATEWAY_HOST", "gateway bind must use a loopback address")
    if not 1 <= args.port <= 65_535:
        raise ResolutionError("GATEWAY_PORT", "gateway port must be between 1 and 65535")
    selection = _network_selection(args, gateway=True)
    if selection is not None:
        network, result, request = selection
        data = {"dry_run": dry_run, "url": _service_url(args.host, args.port),
                "network_digest": network.digest, "backend": network.backend,
                "mode": "fixed-plan" if result is not None else "per-response-request"}
        if dry_run:
            if output_format == "human":
                print(f"Would start selected network gateway on {data['url']}")
            else:
                emit_machine("gateway", data, output_format)
            return
        from pllm.runtime.network_benchmark import SelectedClientFactory
        from pllm.runtime.sidecar import create_sidecar_app
        import uvicorn

        factory = SelectedClientFactory(network, result=result, request=request)
        uvicorn.run(create_sidecar_app(client_factory=factory, local_api_key=args.api_key),
                    host=args.host, port=args.port, access_log=False)
        return
    experiment = None
    experiment_data = None
    resolved_experiment = None
    if args.experiment:
        if not args.local:
            raise ResolutionError("GATEWAY_EXPERIMENT_LOCAL", "--experiment requires --local")
        legacy_flags = {
            "--config": args.config,
            "--model": args.model,
            "--model-id": args.model_id,
            "--tiny": args.tiny,
            "--revision": args.revision,
            "--local-files-only": args.local_files_only,
            "--weight-bits": args.weight_bits,
            "--activation-bits": args.activation_bits,
        }
        conflicts = [name for name, value in legacy_flags.items() if value not in (None, False)]
        if conflicts:
            raise ResolutionError(
                "GATEWAY_EXPERIMENT_CONFLICT",
                "--experiment owns model and quantization settings; remove " + ", ".join(conflicts),
            )
        from pllm.configuration import Experiment

        from .targets import resolve_target

        target = resolve_target(
            args.experiment,
            factory=args.factory,
            no_input=no_input,
            trust_python=args.trust_python,
            output_format=output_format,
        )
        if not isinstance(target.configuration, Experiment):
            raise ResolutionError(
                "GATEWAY_EXPERIMENT_TYPE", f"target is not an Experiment: {args.experiment}"
            )
        experiment = target.configuration
        try:
            resolved_experiment = experiment.resolve()
        except (TypeError, ValueError) as exc:
            raise ResolutionError("GATEWAY_EXPERIMENT_INVALID", str(exc)) from exc
        from pllm.profiles import resolve_runtime_composition

        if resolve_runtime_composition(experiment.pipeline) is None:
            raise ResolutionError(
                "GATEWAY_EXPERIMENT_INVALID",
                "experiment profile is not supported by local serving",
            )
        args.resolved_experiment = experiment
        experiment_data = {
            "name": experiment.name,
            "profile": resolved_experiment.profile,
            "configuration_digest": experiment.configuration_digest(),
            "pipeline_digest": experiment.pipeline.digest(),
        }
    if args.local and not (args.model or args.tiny or experiment is not None):
        raise ResolutionError(
            "GATEWAY_LOCAL_MODEL", "--local requires --model MODEL or --experiment TARGET"
        )
    if args.tiny and not args.local:
        raise ResolutionError("GATEWAY_TINY", "--tiny requires --local")

    url = _service_url(args.host, args.port)
    if resolved_experiment is None:
        separated_roles = True
    else:
        graph = resolved_experiment.role_graph
        separated_roles = graph is not None and bool(graph.separate_operators)
    if (
        output_format == "human"
        and args.local
        and separated_roles
        and not getattr(args, "quiet", False)
    ):
        print(
            "Trust limitation: --local co-locates preparation and inference; "
            "it does not provide role separation or non-collusion.",
            file=sys.stderr,
            flush=True,
        )
    if dry_run:
        model = (
            experiment.pipeline.model.model_id or experiment.pipeline.model.source
            if experiment is not None
            else args.model_id or args.model
        )
        transport = args.transport
        bundle_cache_mode = args.bundle_cache_mode
        if not args.local:
            from pllm.settings import ClientSettings

            try:
                settings = ClientSettings.load(Path(args.config) if args.config else None)
            except (OSError, TypeError, ValueError) as exc:
                raise LocalIOError(
                    "GATEWAY_CONFIG_READ", f"Cannot read client config: {exc}"
                ) from exc
            model = model or settings.model
            transport = transport or settings.transport
            bundle_cache_mode = bundle_cache_mode or settings.bundle_cache_mode
        data = {
            "bundle_cache_mode": bundle_cache_mode,
            "config": args.config,
            "dry_run": True,
            "experiment": experiment_data,
            "local": args.local,
            "model": model,
            "transport": transport,
            "url": url,
        }
        if output_format == "human":
            if not getattr(args, "quiet", False):
                action = "local inference, preparation, and gateway" if args.local else "gateway"
                print(f"Would start {action} on {url}")
        else:
            emit_machine("gateway", data, output_format)
        return

    if output_format == "human" and not getattr(args, "quiet", False):
        print(f"Gateway URL: {url}", flush=True)
    try:
        from pllm.runtime.cli import RuntimeCLIError, run_local_gateway, run_sidecar
    except Exception as exc:
        raise RuntimeFailure("GATEWAY_FAILED", f"gateway failed ({type(exc).__name__})") from exc

    try:
        if args.local:
            run_local_gateway(args)
        else:
            run_sidecar(args)
    except KeyboardInterrupt:
        raise
    except RuntimeCLIError as exc:
        raise ResolutionError("GATEWAY_CONFIGURATION", str(exc)) from exc
    except Exception as exc:
        raise RuntimeFailure("GATEWAY_FAILED", f"gateway failed ({type(exc).__name__})") from exc


def _network_selection(args: argparse.Namespace, *, gateway: bool):
    """Offline selection checks shared by gateway/benchmark; dry-run never discovers."""
    mode = args.plan is not None or args.request is not None
    if not mode:
        if args.network is not None or getattr(args, "compare_feasible", False):
            raise ResolutionError("NETWORK_SELECTION", "--network/--compare-feasible requires --plan or --request")
        return None
    if args.network is None:
        raise ResolutionError("NETWORK_REQUIRED", "--plan/--request requires --network")
    names = ["experiment", "factory", "trust_python", "model", "model_id", "tiny"]
    if not gateway and not getattr(args, "benchmark_model_override", False):
        names.remove("model")
    names += (["local", "config", "inference_url", "preparation_url", "weight_bits", "activation_bits",
               "transport", "correlation_mode", "correlation_prefetch", "prepared_inventory_rows",
               "token_cache_size", "bundle_cache_mode", "bundle_cache_dir", "local_files_only", "timeout",
               "revision"]
              if gateway else ["save_best", "show_dashboard", "inventory_policy", "bundle_compression",
                               "prefill_cache_mib", "prefill_cache_bound_tokens", "prompt_sequence_file",
                               "warmup_prompt_file"])
    conflicts = ["--" + name.replace("_", "-") for name in names if getattr(args, name, None)]
    if not gateway and args.prefill_cache_mode != "exact":
        conflicts.append("--prefill-cache-mode")
    if conflicts:
        raise ResolutionError("NETWORK_OVERRIDE_CONFLICT", "selected network owns execution settings; remove " + ", ".join(conflicts))
    from pllm.deployment import NetworkSpec
    from pllm.plan import PlanningResult
    from pllm.search import PlanningRequest
    from .network import load_record

    network = load_record(NetworkSpec, args.network)
    result = load_record(PlanningResult, args.plan) if args.plan is not None else None
    request = load_record(PlanningRequest, args.request) if args.request is not None else None
    if result is not None:
        if result.status != "feasible":
            raise ResolutionError("NO_FEASIBLE_PLACEMENT", "selected plan is not feasible")
        import time

        try:
            result.validate(snapshot=result.snapshot if network.backend == "http" else network.snapshot,
                            evaluated_at_ms=time.time_ns() // 1_000_000)
        except (TypeError, ValueError) as exc:
            raise ResolutionError("PLAN_VALIDATION", str(exc)) from exc
    candidates = (result.experiment,) if result is not None else request.candidates
    for item in candidates:
        intent = item.deployment
        if network.backend == "http" and (intent.kind != "network" or intent.network_id != network.network_id
                                         or intent.network_spec_digest != network.digest):
            raise ResolutionError("NETWORK_INTENT", "selected candidate network intent differs")
        if network.backend == "local" and intent.kind != "local":
            raise ResolutionError("NETWORK_INTENT", "local backend requires local deployment")
    return network, result, request


def _serve(args: argparse.Namespace, output_format: str, no_input: bool, dry_run: bool) -> None:
    args.host = args.host or os.getenv("PLLM_HOST", "127.0.0.1")
    args.port = args.port if args.port is not None else _env_int("PLLM_PORT", 8000)
    role = args.serve_role
    experiment = None
    if args.experiment:
        from pllm.configuration import Experiment

        from .targets import resolve_target

        if (
            args.config
            or args.model
            or args.model_id
            or args.engine_threads is not None
            or args.metal_min_rows is not None
            or args.privacy_mode is not None
            or args.proprietary_protocol is not None
            or args.guard_max_rows_per_request is not None
            or args.guard_max_rows_per_stage is not None
            or args.guard_max_requests_per_minute is not None
            or args.guard_output_dither is not None
        ):
            raise ResolutionError(
                "SERVE_EXPERIMENT_CONFLICT",
                "--experiment cannot be combined with config, model, engine, privacy, protocol, or guard overrides",
            )
        target = resolve_target(
            args.experiment,
            factory=args.factory,
            no_input=no_input,
            trust_python=args.trust_python,
            output_format=output_format,
        )
        if not isinstance(target.configuration, Experiment):
            raise ResolutionError(
                "SERVE_EXPERIMENT_TYPE", f"target is not an Experiment: {args.experiment}"
            )
        experiment = target.configuration
        try:
            experiment.resolve()
        except (TypeError, ValueError) as exc:
            raise ResolutionError("SERVE_EXPERIMENT_INVALID", str(exc)) from exc
        from pllm.profiles import resolve_runtime_composition

        runtime_options = resolve_runtime_composition(experiment.pipeline)
        if runtime_options is None:
            raise ResolutionError(
                "SERVE_EXPERIMENT_INVALID",
                "experiment profile is not supported by local serving",
            )
        for name in ("weight_bits", "activation_bits"):
            selected = getattr(runtime_options, name)
            explicit = getattr(args, name)
            if explicit is not None and explicit != selected:
                raise ResolutionError(
                    "SERVE_EXPERIMENT_CONFLICT",
                    f"--{name.replace('_', '-')} conflicts with the experiment quantization",
                )
            setattr(args, name, selected)
        if (
            args.public_equalization_digest is not None
            and args.public_equalization_digest != runtime_options.public_equalization_digest
        ):
            raise ResolutionError(
                "SERVE_EXPERIMENT_CONFLICT",
                "--public-equalization-digest conflicts with experiment quantization",
            )
        args.public_equalization_digest = runtime_options.public_equalization_digest
        if args.remote_output_head and not runtime_options.remote_output_head:
            raise ResolutionError(
                "SERVE_EXPERIMENT_CONFLICT",
                "remote output head conflicts with Experiment placement",
            )
        args.remote_output_head = runtime_options.remote_output_head
        if (
            args.client_prefix_layers
            and args.client_prefix_layers != runtime_options.client_prefix_layers
        ):
            raise ResolutionError(
                "SERVE_EXPERIMENT_CONFLICT",
                "client prefix layers conflict with Experiment placement",
            )
        args.client_prefix_layers = runtime_options.client_prefix_layers
        if (
            args.client_linear_roles
            and tuple(args.client_linear_roles.split(",")) != runtime_options.client_linear_roles
        ):
            raise ValueError("client linear roles conflict with Experiment")
        args.client_linear_roles = ",".join(runtime_options.client_linear_roles)
        if getattr(args, "prepared_output_encoding", None) not in (None, runtime_options.prepared_output_encoding):
            raise ResolutionError("SERVE_EXPERIMENT_CONFLICT", "prepared output encoding differs from its Experiment")
        args.prepared_output_encoding = runtime_options.prepared_output_encoding
        if role == "preparation" and not runtime_options.requires_preparation:
            raise ResolutionError(
                "SERVE_CONFIGURATION",
                "experiment profile has no preparation role",
            )
        args.privacy_mode = runtime_options.privacy_mode
        args.proprietary_protocol = runtime_options.proprietary_protocol
        args.guard_max_rows_per_request = runtime_options.guard_max_rows_per_request
        args.guard_max_rows_per_stage = runtime_options.guard_max_rows_per_owner_stage
        args.guard_max_requests_per_minute = runtime_options.guard_max_requests_per_minute
        args.guard_output_dither = runtime_options.output_dither_bound
        args.verification_component = runtime_options.verification_component
        args.verification_target_failure_bits = runtime_options.verification_target_failure_bits
        args.model = [experiment.pipeline.model.source]
        args.model_id = [experiment.pipeline.model.model_id or experiment.pipeline.model.source]
        args.model_kind = experiment.pipeline.model.kind
        args.revision = experiment.pipeline.model.revision
        args.local_files_only = experiment.pipeline.model.local_files_only
        kernels = experiment.pipeline.components.get("kernels")
        if kernels is not None and kernels.component == "pllm/cpu":
            args.engine_threads = int(kernels.params["threads"])
        if kernels is not None and kernels.component == "pllm/apple-metal-int8/v1":
            args.metal_min_rows = int(kernels.params["min_rows"])
    preview: dict[str, Any] = {}
    if args.config:
        try:
            value = json.loads(Path(args.config).expanduser().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalIOError(
                "SERVE_CONFIG_READ", f"Cannot read role configuration: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise UsageError("Role configuration must be a JSON object")
        try:
            from pllm.runtime.config import GatewayConfig

            preview = GatewayConfig.from_dict(value).to_dict()
        except ValueError as exc:
            raise ResolutionError("SERVE_CONFIGURATION", str(exc)) from exc
    if not 1 <= args.port <= 65_535:
        raise ResolutionError("SERVE_PORT", "service port must be between 1 and 65535")
    if args.model_id and len(args.model_id) != len(args.model):
        raise ResolutionError("SERVE_MODEL_ID", "--model-id must be supplied once per --model")
    effective_mode = (
        args.privacy_mode or os.getenv("PLLM_PRIVACY_MODE") or preview.get("privacy_mode", "public")
    )
    if role == "preparation" and effective_mode != "public":
        raise ResolutionError(
            "SERVE_CONFIGURATION", "the preparation service supports public-weight models only"
        )
    if effective_mode == "secure" or str(effective_mode).startswith("shared"):
        raise ResolutionError(
            "SERVE_CONFIGURATION", "production shared-transformer routing is disabled"
        )
    effective_protocol = (
        args.proprietary_protocol
        or os.getenv("PLLM_PROPRIETARY_PROTOCOL")
        or preview.get("proprietary_protocol", "guarded")
    )
    if dry_run:
        data = {
            "config": args.config,
            "dry_run": True,
            "experiment": None
            if experiment is None
            else {
                "configuration_digest": experiment.configuration_digest(),
                "name": experiment.name,
            },
            "host": args.host,
            "models": args.model_id
            or args.model
            or [
                str(item.get("model_id") or item.get("path") or item.get("name"))
                for item in preview.get("engine_models", [])
                if isinstance(item, dict)
                and (item.get("model_id") or item.get("path") or item.get("name"))
            ],
            "port": args.port,
            "weight_bits": args.weight_bits,
            "activation_bits": args.activation_bits,
            "privacy_mode": effective_mode,
            "protocol": effective_protocol,
            "role": role,
        }
        if effective_mode == "proprietary" and effective_protocol == "guarded":
            data["guard_policy"] = {
                "max_rows_per_request": args.guard_max_rows_per_request,
                "max_rows_per_owner_stage": args.guard_max_rows_per_stage,
                "max_requests_per_minute": args.guard_max_requests_per_minute,
                "output_dither_bound": args.guard_output_dither,
            }
        if output_format == "human":
            if not getattr(args, "quiet", False):
                print(f"Would start {role} service on {_service_url(args.host, args.port)}")
        else:
            emit_machine(f"serve.{role}", data, output_format)
        return

    try:
        from pllm.runtime.cli import RuntimeCLIError, run_server
    except Exception as exc:
        raise RuntimeFailure(
            "SERVE_FAILED", f"{role} service failed ({type(exc).__name__})"
        ) from exc

    try:
        run_server(args, preparation=role == "preparation")
    except KeyboardInterrupt:
        raise
    except RuntimeCLIError as exc:
        raise ResolutionError("SERVE_CONFIGURATION", str(exc)) from exc
    except Exception as exc:
        raise RuntimeFailure(
            "SERVE_FAILED", f"{role} service failed ({type(exc).__name__})"
        ) from exc


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise UsageError(f"{name} must be an integer") from exc


def _detect_format(arguments: Sequence[str]) -> str:
    selected = "human"
    for index, argument in enumerate(arguments):
        if argument.startswith("--format="):
            value = argument.partition("=")[2]
            selected = value if value in {"human", "json", "jsonl"} else "human"
        if argument == "--format" and index + 1 < len(arguments):
            value = arguments[index + 1]
            selected = value if value in {"human", "json", "jsonl"} else "human"
    return selected


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    output_format = _detect_format(arguments)
    parser = build_parser()
    try:
        args = parser.parse_args(arguments)
        output_format, no_input, dry_run = _globals(args)
        if args.command is None:
            parser.print_help()
            return
        if args.command == "network":
            from .network import run

            run(args, output_format, dry_run)
        elif args.command == "plan":
            from .plan import run

            run(args, output_format, dry_run)
        elif args.command == "config":
            _config(args, output_format, no_input, dry_run)
        elif args.command == "components":
            _components(args, output_format, dry_run)
        elif args.command == "topology":
            _topology(args, output_format, no_input, dry_run)
        elif args.command == "gateway":
            _gateway(args, output_format, no_input, dry_run)
        elif args.command == "serve":
            if args.serve_role in {"party", "directory"}:
                from .network import serve

                serve(args, output_format, dry_run)
            else:
                _serve(args, output_format, no_input, dry_run)
        elif args.command == "benchmark":
            if args.benchmark_command == "quality":
                _benchmark_quality(args, output_format, no_input, dry_run)
            else:
                _benchmark(args, output_format, no_input, dry_run)
        elif args.command == "dev":
            _dev(args, output_format, dry_run)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except CLIError as exc:
        emit_error(exc, output_format)
        raise SystemExit(exc.exit_code) from None
