"""Parser and command dispatch for PLLM's 0.1 CLI."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
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
        "TARGET", help="experiment .json/.yaml or explicit path.py:object/module:object"
    )
    parser.add_argument(
        "--factory", action="store_true", help="call explicit zero-argument Python factory"
    )
    parser.add_argument(
        "--trust-python",
        action="store_true",
        help="approve execution of the explicit local Python target",
    )


def _add_server_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="JSON GatewayConfig file")
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
    parser.add_argument("--native-library")
    parser.add_argument("--compiled-cache-dir")
    parser.add_argument("--streaming-threshold-elements", type=int)
    parser.add_argument("--quantization-chunk-rows", type=int)
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
        description="Inspect PLLM configuration, run benchmarks, and inspect research metadata",
    )
    _add_globals(parser)
    parser.add_argument("--version", action="version", version=f"pllm {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

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

    research = _command(commands, "research", help="inspect or assess read-only research records")
    research_kinds = research.add_subparsers(dest="research_kind", metavar="KIND", required=True)
    for kind in ("sources", "methods", "recipes"):
        family = _command(research_kinds, kind, help=f"inspect research {kind}")
        actions = family.add_subparsers(dest="research_action", metavar="COMMAND", required=True)
        _command(actions, "list", help=f"list research {kind}")
        item = _command(actions, "show", help=f"show one research {kind[:-1]} record")
        item.add_argument("ID", help="stable record ID or registry alias")
    assess = _command(
        research_kinds,
        "assess",
        help="assess a publication request without executing research workflows",
    )
    assess.add_argument("REQUEST", help="publication assessment request JSON path")
    agents = _command(
        research_kinds,
        "agents",
        help="write repository-root guidance for research coding agents",
    )
    agents.add_argument("--output", required=True, metavar="PATH", help="new Markdown output path")
    agents.add_argument("--force", action="store_true", help="replace an existing output file")

    gateway = _command(commands, "gateway", help="run the trusted local Responses API gateway")
    gateway.add_argument("--config", help="client TOML file")
    gateway.add_argument("--host", default="127.0.0.1")
    gateway.add_argument("--port", type=int, default=8080)
    gateway.add_argument("--api-key", default=os.getenv("PLLM_GATEWAY_API_KEY", "local"))
    gateway.add_argument("--inference-url")
    gateway.add_argument("--inference-key", default=os.getenv("PLLM_INFERENCE_API_KEY"))
    gateway.add_argument("--preparation-url")
    gateway.add_argument("--preparation-key", default=os.getenv("PLLM_PREPARATION_API_KEY"))
    gateway.add_argument("--model")
    gateway.add_argument("--model-id")
    gateway.add_argument("--local", action="store_true", help="co-locate both server roles locally")
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
    gateway.add_argument("--weight-bits", type=int, choices=(4, 8), default=8)
    gateway.add_argument("--activation-bits", type=int, choices=(4, 8), default=8)

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

    benchmark = _command(commands, "benchmark", help="run reproducible local benchmarks")
    benchmark_commands = benchmark.add_subparsers(
        dest="benchmark_command", metavar="COMMAND", required=True
    )
    benchmark_run = _command(
        benchmark_commands,
        "run",
        help="run the real client, preparation, and inference roles on loopback",
    )
    benchmark_run.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
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
        "--max-output-tokens",
        type=int,
        default=24,
        help="maximum generated tokens per run (default: 24)",
    )
    benchmark_run.add_argument(
        "--warmups", type=int, default=0, help="warmup runs retained in the report (default: 0)"
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
    benchmark_run.add_argument("--output", type=Path, help="write the sanitized JSON report")
    benchmark_run.add_argument("--force", action="store_true", help="replace --output if it exists")

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


def _research(args: argparse.Namespace, output_format: str, dry_run: bool) -> None:
    from pllm.research import (
        ResearchMetadataError,
        ResearchMetadataIOError,
        assess_publication,
        get_method,
        get_recipe,
        get_source,
        list_methods,
        list_recipes,
        list_sources,
        render_agents_guide,
    )

    if args.research_kind == "agents":
        output = Path(args.output).expanduser()
        content = render_agents_guide().encode("utf-8")
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
            "digest": f"sha256:{hashlib.sha256(content).hexdigest()}",
            "dry_run": dry_run,
            "output": str(output),
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
            emit_machine("research.agents", data, output_format)
        return

    if args.research_kind == "assess":
        try:
            payload = Path(args.REQUEST).read_bytes()
        except OSError:
            raise LocalIOError(
                "RESEARCH_ASSESSMENT_IO", "assessment request is unavailable"
            ) from None
        try:
            assessment = assess_publication(payload)
        except ResearchMetadataError as exc:
            raise ResolutionError("RESEARCH_ASSESSMENT_INVALID", str(exc)) from exc
        data = assessment.to_dict()
        if output_format == "human":
            print(json.dumps(data, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            emit_machine(
                "research.assess",
                {"assessment": data, "digest": assessment.digest, "dry_run": dry_run},
                output_format,
            )
        return

    listers = {"sources": list_sources, "methods": list_methods, "recipes": list_recipes}
    getters = {"sources": get_source, "methods": get_method, "recipes": get_recipe}
    try:
        records = listers[args.research_kind]()
    except ResearchMetadataIOError as exc:
        raise LocalIOError("RESEARCH_METADATA_IO", str(exc)) from exc
    except ResearchMetadataError as exc:
        raise ResolutionError("RESEARCH_METADATA_INVALID", str(exc)) from exc

    command = f"research.{args.research_kind}.{args.research_action}"
    if args.research_action == "show":
        try:
            record = getters[args.research_kind](args.ID)
        except KeyError:
            raise ResolutionError(
                "RESEARCH_RECORD_NOT_FOUND",
                f"unknown research {args.research_kind[:-1]}: {args.ID}",
            )
        data = record.to_dict()
        if output_format == "human":
            print(json.dumps(data, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            emit_machine(command, {"dry_run": dry_run, "record": data}, output_format)
        return

    items = [record.to_dict() for record in records]
    if output_format == "human":
        for item in items:
            identifier = item.get("id", item.get("registry_alias"))
            detail = item.get(
                "title", item.get("implementation_status", item.get("workflow_status", ""))
            )
            print(f"{identifier}\t{detail}")
    else:
        emit_machine(
            command,
            {"count": len(items), "dry_run": dry_run, "items": items},
            output_format,
            items=items,
        )


def _benchmark(args: argparse.Namespace, output_format: str, dry_run: bool) -> None:
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

    configuration = {
        "model": args.model,
        "model_id": args.model_id,
        "tiny": args.tiny,
        "max_output_tokens": args.max_output_tokens,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "timeout_seconds": args.timeout,
        "show_dashboard": args.show_dashboard,
        "output": str(output) if output is not None else None,
    }
    if dry_run:
        data = {"configuration": configuration, "dry_run": True}
        if output_format == "human":
            print("Would run the client, preparation, and inference roles on loopback")
        else:
            emit_machine("benchmark.run", data, output_format)
        return

    from pllm.runtime.benchmark_cli import LoopbackBenchmarkError, run_loopback_benchmark

    try:
        report = run_loopback_benchmark(
            model=args.model,
            model_id=args.model_id,
            tiny=args.tiny,
            prompt=prompt,
            max_output_tokens=args.max_output_tokens,
            warmups=args.warmups,
            repetitions=args.repetitions,
            timeout_seconds=args.timeout,
            show_dashboard=args.show_dashboard,
            progress=(
                lambda message: (
                    print(message, file=sys.stderr, flush=True)
                    if output_format == "human"
                    else None
                )
            ),
        )
    except LoopbackBenchmarkError as exc:
        raise RuntimeFailure("BENCHMARK_FAILED", str(exc)) from exc

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
        summary = report["summary"]
        ttft = summary["median_ttft_seconds"]
        throughput = summary["median_tokens_per_second"]
        print(f"{report['configuration']['model_id']}: {summary['completed_runs']} run(s)")
        if ttft is not None:
            print(f"Median TTFT: {ttft:.3f}s")
        if throughput is not None:
            print(f"Median throughput: {throughput:.2f} token/s")
        print("Privacy/runtime checks: " + ("passed" if report["checks"]["passed"] else "failed"))
        if output is not None:
            print(f"Wrote {output}")
    else:
        emit_machine("benchmark.run", data, output_format)


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


def _gateway(args: argparse.Namespace, output_format: str, dry_run: bool) -> None:
    if not _is_loopback(args.host):
        raise ResolutionError("GATEWAY_HOST", "gateway bind must use a loopback address")
    if not 1 <= args.port <= 65_535:
        raise ResolutionError("GATEWAY_PORT", "gateway port must be between 1 and 65535")
    if args.local and not (args.model or args.tiny):
        raise ResolutionError("GATEWAY_LOCAL_MODEL", "--local requires --model MODEL")
    if args.tiny and not args.local:
        raise ResolutionError("GATEWAY_TINY", "--tiny requires --local")

    url = _service_url(args.host, args.port)
    if output_format == "human" and args.local and not getattr(args, "quiet", False):
        print(
            "Trust limitation: --local co-locates preparation and inference; "
            "it does not provide role separation or non-collusion.",
            file=sys.stderr,
            flush=True,
        )
    if dry_run:
        model = args.model_id or args.model
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


def _serve(args: argparse.Namespace, output_format: str, dry_run: bool) -> None:
    args.host = args.host or os.getenv("PLLM_HOST", "127.0.0.1")
    args.port = args.port if args.port is not None else _env_int("PLLM_PORT", 8000)
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
    role = args.serve_role
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
    if dry_run:
        data = {
            "config": args.config,
            "dry_run": True,
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
            "privacy_mode": effective_mode,
            "protocol": args.proprietary_protocol
            or os.getenv("PLLM_PROPRIETARY_PROTOCOL")
            or preview.get("proprietary_protocol", "guarded"),
            "role": role,
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
        if args.command == "config":
            _config(args, output_format, no_input, dry_run)
        elif args.command == "components":
            _components(args, output_format, dry_run)
        elif args.command == "research":
            _research(args, output_format, dry_run)
        elif args.command == "gateway":
            _gateway(args, output_format, dry_run)
        elif args.command == "serve":
            _serve(args, output_format, dry_run)
        elif args.command == "benchmark":
            _benchmark(args, output_format, dry_run)
        elif args.command == "dev":
            _dev(args, output_format, dry_run)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except CLIError as exc:
        emit_error(exc, output_format)
        raise SystemExit(exc.exit_code) from None
