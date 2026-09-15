"""Parser and command dispatch for PLLM's 0.1 CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    parser.add_argument("TARGET", help="experiment .json/.yaml or explicit path.py:object/module:object")
    parser.add_argument("--factory", action="store_true", help="call explicit zero-argument Python factory")
    parser.add_argument(
        "--trust-python",
        action="store_true",
        help="approve execution of the explicit local Python target",
    )


def build_parser() -> _Parser:
    parser = _Parser(
        prog="pllm",
        description="Inspect PLLM configurations, components, and research metadata",
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
    export.add_argument("--output", required=True, metavar="PATH", help="new output .json/.yaml path")
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
            raise LocalIOError("RESEARCH_ASSESSMENT_IO", "assessment request is unavailable") from None
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
    if dry_run:
        print(f"Would start local dashboard on http://{args.host}:{args.port}")
        return
    try:
        from pllm.runtime.dashboard import run_dashboard

        run_dashboard(args)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise RuntimeFailure("DASHBOARD_FAILED", f"dashboard failed ({type(exc).__name__})") from exc


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
        elif args.command == "dev":
            _dev(args, output_format, dry_run)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except CLIError as exc:
        emit_error(exc, output_format)
        raise SystemExit(exc.exit_code) from None
