"""Thin static-network CLI adapters."""

from __future__ import annotations

import json
from pathlib import Path

from pllm.deployment import NetworkError, NetworkSpec, PartySpec, discover
from pllm.deployment.network import control_request

from .errors import LocalIOError, ResolutionError
from .output import emit_machine


def load_record(cls, path):
    try:
        return cls.from_file(path)
    except OSError as exc:
        raise LocalIOError("INPUT_READ", f"cannot read input: {path}") from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise ResolutionError("NETWORK_CONFIGURATION", str(exc)) from exc


def write_record(record, output, *, force, dry_run):
    path = Path(output).expanduser()
    try:
        if path.exists() and not force:
            raise LocalIOError("OUTPUT_EXISTS", f"output already exists; use --force: {path}")
        if not dry_run:
            with path.open("wb" if force else "xb") as stream:
                stream.write(record.canonical_bytes() + b"\n")
    except FileExistsError as exc:
        raise LocalIOError("OUTPUT_EXISTS", f"output already exists: {path}") from exc
    except OSError as exc:
        raise LocalIOError("OUTPUT_WRITE", f"cannot write output: {path}") from exc
    return {"output": str(path), "written": not dry_run}


def emit(command, data, args, output_format):
    if output_format == "human":
        if not getattr(args, "quiet", False):
            print(json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2))
    else:
        emit_machine(command, data, output_format)


def run(args, output_format, dry_run):
    network = load_record(NetworkSpec, args.NETWORK)
    command = args.network_command
    try:
        return _run(network, command, args, output_format, dry_run)
    except (NetworkError, TypeError) as exc:
        raise ResolutionError("NETWORK_EXECUTION", str(exc)) from exc


def _run(network, command, args, output_format, dry_run):
    snapshot = network.snapshot
    if command != "inspect" and not dry_run:
        if command in {"parties", "snapshot"}:
            snapshot = discover(network)
    if command == "inspect":
        data = {"network": network.to_spec(), "network_digest": network.digest}
    elif command == "parties":
        import time

        now = time.time_ns() // 1_000_000
        data = {
            "network_id": network.network_id,
            "snapshot_digest": snapshot.digest,
            "parties": [
                {"offer": item.to_spec(), "expired": item.expires_at_ms <= now}
                for item in snapshot.offers
            ],
        }
    elif command in {"drain", "leave"}:
        if network.backend != "http":
            raise NetworkError("administration requires live http backend")
        trust = next((item for item in network.parties if item.party_id == args.party), None)
        if trust is None:
            raise NetworkError("party not in approved trust roots")
        data = {"party_id": trust.party_id, "status": "validated" if dry_run else
                control_request(trust.origin, trust.credential_env, f"/v1/party/{command}", body={})["status"]}
    else:
        data = {
            "snapshot_digest": snapshot.digest,
            **write_record(snapshot, args.output, force=args.force, dry_run=dry_run),
        }
    emit(f"network.{command}", {**data, "dry_run": dry_run}, args, output_format)


def serve(args, output_format, dry_run):
    network = load_record(NetworkSpec, args.network)
    spec = load_record(PartySpec, args.party) if args.serve_role == "party" else None
    try:
        if not 1 <= args.port <= 65535:
            raise NetworkError("invalid service port")
        if args.serve_role == "party":
            from pllm.runtime.party import create_party_app

            app = create_party_app(spec, network)
        else:
            from pllm.runtime.directory import create_directory_app

            app = create_directory_app(network)
    except (ValueError, TypeError) as exc:
        raise ResolutionError("NETWORK_CONFIGURATION", str(exc)) from exc
    if dry_run:
        emit(f"serve.{args.serve_role}", {"validated": True, "dry_run": True,
                                       "network_digest": network.digest}, args, output_format)
        return
    import uvicorn

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)
    except OSError as exc:
        raise LocalIOError("SERVICE_START", "cannot start network service") from exc
