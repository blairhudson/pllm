"""Read-only pacing replay with exact archive hashes and freshly derived summaries."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.research import private_boundary_pacing_profile as profile

MAX_BYTES = 64 << 20


def read_json(path, expected=None):
    path = Path(path)
    with path.open("rb") as handle:
        raw = handle.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES or expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("archive size/hash mismatch")
    if path.suffix == ".gz":
        import io
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as handle:
            raw = handle.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("expanded report exceeds bound")
    return json.loads(raw)


def replay(report, plan, expected):
    from pllm._benchmark_contract import digest
    sources = plan["frozen_sources"]
    if (plan["schema"] != profile.IDENTITY or type(plan["tiny"]) is not bool
            or plan["suite"] != report["suite"] or not isinstance(sources, dict) or len(sources) < 10
            or any(not isinstance(k, str) or not isinstance(v, str) or len(v) != 64
                or any(c not in "0123456789abcdef" for c in v) for k, v in sources.items())
            or digest({"schema": profile.IDENTITY, "sources": sources}) != plan["research_source_identity"]):
        raise ValueError("frozen source/suite identity differs")
    summary = profile.audit(report, source_identity=plan["research_source_identity"], tiny=plan["tiny"])
    validate_trace_order(report)
    if summary != expected:
        raise ValueError("saved summary differs from read-only reconstruction")
    return summary


def validate_trace_order(document):
    """Independent cross-head counter and timer checks on the original raw trace."""
    for row in document["results"]:
        details = row["execution"]["details"]
        trace = details.get(profile.TRACE_KEY)
        if trace is None:
            continue
        sample = details["report"]["native_network_accounting"]["samples"]["runs"][0]
        before = sample["before"]["wan_emulation"]["directed_stream_bytes"]
        after = sample["after"]["wan_emulation"]["directed_stream_bytes"]
        for role, heads in trace["heads"].items():
            for direction in ("upload", "download"):
                edge = f"client->{role}" if direction == "upload" else f"{role}->client"
                previous = before.get(edge, 0)
                for head in heads:
                    chunks = [chunk for chunk in head["chunks"] if chunk["direction"] == direction]
                    if chunks[0]["counter_after"] - chunks[0]["bytes"] < previous:
                        raise ValueError("head counter regressed/reused across requests")
                    previous = chunks[-1]["counter_after"]
                    if previous > after[edge]:
                        raise ValueError("head counter exceeds original scenario window")
                    for chunk in chunks:
                        for sleep in chunk["sleeps"]:
                            elapsed = sleep["end_ns"] - sleep["start_ns"]
                            # Microsecond tolerance exceeds float/clock resolution;
                            # wakeup and wrapper work may only lengthen the timer.
                            if elapsed + 1000 < sleep["requested_seconds"] * 1e9:
                                raise ValueError("observed sleep shorter than requested timer")


def archive(path):
    path = Path(path)
    record = read_json(path)
    if (set(record) != {"schema", "files", "sha256"} or record["schema"] != profile.IDENTITY + ".archive"
            or set(record["files"]) != {"report", "plan", "summary"}
            or set(record["sha256"]) != set(record["files"])):
        raise ValueError("archive contract differs")
    values = {}
    for key, name in record["files"].items():
        if not isinstance(name, str) or name != Path(name).name or not name or name in {".", ".."}:
            raise ValueError("archive path escapes its directory")
        digest = record["sha256"][key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("archive hash missing")
        values[key] = read_json(path.parent / name, digest)
    return replay(values["report"], values["plan"], values["summary"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    if args.archive:
        if any((args.report, args.plan, args.summary)):
            parser.error("archive and live paths are mutually exclusive")
        result = archive(args.archive)
    else:
        if not all((args.report, args.plan, args.summary)):
            parser.error("report, plan and summary are all required")
        result = replay(read_json(args.report), read_json(args.plan), read_json(args.summary))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
