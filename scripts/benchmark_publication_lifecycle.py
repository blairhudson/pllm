"""Serial, source-present research pilots with streamed phase/role observations.

Per-party 100/40 local emulation is resource-unnormalized. These pilots establish
measurement capability, not a fixed-provider-budget or ten-method serving rank.
"""
from __future__ import annotations

import argparse
import atexit
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import secrets
import sys
import time
from unittest.mock import patch

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.research.publication_lifecycle import (  # noqa: E402
    InvalidObservation, MAX_SAMPLE_BYTES, MAX_SAMPLES, MAX_TRACE_FILE_BYTES,
    SCHEMA, clock_tolerance_ns, duration_matches, reconcile_network_snapshots,
    recorder_capacity, request_durations, summarize,
)
from benchmarks.research.publication_lifecycle_live import (  # noqa: E402
    FINAL_FILE, REQUEST_FILE, SAMPLES_FILE, STATE_FILE, canonical, digest, failure,
    file_digest, observation_clock, priced_admission, process_identity, runtime_provenance, write_private,
    check_commitment, sidecar_directory, sidecar_file, sidecar_json,
)

PROFILE = {"name": "resource_unnormalized_per_party_100_down_40_up",
    "network": "local TCP payload pacing, distinct parties, no installed RTT/loss emulator",
    "provider_cpu_quota": None, "provider_rss_quota": None,
    "aggregate_provider_access_equal": False, "serving_rank": "unqualified_pilot"}
DETAILS_LIMIT = 8 << 20  # Installed ordinary BenchmarkResult contract, unchanged.
LIFECYCLE_DETAILS_LIMIT = 512 << 10
MANIFEST_FILE = "manifest.json"
PROVIDER_ROLES = ("inference", "preparation", "boundary_a", "boundary_b")
ARCHIVE_LIMITS = {name: 1 << 20 for name in (REQUEST_FILE, STATE_FILE, FINAL_FILE,
    MANIFEST_FILE, "parent.json", "execution-failure.json", "acceptance.json", "owner.json",
    *(f"publication-child-{role}.json" for role in PROVIDER_ROLES))}
ARCHIVE_LIMITS.update({SAMPLES_FILE: MAX_TRACE_FILE_BYTES, "ordinary-payload.json": 16 << 20})


def sources():
    # CPython can remove __main__.__file__ before atexit. Keep the immutable
    # location, but re-read its bytes for the final source-integrity check.
    paths = [SCRIPT, ROOT / "scripts/benchmark_sota_combinations.py",
        *(ROOT / "benchmarks/research" / name for name in (
            "publication_lifecycle.py", "publication_lifecycle_live.py", "private_boundary_tiers.py",
            "private_boundary_decode.py", "private_boundary_decode_profile.py", "historical_evidence.py",
            "sota_combinations.py")), ROOT / "examples/prepared_sdk.py"]
    return {path.relative_to(ROOT).as_posix(): file_digest(path) for path in paths}


def public_request_counts(model, tokenizer, prompt, *, tiny):
    """Count the dashboard-prepared request; retain raw fixture bytes elsewhere."""
    from pllm.runtime.benchmark_context import validate_context
    from benchmarks.research.sota_combinations import public_input_counts

    # The ordinary dashboard /api/run route strips string input before validation.
    context = validate_context(prompt.strip())
    return public_input_counts(model, tokenizer, {"tiny" if tiny else "150": {"prompt": context}})


def artifact_limit(name, maximum=None):
    cap = ARCHIVE_LIMITS.get(name, 1 << 20)
    if maximum is not None:
        if type(maximum) is not int or maximum < 0:
            raise InvalidObservation("sidecar_size")
        cap = min(cap, maximum)
    return cap


def read_bounded(path, maximum=None):
    with sidecar_file(path, artifact_limit(Path(path).name, maximum)) as reader:
        return sidecar_json(reader)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def immutable_json(path, value, *, maximum=1 << 20):
    raw = canonical(value)
    if len(raw) > maximum:
        raise InvalidObservation("sidecar_metadata_bound")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as file:
        file.write(raw)
        file.flush()
        os.fchmod(file.fileno(), 0o400)
        os.fsync(file.fileno())
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def begin_trial(store, configuration):
    """An exclusively reserved durable owner exists before launching a worker."""
    store = Path(store)
    if store.is_symlink() or not store.is_dir() or store.stat().st_uid != os.getuid():
        raise InvalidObservation("evidence_store_owner")
    owner = read_bounded(store / "owner.json")
    output = Path(owner["output"])
    if (store != output.with_suffix(".traces")
            or owner != read_bounded(output.with_suffix(".reservation"))):
        raise InvalidObservation("evidence_store_reservation")
    capacity = recorder_capacity(configuration["timeout_seconds"])
    disk = os.statvfs(store)
    if disk.f_bavail * disk.f_frsize < capacity["maximum_trace_disk_bytes"] + (20 << 20):
        raise InvalidObservation("durable_trace_disk_admission")
    nonce = configuration["binding"]["nonce"]
    if type(nonce) is not str or len(nonce) != 64 or any(c not in "0123456789abcdef" for c in nonce):
        raise InvalidObservation("trial_nonce")
    directory = store / nonce
    directory.mkdir(mode=0o700)
    immutable_json(directory / REQUEST_FILE, configuration)
    sync_directory(directory)
    sync_directory(store)
    return directory


def retain_trial(source, destination, configuration, parent, payload, error=None):
    """Seal public measurement files before the ordinary runner removes its tempdir.

    Never delete a trial, including rejected trials and result-construction failures.
    Files are read-only, exclusive, hash-bound and fsynced. Inputs and role logs are
    deliberately absent; only the public configuration and ordinary result persist.
    """
    destination = Path(destination)
    with sidecar_file(destination / REQUEST_FILE, artifact_limit(REQUEST_FILE)) as reader:
        if sidecar_json(reader) != configuration:
            raise InvalidObservation("retained_configuration_binding")
        files = {REQUEST_FILE: reader.commitment}
    errors = []
    roles = configuration["expected_roles"]
    if any(role not in {"client", *PROVIDER_ROLES} for role in roles):
        raise InvalidObservation("sidecar_roles")
    names = [STATE_FILE, SAMPLES_FILE, FINAL_FILE,
             *(f"publication-child-{role}.json" for role in roles if role != "client")]
    for name in names:
        path = Path(source) / name
        try:
            with sidecar_file(path, artifact_limit(name)) as incoming:
                out = os.open(destination / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(out, "wb") as file:
                    while chunk := incoming.read(65536):
                        file.write(chunk)
                    file.flush()
                    os.fchmod(file.fileno(), 0o400)
                    os.fsync(file.fileno())
                incoming.verify()
                files[name] = incoming.commitment
        except FileNotFoundError:
            errors.append({"file": name, "code": "missing"})
        except Exception as copy_error:
            errors.append({"file": name, **failure(copy_error, "retention")})
    files["parent.json"] = immutable_json(destination / "parent.json", parent)
    if payload is not None:
        files["ordinary-payload.json"] = immutable_json(destination / "ordinary-payload.json", payload, maximum=16 << 20)
    if error is not None:
        files["execution-failure.json"] = immutable_json(destination / "execution-failure.json", failure(error, "supervisor"))
    manifest = {"schema": "pllm.publication_trace_archive.v1", "binding": configuration["binding"],
        "files": files, "retention_errors": errors, "retention": "immutable; no automatic deletion"}
    record = immutable_json(destination / MANIFEST_FILE, manifest)
    sync_directory(destination)
    return {"schema": "pllm.publication_trace_reference.v1", "path": str(destination / MANIFEST_FILE),
        "sha256": record["sha256"], "bytes": record["bytes"], "binding": configuration["binding"]}


def archive_files(roles):
    if (not isinstance(roles, list) or len(roles) != len(set(roles))
            or set(roles) not in ({"client", "inference", "preparation"}, {"client", *PROVIDER_ROLES})):
        raise InvalidObservation("sidecar_roles")
    return {REQUEST_FILE, STATE_FILE, SAMPLES_FILE, FINAL_FILE, "parent.json", "ordinary-payload.json",
            *(f"publication-child-{role}.json" for role in roles if role != "client")}


def load_trace(reference, *, allow_partial=False):
    """Replay archives with the same per-trial qualification used by the parent.

    Diagnostic recovery is explicit and always returns complete=False on any
    qualification failure. Unlisted files are never consumed. Legacy inline v1
    documents belong to summarize(), not to this archive guarantee.
    """
    try:
        return _load_trace(reference, allow_partial=allow_partial)
    except InvalidObservation:
        raise
    except OSError as error:
        raise InvalidObservation("sidecar_unavailable_or_unsafe") from error
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, RecursionError) as error:
        raise InvalidObservation("invalid_archive_evidence") from error


def _load_trace(reference, *, allow_partial):
    path = Path(reference["path"])
    if reference["schema"] != "pllm.publication_trace_reference.v1" or path.name != MANIFEST_FILE:
        raise InvalidObservation("trace_manifest_binding")
    with ExitStack() as stack:
        directory = stack.enter_context(sidecar_directory(path.parent))
        def opened(name, bound, maximum=None):
            return stack.enter_context(sidecar_file(name, artifact_limit(name, maximum),
                expected=bound, directory=directory))
        manifest = sidecar_json(opened(MANIFEST_FILE, {key: reference[key] for key in ("bytes", "sha256")}))
        if (manifest["schema"] != "pllm.publication_trace_archive.v1"
                or manifest["binding"] != reference["binding"]
                or path.parent.name != reference["binding"]["nonce"]):
            raise InvalidObservation("trace_manifest_binding")
        files = manifest["files"]
        allowed = archive_files(["client", *PROVIDER_ROLES]) | {"execution-failure.json"}
        if not {REQUEST_FILE, "parent.json"} <= set(files) or set(files) - allowed:
            raise InvalidObservation("trace_manifest_files")
        for name, bound in files.items():
            check_commitment(bound, artifact_limit(name))
        configuration = sidecar_json(opened(REQUEST_FILE, files[REQUEST_FILE]))
        required = archive_files(configuration["expected_roles"])
        errors = manifest["retention_errors"]
        if not isinstance(errors, list) or len(errors) > len(required):
            raise InvalidObservation("trace_manifest_coverage")
        missing = {row["file"] for row in errors if row["file"] not in files}
        # Older failed-supervisor archives express absent ordinary output with
        # execution-failure.json instead of a duplicate retention error.
        if "execution-failure.json" in files and "ordinary-payload.json" not in files:
            missing.add("ordinary-payload.json")
        if (required - set(files) != missing or set(files) - required - {"execution-failure.json"}
                or any(row["file"] not in required for row in errors)):
            raise InvalidObservation("trace_manifest_coverage")
        artifacts = {REQUEST_FILE: configuration}
        for name in sorted(set(files) - {REQUEST_FILE, SAMPLES_FILE}):
            artifacts[name] = sidecar_json(opened(name, files[name]))
        capacity = recorder_capacity(configuration["timeout_seconds"])
        receipt = artifacts.get(FINAL_FILE)
        state = artifacts.get(STATE_FILE, {})
        document = receipt["document"] if receipt is not None else state
        sample_cap = capacity["maximum_trace_disk_bytes"]
        if receipt is not None and document.get("complete") is True and not errors:
            count = document["sample_count"]
            if (type(count) is not int or not 2 <= count <= capacity["maximum_samples"]
                    or SAMPLES_FILE not in files or type(receipt["samples_bytes"]) is not int
                    or receipt["samples_bytes"] != files[SAMPLES_FILE]["bytes"]
                    or receipt["samples_sha256"] != files[SAMPLES_FILE]["sha256"]
                    or not count * 3 <= receipt["samples_bytes"] <= count * MAX_SAMPLE_BYTES):
                raise InvalidObservation("receipt_sample_binding")
            sample_cap = min(sample_cap, count * MAX_SAMPLE_BYTES)
        if SAMPLES_FILE in files:
            rows, sha, size, truncated = _sample_rows(opened(SAMPLES_FILE, files[SAMPLES_FILE], sample_cap))
        else:
            rows, sha, size, truncated = [], None, 0, False
        trace = {**document, "samples": rows}
        retained = {"manifest": manifest, "sha256": sha, "bytes": size, "truncated": truncated,
            "artifacts": artifacts, "qualified": False}
        try:
            summary = _qualify_archive(artifacts.get("ordinary-payload.json"), configuration,
                artifacts["parent.json"], reference, trace, receipt, retained)
        except (InvalidObservation, KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
            if not allow_partial:
                raise
            trace["complete"] = False
            retained["qualification_failure"] = failure(error, "archive_qualification")
        else:
            retained.update(qualified=True, summary=summary)
        return trace, receipt, retained


def role_ids(target):
    from pllm import Experiment
    from pllm.profiles import resolve_runtime_composition
    from pllm.roles.topology import graph_for_runtime
    p = target.to_spec()["parameters"]
    exp = Experiment.from_spec(p["model"]["experiment"]).with_params(pipeline__profile=p["model"]["profile"])
    roles = [role.id for role in graph_for_runtime(resolve_runtime_composition(exp.pipeline)).roles]
    if set(roles) not in ({"client", "inference", "preparation"},
                          {"client", "inference", "preparation", "boundary_a", "boundary_b"}):
        raise InvalidObservation("unsupported_role_graph")
    return sorted(roles)


def provider_main(role, directory, command):
    """Independent child receipt, then the original installed module entry point."""
    configuration = read_bounded(Path(directory) / REQUEST_FILE)
    if role not in configuration["expected_roles"] or role == "client":
        raise InvalidObservation("provider_role")
    expected_module = "pllm.runtime.private_boundary_role" if role.startswith("boundary_") else "pllm"
    if command[:2] != ["-m", expected_module]:
        raise InvalidObservation("provider_entry_point")
    if role in {"inference", "preparation"} and command[2:4] != ["serve", role]:
        raise InvalidObservation("provider_entry_point")
    if role.startswith("boundary_") and command[2:4] != ["--role", role]:
        raise InvalidObservation("provider_entry_point")
    identity = configuration["binding"]["source_identity"]
    if digest(sources()) != identity:
        raise InvalidObservation("provider_observer_source")
    receipt = {**process_identity(os.getpid()), "ppid": os.getppid(), "role": role,
        "nonce": configuration["binding"]["nonce"], "source_identity": identity,
        "command_digest": digest(command), "runtime": runtime_provenance(configuration["build"])}
    write_private(Path(directory) / f"publication-child-{role}.json", receipt)
    sys.argv = [command[1], *command[2:]]
    runpy.run_module(command[1], run_name="__main__", alter_sys=True)


@contextmanager
def worker(identity):
    from pllm.runtime import benchmark_targets
    from benchmarks.research.publication_lifecycle_live import instrument
    directory = Path.cwd()
    configuration = read_bounded(directory / REQUEST_FILE)
    if digest(sources()) != identity or configuration["binding"]["source_identity"] != identity:
        raise InvalidObservation("observer_source_changed")
    if digest(configuration["prompt_manifest"]) != configuration["binding"]["prompt_manifest_digest"]:
        raise InvalidObservation("prompt_manifest_binding")
    if runtime_provenance(configuration["build"]) != configuration["runtime"]:
        raise InvalidObservation("worker_runtime_changed")
    original = benchmark_targets.execute_target
    state = {"capture": None, "serialized": False}

    def finalization():
        capture = state["capture"]
        if capture is not None:
            try:
                if digest(sources()) != identity:
                    raise InvalidObservation("observer_source_changed")
            except BaseException as error:
                capture.error(error, "atexit")
            finally:
                capture.finish(after_serialization=state["serialized"])
    atexit.register(finalization)

    def execute(target, measurement):
        if (target.kind != "response" or target.digest != configuration["binding"]["target_digest"]
                or role_ids(target) != configuration["expected_roles"]
                or set(measurement.inputs) != {"prompt"}
                or hashlib.sha256(measurement.inputs["prompt"].encode()).hexdigest()
                    != configuration["prompt_manifest"]["sha256"]):
            raise InvalidObservation("worker_request_binding")
        session = {"capture": None}
        try:
            with instrument(configuration, directory, Path(__file__).resolve()) as session:
                original(target, measurement)
                capture = session["capture"]
                if capture is None or digest(sources()) != identity:
                    raise InvalidObservation("observer_source_or_owner_changed")
                capture.sample()
                capture.target_done = True
                capture.document["target_succeeded"] = True
                capture.checkpoint()
        except BaseException as error:
            measurement.details["lifecycle_failure"] = failure(error, "target")
            raise
        finally:
            state["capture"] = session["capture"]
            if state["capture"] is not None:
                state["capture"].target_done = True
        measurement.details["publication_lifecycle_binding"] = configuration["binding"]
        measurement.details["publication_lifecycle_sources"] = sources()
        measurement.details["publication_profile"] = PROFILE
        measurement.comparison["measurement_digest"] = digest({
            "ordinary": measurement.comparison["measurement_digest"],
            "lifecycle_observer": identity, "schema": SCHEMA, "profile": PROFILE})
        measurement.limitations += [
            "Research instrumentation and provider provenance bootstrap overhead included.",
            "Sampled phase RSS is not process-lifetime high-water or an enforced bound.",
            "Per-party access only; provider CPU/RSS and aggregate access are unnormalized."]

    with patch.object(benchmark_targets, "execute_target", execute):
        yield state


def read_samples(directory):
    """Recover bounded complete rows even from a killed worker's partial journal."""
    try:
        with sidecar_file(Path(directory) / SAMPLES_FILE, artifact_limit(SAMPLES_FILE)) as reader:
            return _sample_rows(reader)
    except FileNotFoundError:
        return [], None, 0, False


def _sample_rows(reader):
    rows, truncated = [], False
    while raw := reader.readline(MAX_SAMPLE_BYTES + 1):
        if len(raw) > MAX_SAMPLE_BYTES or len(rows) >= MAX_SAMPLES:
            raise InvalidObservation("trace_row_or_count_bound")
        if not raw.endswith(b"\n"):
            truncated = True
            break
        try:
            rows.append(json.loads(raw))
        except (ValueError, UnicodeDecodeError):
            truncated = True
            break
    # Commit the bounded corrupt tail too; its valid prefix is diagnostic only.
    while reader.read(65536):
        pass
    reader.verify()
    return rows, reader.commitment["sha256"], reader.commitment["bytes"], truncated


def accept_receipt(payload, configuration, parent, reference):
    """Parent acceptance happens after ordinary supervision has reaped the worker."""
    ordinary_digest = digest(payload)
    details = payload.setdefault("details", {})
    details["publication_lifecycle"] = {"trace_ref": reference, "partial": True}
    _, receipt, retained = load_trace(reference)
    if retained["manifest"]["files"].get("ordinary-payload.json", {}).get("sha256") != ordinary_digest:
        raise InvalidObservation("ordinary_payload_binding")
    artifacts = retained["artifacts"]
    if artifacts[REQUEST_FILE] != configuration or artifacts["parent.json"] != parent:
        raise InvalidObservation("archived_parent_configuration")
    summary = retained["summary"]
    details["publication_lifecycle"] = qualified_details(reference, receipt, configuration, parent, summary)
    return summary


def qualified_details(reference, receipt, configuration, parent, summary):
    return {"trace_ref": reference, "partial": False, "summary": summary,
        "finalization_receipt": {k: v for k, v in receipt.items() if k != "document"},
        "parent_receipt": {**parent, "ordinary_supervisor_returned": True,
            "binding": configuration["binding"], "runtime": configuration["runtime"]}}


def _qualify_archive(payload, configuration, parent, reference, trace, receipt, retained):
    """One qualification path for parent acceptance and standalone archive replay."""
    if (receipt is None or retained["manifest"]["retention_errors"] or payload is None
            or "execution-failure.json" in retained["manifest"]["files"]):
        raise InvalidObservation("incomplete_durable_receipt")
    rows, artifacts, doc = trace["samples"], retained["artifacts"], receipt["document"]
    if doc.get("schema") != SCHEMA or artifacts[STATE_FILE].get("schema") != SCHEMA:
        raise InvalidObservation("archive_trace_schema")
    binding = configuration["binding"]
    if (set(binding) != {"nonce", "target_digest", "source_identity", "prompt_manifest_digest"}
            or any(type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                   for value in binding.values())
            or not binding == reference["binding"] == retained["manifest"]["binding"]
                == doc["binding"] == artifacts[STATE_FILE]["binding"]):
        raise InvalidObservation("archive_binding")
    if digest(configuration["prompt_manifest"]) != configuration["binding"]["prompt_manifest_digest"]:
        raise InvalidObservation("prompt_manifest_binding")
    if (type(doc["sample_count"]) is not int or doc["sample_count"] != len(rows)
            or receipt["samples_sha256"] != retained["sha256"] or receipt["samples_bytes"] != retained["bytes"]
            or retained["truncated"]):
        raise InvalidObservation("receipt_sample_binding")
    details = payload.get("details", {})
    client = doc["owners"]["client"]
    if (receipt["schema"] != "pllm.publication_finalization.v2"
            or receipt["after_worker_serialization"] is not True
            or doc["runtime_provenance"] != configuration["runtime"]
            or doc["expected_roles"] != configuration["expected_roles"]
            or doc["network_conditions"] != configuration["wan"]
            or {k: client[k] for k in ("pid", "create_time_ns")} != parent["worker_identity"]
            or details.get("publication_lifecycle_binding") != configuration["binding"]
            or digest(details.get("publication_lifecycle_sources")) != configuration["binding"]["source_identity"]):
        raise InvalidObservation("parent_child_receipt_binding")
    capacity = recorder_capacity(configuration["timeout_seconds"])
    if doc["capacity"] != capacity or doc["clock"] != configuration["clock"]:
        raise InvalidObservation("parent_observation_contract")
    tolerance = clock_tolerance_ns(configuration["clock"])
    start, timeout_start, end = (parent.get(k) for k in ("launch_started_ns", "timeout_started_ns", "supervisor_returned_ns"))
    if (any(type(t) is not int or t < 0 for t in (start, timeout_start, end, receipt["observed_ns"]))
            or not start <= timeout_start <= end
            or rows[0]["start_ns"] < start - tolerance
            or not rows[-1]["end_ns"] <= receipt["observed_ns"] <= end + tolerance
            or receipt["observed_ns"] - timeout_start >
                (configuration["timeout_seconds"] + capacity["shutdown_grace_seconds"]) * 1e9 + tolerance):
        raise InvalidObservation("parent_observation_window")
    for role, owner in doc["owners"].items():
        if role != "client" and owner["child_receipt"] != artifacts[f"publication-child-{role}.json"]:
            raise InvalidObservation("archived_child_receipt")
    summary = summarize(trace)
    if payload["status"] != "completed":
        raise InvalidObservation("ordinary_target_failed")
    metrics = {m["id"]: m["value"] for m in payload["metrics"]}
    report = details["report"]
    run, = report["runs"]
    emission = summary["emission"]
    manifest = configuration["prompt_manifest"]
    if (run["tokens"]["authoritative"] is not True
            or run["tokens"]["input_tokens"] != manifest["input_tokens"]
            or run["tokens"]["input_tokens"] != emission["input_tokens"]
            or run["tokens"]["output_tokens"] != emission["output_tokens"]
            or emission["output_tokens"] > configuration["output_cap"]
            or (not trace["semantic_work"]["terminal_eos"] and emission["output_tokens"] != configuration["output_cap"])
            or report["configuration"]["source_lock_digest"] != manifest["source_lock_digest"]):
        raise InvalidObservation("ordinary_request_reconciliation")
    durations = request_durations(trace["request"])
    if any(not duration_matches(run["durations"][key], value, doc["clock"]) for key, value in durations.items()):
        raise InvalidObservation("ordinary_request_duration")
    for name, value in (("request.0.full", durations["full_seconds"]),
                        ("request.0.preparation", durations["preparation_seconds"]),
                        ("request.0.ttft", durations["ttft_seconds"]), ("response.ttft", durations["ttft_seconds"])):
        if not duration_matches(metrics[name], value, doc["clock"]):
            raise InvalidObservation("ordinary_request_metric")
    for rate in (metrics["response.decode_tps"], metrics["request.0.decode_tps"], run["durations"]["tokens_per_second"]):
        if emission["decode_intervals"] == 0:
            if rate is not None:
                raise InvalidObservation("ordinary_decode_rate")
        elif (type(rate) not in (int, float) or not math.isfinite(rate) or not rate > 0
                or not duration_matches(emission["decode_intervals"] / rate, emission["decode_seconds"], doc["clock"])):
            raise InvalidObservation("ordinary_decode_rate")
    ordinary = report["native_network_accounting"]["samples"]
    snapshots = [row["wan_emulation"] for row in [ordinary["startup"],
        *(row[when] for row in ordinary["runs"] for when in ("before", "after"))]]
    summary["ordinary_network_reconciliation"] = reconcile_network_snapshots(trace, snapshots)
    old = snapshots[-1]
    if (old["total_stream_bytes"] != metrics["network.all_link_stream_bytes"]
            or report["checks"]["passed"] is not True
            or report["wan_readiness"]["emulation"]["native_stream_rate_snapshots_checked"] is not True
            or report["memory_guard"]["tripped"] is not False
            or report["memory_guard"]["maximum_host_swap_growth_bytes"] != 0):
        raise InvalidObservation("ordinary_traffic_or_pressure_reconciliation")
    complete_target = metrics["process.peak_rss"]
    final_peak = receipt["process_self_high_water_bytes"]
    if (type(complete_target) not in (int, float) or type(final_peak) is not int
            or not final_peak >= complete_target >= metrics["client.peak_rss"] > 0
            or final_peak < summary["sampled_window_memory"]["client"]["simultaneous_peak_bytes"]
            or receipt["observed_ns"] < rows[-1]["end_ns"]):
        raise InvalidObservation("client_high_water_reconciliation")
    summary["client_high_water"] = {
        "ordinary_process_peak_rss_bytes": int(complete_target),
        "post_serialization_atexit_self_peak_bytes": final_peak,
        "post_serialization_scope": receipt["high_water_scope"],
        "tier_observation_bytes": int(complete_target),
        "tier_rule": "unchanged ordinary process.peak_rss maximum across accepted fresh workers",
        "additional_post_serialization_observation_bytes": final_peak,
        "extended_observed_window_peak_bytes": max(int(complete_target), final_peak),
        "ordinary_tier_window": "process_birth_through_target",
        "entire_process_lifetime_tier_qualified": None,
        "later_interpreter_teardown_peak": None}
    summary["ordinary_scenario_stream_bytes"] = old["total_stream_bytes"]
    summary["post_scenario_stream_bytes"] = summary["all_link_stream_bytes"] - old["total_stream_bytes"]
    summary["post_scenario_directed_stream_bytes"] = {key: value - old["directed_stream_bytes"].get(key, 0)
        for key, value in trace["final_network"]["directed_stream_bytes"].items()}
    if doc != artifacts[STATE_FILE]:
        raise InvalidObservation("final_state_binding")
    summary["scope"]["archive_qualification"] = "per-trial reference, manifest, receipts and ordinary-envelope reconciliation; not cohort qualification"
    lifecycle = qualified_details(reference, receipt, configuration, parent, summary)
    if len(canonical(lifecycle)) > LIFECYCLE_DETAILS_LIMIT:
        raise InvalidObservation("lifecycle_summary_bound")
    if len(canonical({**details, "publication_lifecycle": lifecycle})) > DETAILS_LIMIT:
        raise InvalidObservation("ordinary_plus_lifecycle_details_bound")
    return summary


def bound_details(payload, reference):
    """Oversized ordinary metadata remains in the archive, never silently truncated."""
    if len(canonical(payload.get("details", {}))) <= DETAILS_LIMIT:
        return
    if payload["status"] == "completed":
        payload.update(status="failed", failure_code="publication_details_bound")
    payload["details"] = {"publication_lifecycle": {"trace_ref": reference, "partial": True},
        "lifecycle_acceptance_failure": {"code": "ordinary_plus_lifecycle_details_bound",
            "ordinary_payload": "hash-bound ordinary-payload.json in trace archive"}}


@contextmanager
def publication_workers(*, identity, build, runtime, manifest, store):
    from pllm.metrics import benchmark_runner as runner
    original_execute, original_popen = runner._execute, runner.subprocess.Popen

    def execute(case, sampling, inputs, salt, directory):
        # The ordinary runner owns this freshly created scratch directory. macOS
        # may name it through /var -> /private/var. Canonicalize that trusted root
        # once; supplied archive references still use strict no-follow reads.
        directory = Path(directory).resolve(strict=True)
        p = case.target.to_spec()["parameters"]
        configuration = {"binding": {"nonce": secrets.token_hex(32), "target_digest": case.target.digest,
            "source_identity": identity, "prompt_manifest_digest": digest(manifest)},
            "expected_roles": role_ids(case.target), "timeout_seconds": sampling.timeout_seconds,
            "output_cap": p["scenario"]["max_output_tokens"], "wan": p["wan"],
            "prompt_manifest": manifest, "build": build, "runtime": runtime, "clock": observation_clock()}
        durable = begin_trial(store, configuration)
        write_private(Path(directory) / REQUEST_FILE, configuration)
        observed = []
        parent = {"worker_identity": None, "launch_started_ns": None, "timeout_started_ns": None,
            "supervisor_returned_ns": None}

        def popen(command, *args, **kwargs):
            is_worker = command[1:] == ["-I", "-m", "pllm._benchmark_worker"]
            if is_worker:
                command = [command[0], "-I", str(Path(__file__).resolve()), "--worker", identity]
                parent["launch_started_ns"] = time.monotonic_ns()
            process = original_popen(command, *args, **kwargs)
            if is_worker:
                try:
                    observed.append(process_identity(process.pid))
                except Exception:
                    observed.append(None)  # Let the ordinary supervisor retain ownership.
                communicate = process.communicate
                def timed_communicate(*args, **kwargs):
                    if parent["timeout_started_ns"] is None:
                        parent["timeout_started_ns"] = time.monotonic_ns()
                    return communicate(*args, **kwargs)
                process.communicate = timed_communicate
            return process

        payload = None
        try:
            with patch.object(runner.subprocess, "Popen", popen):
                payload = original_execute(case, sampling, inputs, salt, directory)
        except BaseException as error:
            parent["supervisor_returned_ns"] = time.monotonic_ns()
            parent["worker_identity"] = observed[0] if len(observed) == 1 else None
            try:
                retain_trial(directory, durable, configuration, parent, payload, error)
            except Exception as retention_error:
                error.add_note("Lifecycle retention failed: " + type(retention_error).__name__)
            raise
        parent["supervisor_returned_ns"] = time.monotonic_ns()
        parent["worker_identity"] = observed[0] if len(observed) == 1 else None
        reference = retain_trial(directory, durable, configuration, parent, payload)
        try:
            accept_receipt(payload, configuration, parent, reference)
        except Exception as error:
            payload.setdefault("details", {})["lifecycle_acceptance_failure"] = failure(error, "parent_acceptance")
            lifecycle = payload["details"].get("publication_lifecycle")
            if lifecycle is not None:
                lifecycle["partial"] = True
            if payload["status"] == "completed":
                payload.update(status="failed", failure_code="publication_receipt_invalid")
        bound_details(payload, reference)
        immutable_json(durable / "acceptance.json", {"status": payload["status"],
            "failure_code": payload.get("failure_code"), "evidence": reference,
            "lifecycle": payload["details"]["publication_lifecycle"],
            "failure": payload["details"].get("lifecycle_acceptance_failure")})
        sync_directory(durable)
        return payload

    with patch.object(runner, "_execute", execute):
        yield


def reserve_output(output):
    """Preserve all historical outputs and companions, including old markerless runs."""
    def paths(path):
        return [path, *(path.with_suffix(suffix) for suffix in (
            ".reservation", ".plan.json", ".progress.json", ".summary.json", ".model", ".prompt.txt", ".traces"))]
    requested = Path(output).expanduser().absolute()
    if any(path.exists() or path.is_symlink() for path in paths(requested)):
        raise FileExistsError("output or companion evidence already exists")
    # Resolve parents only: resolving the leaf would follow (and hide) a dangling symlink.
    output = requested.parent.resolve() / requested.name
    if any(path.exists() or path.is_symlink() for path in paths(output)):
        raise FileExistsError("output or companion evidence already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    owner = {"schema": "pllm.publication_store.v1", "output": str(output), "nonce": secrets.token_hex(32)}
    immutable_json(output.with_suffix(".reservation"), owner)
    if any(path.exists() or path.is_symlink() for path in paths(output) if path != output.with_suffix(".reservation")):
        raise FileExistsError("companion appeared during reservation")
    output.with_suffix(".traces").mkdir(mode=0o700)
    immutable_json(output.with_suffix(".traces") / "owner.json", owner)
    sync_directory(output.with_suffix(".traces"))
    sync_directory(output.parent)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--replicates", type=int, choices=(1, 3), default=3)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "target/prepared-frontier-artifacts")
    args = parser.parse_args()
    from filelock import FileLock
    from pllm import Model, Pipeline
    from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, BuildRef, Sampling, Scenario, benchmark_suite
    from pllm.metrics.benchmark_runner import inspect_build
    from pllm.deployment import WanConditions
    from pllm.roles import PrivateLookupAndFullHead
    from benchmarks.research import private_boundary_tiers as tiers
    from benchmarks.research.sota_combinations import atomic_json
    from scripts.benchmark_sota_combinations import read_fixture

    lock = Path.home() / ".cache/pllm/frontier-probe.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock), timeout=0):
        output = reserve_output(args.output)
        progress = {"status": "preflight", "profile": PROFILE}
        atomic_json(output.with_suffix(".progress.json"), progress)
        try:
            generated = None
            if args.tiny:
                from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
                generated = Model.tiny() if args.preflight_only else Model.path(str(
                    create_tiny_llama_checkpoint(output.with_suffix(".model"),
                        model_type="qwen2", with_qkv_bias=True).resolve()), model_id="publication-lifecycle-tiny")
                prompt = "abcde " * 14
            else:
                prompt, _ = read_fixture(args.artifacts / "prompt.txt")
            with output.with_suffix(".prompt.txt").open("x") as file:
                file.write(prompt)
            recipes = tiers.recipes(tiny=args.tiny, artifacts=args.artifacts, generated_source=generated)
            recipes = {name: recipes[name] for name in ("maintained", "tiling", "coupled-boundary")}
            boundary = recipes["coupled-boundary"]
            components = dict(boundary.pipeline.components)
            components["boundary"] = PrivateLookupAndFullHead(head_dispatch="parallel", head_output_encoding="uniform_residues")
            recipes["coupled-boundary"] = boundary.with_params(pipeline=Pipeline.from_profile(
                boundary.pipeline.profile, model=boundary.pipeline.model, components=components))
            base = recipes["maintained"].pipeline
            counts, source_lock = public_request_counts(base.model, base.components.get("tokenizer"),
                prompt, tiny=args.tiny)
            manifest = {"kind": "public_performance_fixture", "path": str(output.with_suffix(".prompt.txt")),
                "sha256": hashlib.sha256(prompt.encode()).hexdigest(), "bytes": len(prompt.encode()),
                "input_tokens": next(iter(counts.values())), "source_lock_digest": source_lock,
                "output_cap": 2 if args.tiny else 8, "stopping": "natural_EOS_or_cap; no forced length"}
            build_ref = BuildRef.capture()
            build = inspect_build(build_ref)
            runtime = runtime_provenance(build)
            suite = BenchmarkSuite([BenchmarkCase(name, BenchmarkTarget.response(exp,
                backend="native", scenario=Scenario.cold(max_output_tokens=manifest["output_cap"]),
                wan=WanConditions(), state_audit=True), build_ref) for name, exp in recipes.items()],
                Sampling(processes=args.replicates, repetitions=1, warmups=0, threads=1,
                    order="balanced", timeout_seconds=1200, memory_bytes=8 << 30), "publication-lifecycle-v2")
            capacity = recorder_capacity(suite.sampling.timeout_seconds)
            admissions = tiers.preflight(suite)
            for row in admissions:
                row["admission"] = priced_admission(row["admission"], suite.sampling.memory_bytes)
            frozen = sources()
            identity = digest(frozen)
            atomic_json(output.with_suffix(".plan.json"), {"schema": "pllm.publication_lifecycle_campaign.v2",
                "suite": suite.to_spec(), "sources": frozen, "source_identity": identity,
                "preflight": admissions, "tiny": args.tiny, "prompt_manifest": manifest,
                "runtime_provenance": runtime, "recorder_capacity": capacity, "provider_profile": PROFILE,
                "clock": observation_clock(),
                "evidence_storage": {"directory": str(output.with_suffix(".traces")),
                    "schema": "pllm.publication_trace_archive.v1", "retention": "immutable; no automatic deletion",
                    "lifecycle_details_limit_bytes": LIFECYCLE_DETAILS_LIMIT,
                    "ordinary_details_limit_bytes": DETAILS_LIMIT,
                    "overflow": "unqualified bounded reference; full ordinary payload retained"},
                "phase_contract": {"public_setup": "observer start through request admission, excluding inventory issuance",
                    "request_setup": "request admission/setup outside explicit issuance/semantic execution",
                    "private_preparation": "inventory issuance calls",
                    "prefill": "initial semantic forward through audited first selection",
                    "decode": "subsequent semantic forwards through audited selections, including unemitted EOS",
                    "unclassified": "post-selection emission and inter-step gaps",
                    "cleanup": "terminal EOS selection or cap emission through post-serialization atexit sample"},
                "emission_contract": "ordinary dashboard first/last output monotonic timestamps; N-1 intervals",
                "measurement": "sampled phase RSS plus separate ordinary-target and post-serialization RUSAGE_SELF peaks",
                "checkpoint_distribution": "source_present", "physical_wire": None})
            if not all(row["admission"]["admitted"] for row in admissions):
                raise InvalidObservation("topology_memory_admission_rejected")
            if args.preflight_only:
                progress["status"] = "admitted"
            else:
                progress["status"] = "running"
                atomic_json(output.with_suffix(".progress.json"), progress)
                with publication_workers(identity=identity, build=build, runtime=runtime, manifest=manifest,
                        store=output.with_suffix(".traces")):
                    run = benchmark_suite(suite, inputs={name: {"prompt": prompt} for name in recipes}, output=output)
                document = run.to_dict()
                if (frozen != sources() or not document["complete"]
                        or file_digest(output.with_suffix(".prompt.txt")) != manifest["sha256"]):
                    raise InvalidObservation("campaign_incomplete_or_sources_changed")
                summaries, states = [], None
                for result in document["results"]:
                    if result["status"] != "completed":
                        raise InvalidObservation("response_failed")
                    details = result["execution"]["details"]
                    lifecycle = details["publication_lifecycle"]
                    replay = load_trace(lifecycle["trace_ref"])[2]["summary"]
                    if lifecycle["summary"] != replay:
                        raise InvalidObservation("lifecycle_replay_mismatch")
                    bindings = details["response_state_audit"]["bindings"]
                    if states is not None and bindings != states:
                        raise InvalidObservation("full_state_mismatch")
                    states = bindings
                    summaries.append({"case": result["execution"]["case"]["name"],
                        "replicate": result["execution"]["replicate"], "summary": lifecycle["summary"],
                        "trace_ref": lifecycle["trace_ref"]})
                atomic_json(output.with_suffix(".summary.json"), {"status": "completed", "profile": PROFILE,
                    "prompt_manifest": manifest, "results": summaries})
                progress.update(status="completed", clients=len(summaries))
        except BaseException as error:
            progress.update(status="rejected", failure=failure(error, "campaign"))
            atomic_json(output.with_suffix(".progress.json"), progress)
            raise
        atomic_json(output.with_suffix(".progress.json"), progress)
        print(json.dumps(progress))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        from pllm._benchmark_worker import main as ordinary_worker
        with worker(sys.argv[2]) as state:
            ordinary_worker()
            state["serialized"] = True
    elif len(sys.argv) >= 7 and sys.argv[1] == "--provider":
        provider_main(sys.argv[2], Path(sys.argv[3]), sys.argv[4:])
    else:
        main()
