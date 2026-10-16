"""Replay Paper 1's exact Listing 1 through installed SDK and local HTTP gateway.

Uses an already cached checkpoint, conservative native admission and the existing
host-pressure watchdog. This is functionality evidence, not a timing/RSS cohort.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import runpy
import site
import sys
import sysconfig
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Explain why the sky is blue in one sentence."


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


class ReplayFailure(RuntimeError):
    """Only driver-owned public reason codes belong in publication records."""


def require(condition, reason):
    if not condition:
        raise ReplayFailure(reason)


def failure(error, phase):
    return {"phase": phase,
            "reason": str(error) if isinstance(error, ReplayFailure) else "operation_failed",
            "type": type(error).__name__}


def isolated_environment(environment):
    # -I is also mandatory: scrubbing alone cannot remove cwd or user-site imports.
    clean = {key: value for key, value in environment.items() if not key.startswith("PYTHON")}
    clean.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONSAFEPATH="1",
                 HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    return clean


def package_identity():
    """Observe modules loaded in this process, including the actual native module."""
    require(sys.flags.isolated == 1 and sys.flags.no_user_site == 1
            and site.ENABLE_USER_SITE is False, "isolated_python_required")
    require(not any(key in os.environ for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE")),
            "python_environment_not_scrubbed")
    package = importlib.import_module("pllm")
    native = importlib.import_module("pllm._native")
    package_file = Path(package.__file__).resolve()
    native_file = Path(native.__file__).resolve()
    roots = sorted({str(Path(sysconfig.get_path(key)).resolve()) for key in ("purelib", "platlib")})
    require(any(package_file.is_relative_to(Path(root)) for root in roots), "installed_package_required")
    require(native_file.is_relative_to(package_file.parent), "native_outside_package")
    return {"schema": "pllm.paper_listing_package.v1", "python_executable": sys.executable,
            "python_sha256": digest(sys.executable), "isolated": sys.flags.isolated,
            "no_user_site": sys.flags.no_user_site, "user_site_enabled": site.ENABLE_USER_SITE,
            "python_environment_scrubbed": True, "site_packages": roots,
            "package_file": str(package_file), "native_file": str(native_file),
            "native_sha256": digest(native_file),
            "sdk_sources": {p.relative_to(package_file.parent).as_posix(): digest(p)
                            for p in sorted(package_file.parent.rglob("*"))
                            if p.is_file() and p.suffix in {".py", ".json"}}}


def provider_child(receipt_path, expected_path, command):
    """Run in the actual provider PID; publish import identity before its CLI runs."""
    receipt = {"schema": "pllm.paper_listing_child.v1", "pid": os.getpid(),
               "parent_pid": os.getppid(), "status": "rejected"}
    try:
        require(len(command) >= 4 and command[:3] == ["-m", "pllm", "serve"]
                and command[3] in {"inference", "preparation"}, "unsupported_provider_command")
        receipt["role"] = command[3]
        identity = package_identity()
        receipt.update(identity=identity, driver_sha256=digest(__file__),
                       expected_parent_sha256=digest(expected_path))
        require(identity == json.loads(expected_path.read_text()), "provider_identity_mismatch")
        receipt.update(status="identified", phase="before_module_execution")
    except BaseException as error:
        receipt["failure"] = failure(error, "provider_identity")
        write(receipt_path, receipt)
        return 1
    write(receipt_path, receipt)
    sys.argv = ["pllm", *command[2:]]
    runpy.run_module("pllm", run_name="__main__", alter_sys=True)
    return 0


@contextmanager
def isolated_providers(servers, output, identity, children):
    """Research-only launch shim; restore the production launcher on every exit."""
    original = servers._spawn_role_process

    def launch(command, *, environment, **kwargs):
        require(len(command) >= 5 and command[0] == identity["python_executable"]
                and command[1:4] == ["-m", "pllm", "serve"]
                and command[4] in {"inference", "preparation"}, "unsupported_provider_command")
        role = command[4]
        require(all(item["role"] != role for item in children), "duplicate_provider_launch")
        name = f"provider-{role}.json"
        process = original([command[0], "-I", "-B", str(output / "driver.py"),
                            "--provider-child", str(output / name),
                            str(output / "parent-identity.json"), *command[1:]],
                           environment=isolated_environment(environment), **kwargs)
        children.append({"role": role, "pid": process.pid, "identity_file": name,
                         "process": process})
        return process

    servers._spawn_role_process = launch
    try:
        yield
    finally:
        servers._spawn_role_process = original


def check_children(output, identity, children):
    require({item["role"] for item in children} == {"inference", "preparation"}
            and len(children) == 2, "provider_children_missing")
    for child in children:
        receipt = json.loads((output / child["identity_file"]).read_text())
        require(receipt.get("schema") == "pllm.paper_listing_child.v1"
                and receipt.get("phase") == "before_module_execution"
                and receipt.get("status") == "identified" and receipt.get("pid") == child["pid"]
                and receipt.get("parent_pid") == os.getpid() and receipt.get("role") == child["role"]
                and receipt.get("identity") == identity
                and receipt.get("driver_sha256") == digest(output / "driver.py")
                and receipt.get("expected_parent_sha256") == digest(output / "parent-identity.json"),
                "provider_identity_mismatch")


def source_lock(source, model):
    lock = json.loads(json.dumps(source.manifest.metadata["source_lock"], allow_nan=False))
    require(lock["schema"] == "pllm.model_source_lock.v1" and lock["kind"] == "huggingface"
            and re.fullmatch(r"[0-9a-f]{40}", model.revision or "") is not None
            and lock["repo_id"] == model.source
            and lock["revision"] == lock["commit"] == model.revision, "source_revision_mismatch")
    rows = lock["files"]
    require(bool(rows) and len({row["path"] for row in rows}) == len(rows)
            and all(isinstance(row["path"], str) and not Path(row["path"]).is_absolute()
                    and ".." not in Path(row["path"]).parts and type(row["size"]) is int
                    and row["size"] >= 0 and re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                    for row in rows), "invalid_source_inventory")
    checkpoint = hashlib.sha256(b"pllm.checkpoint.v1\0" + canonical({"files": rows})).hexdigest()
    locked = hashlib.sha256(b"pllm.model_source_lock.v1\0" + canonical(lock)).hexdigest()
    require(checkpoint == lock["checkpoint_digest"] == source.checkpoint_digest
            and locked == source.source_lock_digest, "source_digest_mismatch")
    return lock


def extract_listing(text):
    matches = [code for code in re.findall(r"```\s*python\n(.*?)\n```", text, re.S)
               if '"prepared-qwen"' in code]
    require(len(matches) == 1, "published_listing_not_unique")
    return matches[0] + "\n"


def runtime():
    from pllm._benchmark_build import inspect
    from pllm.model_loader import resolve_model
    from pllm.runtime import benchmark_memory as memory, servers
    from starlette.testclient import TestClient
    return SimpleNamespace(inspect=inspect, resolve_model=resolve_model, memory=memory,
                           servers=servers, TestClient=TestClient)


def replay(output, *, preflight_only=False):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    (output / "driver.py").write_bytes(script.read_bytes())
    record = {"schema": "pllm.paper_listing_replay.v2", "status": "incomplete", "pid": os.getpid(),
              "scope": "Cached public checkpoint functionality; no performance or memory-tier claim",
              "public_prompt": PROMPT, "argv": sys.argv, "driver_sha256": digest(output / "driver.py")}
    watchdog = roles = rt = identity = before = experiment = None
    children = []
    seen_errors = []
    phase = "parent_identity"

    def reject(error, at):
        if any(error is previous for previous in seen_errors):
            return
        seen_errors.append(error)
        detail = failure(error, at)
        if record["status"] == "rejected":
            record.setdefault("secondary_failures", []).append(detail)
        else:
            record.update(status="rejected", failure_phase=detail["phase"],
                          failure_reason=detail["reason"], failure_type=detail["type"])

    @contextmanager
    def owned(resource, label):
        value = resource.__enter__()
        try:
            yield value
        except BaseException as error:
            reject(error, phase)
            raise
        finally:
            active = sys.exc_info()
            try:
                resource.__exit__(*active)
            except BaseException as error:
                reject(error, label + "_cleanup")
                if active[0] is None:
                    raise

    try:
        for key in list(os.environ):
            if key.startswith("PYTHON"):
                del os.environ[key]
        os.environ.update(isolated_environment(os.environ))
        sys.dont_write_bytecode = True
        identity = package_identity()
        write(output / "parent-identity.json", identity)
        record["installed_package"] = identity["package_file"]
        rt = runtime()
        phase = "listing"
        text = (ROOT / "paper/manuscript.md").read_text()
        listing = extract_listing(text)
        (output / "listing.py").write_text(listing)
        namespace = {}
        exec(compile(listing, "paper/manuscript.md:Listing1", "exec"), namespace)
        experiment = namespace["experiment"]
        write(output / "experiment.json", experiment.to_spec())
        phase = "source_before"
        source = rt.resolve_model(experiment.pipeline.model)
        before = source_lock(source, experiment.pipeline.model)
        write(output / "source-lock-before.json", before)
        phase = "build_identity"
        build = rt.inspect()
        write(output / "build.json", build)
        require(build["attributes"]["sdk_sources"] == identity["sdk_sources"]
                and build["attributes"]["native_sha256"] == identity["native_sha256"]
                and build["attributes"]["python_binary"] == identity["python_sha256"],
                "parent_build_mismatch")
        record.update({"listing_sha256": digest(output / "listing.py"),
                       "manuscript_sha256_at_execution": hashlib.sha256(text.encode()).hexdigest(),
                       "source_lock_digest": source.source_lock_digest,
                       "checkpoint_digest": source.checkpoint_digest})
        phase = "admission"
        admission = rt.memory.benchmark_memory(experiment, backend="native")
        write(output / "admission.json", admission)
        rt.memory.require_admission(admission)
        if not preflight_only:
            phase = "provider_setup"
            roles = rt.servers.build_roles(experiment, log_dir=output / "role-logs")
            watchdog = rt.memory.MemoryWatchdog(admission, lambda _reason: roles.close())
            with isolated_providers(rt.servers, output, identity, children):
                phase = "watchdog_start"
                watchdog.start()
                phase = "provider_startup"
                with owned(roles, "provider"):
                    phase = "provider_identity"
                    check_children(output, identity, children)
                    phase = "sdk_request"
                    with owned(roles.client(bundle_cache_dir=output / "client-cache"), "sdk") as client:
                        response = client.responses.create(input=PROMPT, max_output_tokens=8, temperature=0)
                        audit = record["sdk_audit"] = client.privacy_audit.to_dict()
                        require(audit["online_steps"] > 0 and audit["plaintext_prompt_bytes_sent"] == 0
                                and audit["plaintext_token_ids_sent"] == 0, "sdk_privacy_counters")
                        require(response.usage is not None, "sdk_usage_missing")
                        record["sdk_usage"] = response.usage.to_dict()
                        require(record["sdk_usage"]["input_tokens"] == 40
                                and record["sdk_usage"]["output_tokens"] == 8, "sdk_token_counts")
                        sdk_text = response.output_text
                        record["output_text_sha256"] = hashlib.sha256(sdk_text.encode()).hexdigest()
                        record["model"] = response.model
                        phase = "sdk_cleanup"
                    phase = "gateway_request"
                    with owned(rt.TestClient(roles.gateway_app(local_api_key="public-paper-replay")), "gateway") as gateway:
                        reply = gateway.post("/v1/responses",
                            headers={"authorization": "Bearer public-paper-replay"},
                            json={"model": response.model, "input": PROMPT,
                                  "max_output_tokens": 8, "temperature": 0})
                        record["gateway_status"] = reply.status_code
                        require(reply.status_code == 200, "gateway_http_status")
                        body = reply.json()
                        record["gateway_usage"] = body["usage"]
                        require(body["usage"]["input_tokens"] == 40
                                and body["usage"]["output_tokens"] == 8, "gateway_token_counts")
                        gateway_text = "".join(content.get("text", "") for item in body["output"]
                                               for content in item.get("content", []))
                        record["gateway_output_text_sha256"] = hashlib.sha256(gateway_text.encode()).hexdigest()
                        record["sdk_gateway_text_equal"] = gateway_text == sdk_text
                        require(gateway_text == sdk_text, "sdk_gateway_output_mismatch")
                        phase = "gateway_cleanup"
                    phase = "provider_cleanup"
            record["transport"] = "In-process ASGI HTTP gateway; separate local provider children over TCP"
    except BaseException as error:
        reject(error, phase)
    finally:
        # Retry idempotent cleanup even when context entry/exit failed. Preserve the
        # first failure and retain later cleanup/source failures separately.
        if roles is not None:
            try:
                roles.close()
            except BaseException as error:
                reject(error, "provider_cleanup")
        if before is not None:
            try:
                after = source_lock(rt.resolve_model(experiment.pipeline.model), experiment.pipeline.model)
                write(output / "source-lock-after.json", after)
                require(after == before, "source_lock_changed")
                record["source_unchanged"] = True
            except BaseException as error:
                record["source_unchanged"] = False
                reject(error, "source_after")
        if identity is not None:
            try:
                after_identity = package_identity()
                write(output / "parent-identity-after.json", after_identity)
                require(after_identity == identity, "installed_package_changed")
                record["installed_package_unchanged"] = True
            except BaseException as error:
                reject(error, "parent_identity_after")
        if watchdog is not None:
            try:
                watchdog.check()
            except BaseException as error:
                reject(error, "host_pressure")
            try:
                watchdog.close()
            except BaseException as error:
                reject(error, "watchdog_cleanup")
            record["maximum_observed_swap_growth_bytes"] = watchdog.maximum_swap_growth_bytes
            if watchdog.error or watchdog.maximum_swap_growth_bytes:
                reject(ReplayFailure("host_pressure_or_swap_growth"), "host_pressure")
        record["provider_children"] = [{key: value for key, value in child.items() if key != "process"}
                                       | {"returncode": child["process"].poll()}
                                       for child in children]
        if any(child["returncode"] is None for child in record["provider_children"]):
            reject(ReplayFailure("provider_not_reaped"), "provider_cleanup")
        if record["status"] != "rejected":
            record["status"] = "admitted_not_executed" if preflight_only else "completed"
        record["files"] = {p.relative_to(output).as_posix(): digest(p)
                           for p in sorted(output.glob("*")) if p.is_file() and p.name != "result.json"}
        write(output / "result.json", record)
    return record


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--provider-child":
        return provider_child(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    record = replay(args.output, preflight_only=args.preflight_only)
    if record["status"] == "rejected":
        print(f"Replay rejected: {record['failure_phase']}/{record['failure_reason']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
