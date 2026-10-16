"""Record an offline, empty-target release build and a fresh frozen environment.

Run with the existing dependency environment's interpreter. This driver imports
only metadata and build tools, never PLLM or model/operator code. Output must be
new; failures and partial logs remain in place. --verify only checks hashes.
"""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import resource
import runpy
import shutil
import subprocess
import sys
import time
import tomllib
import traceback
from urllib.parse import urlparse, unquote
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NATIVE = (".so", ".pyd", ".dylib")


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def helpers(path):
    return runpy.run_path(str(path), run_name="clean_build_helpers")


def environment_inventory(helper):
    """Read distribution files without importing any installed package."""
    distributions = {}
    for dist in metadata.distributions():
        files = {}
        for entry in dist.files or ():
            path = Path(dist.locate_file(entry))
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_file():
                files[str(entry)] = helper["sha"](path)
        distributions[normalized(dist.metadata["Name"])] = {
            "name": dist.metadata["Name"], "version": dist.version,
            "location": str(dist.locate_file("")), "files": files,
            "files_digest": helper["canonical_digest"](files),
            "direct_url": json.loads(dist.read_text("direct_url.json") or "{}"),
        }
    return {"python": sys.executable, "base_python": sys._base_executable,
            "python_version": sys.version, "python_sha256": helper["sha"](sys.executable),
            "prefix": sys.prefix, "base_prefix": sys.base_prefix, "sys_path": sys.path,
            "distributions": distributions,
            "digest": helper["canonical_digest"](distributions),
            "scope": "All RECORD-listed existing distribution files except generated Python bytecode"}


def cache_inventory(cache, versions, helper):
    """Bound cache hashing to compatible wheels for the frozen name/version set."""
    from email.parser import Parser
    from packaging.tags import parse_tag, sys_tags

    supported = set(sys_tags())
    archives = {}
    for package, version in sorted(versions.items()):
        directories = list(cache.glob(f"wheels-v*/pypi/{package}"))
        directories += list(cache.glob(f"wheels-v*/index/*/{package}"))
        for directory in directories:
            for candidate in sorted(directory.iterdir()):
                if not candidate.is_dir() or not candidate.name.startswith(version + "-"):
                    continue
                distributions = list(metadata.distributions(path=[str(candidate)]))
                if len(distributions) != 1:
                    raise RuntimeError(f"Expected one cached wheel distribution: {candidate}")
                dist = distributions[0]
                if normalized(dist.metadata["Name"]) != package or dist.version != version:
                    raise RuntimeError(f"Cache directory and metadata disagree: {candidate}")
                wheel = Parser().parsestr(dist.read_text("WHEEL"))
                tags = set().union(*(parse_tag(tag) for tag in wheel.get_all("Tag", [])))
                if not supported.intersection(tags):
                    continue
                archives[str(candidate)] = {"package": package, "version": version,
                    "resolved_directory": str(candidate.resolve()), **helper["tree"](candidate)}
    missing = sorted(set(versions) - {v["package"] for v in archives.values()})
    return {"archives": archives, "missing_packages": missing,
            "digest": helper["canonical_digest"](archives),
            "scope": "Compatible unpacked cached wheel trees for exactly the frozen dependency versions; registry resolution metadata excluded"}


def local_dependency_wheels(snapshot, locked, directory, helper):
    """Re-archive verified cached payloads, giving each archive its own new hash.

    Offline index metadata may be absent even when all wheel payloads are cached.
    These are explicitly local reconstructed archives, not the original wheels.
    """
    from email.parser import Parser
    from packaging.tags import parse_tag
    from packaging.utils import parse_wheel_filename

    directory.mkdir()
    chosen = {}
    for path, info in sorted(snapshot["archives"].items()):
        chosen.setdefault(info["package"], (Path(path), info["version"]))
    outputs = {}
    for name, (root, version) in sorted(chosen.items()):
        dist = next(metadata.distributions(path=[str(root)]))
        tags = set().union(*(parse_tag(t) for t in Parser().parsestr(dist.read_text("WHEEL")).get_all("Tag", [])))
        matches = [w for w in locked[(name, version)]["wheels"]
                   if parse_wheel_filename(w["url"].rsplit("/", 1)[-1])[3] == tags]
        if len(matches) != 1:
            raise RuntimeError(f"Cannot identify original locked wheel metadata: {name}")
        upstream = matches[0]
        wheel = directory / upstream["url"].rsplit("/", 1)[-1]
        record = next(root.glob("*.dist-info/RECORD"))
        files = {}
        with zipfile.ZipFile(wheel, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for relative, expected_hash, size in csv.reader(record.read_text().splitlines()):
                path = root / relative
                if Path(relative).is_absolute() or ".." in Path(relative).parts or relative in files:
                    raise RuntimeError(f"Invalid or duplicate cached wheel path: {relative}")
                if expected_hash:
                    algorithm, expected = expected_hash.split("=", 1)
                    with path.open("rb") as stream:
                        actual = hashlib.file_digest(stream, algorithm).digest()
                    if base64.urlsafe_b64encode(actual).rstrip(b"=").decode() != expected:
                        raise RuntimeError(f"Cached RECORD hash mismatch: {path}")
                elif path != record:
                    raise RuntimeError(f"Unexpected unhashed cached file: {path}")
                if size and path.stat().st_size != int(size):
                    raise RuntimeError(f"Cached RECORD size mismatch: {path}")
                files[relative] = helper["sha"](path)
                archive.write(path, relative)
        if helper["zip_hashes"](wheel) != files:
            raise RuntimeError(f"Re-archived payload mismatch: {wheel}")
        outputs[name] = {"version": version, "cache_directory": str(root),
                         "wheel": str(wheel), "sha256": helper["sha"](wheel),
                         "payload_files": len(files), "payload_digest": helper["canonical_digest"](files),
                         "upstream_lock_record": upstream,
                         "identity": "Locally reconstructed ZIP; own archive hash, verified original cached RECORD payloads. Upstream archive hash is historical metadata, not asserted equal."}
    return outputs


def verify(receipt):
    receipt = receipt.resolve()
    data = json.loads(receipt.read_text())
    directory = receipt.parent
    h = helpers(directory / "build-source-receipt.helpers.py")
    checked = 0
    for name, expected in data["artifacts"].items():
        assert h["sha"](directory / name) == expected, name
        checked += 1
    for command in data["commands"]:
        assert h["sha"](directory / command["log"]) == command["log_sha256"]
    for phase in ("before", "after"):
        sources = json.loads((directory / f"sources-{phase}.json").read_text())
        assert h["canonical_digest"](sources["files"]) == sources["digest"]
        deps = json.loads((directory / f"dependencies-{phase}.json").read_text())
        for package in deps["packages"].values():
            assert h["canonical_digest"](package["files"]) == package["digest"]
        assert h["canonical_digest"](deps["packages"]) == deps["digest"]
        cache = json.loads((directory / f"python-cache-{phase}.json").read_text())
        for archive in cache["archives"].values():
            assert h["canonical_digest"](archive["files"]) == archive["digest"]
        assert h["canonical_digest"](cache["archives"]) == cache["digest"]
    wheel = Path(data["wheel"])
    assert h["sha"](wheel) == data["wheel_sha256"]
    assert h["zip_hashes"](wheel) == json.loads((directory / "wheel-files.json").read_text())
    installed = json.loads((directory / "environment-after.json").read_text())
    installed_files = 0
    for dist in installed["distributions"].values():
        assert h["canonical_digest"](dist["files"]) == dist["files_digest"]
        for relative, expected in dist["files"].items():
            assert h["sha"](Path(dist["location"]) / relative) == expected, relative
            installed_files += 1
    assert h["canonical_digest"](installed["distributions"]) == installed["digest"]
    assert h["sha"](installed["python"]) == installed["python_sha256"]
    assert data["source_file_changes_during_build"] == {}
    assert data["cargo_configuration_changes_during_build"] == {}
    assert data["dependency_source_trees_unchanged"]
    assert data["python_cache_payloads_unchanged"]
    assert data["original_environment_unchanged"]
    assert data["installed_build_dependencies_unchanged"]
    inputs = json.loads((directory / "python-dependency-wheel-inputs.json").read_text())
    for value in inputs.values():
        assert h["sha"](value["wheel"]) == value["sha256"]
        assert h["canonical_digest"](h["zip_hashes"](value["wheel"])) == value["payload_digest"]
    assert data["installed_package_differences"] == {}
    assert data["frozen_version_differences"] == {}
    assert data["frozen_dependency_code_differences"] == {}
    empty = json.loads((directory / "target-before-build.json").read_text())
    assert empty["entries"] == [] and empty["created_new"]
    assert data["status"] == "built_and_installed_execution_pending"
    return {"status": "passed", "receipt": str(receipt),
            "receipt_sha256": h["sha"](receipt), "artifacts_checked": checked,
            "installed_distribution_files_checked": installed_files,
            "runtime_execution": False}


def build(output, uv):
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    work = output / "work"
    work.mkdir()
    target, venv, wheels = work / "cargo-target", work / "venv", output / "wheels"
    wheels.mkdir()
    target.mkdir()
    for source, destination in (
        (Path(__file__), "driver.py"),
        (ROOT / "paper/artifacts/conformance/build-source-receipt.py", "build-source-receipt.helpers.py"),
        (ROOT / "paper/artifacts/conformance/2026-10-16-local-build-v1/receipt.json", "prior-cached-receipt.json"),
        (ROOT / "paper/artifacts/conformance/2026-10-16-installed-v4/provenance.json", "v4-provenance.json"),
        (ROOT / "paper/artifacts/conformance/2026-10-16-installed-v4/manifest.json", "v4-manifest.json"),
    ):
        with (output / destination).open("xb") as stream:
            stream.write(source.read_bytes())
    h = helpers(output / "build-source-receipt.helpers.py")
    sha, write, digest, differences = (h[name] for name in ("sha", "write", "canonical_digest", "differences"))
    commands = []
    overrides = {"CARGO_BUILD_JOBS": "2", "CARGO_NET_OFFLINE": "true",
                 "CARGO_INCREMENTAL": "0", "CARGO_TARGET_DIR": str(target),
                 "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never",
                 "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
    removed = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER")
    env = {k: v for k, v in os.environ.items() if k not in removed}
    env.update(overrides)
    selected = {k: v for k, v in env.items()
                if k.startswith(("CARGO_", "RUST", "PYO3_", "MATURIN_", "UV_", "PYTHON"))
                and not any(word in k for word in ("TOKEN", "PASSWORD", "SECRET", "KEY"))}
    for name in ("PATH", "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "AR", "RANLIB", "SDKROOT",
                 "MACOSX_DEPLOYMENT_TARGET", "ARCHFLAGS", "SOURCE_DATE_EPOCH", "DEVELOPER_DIR"):
        selected[name] = env.get(name)
    write(output / "build-environment.json", {"environment": selected, "overrides": overrides,
          "removed_environment_variables": list(removed), "limits_changed": False,
          "swap_configuration_changed": False})

    def run(name, argv, timeout=120, environment_overrides=None):
        argv = list(map(str, argv))
        record = {"name": name, "argv": argv, "cwd": str(ROOT),
                  "started_utc": datetime.now(timezone.utc).isoformat(),
                  "log": name + ".log", "environment_file": "build-environment.json",
                  "environment_overrides": environment_overrides or {}}
        write(output / (name + ".request.json"), record)
        start = time.monotonic()
        try:
            with (output / record["log"]).open("xb") as log:
                result = subprocess.run(argv, cwd=ROOT, env={**env, **(environment_overrides or {})}, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=timeout, check=False)
            record["exit_code"] = result.returncode
        except BaseException as exc:
            record.update(exit_code=None, error=repr(exc))
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - start
            record["log_sha256"] = sha(output / record["log"])
            commands.append(record)
            write(output / (name + ".command.json"), record)
        if record["exit_code"]:
            raise RuntimeError(f"{name} exited {record['exit_code']}; preserved {record['log']}")
        return (output / record["log"]).read_text()

    summary = {"schema": "pllm.local_clean_build_receipt.v1", "status": "failed",
               "driver_argv": [sys.executable, *sys.argv],
               "attestation": "Local agent-recorded receipt; not CI or independent third-party attestation",
               "output": str(output), "cargo_target_dir": str(target), "venv": str(venv),
               "python": str(venv / "bin/python"), "commands": commands,
               "incremental_cache_reused": False, "compiler_wrapper_used": False,
               "cache_scope": "Initially empty Cargo target; pre-existing registry sources, toolchain, OS SDK and Python wheel cache reused",
               "python_dependency_archive_scope": "Locally reconstructed wheels from RECORD-verified cached payloads; distinct archive hashes, locked versions preserved",
               "runtime_execution": False, "existing_conformance_transferred": False,
               "limits": ["Bounded source/config and resolved dependency inventory, not a hermetic whole-host image",
                          "Before/after hashes do not observe transient concurrent modifications",
                          "Tool binaries identified; entire OS SDK and Rust sysroot not inventoried",
                          "Offline flags prohibit Cargo/uv fetching, not an OS-level network sandbox",
                          "No models, operator samples, SDK tests or benchmarks executed"]}
    try:
        import psutil

        preflight = {"time_utc": datetime.now(timezone.utc).isoformat(),
                     "disk": shutil.disk_usage(output)._asdict(),
                     "memory": psutil.virtual_memory()._asdict(),
                     "swap": psutil.swap_memory()._asdict(),
                     "rlimits": {name: resource.getrlimit(getattr(resource, name))
                                 for name in ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_NOFILE", "RLIMIT_CPU")},
                     "minimum_available_memory_bytes": 4 * 1024**3,
                     "minimum_free_disk_bytes": 8 * 1024**3}
        write(output / "preflight.json", preflight)
        for name, argv in {"memory-pressure-before": ["memory_pressure", "-Q"],
                           "vm-stat-before": ["vm_stat"],
                           "swap-before": ["sysctl", "hw.memsize", "vm.swapusage"],
                           "processes-before": ["ps", "-axo", "pid,rss,%cpu,comm"]}.items():
            run(name, argv)
        if preflight["memory"]["available"] < preflight["minimum_available_memory_bytes"]:
            raise RuntimeError("Insufficient available memory; no build attempted")
        if preflight["disk"]["free"] < preflight["minimum_free_disk_bytes"]:
            raise RuntimeError("Insufficient disk; no build attempted")
        run("git-before", ["git", "status", "--short"])
        summary["source_commit"] = run("git-commit", ["git", "rev-parse", "HEAD"]).strip()
        original = environment_inventory(h)
        write(output / "original-environment-before.json", original)
        v4 = json.loads((output / "v4-provenance.json").read_text())
        original_versions = {name: d["version"] for name, d in original["distributions"].items()}
        summary["original_versions_vs_v4"] = differences(
            {normalized(k): v for k, v in v4["installed_distributions"].items()}, original_versions)
        pins = {name: version for name, version in original_versions.items() if name != "pllm-run"}
        lock = tomllib.loads((ROOT / "uv.lock").read_text())
        locked = {(normalized(p["name"]), p.get("version")): p for p in lock["package"]}
        requirements = []
        for name, version in sorted(pins.items()):
            package = locked[(name, version)]
            hashes = sorted({wheel["hash"] for wheel in package["wheels"]})
            if not hashes:
                raise RuntimeError(f"No locked wheels for {name}=={version}")
            requirements.append(f"{name}=={version} " + " ".join("--hash=" + value for value in hashes))
        requirements_path = output / "upstream-frozen-dependencies.txt"
        with requirements_path.open("x") as stream:
            stream.write("# Existing v4-compatible versions; wheel hashes from repository uv.lock.\n")
            stream.write("\n".join(requirements) + "\n")
        uv = str(Path(uv).absolute())
        write(output / "installer-tool.json", {"path": uv, "resolved_path": str(Path(uv).resolve()),
              "sha256": sha(uv), "selection": "Explicit --uv tool, or PATH default; existing local executable"})
        run("uv-version", [uv, "--version"])
        cache = Path(run("uv-cache-dir", [uv, "cache", "dir"]).strip())
        cache_before = cache_inventory(cache, pins, h)
        write(output / "python-cache-before.json", cache_before)
        if cache_before["missing_packages"]:
            raise RuntimeError(f"Missing compatible cached dependencies: {cache_before['missing_packages']}")
        dependency_wheels = local_dependency_wheels(cache_before, locked, work / "dependency-wheels", h)
        write(output / "python-dependency-wheel-inputs.json", dependency_wheels)
        requirements_path = output / "frozen-dependencies.txt"
        with requirements_path.open("x") as stream:
            stream.write("# Local reconstructed archives; original locked versions and RECORD payloads retained.\n")
            for value in dependency_wheels.values():
                stream.write(f"{Path(value['wheel']).as_uri()} --hash=sha256:{value['sha256']}\n")
        run("venv-create", [sys.executable, "-I", "-B", "-m", "venv", "--without-pip", venv])
        python = venv / "bin/python"
        inspect_command = [python, "-I", "-B", Path(__file__).resolve(), "--inspect-environment",
                           "--helper", output / "build-source-receipt.helpers.py"]
        initial = json.loads(run("environment-empty", inspect_command))
        write(output / "environment-empty.json", initial)
        if initial["distributions"] or initial["prefix"] == initial["base_prefix"]:
            raise RuntimeError("New environment is not empty and isolated")
        write(output / "venv-configuration.json", {"path": str(venv / "pyvenv.cfg"),
              "sha256": sha(venv / "pyvenv.cfg"), "text": (venv / "pyvenv.cfg").read_text()})
        run("dependencies-install", [uv, "--offline", "--no-config", "pip", "install",
            "--python", python, "--no-index", "--cache-dir", work / "uv-install-cache",
            "--require-hashes", "--no-build", "--no-deps", "--link-mode", "copy",
            "--requirements", requirements_path, "--verbose"], timeout=600)
        cache_after = cache_inventory(cache, pins, h)
        write(output / "python-cache-after.json", cache_after)
        summary["python_cache_payloads_unchanged"] = cache_before == cache_after
        before_env = json.loads(run("environment-before-build", inspect_command))
        write(output / "environment-before-build.json", before_env)
        assert {n: d["version"] for n, d in before_env["distributions"].items()} == pins
        tool_commands = {
            "rustc-version": ["rustc", "--version", "--verbose"],
            "cargo-version": ["cargo", "--version", "--verbose"],
            "rustup-toolchain": ["rustup", "show", "active-toolchain"],
            "rustc-path": ["rustup", "which", "rustc"],
            "cargo-path": ["rustup", "which", "cargo"],
            "maturin-version": [python, "-I", "-B", "-m", "maturin", "--version"],
            "clang-version": ["/usr/bin/xcrun", "clang", "--version"],
            "clang-path": ["/usr/bin/xcrun", "--find", "clang"],
            "linker-path": ["/usr/bin/xcrun", "--find", "ld"],
            "sdk-path": ["/usr/bin/xcrun", "--show-sdk-path"],
            "sdk-version": ["/usr/bin/xcrun", "--show-sdk-version"],
        }
        tools = {name: run(name, argv).strip() for name, argv in tool_commands.items()}
        binaries = [str(python), str(venv / "bin/maturin"), uv, "/usr/bin/strip", "/usr/bin/xcrun"]
        binaries += [tools[name] for name in ("rustc-path", "cargo-path", "clang-path", "linker-path")]
        write(output / "tools.json", {"tools": tools,
              "tool_binary_sha256": {path: sha(path) for path in binaries}})
        cargo = json.loads(run("cargo-metadata", ["cargo", "metadata", "--format-version", "1", "--locked", "--offline"],
            environment_overrides={"CARGO_TARGET_DIR": str(work / "metadata-target")}))
        write(output / "cargo-metadata.json", cargo)
        before, deps_before = h["source_snapshot"](ROOT), h["dependency_snapshot"](cargo)
        write(output / "sources-before.json", before)
        write(output / "dependencies-before.json", deps_before)
        empty = {"path": str(target), "created_new": True,
                 "entries": sorted(p.name for p in target.iterdir()),
                 "time_utc": datetime.now(timezone.utc).isoformat()}
        write(output / "target-before-build.json", empty)
        if empty["entries"]:
            raise RuntimeError("Cargo target is not empty immediately before build")
        prebuild_resources = {"disk": shutil.disk_usage(output)._asdict(),
                              "memory": psutil.virtual_memory()._asdict(),
                              "swap": psutil.swap_memory()._asdict()}
        write(output / "resources-before-build.json", prebuild_resources)
        if (prebuild_resources["memory"]["available"] < preflight["minimum_available_memory_bytes"]
                or prebuild_resources["disk"]["free"] < preflight["minimum_free_disk_bytes"]):
            raise RuntimeError("Insufficient resources immediately before build")
        build_argv = [python, "-I", "-B", "-m", "maturin", "build", "--release", "--locked",
                      "--offline", "--interpreter", python, "--out", wheels, "--verbose"]
        print(json.dumps({"build": list(map(str, build_argv)), "overrides": overrides}), flush=True)
        try:
            run("wheel-build", build_argv, timeout=3600)
        finally:
            after, deps_after = h["source_snapshot"](ROOT), h["dependency_snapshot"](cargo)
            write(output / "sources-after.json", after)
            write(output / "dependencies-after.json", deps_after)
            summary.update(source_file_changes_during_build=differences(before["files"], after["files"]),
                cargo_configuration_changes_during_build=differences(before["cargo_configuration_files"], after["cargo_configuration_files"]),
                dependency_source_trees_unchanged=deps_before == deps_after,
                source_inventory_digest=before["digest"], dependency_inventory_digest=deps_before["digest"])
        artifacts = list(wheels.glob("*.whl"))
        if len(artifacts) != 1:
            raise RuntimeError(f"Expected one wheel, found {len(artifacts)}")
        wheel = artifacts[0]
        new_files = h["zip_hashes"](wheel)
        write(output / "wheel-files.json", new_files)
        summary.update(wheel=str(wheel), wheel_sha256=sha(wheel), wheel_bytes=wheel.stat().st_size,
                       native_files={n: v for n, v in new_files.items() if n.endswith(NATIVE)})
        old_wheel = Path(v4["wheel"])
        assert sha(old_wheel) == v4["wheel_sha256"]
        old_files = h["zip_hashes"](old_wheel)
        assert {n: d for n, d in old_files.items() if n.startswith("pllm/")} == v4["wheel_package_files"]
        write(output / "v4-wheel-files.json", old_files)
        comparison = {"v4_wheel": str(old_wheel), "v4_wheel_sha256": v4["wheel_sha256"],
                      "v4_native_sha256": v4["native_sha256"], "v4_build_digest": v4["build_digest"],
                      "new_wheel_sha256": sha(wheel), "new_native_files": summary["native_files"],
                      "archive_bytes_identical": sha(wheel) == v4["wheel_sha256"],
                      "wheel_payload_differences": differences(old_files, new_files),
                      "package_payload_differences": differences(v4["wheel_package_files"],
                          {n: d for n, d in new_files.items() if n.startswith("pllm/")}),
                      "conformance_transferred": False,
                      "note": "Comparison is byte-level only. v4 retains its original execution/build identity; this wheel needs fresh execution."}
        write(output / "v4-comparison.json", comparison)
        with (output / "wheel-install.txt").open("x") as stream:
            stream.write(f"{wheel.as_uri()} --hash=sha256:{sha(wheel)}\n")
        run("wheel-install", [uv, "--offline", "--no-config", "pip", "install", "--python", python,
            "--require-hashes", "--no-deps", "--no-build", "--link-mode", "copy",
            "--requirements", output / "wheel-install.txt", "--verbose"], timeout=300)
        run("dependency-check", [uv, "--offline", "--no-config", "pip", "check", "--python", python])
        installed = json.loads(run("environment-after", inspect_command))
        write(output / "environment-after.json", installed)
        summary["installed_build_dependencies_unchanged"] = before_env["distributions"] == {
            name: dist for name, dist in installed["distributions"].items() if name != "pllm-run"}
        dist = installed["distributions"]["pllm-run"]
        package = {n: d for n, d in dist["files"].items() if n.startswith("pllm/")}
        write(output / "installed-package-files.json", package)
        summary["installed_package_differences"] = differences(
            {n: d for n, d in new_files.items() if n.startswith("pllm/")}, package)
        summary["frozen_version_differences"] = differences(original_versions,
            {n: d["version"] for n, d in installed["distributions"].items()})
        dependency_code = lambda inventory: {name: {p: d for p, d in dist["files"].items()
            if p.endswith((".py", *NATIVE))} for name, dist in inventory["distributions"].items()
            if name != "pllm-run"}
        summary["frozen_dependency_code_differences"] = differences(
            dependency_code(original), dependency_code(installed))
        direct = dist["direct_url"]
        assert unquote(urlparse(direct["url"]).path) == str(wheel)
        summary["installed_direct_url"] = direct
        original_after = environment_inventory(h)
        write(output / "original-environment-after.json", original_after)
        summary["original_environment_unchanged"] = original == original_after
        summary["fresh_install_identity"] = digest({"wheel_sha256": sha(wheel),
            "native_files": summary["native_files"], "environment_digest": installed["digest"],
            "python_sha256": installed["python_sha256"], "source_digest": before["digest"]})
        summary["identity_scope"] = "Receipt content identity; not the runtime execution build_digest"
        run("memory-pressure-after", ["memory_pressure", "-Q"])
        run("swap-after", ["sysctl", "vm.swapusage"])
        run("git-after", ["git", "status", "--short"])
        write(output / "resources-after.json", {"disk": shutil.disk_usage(output)._asdict(),
              "memory": psutil.virtual_memory()._asdict(), "swap": psutil.swap_memory()._asdict()})
        assert not summary["source_file_changes_during_build"]
        assert not summary["cargo_configuration_changes_during_build"]
        assert summary["dependency_source_trees_unchanged"]
        assert summary["python_cache_payloads_unchanged"]
        assert summary["original_environment_unchanged"]
        assert summary["installed_build_dependencies_unchanged"]
        assert not summary["installed_package_differences"]
        assert not summary["frozen_version_differences"]
        assert not summary["frozen_dependency_code_differences"]
        summary["status"] = "built_and_installed_execution_pending"
    except BaseException as exc:
        summary["failure"] = repr(exc)
        with (output / "failure.txt").open("x") as stream:
            stream.write(traceback.format_exc())
        raise
    finally:
        summary["artifacts"] = {p.relative_to(output).as_posix(): sha(p)
            for p in sorted(output.rglob("*")) if p.is_file() and work not in p.parents}
        write(output / "receipt.json", summary)
        print(json.dumps({"status": summary["status"], "receipt": str(output / "receipt.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--uv", default=shutil.which("uv"), help="Existing offline-capable uv executable")
    parser.add_argument("--inspect-environment", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--helper", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inspect_environment:
        print(json.dumps(environment_inventory(helpers(args.helper)), sort_keys=True))
    elif args.verify:
        print(json.dumps(verify(args.verify), indent=2, sort_keys=True))
    elif args.output:
        build(args.output, args.uv)
    else:
        parser.error("--output or --verify is required")


if __name__ == "__main__":
    main()
