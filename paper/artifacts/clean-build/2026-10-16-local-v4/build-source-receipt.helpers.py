"""Create a new local, offline, cache-reusing Maturin build receipt without installing.

All output directories must be new. The receipt hashes build inputs, resolved
Cargo dependency source trees, tool binaries, and outputs. It compares full wheel
payloads and the installed PLLM package. This is not a clean or remote attestation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def tree(root, *, skip_generated=False):
    files, omitted = {}, []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(p in {"target", "__pycache__", ".git"} for p in relative.parts):
            omitted.append(relative.as_posix())
            continue
        if skip_generated and path.suffix in {".pyc", ".pyo", ".so", ".dylib", ".pyd"}:
            omitted.append(relative.as_posix())
            continue
        files[relative.as_posix()] = sha(path)
    return {"files": files, "excluded_generated_files": omitted, "digest": canonical_digest(files)}


def source_snapshot(root):
    files, omitted = {}, []
    roots = (".cargo", "crates", "python", "schemas")
    root_files = ("Cargo.toml", "Cargo.lock", "pyproject.toml", "rust-toolchain.toml",
                  "uv.lock", ".python-version", "README.md", "LICENSE")
    for name in roots:
        result = tree(root / name, skip_generated=True)
        files.update({name + "/" + path: digest for path, digest in result["files"].items()})
        omitted.extend(name + "/" + path for path in result["excluded_generated_files"])
    for name in root_files:
        files[name] = sha(root / name)
    config_paths = set()
    for parent in (root, *root.parents):
        for name in (".cargo/config", ".cargo/config.toml"):
            if (parent / name).is_file():
                config_paths.add(parent / name)
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    for name in ("config", "config.toml"):
        if (cargo_home / name).is_file():
            config_paths.add(cargo_home / name)
    return {"files": files, "digest": canonical_digest(files),
            "inventory_roots": roots, "root_files": root_files,
            "excluded_generated_files": omitted,
            "cargo_configuration_files": {str(p): sha(p) for p in sorted(config_paths)},
            "scope": "All files in repository build-source/config roots, including untracked files; generated caches/binaries excluded explicitly"}


def dependency_snapshot(cargo_metadata):
    packages = {}
    for package in cargo_metadata["packages"]:
        if package["source"] is None:
            continue
        directory = Path(package["manifest_path"]).parent
        packages[package["id"]] = {"source": package["source"], "directory": str(directory),
                                  **tree(directory)}
    return {"packages": packages, "digest": canonical_digest(packages),
            "scope": "Every file in each resolved non-workspace Cargo package source directory, excluding generated target/cache directories"}


def zip_hashes(path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.namelist()
        if len(entries) != len(set(entries)):
            raise RuntimeError("Duplicate wheel entries")
        return {name: hashlib.sha256(archive.read(name)).hexdigest()
                for name in sorted(entries) if not name.endswith("/")}


def differences(before, after):
    return {name: {"before": before.get(name), "after": after.get(name)}
            for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--historical-build", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    build_root, receipt = args.build_root.absolute(), args.receipt.absolute()
    historical = args.historical_build.absolute()
    if build_root.exists() or receipt.exists():
        parser.error("build-root and receipt must both be new")
    build_root.mkdir(parents=True)
    receipt.mkdir(parents=True)
    wheels = build_root / "wheels"
    wheels.mkdir()
    shutil.copyfile(__file__, receipt / "driver.py")
    archive = Path(__file__).resolve().parent
    shutil.copyfile(archive / "index.json", receipt / "prior-archive-index.json")
    for name in ("sources.json", "progress.json", "wheel-build.log"):
        shutil.copyfile(historical / name, receipt / ("historical-" + name))
    env = {**os.environ, "CARGO_BUILD_JOBS": "2", "CARGO_NET_OFFLINE": "true"}
    commands = []

    def run(name, argv, *, timeout=120):
        start = datetime.now(timezone.utc).isoformat()
        monotonic = time.monotonic()
        result = subprocess.run(argv, cwd=root, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
        (receipt / (name + ".log")).write_text(result.stdout)
        record = {"name": name, "argv": argv, "cwd": str(root), "started_utc": start,
                  "elapsed_seconds": time.monotonic() - monotonic, "exit_code": result.returncode,
                  "log": name + ".log", "log_sha256": sha(receipt / (name + ".log"))}
        commands.append(record)
        write(receipt / (name + ".command.json"), record)
        return result

    tool_commands = {
        "rustc-version": ["rustc", "--version", "--verbose"],
        "cargo-version": ["cargo", "--version", "--verbose"],
        "rustup-toolchain": ["rustup", "show", "active-toolchain"],
        "rustc-path": ["rustup", "which", "rustc"],
        "cargo-path": ["rustup", "which", "cargo"],
        "maturin-version": [sys.executable, "-m", "maturin", "--version"],
        "clang-version": ["/usr/bin/xcrun", "clang", "--version"],
        "clang-path": ["/usr/bin/xcrun", "--find", "clang"],
        "linker-path": ["/usr/bin/xcrun", "--find", "ld"],
        "sdk-path": ["/usr/bin/xcrun", "--show-sdk-path"],
        "sdk-version": ["/usr/bin/xcrun", "--show-sdk-version"],
    }
    tools = {}
    for name, argv in tool_commands.items():
        observed = run(name, argv)
        if observed.returncode:
            raise RuntimeError(f"Tool identity failed: {name}")
        tools[name] = observed.stdout.strip()
    tool_files = {sys.executable: sha(sys.executable)}
    for name in ("rustc-path", "cargo-path", "clang-path", "linker-path"):
        tool_files[tools[name]] = sha(tools[name])
    for path in ("/usr/bin/strip", "/usr/bin/xcrun"):
        tool_files[path] = sha(path)
    dist = metadata.distribution("maturin")
    maturin_files = {str(entry): sha(dist.locate_file(entry)) for entry in dist.files or ()
                     if dist.locate_file(entry).is_file() and not str(entry).endswith(".pyc")}
    selected_env = {name: value for name, value in env.items()
                    if name.startswith(("CARGO_", "RUST", "PYO3_", "MATURIN_"))
                    and not any(word in name for word in ("TOKEN", "PASSWORD", "SECRET", "KEY"))}
    for name in ("PATH", "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "AR", "RANLIB",
                 "SDKROOT", "MACOSX_DEPLOYMENT_TARGET", "ARCHFLAGS", "SOURCE_DATE_EPOCH",
                 "VIRTUAL_ENV", "DEVELOPER_DIR"):
        selected_env[name] = env.get(name)
    write(receipt / "tools.json", {"python": sys.version, "tools": tools, "tool_binary_sha256": tool_files,
          "maturin_version": dist.version, "maturin_files": maturin_files, "build_environment": selected_env})
    git_before = run("git-before", ["git", "status", "--short"])
    commit = run("git-commit", ["git", "rev-parse", "HEAD"]).stdout.strip()
    cargo = run("cargo-metadata", ["cargo", "metadata", "--format-version", "1", "--locked", "--offline"])
    if cargo.returncode:
        raise RuntimeError("Offline Cargo metadata failed; no build attempted")
    cargo_metadata = json.loads(cargo.stdout)
    write(receipt / "cargo-metadata.json", cargo_metadata)
    before = source_snapshot(root)
    deps_before = dependency_snapshot(cargo_metadata)
    write(receipt / "sources-before.json", before)
    write(receipt / "dependencies-before.json", deps_before)
    old = json.loads((historical / "sources.json").read_text())
    old_mismatches = {path: {"recorded": digest, "current": sha(root / path) if (root / path).is_file() else None}
                      for path, digest in old.items() if not (root / path).is_file() or sha(root / path) != digest}
    command = [sys.executable, "-m", "maturin", "build", "--release", "--locked", "--offline", "--out", str(wheels)]
    print(json.dumps({"build": command, "environment_overrides": {"CARGO_BUILD_JOBS": "2", "CARGO_NET_OFFLINE": "true"}}), flush=True)
    built = run("wheel-build", command, timeout=1800)
    after = source_snapshot(root)
    deps_after = dependency_snapshot(cargo_metadata)
    write(receipt / "sources-after.json", after)
    write(receipt / "dependencies-after.json", deps_after)
    run("git-after", ["git", "status", "--short"])
    summary = {"schema": "pllm.local_build_receipt.v1", "source_commit": commit,
               "build_exit_code": built.returncode, "incremental_cache_reused": True,
               "cache_scope": "Existing Cargo target and registry caches; not a clean build. CARGO_INCREMENTAL is recorded separately.",
               "attestation": "Local agent-recorded receipt; no third-party or CI attestation",
               "historical_source_files_checked": len(old), "historical_source_mismatches": old_mismatches,
               "source_file_changes_during_build": differences(before["files"], after["files"]),
               "cargo_configuration_changes_during_build": differences(before["cargo_configuration_files"], after["cargo_configuration_files"]),
               "dependency_source_trees_unchanged": deps_before == deps_after,
               "source_inventory_digest": before["digest"], "dependency_inventory_digest": deps_before["digest"],
               "git_before_exit_code": git_before.returncode, "commands": commands}
    if built.returncode == 0:
        artifacts = list(wheels.glob("*.whl"))
        if len(artifacts) != 1:
            raise RuntimeError(f"Expected one wheel, got {len(artifacts)}")
        wheel = artifacts[0]
        progress = json.loads((historical / "progress.json").read_text())
        old_wheel = Path(progress["wheel"])
        if sha(old_wheel) != progress["sha256"]:
            raise RuntimeError("Historical wheel differs from its existing receipt")
        new_files, old_files = zip_hashes(wheel), zip_hashes(old_wheel)
        package = metadata.distribution("pllm.run").locate_file("pllm")
        installed = tree(package, skip_generated=True)["files"]
        # Native binaries are required comparison targets, not source inventory inputs.
        for path in package.rglob("*"):
            if path.is_file() and path.suffix in {".so", ".pyd", ".dylib"}:
                installed[path.relative_to(package).as_posix()] = sha(path)
        installed = {"pllm/" + name: digest for name, digest in installed.items()}
        new_package = {name: digest for name, digest in new_files.items() if name.startswith("pllm/")}
        write(receipt / "wheel-files.json", new_files)
        write(receipt / "historical-wheel-files.json", old_files)
        write(receipt / "installed-package-files.json", installed)
        summary.update({"wheel": str(wheel), "wheel_sha256": sha(wheel), "wheel_bytes": wheel.stat().st_size,
                        "historical_wheel_sha256": sha(old_wheel),
                        "archive_bytes_identical": sha(wheel) == sha(old_wheel),
                        "wheel_payload_differences": differences(old_files, new_files),
                        "installed_package_differences": differences(installed, new_package),
                        "wheel_payload_files": len(new_files), "package_files": len(new_package),
                        "native_files": {name: digest for name, digest in new_files.items() if name.endswith((".so", ".pyd", ".dylib"))}})
        conform = archive / "2026-10-16-installed-v4"
        summary["existing_conformance"] = {"manifest_sha256": sha(conform / "manifest.json"),
                "provenance_sha256": sha(conform / "provenance.json"),
                "native_execution_build_digest": json.loads((conform / "provenance.json").read_text())["build_digest"]}
        summary["existing_conformance_bound_to_identical_outputs"] = (
            not summary["wheel_payload_differences"] and not summary["installed_package_differences"]
            and not summary["source_file_changes_during_build"]
            and not summary["cargo_configuration_changes_during_build"]
            and summary["dependency_source_trees_unchanged"]
            and sha(old_wheel) == json.loads((conform / "provenance.json").read_text())["wheel_sha256"])
    summary["artifacts"] = {p.name: sha(p) for p in sorted(receipt.iterdir()) if p.is_file()}
    write(receipt / "receipt.json", summary)
    print(json.dumps({"build_exit_code": built.returncode,
          "conformance_bound": summary.get("existing_conformance_bound_to_identical_outputs"),
          "receipt": str(receipt)}), flush=True)
    if built.returncode:
        raise SystemExit(built.returncode)


if __name__ == "__main__":
    main()
