"""Read-only release-archive verifier; no imports of PLLM, builds, or test execution.

Default checks archived bytes and cross-record consistency. --live also checks
the recorded local target, tool/dependency sources, fresh installation, current
repository inputs and pre-existing conformance files. Historical files are never
rewritten. The manifest is an integrity inventory, not a signed attestation.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import tomllib
import zipfile


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def verify(archive, live=False):
    def read(name):
        return json.loads((archive / name).read_text())

    manifest = read("manifest.json")
    require(manifest["digest"] == digest(manifest["files"]), "Manifest digest")
    actual_paths = {p.relative_to(archive).as_posix() for p in archive.rglob("*") if p.is_file()}
    require(actual_paths == set(manifest["files"]) | {"manifest.json"}, "Archive file set changed")
    for name, expected in manifest["files"].items():
        require(sha(archive / name) == expected, f"Archive hash: {name}")
    receipt = read("receipt.json")
    require(receipt["status"] == "passed", f"Run not complete: {receipt.get('blocker')}")
    for command in receipt["commands"]:
        require(command["exit_code"] == 0 and not command["stop_reason"], f"Command: {command['argv']}")
        require(sha(archive / command["log"]) == command["log_sha256"], f"Command log: {command['log']}")
    sources = read("sources-before.json")
    for prefix in ("sources", "cargo-dependencies", "tools"):
        before, after = read(prefix + "-before.json"), read(prefix + "-after.json")
        require(before == after, f"Before/after mismatch: {prefix}")
        key = "packages" if prefix == "cargo-dependencies" else "files"
        require(before["digest"] == digest(before[key]), f"Inventory digest: {prefix}")
    with tarfile.open(archive / "source.tar.gz", "r:gz") as source:
        require(len(source.getmembers()) == len(sources["files"]), "Source tar entry count")
        archived = {member.name: hashlib.sha256(source.extractfile(member).read()).hexdigest()
                    for member in source.getmembers() if member.isfile()}
        require(archived == sources["files"], "Source archive does not match pre-build inputs")
        lock = tomllib.loads(source.extractfile("Cargo.lock").read().decode())
        locked = {(p["name"], p["version"], p.get("source")): p.get("checksum") for p in lock["package"]}
    cargo = read("cargo-metadata.json")
    dependencies_by_id = read("cargo-dependencies-before.json")["packages"]
    for package_row in cargo["packages"]:
        if package_row["source"] is None:
            continue
        row = dependencies_by_id[package_row["id"]]
        require(row["digest"] == digest(row["files"]), "Cargo package source digest")
        checksum = locked[(package_row["name"], package_row["version"], package_row["source"])]
        if row["cached_crate"]:
            require(row["cached_crate"]["sha256"] == checksum, "Cached crate differs from Cargo.lock")
    for index, path in enumerate(sorted(sources["cargo_configuration_files"])):
        require(sha(archive / "cargo-configs" / f"{index}.toml") == sources["cargo_configuration_files"][path], "Cargo config copy")
    for name, frozen in read("frozen-sources.json").items():
        require(sha(archive / name) == frozen["sha256"], f"Frozen source: {name}")
    empty = read("target-before.json")
    require(empty["created_by_this_run"] and not empty["is_symlink"] and empty["entries"] == [], "Target not recorded empty")
    require(empty["path"] == receipt["cargo_target_dir"] == empty["cargo_metadata_target_directory"], "Target identity")
    require(read("cargo-metadata.json")["target_directory"] == empty["path"], "Cargo metadata target")
    env = read("environment.json")["build"]
    for key, expected in {"CARGO_TARGET_DIR": empty["path"], "CARGO_BUILD_JOBS": "2",
                          "CARGO_NET_OFFLINE": "true", "CARGO_INCREMENTAL": "0", "RUSTC_WRAPPER": "",
                          "RUSTC_WORKSPACE_WRAPPER": ""}.items():
        require(env[key] == expected, f"Build environment {key}")
    command = read("wheel-build.command.json")
    require(all(flag in command["argv"] for flag in ("--release", "--locked", "--offline", "--verbose")), "Build flags")
    compilation = read("compilation-evidence.json")
    log = (archive / "wheel-build.log").read_text()
    compiled = [line for line in log.splitlines() if line.lstrip().startswith("Compiling ")]
    rustc = [line for line in log.splitlines() if "Running `" in line and "rustc " in line]
    require(compilation["compiling_lines"] == compiled and len(compiled) == receipt["compiled_crates"] > 0, "Compilation lines")
    require(compilation["rustc_invocations"] == rustc and len(rustc) == receipt["rustc_invocations"] > 0, "Rust invocations")
    native_crate = next(t["name"] for p in cargo["packages"] if p["name"] == "pllm-python"
                        for t in p["targets"] if "cdylib" in t["crate_types"])
    require(compilation["native_crate"] == native_crate, "Native crate identity")
    require(any(f"--crate-name {native_crate} " in line for line in rustc), "Native Rust compilation absent")
    require(not compilation["cargo_fresh_lines"] and not any(line.lstrip().startswith("Fresh ") for line in log.splitlines()), "Cargo cached output present")
    require(digest(compilation["target_artifacts"]["files"]) == compilation["target_artifacts"]["digest"], "Target artifact digest")
    headroom = read("headroom-prebuild.json")
    require(headroom["memory_available_bytes"] >= headroom["admission"]["minimum_available_bytes"], "Memory admission")
    require(headroom["disk_free_bytes"] >= headroom["admission"]["minimum_disk_free_bytes"], "Disk admission")
    require(headroom["cpu_percent_1s"] <= headroom["admission"]["maximum_cpu_percent"], "CPU admission")
    wheel = archive / receipt["wheel"]
    require(sha(wheel) == receipt["wheel_sha256"] and wheel.stat().st_size == receipt["wheel_bytes"], "Wheel identity")
    with zipfile.ZipFile(wheel) as zipped:
        require(len(zipped.namelist()) == len(set(zipped.namelist())), "Duplicate wheel entries")
        payload = {name: hashlib.sha256(zipped.read(name)).hexdigest()
                   for name in zipped.namelist() if not name.endswith("/")}
        require(payload == read("wheel-files.json"), "Wheel payload")
        record_name = next(name for name in payload if name.endswith(".dist-info/RECORD"))
        records = list(csv.reader(io.StringIO(zipped.read(record_name).decode())))
        require({row[0] for row in records} == set(payload), "Wheel RECORD coverage")
        for name, encoded, size in records:
            if name == record_name:
                require(not encoded and not size, "Wheel RECORD self-entry")
                continue
            wanted = "sha256=" + base64.urlsafe_b64encode(bytes.fromhex(payload[name])).decode().rstrip("=")
            require(encoded == wanted and int(size) == len(zipped.read(name)), f"Wheel RECORD: {name}")
    package = {name: value for name, value in payload.items() if name.startswith("pllm/")}
    native = {name: value for name, value in package.items() if name.endswith((".so", ".pyd", ".dylib"))}
    require(package and native == receipt["native_files"], "Native digest")
    require(digest(package) == receipt["package_digest"], "Package digest")
    empty_env = read("venv-empty.json")
    require(empty_env["distributions"] == [] and empty_env["pllm_spec"] == "None" and not empty_env["user_site"], "Environment not recorded empty")
    require(empty_env["prefix"] == receipt["fresh_environment"], "Fresh venv identity")
    provenance = read("replay/provenance.json")
    require(provenance["wheel_sha256"] == receipt["wheel_sha256"], "Replay wheel identity")
    require(provenance["package_files"] == package and provenance["package_digest"] == receipt["package_digest"], "Installed package identity")
    require(provenance["native_sha256"] in native.values(), "Installed native identity")
    require(provenance["prefix"] == receipt["fresh_environment"] != provenance["base_prefix"], "Isolated venv prefix")
    require(provenance["isolated_flag"] == 1 and not provenance["user_site_enabled"], "Interpreter isolation")
    require(provenance["upstream_runtime_distributions_absent"] and not provenance["dependency_errors"], "Fresh dependency checks")
    dependencies = read("replay/dependencies-before.json")
    require(dependencies == read("replay/dependencies-after.json") and not dependencies["errors"], "Installed dependency drift/RECORD errors")
    require(dependencies["digest"] == digest(dependencies["distributions"]), "Installed dependency digest")
    for name, dist in dependencies["distributions"].items():
        require(dist["digest"] == digest(dist["files"]), f"Distribution digest: {name}")
    installed_versions = {re.sub(r"[-_.]+", "-", name).lower(): dist["version"]
                          for name, dist in dependencies["distributions"].items()}
    pinned_versions = read("python-pins.json")["packages"]
    require({name: version for name, version in installed_versions.items() if name != "pllm-run"} == pinned_versions,
            "Fresh dependency versions differ from frozen pins")
    cached = None
    if (archive / "dependency-cache-receipt.json").is_file():
        cached = read("dependency-cache-receipt.json")
        require(cached["status"] == "passed" and cached["packages_digest"] == digest(cached["packages"]), "Dependency rewrap receipt")
        cache_manifest = read("dependency-cache-manifest.json")
        require(cache_manifest["digest"] == digest(cache_manifest["files"]), "Dependency cache manifest")
        for original, copied in (("driver.py", "dependency-cache-driver.py"), ("receipt.json", "dependency-cache-receipt.json"),
                                 ("pins.json", "python-pins.json")):
            require(cache_manifest["files"][original] == sha(archive / copied), "Dependency cache source binding")
        after_install = read("dependency-cache-after-install.json")
        normalized = {re.sub(r"[-_.]+", "-", name).lower(): dist for name, dist in dependencies["distributions"].items()}
        for name, row in cached["packages"].items():
            require(row["version"] == pinned_versions[name], "Cache dependency version")
            require(row["files_before"] == row["files_after"] == after_install[name]["files"], "Extracted wheel cache drift")
            require(row["files_digest"] == digest(row["files_before"]) and row["record_validated"], "Cache payload digest")
            require(row["wheel_sha256"] == after_install[name]["wheel_sha256"], "Rewrapped wheel drift")
            for relative, expected in row["files_before"].items():
                if relative.endswith(".dist-info/RECORD"):
                    continue
                require(normalized[name]["files"].get(relative) == expected, f"Installed dependency differs from rewrapped cache: {name}/{relative}")
        install_argv = read("wheel-install.command.json")["argv"]
        require(all(flag in install_argv for flag in ("--offline", "--no-index", "--no-build", "--find-links")), "Offline cache-install flags")
    replay = read("replay/result.json")
    require(replay["status"] == "passed" and len(replay["cases"]) == 13, "Expected replay cases incomplete")
    require(all(case["status"] == "passed" for case in replay["cases"]), "Replay failed case")
    require(replay["dependency_files_unchanged"] and replay["no_upstream_runtime_imported"], "Replay execution constraints")
    require(replay["frozen_boundary_sha256"] == sha(archive / "frozen-boundary.py"), "Boundary source identity")
    generated = read("replay/generated-trajectories.json")
    require({row["family"] for row in generated} == {"qwen2", "qwen3", "llama"}, "Generated family set")
    array_pairs = 0
    for row in generated:
        family = row["family"]
        require(row["wrong_model_rejected"] and row["post_binding_norm_mutation_rejected"], "Binding checks")
        for name, expected in row["fixture_files"].items():
            require(sha(archive / "replay/fixtures" / family / name) == expected, "Generated fixture identity")
        with zipfile.ZipFile(archive / "replay" / f"trajectory-{family}-None.npz") as zipped:
            baseline = {name: zipped.read(name) for name in zipped.namelist()}
        require(len(baseline) == 20 and row["tile_rows"] == [1, 7, 16], "Trajectory scope")
        for tile in row["tile_rows"]:
            with zipfile.ZipFile(archive / "replay" / f"trajectory-{family}-{tile}.npz") as zipped:
                actual = {name: zipped.read(name) for name in zipped.namelist()}
            require(actual == baseline, f"Archived trajectory bytes differ: {family}, tile={tile}")
            array_pairs += len(baseline)
        require(row["array_pairs_compared"] == 3 * len(baseline), "Trajectory array count")
    for path in replay["imported_pllm_modules"].values():
        require(Path(path).is_relative_to(Path(provenance["package_path"])), f"Source import: {path}")
    old = read("historical-files-before.json")
    require(old == read("historical-files-after.json") and receipt["previous_conformance_files_unchanged"], "Historical archive drift")
    if live:
        root = archive.parents[3]
        for name, expected in sources["files"].items():
            require(sha(root / name) == expected, f"Live source: {name}")
        for name, expected in sources["cargo_configuration_files"].items():
            require(sha(name) == expected, f"Live Cargo config: {name}")
        for name, expected in read("tools-before.json")["files"].items():
            require(sha(name) == expected, f"Live tool: {name}")
        for row in read("cargo-dependencies-before.json")["packages"].values():
            for name, expected in row["files"].items():
                require(sha(Path(row["directory"]) / name) == expected, f"Live Cargo dependency: {name}")
            if row["cached_crate"]:
                require(sha(row["cached_crate"]["path"]) == row["cached_crate"]["sha256"], "Live cached crate")
        for name, expected in compilation["target_artifacts"]["files"].items():
            require(sha(Path(receipt["cargo_target_dir"]) / name) == expected, f"Live target artifact: {name}")
        site_packages = Path(provenance["package_path"]).parent
        for dist in dependencies["distributions"].values():
            for name, expected in dist["files"].items():
                require(sha(site_packages / name) == expected, f"Live Python distribution: {name}")
        if cached is not None:
            for row in cached["packages"].values():
                require(sha(row["wheel"]) == row["wheel_sha256"], "Live rewrapped dependency wheel")
                for name, expected in row["files_before"].items():
                    require(sha(Path(row["extracted_cache_path"]) / name) == expected, f"Live extracted dependency cache: {name}")
        for name, expected in old.items():
            require(sha(root / "paper/artifacts/conformance" / name) == expected, f"Live historical file: {name}")
    return {"status": "verified", "archive": str(archive), "live_checked": live,
            "archive_files": len(manifest["files"]), "compiled_crates": len(compiled),
            "wheel_sha256": receipt["wheel_sha256"], "replay_case_groups": len(replay["cases"]),
            "trajectory_array_pairs_verified": array_pairs,
            "historical_files_preserved": len(old)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.archive.resolve(), args.live), indent=2, sort_keys=True))
