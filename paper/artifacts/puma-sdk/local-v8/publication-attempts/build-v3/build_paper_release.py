"""Archive a bounded offline release build from an empty Cargo target, then replay.

Run with target/frontier-sdk-env/bin/python -I. Output must not exist. Scratch
Cargo/venv state is created only beneath paper/artifacts/release/.work. Existing
registry source and Python wheel caches are used; this is not a hermetic build.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import sysconfig
import tarfile
import time
import traceback
import uuid
import zipfile

import psutil


ROOT = Path(__file__).resolve().parents[1]
THREADS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "VECLIB_MAXIMUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
           "RAYON_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "UV_CONCURRENT_INSTALLS": "2", "UV_CONCURRENT_BUILDS": "1", "UV_CONCURRENT_DOWNLOADS": "1"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def tree(root, source=False):
    files, excluded = {}, []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if not path.is_file():
            continue
        if any(p in {"target", "__pycache__", ".git"} for p in path.relative_to(root).parts) or (
                source and path.suffix in {".so", ".dylib", ".pyd", ".pyc", ".pyo"}):
            excluded.append(relative)
        else:
            files[relative] = sha(path)
    return {"files": files, "digest": digest(files), "excluded_generated_files": excluded}


def source_snapshot(root):
    files, excluded = {}, []
    for name in ("crates", "python", "schemas", ".cargo"):
        result = tree(root / name, source=True)
        files.update({name + "/" + path: value for path, value in result["files"].items()})
        excluded.extend(name + "/" + path for path in result["excluded_generated_files"])
    for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "pyproject.toml", "uv.lock",
                 ".python-version", "README.md", "LICENSE"):
        files[name] = sha(root / name)
    configs = {}
    for parent in (root, *root.parents):
        for suffix in (".cargo/config", ".cargo/config.toml"):
            path = parent / suffix
            if path.is_file():
                configs[str(path)] = sha(path)
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    for suffix in ("config", "config.toml"):
        path = cargo_home / suffix
        if path.is_file():
            configs[str(path)] = sha(path)
    return {"files": files, "digest": digest(files), "cargo_configuration_files": configs,
            "excluded_generated_files": excluded,
            "scope": "All files in crates/python/schemas/.cargo plus named root inputs, including untracked sources; generated native files explicitly excluded"}


def dependency_snapshot(cargo):
    packages = {}
    for package in cargo["packages"]:
        if package["source"] is None:
            continue
        directory = Path(package["manifest_path"]).parent
        row = {"source": package["source"], "directory": str(directory), **tree(directory)}
        registry = directory.parent.name
        cache = directory.parents[2] / "cache" / registry / (directory.name + ".crate")
        row["cached_crate"] = {"path": str(cache), "sha256": sha(cache)} if cache.is_file() else None
        packages[package["id"]] = row
    return {"packages": packages, "digest": digest(packages),
            "scope": "All resolved non-workspace Cargo source files and available cached crate archives; source cache reused, compiled target empty"}


def wheel_files(wheel):
    with zipfile.ZipFile(wheel) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        return {name: hashlib.sha256(archive.read(name)).hexdigest()
                for name in sorted(archive.namelist()) if not name.endswith("/")}


def headroom():
    memory = psutil.virtual_memory()
    return {"observed_utc": datetime.now(timezone.utc).isoformat(), "cpu_count": os.cpu_count(),
            "cpu_percent_1s": psutil.cpu_percent(interval=1), "loadavg": os.getloadavg(),
            "memory_total_bytes": memory.total, "memory_available_bytes": memory.available,
            "swap": psutil.swap_memory()._asdict(), "disk_free_bytes": shutil.disk_usage(ROOT).free,
            "largest_processes": sorted(
                ({"pid": p.pid, "name": p.info["name"], "rss": p.info["memory_info"].rss}
                 for p in psutil.process_iter(["name", "memory_info"]) if p.info["memory_info"]),
                key=lambda row: row["rss"], reverse=True)[:8]}


def python_pins():
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    pending, seen = ["pllm.run", "pytest"], {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in seen:
            continue
        dist = metadata.distribution(name)
        seen[name] = dist.version
        for text in dist.requires or ():
            req = Requirement(text)
            if not req.marker or req.marker.evaluate({"extra": ""}):
                pending.append(req.name)
    assert not set(seen) & {"torch", "transformers", "mlx", "tenseal"}
    seen.pop("pllm-run")
    return dict(sorted(seen.items()))


def execute_release(output, dependency_cache=None):
    release = ROOT / "paper/artifacts/release"
    assert output.parent == release and not output.exists()
    assert Path(sys.executable).absolute() == ROOT / "target/frontier-sdk-env/bin/python"
    output.mkdir()
    scratch = release / ".work" / ("fresh-target-" + uuid.uuid4().hex)
    scratch.mkdir(parents=True)
    target = scratch / "cargo-target"
    target.mkdir()
    environment = scratch / "venv"
    env = {**os.environ, **THREADS, "CARGO_TARGET_DIR": str(target), "CARGO_BUILD_JOBS": "2",
           "CARGO_NET_OFFLINE": "true", "CARGO_INCREMENTAL": "0", "RUSTC_WRAPPER": "",
           "RUSTC_WORKSPACE_WRAPPER": "", "CARGO_BUILD_RUSTC_WRAPPER": "",
           "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER": "", "UV_OFFLINE": "1",
           "UV_PYTHON_DOWNLOADS": "never", "UV_LINK_MODE": "copy", "CARGO_TERM_COLOR": "never"}
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(name, None)
    command_records = []
    result = {"schema": "pllm.fresh_target_release.v1", "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(), "scratch": str(scratch),
              "cargo_target_dir": str(target), "fresh_environment": str(environment),
              "cache_distinction": "New empty Cargo target; installed registry source/archive cache reused offline. New venv; prebuilt Python wheel cache reused offline with copy installation.",
              "attestation": "Local recorded evidence, not independent/CI attestation or whole-OS hermetic reproducibility",
              "historical_execution_relabelled": False}

    def run(name, argv, *, cwd=ROOT, timeout=120, monitor=False):
        argv = list(map(str, argv))
        started = datetime.now(timezone.utc).isoformat()
        start = time.monotonic()
        samples, reason = [], None
        with (output / (name + ".log")).open("x") as stream:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            while process.poll() is None:
                if monitor:
                    children = psutil.Process(process.pid).children(recursive=True)
                    live = []
                    for child in children:
                        try:
                            live.append({"pid": child.pid, "name": child.name(), "rss": child.memory_info().rss})
                        except psutil.Error:
                            pass
                    available = psutil.virtual_memory().available
                    samples.append({"elapsed_seconds": time.monotonic() - start,
                                    "memory_available_bytes": available, "children": live})
                    if available < 2 << 30:
                        reason = "Live available memory below 2 GiB stop threshold"
                if time.monotonic() - start > timeout:
                    reason = f"Command exceeded {timeout}s timeout"
                if reason:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(1 if monitor else 0.05)
            code = process.wait()
        record = {"argv": argv, "cwd": str(cwd), "started_utc": started,
                  "elapsed_seconds": time.monotonic() - start, "exit_code": code,
                  "log": name + ".log", "log_sha256": sha(output / (name + ".log")), "stop_reason": reason}
        write(output / (name + ".command.json"), record)
        command_records.append(record)
        if monitor:
            write(output / (name + ".monitor.json"), samples)
        if code or reason:
            raise RuntimeError(f"{name} failed: exit={code}, reason={reason}; log retained")
        return (output / (name + ".log")).read_text()

    try:
        for source, name in ((Path(__file__), "driver.py"), (release / "replay.py", "replay.py"),
                             (release / "verify.py", "verify.py"),
                             (ROOT / "tests/test_paper_boundary_security.py", "frozen-boundary.py")):
            shutil.copyfile(source, output / name)
        write(output / "frozen-sources.json", {
            name: {"original": str(source), "sha256": sha(source)}
            for source, name in ((Path(__file__), "driver.py"), (release / "replay.py", "replay.py"),
                                 (release / "verify.py", "verify.py"),
                                 (ROOT / "tests/test_paper_boundary_security.py", "frozen-boundary.py"))})
        selected = {key: value for key, value in env.items() if key.startswith(("CARGO", "RUST", "PYO3", "MATURIN", "UV_"))
                    and not any(word in key for word in ("TOKEN", "PASSWORD", "SECRET", "KEY"))}
        selected.update({key: env.get(key) for key in ("PATH", "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "AR", "RANLIB",
                        "SDKROOT", "MACOSX_DEPLOYMENT_TARGET", "ARCHFLAGS", "SOURCE_DATE_EPOCH", "DEVELOPER_DIR", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")})
        write(output / "environment.json", {"build": selected, "thread_limits": THREADS,
              "platform": platform.platform(), "python": sys.version})
        write(output / "headroom-initial.json", headroom())
        commit = run("git-commit", ["git", "rev-parse", "HEAD"]).strip()
        result["source_commit"] = commit
        run("git-before", ["git", "status", "--short"])
        historical = tree(ROOT / "paper/artifacts/conformance")["files"]
        write(output / "historical-files-before.json", historical)
        identity_commands = {
            "rustc-version": ["rustc", "--version", "--verbose"],
            "cargo-version": ["cargo", "--version", "--verbose"],
            "rustup-toolchain": ["rustup", "show", "active-toolchain"],
            "rustc-path": ["rustup", "which", "rustc"], "cargo-path": ["rustup", "which", "cargo"],
            "rust-sysroot": ["rustc", "--print", "sysroot"],
            "maturin-version": [sys.executable, "-I", "-m", "maturin", "--version"],
            "uv-version": ["uv", "--version"], "clang-version": ["/usr/bin/xcrun", "clang", "--version"],
            "clang-path": ["/usr/bin/xcrun", "--find", "clang"], "linker-path": ["/usr/bin/xcrun", "--find", "ld"],
            "sdk-path": ["/usr/bin/xcrun", "--show-sdk-path"], "sdk-version": ["/usr/bin/xcrun", "--show-sdk-version"],
        }
        identities = {name: run(name, argv).strip() for name, argv in identity_commands.items()}
        tool_paths = {Path(sys.executable), Path(sys._base_executable), Path(shutil.which("uv")),
                      Path(shutil.which("rustup")), Path("/usr/bin/strip"), Path("/usr/bin/xcrun")}
        tool_paths.update(Path(identities[name]) for name in ("rustc-path", "cargo-path", "clang-path", "linker-path"))
        tool_paths.update(p for p in (Path(identities["rust-sysroot"]) / "lib").rglob("*") if p.is_file())
        libpython = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var("LDLIBRARY")
        if libpython.is_file():
            tool_paths.add(libpython)
        for name in ("SDKSettings.json", "SDKSettings.plist", "usr/lib/libSystem.tbd"):
            path = Path(identities["sdk-path"]) / name
            if path.is_file():
                tool_paths.add(path)
        for entry in metadata.distribution("maturin").files or ():
            path = Path(metadata.distribution("maturin").locate_file(entry))
            if path.is_file() and path.suffix != ".pyc":
                tool_paths.add(path)

        def tools_snapshot():
            files = {str(p): sha(p) for p in sorted(tool_paths)}
            return {"files": files, "digest": digest(files), "identities": identities,
                    "scope": "Python executable/libpython, Maturin distribution, Rust compiler/Cargo/rustup and full Rust lib tree, clang/linker/strip/xcrun/uv; SDK metadata and libSystem stub. OS and entire SDK/Clang resource tree not frozen."}

        cargo = json.loads(run("cargo-metadata", ["cargo", "metadata", "--format-version", "1", "--locked", "--offline"]))
        assert Path(cargo["target_directory"]) == target
        write(output / "cargo-metadata.json", cargo)
        before = source_snapshot(ROOT)
        deps_before = dependency_snapshot(cargo)
        tools_before = tools_snapshot()
        for name, value in (("sources-before", before), ("cargo-dependencies-before", deps_before), ("tools-before", tools_before)):
            write(output / (name + ".json"), value)
        with tarfile.open(output / "source.tar.gz", "w:gz", dereference=True) as archive:
            for name in sorted(before["files"]):
                archive.add(ROOT / name, arcname=name, recursive=False)
        config_dir = output / "cargo-configs"
        config_dir.mkdir()
        for index, path in enumerate(sorted(before["cargo_configuration_files"])):
            shutil.copyfile(path, config_dir / f"{index}.toml")
        write(output / "target-before.json", {"path": str(target), "created_by_this_run": True,
              "is_symlink": target.is_symlink(), "entries": sorted(p.name for p in target.iterdir()),
              "cargo_metadata_target_directory": cargo["target_directory"], "observed_utc": datetime.now(timezone.utc).isoformat()})
        assert not list(target.iterdir()) and not target.is_symlink()
        live = headroom()
        write(output / "headroom-prebuild.json", {**live, "admission": {"minimum_available_bytes": 4 << 30,
              "minimum_disk_free_bytes": 8 << 30, "maximum_cpu_percent": 90, "cargo_jobs": 2}})
        assert live["memory_available_bytes"] >= 4 << 30 and live["disk_free_bytes"] >= 8 << 30
        assert live["cpu_percent_1s"] <= 90
        print(json.dumps({"phase": "fresh-target-build", "target": str(target), "jobs": 2}), flush=True)
        build_log = run("wheel-build", [sys.executable, "-I", "-m", "maturin", "build", "--release", "--locked", "--offline",
                        "--verbose", "--interpreter", sys.executable, "--out", output / "wheels"], timeout=1800, monitor=True)
        after, deps_after, tools_after = source_snapshot(ROOT), dependency_snapshot(cargo), tools_snapshot()
        for name, value in (("sources-after", after), ("cargo-dependencies-after", deps_after), ("tools-after", tools_after)):
            write(output / (name + ".json"), value)
        assert before == after, "Source/config changed during compilation"
        assert deps_before == deps_after, "Cargo dependency cache changed during compilation"
        assert tools_before == tools_after, "Tool bytes changed during compilation"
        compilation = [line for line in build_log.splitlines() if line.lstrip().startswith("Compiling ")]
        rustc = [line for line in build_log.splitlines() if "Running `" in line and "rustc " in line]
        fresh = [line for line in build_log.splitlines() if line.lstrip().startswith("Fresh ")]
        native_crate = next(t["name"] for p in cargo["packages"] if p["name"] == "pllm-python"
                            for t in p["targets"] if "cdylib" in t["crate_types"])
        assert compilation and rustc and any(f"--crate-name {native_crate} " in line for line in rustc)
        assert not fresh, "Unexpected Cargo Fresh entries in empty target"
        write(output / "compilation-evidence.json", {"compiling_lines": compilation, "rustc_invocations": rustc,
              "cargo_fresh_lines": fresh, "native_crate": native_crate, "target_artifacts": tree(target),
              "note": "Empty directory observed immediately before build; wrappers disabled; verbose Rust invocations and resulting artifact hashes retained"})
        wheels = list((output / "wheels").glob("*.whl"))
        assert len(wheels) == 1
        wheel = wheels[0]
        payload = wheel_files(wheel)
        write(output / "wheel-files.json", payload)
        package = {name: value for name, value in payload.items() if name.startswith("pllm/")}
        native = {name: value for name, value in package.items() if name.endswith((".so", ".dylib", ".pyd"))}
        assert native
        old_payload = json.loads((ROOT / "paper/artifacts/conformance/2026-10-16-local-build-v1/wheel-files.json").read_text())
        old_native = json.loads((ROOT / "paper/artifacts/conformance/2026-10-16-installed-v4/provenance.json").read_text())["native_sha256"]
        write(output / "historical-comparison.json", {"historical_native_sha256": old_native,
              "fresh_native": native, "native_matches_historical": list(native.values()) == [old_native],
              "payload_differences_from_cached_build": {name: {"cached": old_payload.get(name), "fresh": payload.get(name)}
                  for name in sorted(old_payload.keys() | payload.keys()) if old_payload.get(name) != payload.get(name)},
              "meaning": "Historical executions retain their original wheel/native identities. These are new build and replay observations."})
        result.update({"wheel": wheel.relative_to(output).as_posix(), "wheel_sha256": sha(wheel),
                       "wheel_bytes": wheel.stat().st_size, "package_digest": digest(package), "native_files": native,
                       "compiled_crates": len(compilation), "rustc_invocations": len(rustc),
                       "sources_unchanged": True, "cargo_dependencies_unchanged": True, "tools_unchanged": True})
        pins = python_pins()
        write(output / "python-pins.json", {"packages": pins, "origin": "Active base requirement closure of installed pllm.run plus pytest, exact versions; excludes PLLM itself"})
        (output / "requirements.txt").write_text("".join(f"{name}=={version}\n" for name, version in pins.items()))
        install_source = []
        cache_receipt = None
        if dependency_cache is not None:
            dependency_cache = dependency_cache.resolve()
            cache_receipt = json.loads((dependency_cache / "receipt.json").read_text())
            assert cache_receipt["status"] == "passed"
            assert {name: row["version"] for name, row in cache_receipt["packages"].items()} == pins
            for row in cache_receipt["packages"].values():
                assert sha(row["wheel"]) == row["wheel_sha256"]
            shutil.copyfile(dependency_cache / "receipt.json", output / "dependency-cache-receipt.json")
            shutil.copyfile(dependency_cache / "driver.py", output / "dependency-cache-driver.py")
            shutil.copyfile(dependency_cache / "manifest.json", output / "dependency-cache-manifest.json")
            install_source = ["--no-index", "--find-links", cache_receipt["wheelhouse"]]
            result["cache_distinction"] = "New empty Cargo target; registry source/archive cache reused offline. New venv; existing extracted prebuilt dependency wheel-cache payloads RECORD-validated and rewrapped as new wheel ZIPs, installed offline by copying. Dependency ZIPs are rewraps, not original download bytes."
        print(json.dumps({"phase": "fresh-environment", "python_dependencies": len(pins)}), flush=True)
        assert not environment.exists()
        run("venv-create", [sys.executable, "-I", "-m", "venv", "--without-pip", environment])
        fresh_python = environment / "bin/python"
        empty = run("venv-empty", [fresh_python, "-I", "-c", "import importlib.metadata as m,importlib.util as u,json,sys,site; print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'distributions':[(d.metadata['Name'],d.version) for d in m.distributions()],'pllm_spec':str(u.find_spec('pllm')),'user_site':site.ENABLE_USER_SITE})); assert not list(m.distributions()); assert u.find_spec('pllm') is None; assert not site.ENABLE_USER_SITE"])
        write(output / "venv-empty.json", json.loads(empty))
        run("wheel-install", ["uv", "--no-config", "--offline", "pip", "install", "--python", fresh_python,
                              "--no-build", "--strict", "--link-mode", "copy", *install_source,
                              "--requirements", output / "requirements.txt", wheel], timeout=300)
        if cache_receipt is not None:
            checked = {}
            for name, row in cache_receipt["packages"].items():
                files = {relative: sha(Path(row["extracted_cache_path"]) / relative) for relative in row["files_before"]}
                assert files == row["files_before"] and sha(row["wheel"]) == row["wheel_sha256"]
                checked[name] = {"files": files, "wheel_sha256": sha(row["wheel"])}
            write(output / "dependency-cache-after-install.json", checked)
        run("dependency-check", ["uv", "--no-config", "--offline", "pip", "check", "--python", fresh_python])
        write(output / "venv-pyvenv-config.json", {"path": str(environment / "pyvenv.cfg"),
              "contents": (environment / "pyvenv.cfg").read_text(), "sha256": sha(environment / "pyvenv.cfg")})
        env["XDG_CACHE_HOME"] = str(scratch / "cache")
        env["TMPDIR"] = str(scratch / "tmp")
        Path(env["TMPDIR"]).mkdir()
        print(json.dumps({"phase": "fresh-installed-wheel-replay"}), flush=True)
        run("fresh-replay", [fresh_python, "-I", output / "replay.py", "--wheel", wheel,
                             "--environment", environment, "--frozen-boundary", output / "frozen-boundary.py",
                             "--output", output / "replay"], cwd=scratch, timeout=300, monitor=True)
        replay = json.loads((output / "replay/result.json").read_text())
        assert replay["status"] == "passed"
        result["replay_status"] = replay["status"]
        result["status"] = "passed"
    except Exception:
        result["status"] = "blocked"
        result["blocker"] = traceback.format_exc()
        print(result["blocker"], file=sys.stderr)
    finally:
        if (output / "historical-files-before.json").is_file():
            old = json.loads((output / "historical-files-before.json").read_text())
            current = {name: sha(ROOT / "paper/artifacts/conformance" / name)
                       if (ROOT / "paper/artifacts/conformance" / name).is_file() else None for name in old}
            write(output / "historical-files-after.json", current)
            result["previous_conformance_files_unchanged"] = old == current
            result["previous_conformance_files_checked"] = len(old)
        write(output / "headroom-after.json", headroom())
        result["commands"] = command_records
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(output / "receipt.json", result)
        files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
        write(output / "manifest.json", {"schema": "pllm.release_archive.v1", "files": files,
              "digest": digest(files), "note": "Integrity inventory; locally generated and not signed"})
    print(json.dumps({"status": result["status"], "archive": str(output)}), flush=True)
    return result["status"] == "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dependency-cache", type=Path, help="Optional RECORD-verified extracted-cache rewrap receipt directory")
    args = parser.parse_args()
    if not execute_release(args.output.absolute(), args.dependency_cache):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
