"""Replay a fully recorded fresh-target wheel after an earlier install blocker.

The original build receipt and all failed attempts remain byte-identical. This
creates a separately identified execution with a new venv and a hash link to
the completed build's original manifest. No compilation is performed here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

import psutil


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--dependency-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build, cache, output = args.build.resolve(), args.dependency_cache.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    release = Path(__file__).resolve().parent
    scratch = release / ".work" / ("fresh-replay-" + uuid.uuid4().hex)
    scratch.mkdir(parents=True)
    environment = scratch / "venv"
    manifest = json.loads((build / "manifest.json").read_text())
    before_build = {name: sha(build / name) for name in manifest["files"]}
    assert before_build == manifest["files"] and digest(before_build) == manifest["digest"]
    original = json.loads((build / "receipt.json").read_text())
    assert json.loads((build / "wheel-build.command.json").read_text())["exit_code"] == 0
    for prefix in ("sources", "cargo-dependencies", "tools"):
        assert (build / (prefix + "-before.json")).read_bytes() == (build / (prefix + "-after.json")).read_bytes()
    assert original["compiled_crates"] > 0 and original["rustc_invocations"] > 0
    wheel = build / original["wheel"]
    assert sha(wheel) == original["wheel_sha256"]
    cache_record = json.loads((cache / "receipt.json").read_text())
    assert cache_record["status"] == "passed"
    for source, name in ((Path(__file__), "driver.py"), (release / "verify.py", "verify.py"),
                         (build / "replay.py", "replay.py"), (build / "frozen-boundary.py", "frozen-boundary.py"),
                         (build / "python-pins.json", "python-pins.json"), (build / "requirements.txt", "requirements.txt"),
                         (cache / "receipt.json", "dependency-cache-receipt.json"),
                         (cache / "driver.py", "dependency-cache-driver.py"),
                         (cache / "manifest.json", "dependency-cache-manifest.json")):
        shutil.copyfile(source, output / name)
    write(output / "build-origin.json", {"archive": str(build), "manifest_sha256": sha(build / "manifest.json"),
          "receipt_sha256": sha(build / "receipt.json"), "original_status": original["status"],
          "original_blocker": original.get("blocker"), "wheel_sha256": sha(wheel),
          "meaning": "New replay of this completed fresh-target build; original failed installation and historical execution labels remain unchanged"})
    tool_paths = [Path(sys.executable), Path(shutil.which("uv"))]
    tools_before = {str(p): sha(p) for p in tool_paths}
    write(output / "tools-before.json", tools_before)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "VECLIB_MAXIMUM_THREADS": "1", "RAYON_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
           "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never", "UV_CONCURRENT_INSTALLS": "2",
           "UV_CONCURRENT_BUILDS": "1", "UV_CONCURRENT_DOWNLOADS": "1", "UV_LINK_MODE": "copy",
           "XDG_CACHE_HOME": str(scratch / "cache"), "TMPDIR": str(scratch / "tmp")}
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(name, None)
    Path(env["TMPDIR"]).mkdir()
    write(output / "environment.json", {k: env.get(k) for k in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS",
        "TOKENIZERS_PARALLELISM", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        "UV_OFFLINE", "UV_CONCURRENT_INSTALLS", "UV_CONCURRENT_BUILDS", "UV_CONCURRENT_DOWNLOADS", "UV_LINK_MODE",
        "XDG_CACHE_HOME", "TMPDIR", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")})
    commands = []

    def run(name, argv, timeout=300):
        argv = list(map(str, argv))
        start = time.monotonic()
        utc = datetime.now(timezone.utc).isoformat()
        completed = subprocess.run(argv, cwd=scratch, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False)
        (output / (name + ".log")).write_text(completed.stdout)
        record = {"argv": argv, "cwd": str(scratch), "started_utc": utc, "exit_code": completed.returncode,
                  "elapsed_seconds": time.monotonic() - start, "log": name + ".log",
                  "log_sha256": sha(output / (name + ".log")), "stop_reason": None}
        write(output / (name + ".command.json"), record)
        commands.append(record)
        if completed.returncode:
            raise RuntimeError(f"{name} failed; see {name}.log")
        return completed.stdout

    result = {"schema": "pllm.fresh_environment_continuation.v1", "status": "running",
              "fresh_environment": str(environment), "build_archive": str(build), "wheel_sha256": sha(wheel),
              "build_manifest_sha256": sha(build / "manifest.json"), "no_build_performed": True}
    try:
        headroom = {"observed_utc": datetime.now(timezone.utc).isoformat(),
                    "memory_available_bytes": psutil.virtual_memory().available,
                    "cpu_percent_1s": psutil.cpu_percent(interval=1), "disk_free_bytes": shutil.disk_usage(scratch).free}
        write(output / "headroom-before.json", headroom)
        assert headroom["memory_available_bytes"] > 4 << 30 and headroom["disk_free_bytes"] > 2 << 30
        run("uv-version", ["uv", "--version"])
        assert not environment.exists()
        run("venv-create", [sys.executable, "-I", "-B", "-m", "venv", "--without-pip", environment])
        python = environment / "bin/python"
        empty = run("venv-empty", [python, "-I", "-B", "-c", "import importlib.metadata as m,importlib.util as u,json,sys,site; print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'distributions':[(d.metadata['Name'],d.version) for d in m.distributions()],'pllm_spec':str(u.find_spec('pllm')),'user_site':site.ENABLE_USER_SITE})); assert not list(m.distributions()); assert u.find_spec('pllm') is None; assert not site.ENABLE_USER_SITE"])
        write(output / "venv-empty.json", json.loads(empty))
        before_cache = {}
        for name, row in cache_record["packages"].items():
            assert sha(row["wheel"]) == row["wheel_sha256"]
            files = {p: sha(Path(row["extracted_cache_path"]) / p) for p in row["files_before"]}
            assert files == row["files_before"]
            before_cache[name] = {"files": files, "wheel_sha256": sha(row["wheel"])}
        write(output / "dependency-cache-before-install.json", before_cache)
        run("wheel-install", ["uv", "--no-config", "--offline", "pip", "install", "--python", python,
                              "--no-build", "--strict", "--link-mode", "copy", "--no-index", "--find-links",
                              cache_record["wheelhouse"], "--requirements", output / "requirements.txt", wheel])
        run("dependency-check", ["uv", "--no-config", "--offline", "pip", "check", "--python", python])
        after_cache = {name: {"files": {p: sha(Path(row["extracted_cache_path"]) / p) for p in row["files_before"]},
                             "wheel_sha256": sha(row["wheel"])} for name, row in cache_record["packages"].items()}
        write(output / "dependency-cache-after-install.json", after_cache)
        assert before_cache == after_cache
        write(output / "venv-pyvenv-config.json", {"path": str(environment / "pyvenv.cfg"),
              "contents": (environment / "pyvenv.cfg").read_text(), "sha256": sha(environment / "pyvenv.cfg")})
        run("fresh-replay", [python, "-I", "-B", output / "replay.py", "--wheel", wheel,
                             "--environment", environment, "--frozen-boundary", output / "frozen-boundary.py",
                             "--output", output / "replay"])
        assert json.loads((output / "replay/result.json").read_text())["status"] == "passed"
        result["status"] = "passed"
    except Exception:
        result["status"] = "blocked"
        result["blocker"] = traceback.format_exc()
        print(result["blocker"], file=sys.stderr)
    finally:
        tools_after = {str(p): sha(p) for p in tool_paths}
        write(output / "tools-after.json", tools_after)
        result["tools_unchanged"] = tools_before == tools_after
        result["original_build_archive_unchanged"] = before_build == {name: sha(build / name) for name in before_build}
        result["commands"] = commands
        write(output / "receipt.json", result)
        files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
        write(output / "manifest.json", {"files": files, "digest": digest(files)})
    print(json.dumps({"status": result["status"], "output": str(output)}), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
