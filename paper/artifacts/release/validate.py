"""Record verifier results and adversarial checks without changing any archive."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import uuid
import zipfile


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(path):
    return {p.relative_to(path).as_posix(): sha(p) for p in sorted(path.rglob("*")) if p.is_file()}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive, output = args.archive.resolve(), args.output.resolve()
    execution = args.replay.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / "driver.py")
    verifier = Path(__file__).resolve().with_name("verify.py")
    shutil.copyfile(verifier, output / "verifier.py")
    verify = runpy.run_path(str(verifier))["verify"]
    before = {"build": snapshot(archive), "replay": snapshot(execution)}
    write(output / "archive-before.json", before)
    receipt = json.loads((archive / "receipt.json").read_text())
    compilation = json.loads((archive / "compilation-evidence.json").read_text())
    target_native = {name: value for name, value in compilation["target_artifacts"]["files"].items()
                     if name.endswith(("lib_native.dylib", "lib_native.so"))}
    assert target_native and set(receipt["native_files"].values()) == set(target_native.values()), "Compiled target native does not match wheel"
    commands = []
    for name, flags in (("archive", []), ("live", ["--live", "--allow-catalog-drift"])):
        argv = [sys.executable, "-I", "-B", str(verifier), str(archive), "--replay", str(execution), *flags]
        command = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        (output / (name + ".log")).write_text(command.stdout)
        commands.append({"argv": argv, "exit_code": command.returncode,
                         "log": name + ".log", "log_sha256": sha(output / (name + ".log"))})
        if command.returncode:
            write(output / "failed-command.json", commands[-1])
            raise RuntimeError(command.stdout)
    scratch = archive.parent / ".work" / ("verifier-negatives-" + uuid.uuid4().hex)
    cases = []
    def reseal(directory):
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["files"] = {name: sha(directory / name) for name in manifest["files"]}
        manifest["digest"] = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        path.write_text(json.dumps(manifest))

    for mode in ("unsealed-log-corruption", "resealed-nonempty-target", "resealed-trajectory-corruption"):
        copied = scratch / mode / "build"
        copied_execution = scratch / mode / "replay"
        shutil.copytree(archive, copied)
        shutil.copytree(execution, copied_execution)
        if mode == "unsealed-log-corruption":
            with (copied / "wheel-build.log").open("a") as stream:
                stream.write("\nDeliberate verifier negative-test corruption\n")
        elif mode == "resealed-nonempty-target":
            path = copied / "target-before.json"
            value = json.loads(path.read_text())
            value["entries"] = ["unexpected-existing-artifact"]
            path.write_text(json.dumps(value))
        else:
            path = copied_execution / "replay/trajectory-qwen2-7.npz"
            with zipfile.ZipFile(path) as source:
                payload = {name: source.read(name) for name in source.namelist()}
            name = sorted(payload)[0]
            data = bytearray(payload[name])
            data[-1] ^= 1
            payload[name] = bytes(data)
            with zipfile.ZipFile(path, "w") as target:
                for name, data in payload.items():
                    target.writestr(name, data)
        if mode == "resealed-nonempty-target":
            reseal(copied)
            for filename, key in (("receipt.json", "build_manifest_sha256"), ("build-origin.json", "manifest_sha256")):
                path = copied_execution / filename
                document = json.loads(path.read_text())
                document[key] = sha(copied / "manifest.json")
                path.write_text(json.dumps(document))
            reseal(copied_execution)
        elif mode == "resealed-trajectory-corruption":
            reseal(copied_execution)
        try:
            verify(copied, replay_archive=copied_execution)
        except ValueError as error:
            expected = {"unsealed-log-corruption": "Archive hash: wheel-build.log",
                        "resealed-nonempty-target": "Target not recorded empty",
                        "resealed-trajectory-corruption": "Archived trajectory bytes differ"}[mode]
            assert str(error).startswith(expected), (mode, str(error))
            cases.append({"name": mode, "status": "rejected", "message": str(error)})
        else:
            raise RuntimeError(f"Verifier accepted deliberate corruption: {mode}")
    after = {"build": snapshot(archive), "replay": snapshot(execution)}
    write(output / "archive-after.json", after)
    assert before == after, "Read-only verifier changed original archive"
    write(output / "result.json", {"schema": "pllm.release_verifier_validation.v1", "status": "passed",
          "observed_utc": datetime.now(timezone.utc).isoformat(), "archive": str(archive),
          "replay_archive": str(execution), "replay_manifest_sha256": sha(execution / "manifest.json"),
          "archive_manifest_sha256": sha(archive / "manifest.json"), "verifier_sha256": sha(verifier),
          "commands": commands, "negative_checks": cases, "original_archive_byte_identical": True,
          "compiled_target_native_matches_wheel": True, "compiled_target_native": target_native,
          "scratch": str(scratch)})
    files = snapshot(output)
    write(output / "manifest.json", {"files": files,
          "digest": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    print(json.dumps({"status": "passed", "negative_checks": len(cases), "output": str(output)}))


if __name__ == "__main__":
    main()
