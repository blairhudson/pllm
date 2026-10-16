"""Verify frozen Listing 1 artifacts and their current manuscript binding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlparse
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def verify_files(folder, files):
    folder = folder.resolve()
    for name, expected in files.items():
        path = (folder / name).resolve()
        require(not Path(name).is_absolute() and path.is_relative_to(folder), "Archive path escapes its root")
        require(file_hash(path) == expected, f"Archive hash changed: {name}")


def verify_current_listing(folder, root=ROOT):
    # Current publication binding is separate from each version's frozen driver.
    text = (root / "paper/manuscript.md").read_text()
    listings = [code + "\n" for code in re.findall(r"```\s*python\n(.*?)\n```", text, re.S)
                if '"prepared-qwen"' in code]
    require(len(listings) == 1, "Published listing is not unique")
    require(listings[0] == (folder / "listing.py").read_text(), "Published listing drift")


def verify_functionality(result):
    require(result["gateway_status"] == 200, "Gateway failed")
    require(result["sdk_gateway_text_equal"] is True, "Output mismatch")
    for key in ("sdk_usage", "gateway_usage"):
        require(result[key]["input_tokens"] == 40, f"Input count: {key}")
        require(result[key]["output_tokens"] == 8, f"Output count: {key}")
    audit = result["sdk_audit"]
    require(audit["online_steps"] > 0, "Missing online execution")
    require(audit["plaintext_prompt_bytes_sent"] == 0, "Plaintext prompt sent")
    require(audit["plaintext_token_ids_sent"] == 0, "Plaintext token IDs sent")
    require(result["maximum_observed_swap_growth_bytes"] == 0, "Observed swap growth")


def verify_source_lock(lock, model, result):
    require(lock["schema"] == "pllm.model_source_lock.v1" and lock["kind"] == "huggingface", "Source lock schema")
    require(re.fullmatch(r"[0-9a-f]{40}", model["revision"]) is not None
            and lock["revision"] == lock["commit"] == model["revision"]
            and lock["repo_id"] == model["source"], "Source revision mismatch")
    rows = lock["files"]
    require(bool(rows) and len({row["path"] for row in rows}) == len(rows), "Source inventory missing/duplicate")
    for row in rows:
        require(isinstance(row["path"], str) and not Path(row["path"]).is_absolute()
                and ".." not in Path(row["path"]).parts and type(row["size"]) is int
                and row["size"] >= 0 and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "Invalid source file")
    checkpoint = hashlib.sha256(b"pllm.checkpoint.v1\0" + canonical({"files": rows})).hexdigest()
    locked = hashlib.sha256(b"pllm.model_source_lock.v1\0" + canonical(lock)).hexdigest()
    require(checkpoint == lock["checkpoint_digest"] == result["checkpoint_digest"], "Checkpoint digest")
    require(locked == result["source_lock_digest"], "Source lock digest")


def verify_identity(identity):
    require(identity["schema"] == "pllm.paper_listing_package.v1", "Package identity schema")
    require(identity["isolated"] == 1 and identity["no_user_site"] == 1
            and identity["user_site_enabled"] is False
            and identity["python_environment_scrubbed"] is True, "Import isolation missing")
    package = Path(identity["package_file"])
    require(package.is_absolute() and any(package.is_relative_to(Path(root))
            for root in identity["site_packages"]), "Package outside installed site-packages")
    require(Path(identity["native_file"]).is_relative_to(package.parent), "Native outside package")
    require("__init__.py" in identity["sdk_sources"], "SDK inventory missing")
    for value in (*identity["sdk_sources"].values(), identity["python_sha256"], identity["native_sha256"]):
        require(re.fullmatch(r"[0-9a-f]{64}", value) is not None, "Invalid identity digest")


def verify_v2(folder, result):
    required = {"driver.py", "listing.py", "experiment.json", "admission.json", "build.json",
                "source-lock-before.json", "source-lock-after.json",
                "parent-identity.json", "parent-identity-after.json"}
    require(required <= result["files"].keys(), "Missing v2 provenance files")
    load = lambda name: json.loads((folder / name).read_text())
    before, after = load("source-lock-before.json"), load("source-lock-after.json")
    verify_source_lock(before, load("experiment.json")["pipeline"]["model"], result)
    require(before == after and result["source_unchanged"] is True, "Source changed during execution")
    identity = load("parent-identity.json")
    verify_identity(identity)
    require(identity == load("parent-identity-after.json")
            and result["installed_package_unchanged"] is True, "Installed package changed")
    require(result["installed_package"] == identity["package_file"], "Installed package path mismatch")
    build = load("build.json")
    require(hashlib.sha256(canonical(build["attributes"])).hexdigest() == build["digest"], "BuildRef digest")
    attributes = build["attributes"]
    require(attributes["sdk_sources"] == identity["sdk_sources"]
            and attributes["native_sha256"] == identity["native_sha256"]
            and attributes["python_binary"] == identity["python_sha256"], "Build/parent identity mismatch")
    admission = load("admission.json")
    require(admission["admitted"] is True and admission["selected_backend"] == "native", "Native admission failed")
    children = result["provider_children"]
    if result["status"] == "admitted_not_executed":
        require(children == [], "Preflight launched providers")
        return
    require(len(children) == 2 and {child["role"] for child in children} == {"inference", "preparation"}
            and len({child["pid"] for child in children}) == 2, "Provider children missing/duplicate")
    for child in children:
        name = f"provider-{child['role']}.json"
        require(child["identity_file"] == name and name in result["files"], "Child receipt missing")
        require(type(child["returncode"]) is int, "Provider was not reaped")
        receipt = load(name)
        require(receipt["schema"] == "pllm.paper_listing_child.v1" and receipt["status"] == "identified"
                and receipt["phase"] == "before_module_execution", "Child did not identify before execution")
        require(receipt["pid"] == child["pid"] and receipt["parent_pid"] == result["pid"]
                and receipt["role"] == child["role"], "Child process binding mismatch")
        require(receipt["identity"] == identity, "Actual child import identity mismatch")
        require(receipt["driver_sha256"] == result["driver_sha256"]
                and receipt["expected_parent_sha256"] == result["files"]["parent-identity.json"], "Child provenance binding")
    require(result["gateway_output_text_sha256"] == result["output_text_sha256"], "Output digest mismatch")


def verify_run(folder, expected_status):
    folder = folder.resolve()
    result = json.loads((folder / "result.json").read_text())
    require(result["schema"] in {"pllm.paper_listing_replay.v1", "pllm.paper_listing_replay.v2"}, "Result schema")
    require(result["status"] == expected_status, f"Unexpected outcome: {folder.name}")
    verify_files(folder, result["files"])
    actual = {p.name for p in folder.iterdir() if p.is_file() and p.name != "result.json"}
    require(actual == set(result["files"]), "Incomplete top-level file inventory")
    # Never compare a historical snapshot to the changing development driver.
    require("driver.py" in result["files"]
            and file_hash(folder / "driver.py") == result["driver_sha256"], "Driver hash")
    if expected_status == "rejected":
        require(result["schema"] == "pllm.paper_listing_replay.v2", "Rejected run requires v2 diagnostics")
        for key in ("failure_phase", "failure_reason", "failure_type"):
            require(isinstance(result.get(key), str) and bool(result[key]), f"Missing {key}")
        return result
    require("listing.py" in result["files"]
            and file_hash(folder / "listing.py") == result["listing_sha256"], "Listing hash")
    if expected_status == "completed":
        verify_functionality(result)
    if result["schema"] == "pllm.paper_listing_replay.v2":
        verify_v2(folder, result)
    return result


def relative_file(root, name):
    path = (root / name).resolve()
    require(not Path(name).is_absolute() and path.is_relative_to(root.resolve()), "Linked path escapes its root")
    return path


def recorded_command(receipt, log):
    matches = [command for command in receipt["commands"] if command["log"] == log]
    require(len(matches) == 1, f"Missing/duplicate recorded command: {log}")
    return matches[0]


def verify_release_linkage(root, links, identity):
    """Join archived observations; never inspect a live environment or run code.

    Full build/tool/dependency internals remain the release verifier's scope.
    Qwen observed only Python, native and the .py/.json SDK projection below.
    """
    require(set(links) == {"build_receipt", "build_manifest", "environment_receipt", "installed_provenance"},
            "Release metadata link set")
    paths, documents = {}, {}
    for role, link in links.items():
        path = relative_file(root, link["path"])
        require(file_hash(path) == link["sha256"], f"Linked metadata hash changed: {role}")
        paths[role], documents[role] = path, json.loads(path.read_text())
    build, manifest = documents["build_receipt"], documents["build_manifest"]
    continued, provenance = documents["environment_receipt"], documents["installed_provenance"]
    require(paths["build_receipt"].name == "receipt.json"
            and paths["build_manifest"] == paths["build_receipt"].with_name("manifest.json")
            and paths["environment_receipt"].name == "receipt.json"
            and paths["installed_provenance"] == paths["environment_receipt"].parent / "replay/provenance.json",
            "Release metadata layout")
    require(build["schema"] == "pllm.fresh_target_release.v1"
            and build["status"] == "blocked" and build["historical_execution_relabelled"] is False,
            "Original build history relabelled")
    failures = [command for command in build["commands"] if command["exit_code"] != 0]
    require(len(failures) == 1 and failures[0]["log"] == "wheel-install.log", "Original install blocker changed")
    compiled = recorded_command(build, "wheel-build.log")
    require(compiled["exit_code"] == 0 and compiled["stop_reason"] is None
            and {"--release", "--locked", "--offline"} <= set(compiled["argv"])
            and build["compiled_crates"] > 0 and build["rustc_invocations"] > 0, "Compilation not recorded successful")
    require(manifest["schema"] == "pllm.release_archive.v1"
            and hashlib.sha256(canonical(manifest["files"])).hexdigest() == manifest["digest"], "Build manifest digest")
    require(manifest["files"]["receipt.json"] == links["build_receipt"]["sha256"], "Build receipt/manifest binding")
    require(continued["schema"] == "pllm.fresh_environment_continuation.v1"
            and continued["status"] == "passed" and continued["no_build_performed"] is True
            and continued["original_build_archive_unchanged"] is True and continued["tools_unchanged"] is True,
            "Fresh environment continuation failed")
    require(continued["build_manifest_sha256"] == links["build_manifest"]["sha256"], "Continuation build manifest binding")
    require(all(command["exit_code"] == 0 and command["stop_reason"] is None
                for command in continued["commands"]), "Continuation command failed")
    installed = recorded_command(continued, "wheel-install.log")
    historical_build = Path(continued["build_archive"])
    historical_root = Path(compiled["cwd"])
    require(historical_build == historical_root / Path(links["build_receipt"]["path"]).parent,
            "Continuation build archive path")
    historical_wheel = historical_build / build["wheel"]
    require(str(historical_wheel) in installed["argv"]
            and {"--offline", "--no-build", "--strict", "--python"} <= set(installed["argv"])
            and installed["argv"][installed["argv"].index("--python") + 1] == provenance["python"],
            "Continuation installed wheel/interpreter binding")
    direct = urlparse(provenance["direct_url"]["url"])
    require(direct.scheme == "file" and direct.netloc == ""
            and Path(unquote(direct.path)) == historical_wheel, "Installed direct wheel URL")
    wheel = relative_file(paths["build_receipt"].parent, build["wheel"])
    require(file_hash(wheel) == build["wheel_sha256"] == continued["wheel_sha256"]
            == provenance["wheel_sha256"] == manifest["files"][build["wheel"]]
            and wheel.stat().st_size == build["wheel_bytes"], "Linked wheel identity")
    with zipfile.ZipFile(wheel) as zipped:
        require(len(zipped.namelist()) == len(set(zipped.namelist())), "Duplicate wheel entries")
        package = {name: hashlib.sha256(zipped.read(name)).hexdigest()
                   for name in zipped.namelist() if name.startswith("pllm/") and not name.endswith("/")}
    require(package == provenance["package_files"]
            and hashlib.sha256(canonical(package)).hexdigest() == build["package_digest"] == provenance["package_digest"],
            "Fresh release package inventory/digest")
    native = {name: value for name, value in package.items() if name.endswith((".so", ".pyd", ".dylib"))}
    require(native == build["native_files"] and bool(native), "Release native inventory")
    observed_native = "pllm/" + Path(identity["native_file"]).name
    require(native.get(observed_native) == provenance["native_sha256"] == identity["native_sha256"],
            "Qwen/release native mismatch")
    sdk = {name.removeprefix("pllm/"): value for name, value in package.items()
           if Path(name).suffix in {".py", ".json"}}
    require(identity["sdk_sources"] == sdk, "Qwen/release SDK inventory mismatch")
    require(identity["python_sha256"] == provenance["python_sha256"], "Qwen/release Python hash mismatch")
    require(identity["python_executable"] == provenance["python"]
            and Path(provenance["python"]).is_relative_to(Path(continued["fresh_environment"]))
            and provenance["prefix"] == continued["fresh_environment"] != provenance["base_prefix"]
            and identity["package_file"] == str(Path(provenance["package_path"]) / "__init__.py")
            and identity["native_file"] == provenance["native_path"]
            and Path(provenance["package_path"]).is_relative_to(Path(provenance["prefix"])),
            "Qwen/fresh environment path binding")
    require(provenance["isolated_flag"] == 1 and provenance["user_site_enabled"] is False,
            "Fresh environment isolation")
    return {"linked_metadata": len(links), "sdk_sources": len(sdk), "release_package_files": len(package),
            "historical_root": historical_root}


def verify_archives(here=HERE, root=ROOT):
    index = json.loads((here / "index.json").read_text())
    require(index["schema"] == "pllm.paper_listing_archive.v1", "Archive schema")
    verify_files(here, index["files"])
    for attempt, status in (("preflight-v1", "admitted_not_executed"), ("qwen-v1", "completed")):
        verify_run(here / attempt, status)
    verify_current_listing(here / "qwen-v1", root)
    fresh = json.loads((here / "index-v2.json").read_text())
    require(fresh["schema"] == "pllm.paper_listing_archive.v2", "Fresh archive schema")
    require(fresh["prior_archive"]["path"] == "index.json"
            and fresh["prior_archive"]["sha256"] == file_hash(here / "index.json"), "Original index changed")
    name = fresh["archive"]
    require(Path(name).name == name and name not in {".", "..", "qwen-v1", "preflight-v1"}, "Fresh archive name")
    folder = relative_file(here, name)
    verify_files(here, fresh["files"])
    result = verify_run(folder, "completed")
    require(result["schema"] == "pllm.paper_listing_replay.v2", "Fresh run schema")
    expected = {f"{name}/{file}" for file in result["files"]} | {f"{name}/result.json"}
    require(set(fresh["files"]) == expected, "Fresh archive index coverage")
    require({p.name for p in folder.iterdir()} == set(result["files"]) | {"result.json"}
            and all(p.is_file() and not p.is_symlink() for p in folder.iterdir()), "Unexpected operational archive entries")
    verify_current_listing(folder, root)
    identity = json.loads((folder / "parent-identity.json").read_text())
    linkage = verify_release_linkage(root, fresh["release"], identity)
    invocation = fresh["accepted_invocation"]
    require(invocation[1:3] == ["-I", "-B"] and invocation[3:] == result["argv"]
            and not Path(invocation[0]).is_absolute()
            and linkage["historical_root"] / invocation[0] == Path(identity["python_executable"])
            and result["argv"][-2:] == ["--output", fresh["original_run"]], "Accepted invocation binding")
    return {"v1_files": len(index["files"]), "v2_files": len(fresh["files"]), **linkage}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=[], help="Also verify a fresh/versioned run directory")
    parser.add_argument("--expect-status", choices=("completed", "admitted_not_executed", "rejected"), default="completed")
    args = parser.parse_args()
    summary = verify_archives(HERE, ROOT)
    print(f"Verified {summary['v1_files']} original + {summary['v2_files']} fresh frozen files, current Listing 1, "
          f"and {summary['linked_metadata']} linked release metadata files.")
    print(f"Qwen-observed {summary['sdk_sources']} SDK sources, native and Python match fresh release; "
          f"{summary['release_package_files']}-file full package inventory belongs to earlier installation evidence.")
    for folder in args.run:
        result = verify_run(folder, args.expect_status)
        if result["status"] != "rejected":
            verify_current_listing(folder)
        print(f"Verified {folder}: {result['status']} ({result['schema']}); original drivers retained.")


if __name__ == "__main__":
    main()
