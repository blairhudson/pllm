"""Verify the frozen conformance archive with Python's standard library only."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import zipfile


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def junit_totals(directory):
    totals = dict.fromkeys(("passed", "failed", "errors", "skipped"), 0)
    for path in sorted(directory.glob("*.xml")):
        for node in ET.parse(path).iter("testcase"):
            status = "errors" if node.find("error") is not None else (
                "failed" if node.find("failure") is not None else (
                    "skipped" if node.find("skipped") is not None else "passed"))
            totals[status] += 1
    return totals


def verify_identity_record(record, expected):
    package = Path(expected["package_path"])
    return (record["status"] == "passed" and not record["errors"]
            and all(record[field] == expected[field] for field in
                    ("package_path", "python", "native_sha256", "wheel_package_files", "wheel_sha256"))
            and all(record["flags"].values()) and not record["user_site_enabled"]
            and not record["remaining_import_environment_keys"] and not record["extra_package_files"]
            and Path(record["native_path"]) == package / "_native.abi3.so"
            and record["loaded_pllm_module_origins"].get("pllm") == str(package / "__init__.py")
            and record["loaded_pllm_module_origins"].get("pllm._native") == record["native_path"]
            and all(Path(origin).is_relative_to(package)
                    for origin in record["loaded_pllm_module_origins"].values()))


def verify_child_identities(directory):
    expected = json.loads((directory / "expected.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    launches = sorted(directory.glob("*.launch.json"))
    if summary["captured_python_processes"] != len(launches):
        raise SystemExit("child identity count differs from launch receipts")
    errors = []
    entries, roles = [], Counter()
    for launch_path in launches:
        launch = json.loads(launch_path.read_text())
        key = launch_path.name.removesuffix(".launch.json")
        if launch["expected_sha256"] != hashlib.sha256((directory / "expected.json").read_bytes()).hexdigest():
            raise SystemExit("child expected-identity lock differs")
        if launch["helper_sha256"] != hashlib.sha256((directory / "python-child-isolation.py").read_bytes()).hexdigest():
            raise SystemExit("child isolation bootstrap lock differs")
        records = []
        phases = ("spawn", "start", "end", "exit") if launch["schema"] == "pllm.paper_python_launch.v2" else ("spawn", "start", "end")
        by_phase = {}
        for phase in phases:
            path = directory / f"{key}.{phase}.json"
            if not path.is_file():
                errors.append(f"missing {key}.{phase}")
                continue
            record = json.loads(path.read_text())
            records.append(record)
            by_phase[phase] = record
            if phase in ("start", "end") and not verify_identity_record(record, expected):
                errors.append(f"invalid identity {key}.{phase}")
            if phase in ("start", "end") and record["parent_pid"] != launch["launcher_pid"]:
                errors.append(f"child parent differs: {key}.{phase}")
        if len(records) == len(phases) and len({record["pid"] for record in records}) != 1:
            errors.append(f"PID differs: {key}")
        if "exit" in by_phase and "end" in by_phase:
            end, exited = by_phase["end"], by_phase["exit"]
            if exited["observed_by_pid"] != launch["launcher_pid"]:
                errors.append(f"exit observer differs: {key}")
            termination = end["termination"]
            if termination["kind"] == "default-signal":
                if exited["returncode"] != -termination["signal"]:
                    errors.append(f"signal exit changed: {key}")
            elif end["entry_exit_code"] is not None and exited["returncode"] != (end["entry_exit_code"] & 255):
                errors.append(f"entry exit changed: {key}")
        if by_phase.get("spawn", {}).get("start_new_session"):
            for phase in ("start", "end"):
                if phase in by_phase and (by_phase[phase]["process_group"] != by_phase[phase]["pid"]
                        or by_phase[phase]["session_id"] != by_phase[phase]["pid"]):
                    errors.append(f"owned child session/group differs: {key}.{phase}")
        if (directory / f"{key}.end-error.json").exists():
            errors.append(f"end capture failed: {key}")
        if "start" in by_phase:
            entries.append({"entry": launch["entry"], "key": key, "pid": by_phase["start"]["pid"]})
            if launch["entry"].get("role"):
                roles[launch["entry"]["role"]] += 1
    for phase in ("start", "end"):
        controller = json.loads((directory / f"controller-{phase}.json").read_text())
        if not verify_identity_record(controller, expected):
            errors.append(f"controller {phase} differs")
    if summary["entries"] != entries or summary["roles"] != dict(roles):
        errors.append("child entry/role counts differ from launch and start receipts")
    if summary["expected_sha256"] != sha(directory / "expected.json"):
        errors.append("summary expected-identity lock differs")
    if summary["status"] == "passed" and (errors or summary["failures"]):
        raise SystemExit("claimed complete child-identity coverage is inconsistent")
    return summary


def float32_array(archive, name):
    """Read the bounded saved float32 NPY arrays; never import a model or NumPy."""
    data = archive.read(name + ".npy")
    if data[:8] not in (b"\x93NUMPY\x01\x00", b"\x93NUMPY\x02\x00"):
        raise SystemExit("unexpected retained array format")
    header_start = 10 if data[6] == 1 else 12
    header_size = int.from_bytes(data[8:header_start], "little")
    header = ast.literal_eval(data[header_start:header_start + header_size].decode())
    payload = data[header_start + header_size:]
    if header["descr"] != "<f4" or header["fortran_order"] or len(payload) != 4 * math.prod(header["shape"]):
        raise SystemExit("retained array dtype/shape/layout differs")
    return header["shape"], [value[0] for value in struct.iter_unpack("<f", payload)]


def verify_generated_arrays(directory):
    rows = json.loads((directory / "generated-conformance.json").read_text())
    for row in rows:
        family, checked = row["family"], 0
        with zipfile.ZipFile(directory / f"candidate-{family}.npz") as candidate, \
                zipfile.ZipFile(directory / f"reference-{family}.npz") as reference:
            for key, expected_error in row["upstream_max_absolute_errors"].items():
                shape, observed = float32_array(candidate, key)
                reference_shape, target = float32_array(reference, key)
                if key.startswith("logits_"):
                    target, reference_shape = target[-shape[-1]:], (1, reference_shape[-1])
                if shape != reference_shape:
                    raise SystemExit("retained candidate/reference shape differs")
                # Match the archived float32 subtraction, not float64 subtraction.
                error = max(abs(struct.unpack("<f", struct.pack("<f", a - b))[0])
                            for a, b in zip(observed, target, strict=True))
                if error != expected_error or not error <= row["upstream_fp32_atol"]:
                    raise SystemExit("retained FP32 comparison differs")
                checked += len(observed)
        if checked != row["exact_tiled_vs_untiled_values"] or not row["tiled_vs_untiled_bit_exact"]:
            raise SystemExit("generated comparison count differs")
    # Untiled arrays were checked during execution, but were not separately saved.
    return sum(row["exact_tiled_vs_untiled_values"] for row in rows)


def verify_build_link(root, index, item):
    link = json.loads((root / item["directory"] / "receipt.json").read_text())
    receipt_path = (root / link["build_receipt_path"]).resolve()
    if not receipt_path.is_relative_to(root.parent) or sha(receipt_path) != link["build_receipt_sha256"]:
        raise SystemExit("current clean-build receipt lock differs")
    receipt = json.loads(receipt_path.read_text())
    if (receipt["status"] != link["build_receipt_original_status"] or receipt["runtime_execution"]
            or receipt["incremental_cache_reused"] or receipt["source_file_changes_during_build"]
            or receipt["cargo_configuration_changes_during_build"] or not receipt["dependency_source_trees_unchanged"]):
        raise SystemExit("clean-build receipt scope/stability differs")
    for relative, digest in receipt["artifacts"].items():
        path = (receipt_path.parent / relative).resolve()
        if not path.is_relative_to(receipt_path.parent) or sha(path) != digest:
            raise SystemExit("linked clean-build artifact differs")
    wheel = (root / link["wheel_path"]).resolve()
    if sha(wheel) != link["wheel_sha256"] or receipt["wheel_sha256"] != link["wheel_sha256"]:
        raise SystemExit("linked clean-build wheel differs")
    with zipfile.ZipFile(wheel) as archive:
        files = {name: hashlib.sha256(archive.read(name)).hexdigest()
                 for name in archive.namelist() if not name.endswith("/")}
    if files != json.loads((receipt_path.parent / "wheel-files.json").read_text()):
        raise SystemExit("clean-build wheel payload inventory differs")
    package = {name: digest for name, digest in files.items() if name.startswith("pllm/")}
    if (len(package) != link["package_file_count"] or package["pllm/_native.abi3.so"] != link["native_sha256"]
            or receipt["native_files"] != {"pllm/_native.abi3.so": link["native_sha256"]}
            or receipt["installed_package_differences"]):
        raise SystemExit("clean-build native/package binding differs")
    for run in link["executions"]:
        directory = root / run["directory"]
        for name, digest in run["locks"].items():
            if index["files"].get(f"{run['directory']}/{name}") != digest:
                raise SystemExit("build link refers to different execution records")
        manifest = json.loads((directory / "manifest.json").read_text())
        provenance = json.loads((directory / "provenance.json").read_text())
        build = json.loads((directory / "installed-build.json").read_text())
        expected = json.loads((directory / "child-identities/expected.json").read_text())
        if (provenance["wheel_package_files"] != package or expected["wheel_package_files"] != package
                or provenance["wheel_sha256"] != link["wheel_sha256"]
                or provenance["native_sha256"] != link["native_sha256"]
                or provenance["build_digest"] != link["runtime_build_digest"]
                or expected["wheel_sha256"] != link["wheel_sha256"]
                or expected["native_sha256"] != link["native_sha256"]
                or expected["python"] != provenance["python"]
                or expected["package_path"] != provenance["package_path"]
                or receipt["source_commit"] != provenance["checkout_commit"]
                or canonical_digest(build["attributes"]) != build["digest"]):
            raise SystemExit("executed package/native/BuildRef differs from clean-build link")
        if build["digest"] != link["runtime_build_digest"] or build["attributes"]["native_sha256"] != link["native_sha256"]:
            raise SystemExit("runtime BuildRef identity differs")
        summary = verify_child_identities(directory / "child-identities")
        if (manifest["status"] != "passed" or summary["status"] != "passed"
                or summary["captured_python_processes"] != run["children"] or summary["roles"] != run["roles"]
                or junit_totals(directory) != run["pytest"]):
            raise SystemExit("linked execution outcomes/counts differ")
        if "historical_receipt_field" in run:
            prior = run["historical_receipt_field"]
            source = json.loads((directory / "source-evidence-lock.json").read_text())
            if (manifest[prior["name"]] != prior["sha256"]
                    or source["prior_build_receipt_sha256"] != prior["sha256"]
                    or index["files"].get("2026-10-16-local-build-v1/receipt.json") != prior["sha256"]
                    or prior["sha256"] == link["build_receipt_sha256"]):
                raise SystemExit("historical/current build receipt scopes were conflated")


def refresh_index(root):
    """Regenerate only explicitly catalogued completed manifests, never discover runs."""
    path = root / "index.json"
    index = json.loads(path.read_text())
    files = dict(index["files"])

    def include(relative, expected=None):
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise SystemExit(f"invalid catalog path: {relative}")
        digest = sha(target)
        if expected is not None and digest != expected:
            raise SystemExit(f"completed manifest artifact changed: {relative}")
        if "/" in relative and relative in files and digest != files[relative]:
            raise SystemExit(f"immutable archive file changed: {relative}")
        files[relative] = digest

    for relative in list(files):
        include(relative)
    for item in index["runs"] + index.get("family_supplements", []):
        relative = f"{item['directory']}/manifest.json"
        include(relative, item.get("manifest_sha256"))
        manifest = json.loads((root / relative).read_text())
        if manifest["status"] != item["status"]:
            raise SystemExit("catalogued run disposition differs")
        for name, digest in manifest["artifacts"].items():
            include(f"{item['directory']}/{name}", digest)
    for item in index.get("current_build_links", []):
        include(f"{item['directory']}/receipt.json")
    index["files"] = dict(sorted(files.items()))
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def main():
    root = Path(__file__).resolve().parent
    index = json.loads((root / "index.json").read_text())
    checked = 0
    for relative, expected in index["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise SystemExit(f"missing/invalid archive path: {relative}")
        with path.open("rb") as handle:
            observed = hashlib.file_digest(handle, "sha256").hexdigest()
        if observed != expected:
            raise SystemExit(f"hash mismatch: {relative}")
        checked += 1
    for run in index["runs"]:
        directory = root / run["directory"]
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["status"] != run["status"]:
            raise SystemExit(f"run disposition mismatch: {run['directory']}")
        observed_status = "passed" if manifest["cases"] and all(
            case["status"] == "passed" for case in manifest["cases"]) else "failed"
        if manifest["status"] != observed_status:
            raise SystemExit("run disposition differs from individual cases")
        if sha(directory / "manifest.json") != run["manifest_sha256"]:
            raise SystemExit("run manifest lock differs")
        if "pytest" in run and junit_totals(directory) != run["pytest"]:
            raise SystemExit("run JUnit counts differ")
        for relative, digest in manifest["artifacts"].items():
            key = f"{run['directory']}/{relative}"
            if index["files"].get(key) != digest:
                raise SystemExit(f"manifest/index mismatch: {key}")
        cases = {case["name"]: case for case in manifest["cases"]}
        if cases.get("generated-independent-conformance", {}).get("status") == "passed":
            if verify_generated_arrays(directory) != cases["generated-independent-conformance"]["result"]["bit_exact_values"]:
                raise SystemExit("generated case count differs from retained arrays")
        if (manifest["schema"] == "pllm.paper_conformance.v2" and manifest["status"] == "passed"
                and not (directory / "child-identities/summary.json").is_file()):
            raise SystemExit("successful new conformance is missing child identities")
        if manifest["schema"] == "pllm.paper_conformance.v2" and (directory / "child-identities/summary.json").is_file():
            summary = verify_child_identities(directory / "child-identities")
            if summary["status"] != cases["provider-child-identities"]["status"]:
                raise SystemExit("conformance child case/summary disposition differs")
            if manifest["status"] == "passed" and (summary["status"] != "passed" or not all(
                    summary["roles"].get(role) for role in ("inference", "preparation"))):
                raise SystemExit("new conformance lacks successful provider identities")
    for item in index.get("build_receipts", []):
        directory = root / item["directory"]
        receipt = json.loads((directory / "receipt.json").read_text())
        for relative, digest in receipt["artifacts"].items():
            key = f"{item['directory']}/{relative}"
            if index["files"].get(key) != digest:
                raise SystemExit(f"build receipt/index mismatch: {key}")
        bound = receipt.get("existing_conformance_bound_to_identical_outputs", False)
        if bound != item["existing_conformance_bound_to_identical_outputs"]:
            raise SystemExit("build receipt disposition mismatch")
        old = json.loads((directory / "historical-wheel-files.json").read_text())
        new = json.loads((directory / "wheel-files.json").read_text())
        installed = json.loads((directory / "installed-package-files.json").read_text())
        changes = {p: {"before": old.get(p), "after": new.get(p)}
                   for p in sorted(old.keys() | new.keys()) if old.get(p) != new.get(p)}
        if changes != receipt["wheel_payload_differences"]:
            raise SystemExit("wheel payload difference record is inconsistent")
        package = {p: h for p, h in new.items() if p.startswith("pllm/")}
        installed_changes = {p: {"before": installed.get(p), "after": package.get(p)}
                             for p in sorted(installed.keys() | package.keys())
                             if installed.get(p) != package.get(p)}
        if installed_changes != receipt["installed_package_differences"]:
            raise SystemExit("installed package comparison is inconsistent")
        comparison_path = directory / "payload-comparison.json"
        if comparison_path.is_file():
            comparison = json.loads(comparison_path.read_text())
            if comparison["receipt_sha256"] != index["files"][f"{item['directory']}/receipt.json"]:
                raise SystemExit("payload comparison refers to a different receipt")
            if comparison["differing_entries"] != changes:
                raise SystemExit("payload comparison entry differences are inconsistent")
            if comparison["namespace_package_identical"] != (not installed_changes):
                raise SystemExit("namespace package identity claim is inconsistent")
            sbom_before = json.loads((directory / "historical-sbom.json").read_text())
            sbom_after = json.loads((directory / "rebuilt-sbom.json").read_text())
            for delta in comparison["sbom_json_differences"]:
                path = delta["path"].strip("/").split("/")
                before_cursor, after_cursor = sbom_before, sbom_after
                for key in path[:-1]:
                    before_cursor, after_cursor = before_cursor[key], after_cursor[key]
                key = path[-1]
                if before_cursor[key] != delta["before"] or after_cursor[key] != delta["after"]:
                    raise SystemExit("SBOM delta differs from archived JSON")
                before_cursor[key] = after_cursor[key]
            if sbom_before != sbom_after:
                raise SystemExit("SBOM has unreported JSON differences")
        if bound:
            before = json.loads((directory / "sources-before.json").read_text())
            after = json.loads((directory / "sources-after.json").read_text())
            if old != new or installed != {p: h for p, h in new.items() if p.startswith("pllm/")}:
                raise SystemExit("claimed identical build payload differs")
            if before["files"] != after["files"] or before["cargo_configuration_files"] != after["cargo_configuration_files"]:
                raise SystemExit("claimed stable build source differs")
            if (directory / "dependencies-before.json").read_bytes() != (directory / "dependencies-after.json").read_bytes():
                raise SystemExit("claimed stable Cargo dependency source differs")
            accepted = index["accepted_run"]
            evidence = receipt["existing_conformance"]
            if evidence["manifest_sha256"] != index["files"][f"{accepted}/manifest.json"]:
                raise SystemExit("build receipt points to different conformance")
            if evidence["provenance_sha256"] != index["files"][f"{accepted}/provenance.json"]:
                raise SystemExit("build receipt points to different conformance provenance")
            provenance = json.loads((root / accepted / "provenance.json").read_text())
            if new["pllm/_native.abi3.so"] != provenance["native_sha256"] or receipt["build_exit_code"] != 0:
                raise SystemExit("build receipt does not bind the executed native binary")
    original = json.loads((root / index["accepted_run"] / "provenance.json").read_text())
    for item in index.get("family_supplements", []):
        directory = root / item["directory"]
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["status"] != item["status"]:
            raise SystemExit("family supplement disposition changed")
        if "pytest" in item and junit_totals(directory) != item["pytest"]:
            raise SystemExit("family supplement JUnit counts differ")
        expected_status = "passed" if manifest["sources_unchanged"] and all(
            case["status"] == "passed" for case in manifest["cases"]) else "failed"
        if manifest["status"] != expected_status or not manifest["cases"]:
            raise SystemExit("family supplement outcome inconsistent")
        for relative, digest in manifest["artifacts"].items():
            if index["files"].get(f"{item['directory']}/{relative}") != digest:
                raise SystemExit("family supplement artifact/index mismatch")
        if manifest["schema"] == "pllm.paper_family_conformance.v1":
            for field in ("wheel_sha256", "native_sha256", "build_digest"):
                if manifest[field] != original[field]:
                    raise SystemExit("historical family supplement execution identity differs from v4")
        else:
            provenance = json.loads((directory / "provenance.json").read_text())
            for field, equal in manifest["identity_equals_original_v4"].items():
                if equal != (provenance[field] == original[field]):
                    raise SystemExit("new family identity comparison differs")
            for field in ("wheel_sha256", "native_sha256", "build_digest"):
                if provenance[field] != manifest[field]:
                    raise SystemExit("new family manifest/provenance mismatch")
            verify_child_identities(directory / "child-identities")
        before = json.loads((directory / "source-evidence-lock.json").read_text())["files"]
        after = json.loads((directory / "sources-after.json").read_text())
        if manifest["sources_unchanged"] != (before == after):
            raise SystemExit("family supplement source stability inconsistent")
        if manifest["original_execution_manifest_sha256"] != index["files"][f"{index['accepted_run']}/manifest.json"]:
            raise SystemExit("family supplement refers to another v4 execution")
        if manifest["local_build_receipt_sha256"] != index["files"]["2026-10-16-local-build-v1/receipt.json"]:
            raise SystemExit("family supplement refers to another local build receipt")
    for item in index.get("source_reviews", []):
        directory = root / item["directory"]
        receipt = json.loads((directory / "receipt.json").read_text())
        for relative, digest in receipt["artifacts"].items():
            if index["files"].get(f"{item['directory']}/{relative}") != digest:
                raise SystemExit("source review artifact/index mismatch")
        ledger = json.loads((directory / "check-ledger.json").read_text())
        totals = {"passed": 0, "failed": 0, "skipped": 0}
        for run in ledger["runs"]:
            run_dir = root / run["directory"]
            if run["manifest_sha256"] != index["files"][f"{run['directory']}/manifest.json"]:
                raise SystemExit("check ledger execution manifest mismatch")
            observed = []
            for path in sorted(run_dir.glob("*.xml")):
                for node in ET.parse(path).iter("testcase"):
                    status = "failed" if node.find("failure") is not None or node.find("error") is not None else (
                        "skipped" if node.find("skipped") is not None else "passed")
                    observed.append({"suite": path.name, "class": node.get("classname"),
                                     "name": node.get("name"), "status": status})
                    totals[status] += 1
            if observed != run["tests"]:
                raise SystemExit("check ledger differs from original JUnit nodes")
            for operator in run["operators"]:
                first = json.loads((run_dir / (operator["case"] + "-initial.json")).read_text())
                replay = json.loads((run_dir / (operator["case"] + "-replay.json")).read_text())
                if operator["initial_quality"] != first["quality"] or operator["replay_quality"] != replay["quality"]:
                    raise SystemExit("check ledger altered operator qualities")
                if operator["resources"] != first["resources"] or operator["resources_equal"] != (first["resources"] == replay["resources"]):
                    raise SystemExit("check ledger altered operator resources")
                if operator["numeric_identity_equal"] != (first["numeric_contract_digest"] == replay["numeric_contract_digest"]):
                    raise SystemExit("check ledger altered numeric identity")
        if totals != ledger["test_totals"]:
            raise SystemExit("check ledger totals multiply or omit JUnit outcomes")
    for item in index.get("isolation_checks", []):
        directory = root / item["directory"]
        receipt = json.loads((directory / "receipt.json").read_text())
        if receipt["status"] != item["status"] or receipt["provider_model_operator_executions"] != 0:
            raise SystemExit("focused isolation scope/disposition differs")
        for relative, digest in receipt["artifacts"].items():
            if index["files"].get(f"{item['directory']}/{relative}") != digest:
                raise SystemExit("isolation receipt/index mismatch")
        summary = verify_child_identities(directory / "child-identities")
        log = (directory / "focused-tests.log").read_text()
        observed_passes = sum(line.endswith(" ... ok") for line in log.splitlines())
        if observed_passes + len(receipt["failed"]) + len(receipt["errors"]) != receipt["tests_run"]:
            raise SystemExit("focused test count differs from retained unittest log")
        expected_processes = receipt.get("expected_captured_python_processes", 3)
        negatives_match = True
        for negative in receipt.get("negative_controls", []):
            negative_dir = directory / negative["directory"]
            if not (negative_dir / "summary.json").is_file():
                negatives_match = False
                continue
            observed = verify_child_identities(negative_dir)
            if observed["status"] != negative["observed_identity_status"]:
                raise SystemExit("negative identity control disposition differs")
            negatives_match &= observed["status"] == negative["expected_identity_status"] == "failed"
            control = json.loads((negative_dir / "control.json").read_text())
            if not control["process_group_absent"]:
                negatives_match = False
            if control["mode"] in {"sigkill", "timeout"}:
                if list(negative_dir.glob("*.end.json")) or not any(".end receipt" in value for value in observed["failures"]):
                    raise SystemExit("forced termination/timeout was relabeled as complete identity")
                if control["returncode"] != (-9 if control["mode"] == "sigkill" else -15):
                    raise SystemExit("negative signal semantics changed")
            else:
                ending = json.loads(next(negative_dir.glob("*.end.json")).read_text())
                if ending["status"] != "failed" or control["returncode"] != 0:
                    raise SystemExit("changed native origin was accepted or original exit changed")
        if receipt["schema"] == "pllm.paper_child_isolation_checks.v2":
            outcomes = [json.loads(path.read_text()) for path in directory.glob("uvicorn-*/outcome.json")]
            if receipt["no_model_uvicorn_executions"] != len(outcomes):
                raise SystemExit("no-model service count differs from outcomes")
            for signum in {item["signal"] for item in outcomes}:
                pair = [item for item in outcomes if item["signal"] == signum]
                if len(pair) != 2 or len({item["captured"] for item in pair}) != 2 or len({item["returncode"] for item in pair}) != 1:
                    raise SystemExit("raw and captured Uvicorn signal exits differ")
                if not all(item["http_status"] == 200 and item["lifespan_shutdown"] and item["process_group_absent"] for item in pair):
                    raise SystemExit("Uvicorn service shutdown/cleanup was not verified")
        passed = (not receipt["failed"] and not receipt["errors"] and summary["status"] == "passed"
                  and summary["captured_python_processes"] == expected_processes and negatives_match)
        if (receipt["status"] == "passed") != passed:
            raise SystemExit("focused receipt outcome inconsistent")
    for item in index.get("provenance_reviews", []):
        directory = root / item["directory"]
        receipt = json.loads((directory / "receipt.json").read_text())
        for relative, digest in receipt["artifacts"].items():
            if index["files"].get(f"{item['directory']}/{relative}") != digest:
                raise SystemExit("provider provenance review/index mismatch")
        for relative, digest in receipt["unchanged_historical_artifacts"].items():
            if index["files"].get(relative) != digest:
                raise SystemExit("historical execution/build/source artifact was rewritten")
    for item in index.get("current_build_links", []):
        verify_build_link(root, index, item)
    print(f"Verified {checked} files and {len(index['runs'])} run manifests.")
    print(f"Verified {len(index.get('build_receipts', []))} local build receipt(s).")
    print(f"Verified {len(index.get('family_supplements', []))} family supplement(s) and {len(index.get('source_reviews', []))} source review(s).")
    print(f"Verified {len(index.get('isolation_checks', []))} subprocess-isolation check receipt(s) and {len(index.get('provenance_reviews', []))} provenance review(s).")
    print(f"Verified {len(index.get('current_build_links', []))} explicit current clean-build execution link(s).")
    print("Byte integrity only; failed attempts remain failed; no execution or publication claim inferred.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-index", action="store_true",
                        help="refresh generated hash catalog for explicitly declared completed runs only")
    if parser.parse_args().refresh_index:
        refresh_index(Path(__file__).resolve().parent)
    main()
