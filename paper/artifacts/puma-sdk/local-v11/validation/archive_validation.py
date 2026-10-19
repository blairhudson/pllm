"""Preserve conformance inputs and derive counts from the original receipts."""
import hashlib
import json
import re
import shutil
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

root = Path.cwd()
validation = Path(__file__).resolve().parent
build = root / "target/puma-guarded-activation-build-v2"
out = root / "paper/artifacts/puma-sdk/local-v11"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
load = lambda p: json.loads(Path(p).read_text())


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


receipt = load(validation / "validation-commands.json")
assert receipt["status"] == "passed" and receipt["inputs_match_before_and_after"]
assert len(receipt["commands"]) == 7 and all(c["returncode"] == 0 for c in receipt["commands"])
before, after = [load(validation / f"installed-files-{label}.json") for label in ("before", "after")]
assert before["files"] == after["files"]
assert before["wheel_sha256"] == after["wheel_sha256"] == receipt["wheel_sha256"]
assert before["all_match_wheel"] and after["all_match_wheel"]
assert len(before["files"]) == receipt["installed_file_count"]
inputs = load(validation / "validation-inputs.json")
assert all(sha(root / p) == h for p, h in inputs.items())

cases = list(ET.parse(validation / "installed-tests.xml").getroot().iter("testcase"))
assert cases and all(c.find("failure") is None and c.find("error") is None for c in cases)
skipped = sum(c.find("skipped") is not None for c in cases)
counts = []
for index in range(1, 5):
    log = (validation / f"validation-{index}.log").read_text()
    assert "test result: FAILED" not in log
    matches = re.findall(r"test result: ok\. (\d+) passed; 0 failed;", log)
    assert matches, index
    counts.append(sum(map(int, matches)))
qwen = load(validation / "qwen-program-admission.json")
assert qwen["status"] == "rejected" and qwen["watchdog_error"] is None
assert qwen["max_swap_used_growth_bytes"] == 0
assert "step: Some(29), operator: Some(RmsNorm), layer: Some(1)" in qwen["reason"]
old_xml = root / "target/puma-guarded-activation-old-wheel.xml"
old_cases = list(ET.parse(old_xml).getroot().iter("testcase"))
assert len(old_cases) == 3
assert all(c.find("failure") is not None and "Some(Silu)" in c.find("failure").text for c in old_cases)

(out / "validation").mkdir(parents=True)
for path in sorted(validation.iterdir()):
    if path.is_file():
        shutil.copy2(path, out / "validation" / path.name)
(out / "old-wheel").mkdir()
shutil.copy2(old_xml, out / "old-wheel/tests.xml")
(out / "build").mkdir()
for path in sorted(build.iterdir()):
    if path.is_file():
        shutil.copy2(path, out / "build" / path.name)
shutil.copy2(validation / "validation-inputs.json", out / "validation-inputs.json")
with tarfile.open(out / "inputs.tar.gz", "w:gz") as archive:
    for path in sorted(inputs):
        archive.add(root / path, arcname=path, recursive=False)
with tarfile.open(out / "inputs.tar.gz", "r:gz") as archive:
    actual = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive.getmembers()}
    assert actual == inputs

prior = load(root / "paper/artifacts/puma-sdk/local-v10/validation/installed-files-after.json")
assert set(prior["files"]) == set(before["files"])
changed = [{"path": p, "before": prior["files"][p], "after": h}
           for p, h in sorted(before["files"].items()) if prior["files"][p] != h]
assert len(changed) == 1 and changed[0]["path"].endswith(".so")
write(out / "package-delta.json", {
    "previous_wheel_sha256": prior["wheel_sha256"], "wheel_sha256": receipt["wheel_sha256"],
    "changed": changed, "unchanged_file_count": len(before["files"]) - len(changed),
})
summary = {
    "schema": "pllm.puma_sdk.guarded_activation_conformance.v1", "status": "passed",
    "scope": "bounded public-weight guarded SiLU/GeLU, generated installed SDK conformance and rejected pinned checkpoint source publication",
    "serving_or_quality_claim": None,
    "wheel_sha256": receipt["wheel_sha256"], "build_receipt": "build/receipt.json",
    "validation_receipt": "validation/validation-commands.json", "package_delta": "package-delta.json",
    "python": {"passed": len(cases) - skipped, "optional_skipped": skipped, "total_cases": len(cases)},
    "native": {"puma_library": counts[0], "replicated_compiler": counts[1], "integration": counts[2],
               "doctests": counts[3], "total": sum(counts), "clippy": "passed",
               "clippy_allowances": ["needless_range_loop", "nonminimal_bool"]},
    "coverage_check": "passed",
    "installed_files": {"count": len(before["files"]), "match_wheel_before_and_after": True},
    "source_inputs": {"count": len(inputs), "manifest": "validation-inputs.json", "archive": "inputs.tar.gz",
                      "archive_matches_manifest": True, "validation_unchanged_before_and_after": True},
    "old_wheel": {"junit": "old-wheel/tests.xml", "expected_failures": 3,
                  "rejecting_operator": "Silu", "supplies_passing_test_count": False,
                  "per_run_installed_file_comparison": None},
    "qwen": {"receipt": "validation/qwen-program-admission.json", "input_tokens": 4, "output_tokens": 2,
             "source_lock": qwen["source_lock"], "checkpoint": qwen["checkpoint"],
             "ordinary_memory_preflight": "admitted", "tensor_values_decoded": True,
             "layer_zero_silu_and_mlp_admitted": True, "complete_numeric_program_admission": "rejected",
             "rejecting_operator": "RmsNorm", "rejecting_step": 29, "rejecting_layer": 1, "selection": 0,
             "workers_launched": False, "online_decoder_executed": False,
             **{k: qwen[k] for k in ("scope", "elapsed_seconds", "process_cpu_seconds", "process_peak_rss_bytes", "max_swap_used_growth_bytes")}},
}
write(out / "summary.json", summary)
print(json.dumps(summary, indent=2, sort_keys=True))
