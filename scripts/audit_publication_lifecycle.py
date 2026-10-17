"""Read-only replay of the archived local lifecycle smoke; no workload execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.research.publication_lifecycle import InvalidObservation  # noqa: E402
from benchmarks.research.publication_lifecycle_live import sidecar_file, sidecar_json  # noqa: E402
from scripts.benchmark_publication_lifecycle import load_trace  # noqa: E402

DEFAULT = ROOT / "paper/artifacts/publication-lifecycle/2026-10-17-tiny-v3"


def audit(directory):
    directory = Path(directory).absolute()
    with sidecar_file(directory / "archive.json", 1 << 20) as reader:
        manifest = sidecar_json(reader)
    if manifest["schema"] != "pllm.publication_lifecycle_archive.v1":
        raise InvalidObservation("archive_schema")

    def read_artifact(name):
        if name not in manifest["files"]:
            raise InvalidObservation("archive_file_coverage")
        with sidecar_file(directory / name, 16 << 20, expected=manifest["files"][name]) as reader:
            return sidecar_json(reader)

    for name, commitment in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise InvalidObservation("archive_path")
        with sidecar_file(directory / relative, 16 << 20, expected=commitment) as reader:
            while reader.read(65536):
                pass
            reader.verify()
    document = read_artifact("v3/cohort.json")
    summary = read_artifact("v3/cohort.summary.json")
    plan = read_artifact("v3/cohort.plan.json")
    for name, expected in plan["sources"].items():
        if manifest["files"].get("sources/" + name, {}).get("sha256") != expected:
            raise InvalidObservation("archive_source_coverage")
    if not document["complete"] or summary["status"] != "completed":
        raise InvalidObservation("campaign_incomplete")
    if len(document["results"]) != 3 or len(summary["results"]) != 3:
        raise InvalidObservation("campaign_count")
    states, cases, samples, owners = None, set(), 0, set()
    for result, row in zip(document["results"], summary["results"], strict=True):
        execution = result["execution"]
        details = execution["details"]
        lifecycle = details["publication_lifecycle"]
        case = execution["case"]["name"]
        if (result["status"] != "completed" or row["case"] != case or case in cases
                or row["replicate"] != execution["replicate"]
                or row["trace_ref"] != lifecycle["trace_ref"]
                or row["summary"] != lifecycle["summary"]):
            raise InvalidObservation("campaign_row_binding")
        cases.add(case)
        reference = dict(row["trace_ref"])
        # Relocate only the external locator; original manifests and receipts stay byte-identical.
        nonce = reference["binding"]["nonce"]
        if len(nonce) != 64 or any(c not in "0123456789abcdef" for c in nonce):
            raise InvalidObservation("archive_nonce")
        relative = f"v3/cohort.traces/{nonce}/manifest.json"
        if relative not in manifest["files"]:
            raise InvalidObservation("archive_trace_coverage")
        reference["path"] = str(directory / relative)
        trace, _, retained = load_trace(reference)
        if not retained["qualified"] or retained["summary"] != row["summary"]:
            raise InvalidObservation("campaign_summary_replay")
        if (details["publication_lifecycle_sources"] != plan["sources"]
                or row["summary"]["maximum_observed_host_swap_growth_bytes"] != 0):
            raise InvalidObservation("campaign_source_or_pressure")
        current = details["response_state_audit"]["bindings"]
        if states is not None and current != states:
            raise InvalidObservation("campaign_state_mismatch")
        states = current
        samples += len(trace["samples"])
        owners.update((owner["pid"], owner["create_time_ns"]) for owner in trace["owners"].values())
    if cases != {"maintained", "tiling", "coupled-boundary"}:
        raise InvalidObservation("campaign_cases")
    for name in ("v1", "v2"):
        failed = read_artifact(f"{name}/cohort.progress.json")
        if failed["status"] != "rejected":
            raise InvalidObservation("historical_attempt_status")
    original_plan = read_artifact("v1/cohort.plan.json")
    for name, expected in original_plan["sources"].items():
        if manifest["files"].get("v1/executed-sources/" + name, {}).get("sha256") != expected:
            raise InvalidObservation("archive_source_coverage")
    return {"status": "passed", "files": len(manifest["files"]), "trials": len(cases),
            "samples": samples, "process_incarnations": len(owners),
            "state_equal": True, "historical_rejections": 2,
            "scope": "archived tiny/local/cold measurement capability; no serving rank"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT)
    print(json.dumps(audit(parser.parse_args().archive), indent=2))


if __name__ == "__main__":
    main()
