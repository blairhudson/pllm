"""Freeze local primary-page locators and an honest ledger of family check outcomes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "paper/artifacts/conformance"
RECORDS = {
    "slalom": {"pages": [5, 6], "locator": "Section 3/Figure 1; Section 3.2/Lemma 3.1; Section 3.3/Theorem 3.2",
               "scope": "trusted software client and trusted Preparation replace original SGX boundary; field, one-use attempts and resource contracts differ"},
    "smoothquant": {"pages": [3, 4], "locator": "Equations (3)-(4), Figures 4-5",
                    "scope": "alpha=3/4 plus floor, geometric normalization, clipping, public source-bound calibration and client divisions"},
    "logrow": {"pages": [10, 12, 13, 14, 15, 16, 17, 18, 19, 20],
               "locator": "Section 3.4/Figure 1; Sections 4.1-4.2; Section 5/Construction 1/Figures 3-5; Section 6/Theorem 1",
               "scope": "implementation retains 2n-1 one-hot tree blocks and implements Figure 5 row reduction; paper Figure 1 states 2n-1 but Section 6 charges n-1; do not claim optimized headline accounting"},
    "sigma": {"pages": [4, 6, 7, 9], "locator": "Sections 2.5-2.6, 4.2.1-4.2.2; Section 5.1/Figure 7",
              "scope": "128 seed bits plus flags, CPU AES, bounded signed-gap shifts and residual SiLU complete gated operator"},
    "fusefss": {"pages": [6], "locator": "Section 4.5/Theorem 4.6, Section 4.6/Theorem 4.7",
                "scope": "specific eight-phase CPU multi-view adaptation; not full gate compiler or transferred proof"},
    "fastquery": {"pages": [2, 3, 4], "locator": "Sections 2.2 and 4.1-4.3, Figures 4-6",
                  "scope": "coefficient one-hot query, mixed width and extracted-LWE packing; local half-gates/OT replaces distributed SiRNN; no fine-tuned model"},
    "bumblebee": {"pages": [6, 7], "locator": "cached ePrint Section IV/Figure 6; final NDSS Section III/Figure 6 is separately reviewed",
                  "scope": "code locks cached ePrint hash, not different final NDSS PDF; RNS division and trace/interleave are bounded BFV adaptation"},
    "nexus": {"pages": [8, 9], "locator": "Section IV softmax, Equation (4), following unnumbered repeated-squaring formula with r=8, Algorithm 4",
              "scope": "SDK calls Equation (4) the exponential, but (4) is normalized exponentiation; constant-shift optimization already appears in original; exact shift/range/reciprocal iterations adapted"},
    "thor": {"pages": [10, 15], "locator": "Section 5.1/Algorithm 3; Appendix C degree-15 exponential coefficients",
             "scope": "delta1=1 delta2=2, balanced powers, narrow public range, two unit-factor reciprocal iterations; no aSOR/bootstrap/full decoder"},
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "driver.py").write_bytes(Path(__file__).read_bytes())
    library = json.loads((ROOT / "docs/data/research/paper-library.json").read_text())
    sources = {p["id"]: p for p in library["papers"]}
    tool = shutil.which("pdftotext")
    if not tool:
        raise RuntimeError("existing pdftotext is required; no dependency installation")
    version = subprocess.run([tool, "-v"], capture_output=True, text=True, check=True)
    records = []
    for name, record in RECORDS.items():
        source = sources[name]
        assert sha(ROOT / source["file"]) == source["sha256"]
        commands = []
        for page in record["pages"]:
            dest = output / f"{name}-pdf-page-{page}.txt"
            argv = [tool, "-layout", "-f", str(page), "-l", str(page), str(ROOT / source["file"]), str(dest)]
            done = subprocess.run(argv, capture_output=True, text=True, check=True)
            commands.append({"argv": argv, "exit_code": done.returncode, "stderr": done.stderr,
                             "text_sha256": sha(dest)})
        records.append({"id": name, "source": source, **record, "extraction_commands": commands})
    save(output / "primary-locators.json", {
        "schema": "pllm.paper_family_primary_locators.v1", "utc": datetime.now(timezone.utc).isoformat(),
        "pdftotext": {"path": tool, "sha256": sha(tool), "version": version.stdout + version.stderr},
        "page_numbering": "PDF page selected with -f/-l; embedded form-feed characters are not page counters",
        "records": records,
        "review_sha256": sha(ROOT / "paper/PRIOR-ART-REVIEW.md"),
        "prior_art_sources_sha256": sha(ROOT / "paper/prior-art-sources.json"),
        "scope": "read existing local primary PDF bytes; no new web search, retrieval or upstream execution",
    })
    runs = []
    for name in ["2026-10-16-six-families-v1", "2026-10-16-family-followup-v1"]:
        directory = ARCHIVE / name
        manifest = json.loads((directory / "manifest.json").read_text())
        tests, operators, commands = [], [], []
        for path in sorted(directory.glob("*.xml")):
            for node in ET.parse(path).iter("testcase"):
                status = "failed" if node.find("failure") is not None or node.find("error") is not None else (
                    "skipped" if node.find("skipped") is not None else "passed")
                tests.append({"suite": path.name, "class": node.get("classname"), "name": node.get("name"), "status": status})
        for path in sorted(directory.glob("*-initial.json")):
            initial = json.loads(path.read_text())
            replay = json.loads(path.with_name(path.name.replace("-initial", "-replay")).read_text())
            operators.append({"case": path.stem.removesuffix("-initial"),
                              "initial_quality": initial["quality"], "replay_quality": replay["quality"],
                              "resources_equal": initial["resources"] == replay["resources"],
                              "numeric_identity_equal": initial["numeric_contract_digest"] == replay["numeric_contract_digest"],
                              "resources": initial["resources"], "limitations": initial["limitations"]})
        for path in sorted(directory.glob("*.command.json")):
            c = json.loads(path.read_text())
            commands.append({"path": path.name, "exit_code": c["exit_code"],
                             "maximum_host_swap_growth_bytes": c["maximum_host_swap_growth_bytes"],
                             "watchdog_error": c["watchdog_error"]})
        runs.append({"directory": name, "manifest_sha256": sha(directory / "manifest.json"),
                     "status": manifest["status"], "tests": tests, "operators": operators, "commands": commands})
    save(output / "check-ledger.json", {
        "schema": "pllm.paper_family_check_ledger.v1", "runs": runs,
        "counting": "pytest node outcomes and operator outputs are distinct; earlier v4 checks/historical cohorts are not added",
        "original_v4_manifest_sha256": sha(ARCHIVE / "2026-10-16-installed-v4/manifest.json"),
        "local_build_receipt_sha256": sha(ARCHIVE / "2026-10-16-local-build-v1/receipt.json"),
        "test_totals": {s: sum(t["status"] == s for r in runs for t in r["tests"]) for s in ("passed", "failed", "skipped")},
    })
    save(output / "receipt.json", {
        "schema": "pllm.paper_family_source_review.v1",
        "artifacts": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        "scope": "local primary-text extraction and derived result ledger; no fresh model execution or security attestation",
    })
    print(f"Archived {len(records)} primary source locators and two separate outcome ledgers at {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output.resolve())
