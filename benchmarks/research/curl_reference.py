"""Rerun the bounded native Curl component reference, outside decoder rankings."""
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CONFIGURATION = {
    "paper": "curl",
    "paper_sha256": "dc930fdb3219c989106f96eb9ae1f00f28aae16ec5d17eef0c5421eb4f241c1e",
    "source_sections": ["4.1.1 / Fig. 4", "4.2.1"],
    "input_rows": 512,
    "input_interval": [-4, 4],
    "upper_endpoint": "excluded",
    "function": "SiLU",
    "output_fraction_bits": 16,
    "haar_levels": [0, 1, 2, 3, 4],
    "scope": "in-process party-local u32 shares; client supplies already-truncated index; no protected truncation, transport or Qwen decoder",
}


def run():
    result = subprocess.run(
        ["cargo", "run", "--release", "-p", "pllm-garble", "--example", "curl_lookup_probe"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    report = json.loads(result.stdout)
    assert report["input_rows"] == CONFIGURATION["input_rows"]
    assert [row["haar_levels"] for row in report["cases"]] == CONFIGURATION["haar_levels"]
    report["configuration"] = CONFIGURATION
    report["environment"] = {"platform": platform.platform(), "machine": platform.machine()}
    report["source_sha256"] = {
        file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest() for file in [
            "benchmarks/research/curl_reference.py",
            "crates/pllm-garble/src/shared_lut_reference.rs",
            "crates/pllm-garble/examples/curl_lookup_probe.rs",
        ]
    }
    return report


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(run(), indent=2) + "\n")
