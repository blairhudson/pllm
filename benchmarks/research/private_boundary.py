"""Frozen private-boundary recipes; the original lifecycle driver owns replay.

The summary is regenerated from the strict raw-evidence validator. Its recipe
must match the Pipeline digest retained by original native memory admission.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

SOURCES = {
    "private-boundary-qwen-48-sdk": "c117dd5f0d5b5706a8bc18451241d779a1871eb06c8b20b3dc55d3219c76f57a",
    "private-boundary-qwen-150-sdk": "5f59094f24105ad26e024ba688c4d0015969d53ea39b1f95eb2d77b2cc5c6c6b",
}


def recorded_contract(source, name):
    """Exact historical Experiment/build, not current-build performance evidence."""
    from scripts.summarize_private_boundary_sdk import summarize

    root = Path(__file__).resolve().parents[2]
    raw = (root / "docs/evidence" / f"{source}.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCES[source]:
        raise ValueError("historical source changed")
    control = (root / "docs/evidence/private-boundary-qwen-control.json").read_bytes()
    summary = summarize(json.loads(raw), SOURCES[source], json.loads(control), hashlib.sha256(control).hexdigest())
    return deepcopy({"experiment": summary["groups"][name]["experiment"], "build": summary["build"],
        "source_lock_digest": summary["source_lock_digest"], "cohort_commitment": summary["cohort_commitment"],
        "measurement_sources": summary["driver_implementation"], "limits": summary["limits"],
        "lifecycle": {"source_case": summary["source_case"], "input_tokens": summary["input_tokens"],
            "max_output_tokens": 8, "fresh_client_and_providers": True, "wan": "native-shared-100/40"},
        "requires_original_measurement_driver": True})


def experiment(source, name):
    """Load the frozen recipe without loading weights or starting providers."""
    from pllm import Experiment

    return Experiment.from_spec(recorded_contract(source, name)["experiment"])
