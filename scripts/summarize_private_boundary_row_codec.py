"""Read-only replay of one complete row-codec cohort and its frozen plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.research import private_boundary_row_codec as study


def _summarize(document, plan, receipt):
    study._require(type(plan) is dict and type(plan.get("tiny")) is bool
        and type(plan.get("max_output_tokens")) is int, "row_codec_plan_invalid")
    study.verify_receipt(document, plan, receipt)
    study.verify_verifier_sources(plan["verifier_sources"])
    study.validate_source_manifest(plan["frozen_sources"])
    study._require(all(plan["frozen_sources"].get(name) == value
        for name, value in plan["verifier_sources"].items()), "row_codec_frozen_verifier_mismatch")
    study._require(plan["schema"] == "pllm.private_boundary_row_codec_plan.v1"
        and document["suite"] == plan["suite"], "row_codec_plan_suite_mismatch")
    pin = study.validate_pair_specs(plan["control"], plan["candidate"],
        max_output_tokens=plan["max_output_tokens"], tiny=plan["tiny"])
    study._require(plan["frozen_sources"][str(Path(pin["path"]))] == pin["digest"],
                   "row_codec_frozen_certificate_digest_mismatch")
    for case, spec in zip(document["suite"]["cases"], (plan["control"], plan["candidate"])):
        study._require(case["target"]["parameters"]["model"]["experiment"] == spec,
                       "row_codec_plan_experiment_mismatch")
    # Replay never reads current checkpoint/certificate files or loads providers.
    expected = study.certificate_accounting(pin, plan["certificate_declaration"], read_artifact=False)
    stored = dict(plan["certificate"])
    stored.pop("artifact_base64")
    expected.pop("artifact_base64")
    study._require(stored.pop("artifact_bytes_verified") is True, "row_codec_certificate_bytes_unverified")
    expected.pop("artifact_bytes_verified")
    study._require(stored == expected, "row_codec_certificate_accounting_mismatch")
    result = study.audit(document, max_output_tokens=plan["max_output_tokens"], tiny=plan["tiny"],
        certificate=plan["certificate"], native_commitments=plan["native_head_commitments"],
        source_manifest=plan["verifier_sources"])
    if plan["source_lock_digest"] is not None:
        study._require(result["identity"]["source_lock_digest"] == plan["source_lock_digest"],
                       "row_codec_planned_source_mismatch")
    # Public file SHA256 and private salted prompt commitment have different
    # domains. The frozen driver reads this bounded file, verifies it around
    # execution, and supplies those exact bytes through private benchmark stdin.
    # Replay checks that declaration; it cannot recover/verify a private preimage.
    public = plan["public_prompt"]
    study._require(type(public) is dict and set(public) == {"path", "sha256", "size_bytes"}
        and type(public["path"]) is str and 0 < len(public["path"]) <= 4096
        and study._sha(public["sha256"]) and type(public["size_bytes"]) is int
        and 0 < public["size_bytes"] <= 4096
        and plan["frozen_sources"].get(public["path"]) == public["sha256"] == plan["prompt_digest"],
        "row_codec_planned_public_prompt_mismatch")
    counts = plan["public_input_counts"]
    study._require(type(counts) is dict and set(counts) == {"tiny" if plan["tiny"] else "150"}
        and all(type(value) is int and value == result["identity"]["tokens"]["input_tokens"]
            for value in counts.values()), "row_codec_planned_input_count_mismatch")
    result["public_prompt"] = {**public, "scope":
        "driver-frozen public fixture; private salted commitment remains separate, not preimage verification"}
    result["certificate"] = plan["certificate"]
    result["verifier_sources"] = plan["verifier_sources"]
    result["frozen_sources"] = plan["frozen_sources"]
    result["pending_integration"] = plan["pending"]
    result["terminal_receipt"] = receipt
    return result


def summarize(document, plan, *, receipt=None):
    try:
        return _summarize(document, plan, receipt)
    except study.Rejected:
        raise
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError, RecursionError, OSError) as exc:
        raise study.Rejected("row_codec_plan_missing_or_invalid") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(summarize(study.bounded_json(args.input, 256 << 20),
        study.bounded_json(args.plan, 8 << 20),
        receipt=study.bounded_json(args.receipt, 64 << 10)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
