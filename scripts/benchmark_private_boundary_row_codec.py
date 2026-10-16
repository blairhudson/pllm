"""Run one source-present matched ordinary-SDK head-codec cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filelock import FileLock, Timeout
from pllm import Experiment
from pllm.metrics import BuildRef, benchmark_suite

from benchmarks.research import private_boundary_row_codec as study
from benchmarks.research.sota_combinations import Rejected, atomic_json, public_input_counts
from scripts.benchmark_sota_combinations import fresh_output_paths

hashes = study.hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path, help="explicit frozen Experiment JSON")
    parser.add_argument("--candidate", required=True, type=Path, help="explicit frozen Experiment JSON")
    parser.add_argument("--certificate-accounting", required=True, type=Path,
        help="public delivery/derivation declaration; not measured publisher traffic")
    parser.add_argument("--native-head-commitments", type=Path,
        help="independently approved per-arm native pipeline/body/descriptor/schedule commitment JSON; required to execute")
    parser.add_argument("--prompt", type=Path, default=Path("target/prepared-frontier-artifacts/prompt.txt"))
    parser.add_argument("--max-output-tokens", required=True, type=int, choices=(8, 64))
    parser.add_argument("--tiny", action="store_true", help="explicit frozen generated-source 128+2 smoke only")
    parser.add_argument("--preflight-only", action="store_true", help="metadata/admission only, no tensors/providers")
    args = parser.parse_args(argv)
    if args.tiny and args.max_output_tokens != 8:
        parser.error("--tiny requires --max-output-tokens 8 (actual smoke cap 2)")
    plan_path, progress_path, summary_path = fresh_output_paths(args.output)
    receipt_path = args.output.with_suffix(".receipt.json")
    if receipt_path.exists():
        raise FileExistsError("terminal receipt already exists")
    progress = {"schema": "pllm.private_boundary_row_codec_progress.v1", "status": "preflight",
        "measurement_complete": False}
    # Reserve companion exclusively, before lock acquisition. Never reuse partial output.
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("x") as stream:
        json.dump(progress, stream, indent=2)
        stream.write("\n")
    lock = Path.home() / ".cache/pllm/frontier-probe.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(lock), timeout=0):
            input_paths = [args.control, args.candidate, args.certificate_accounting, args.prompt]
            # Bound every operator input before hashing, admission or allocation.
            limits = (1 << 20, 1 << 20, 64 << 10, 4096)
            if args.native_head_commitments is not None:
                input_paths.append(args.native_head_commitments)
                limits += (64 << 10,)
            raw = {str(path): study.bounded_read(path, limit) for path, limit in zip(input_paths, limits)}
            input_hashes = {path: hashlib.sha256(data).hexdigest() for path, data in raw.items()}
            control_spec = json.loads(raw[str(args.control)])
            candidate_spec = json.loads(raw[str(args.candidate)])
            pin = study.validate_pair_specs(control_spec, candidate_spec,
                max_output_tokens=args.max_output_tokens, tiny=args.tiny)
            declaration = json.loads(raw[str(args.certificate_accounting)])
            certificate = study.certificate_accounting(pin, declaration)
            study.certificate_evidence(certificate, pin)
            native = None
            if args.native_head_commitments is not None:
                native = json.loads(raw[str(args.native_head_commitments)])
                study.validate_native_commitments(native)
            if native is None and not args.preflight_only:
                raise Rejected("row_codec_native_commitments_missing_or_invalid")
            prompt = raw[str(args.prompt)].decode("utf-8")
            prompt_digest = input_hashes[str(args.prompt)]
            verifier = study.verifier_sources()
            sources = input_paths + [Path(pin["path"])]
            frozen = study.verifier_sources() | hashes(sources)
            if (hashes(input_paths) != input_hashes or frozen[str(Path(pin["path"]))] != pin["digest"]
                    or any(frozen[name] != value for name, value in verifier.items())):
                raise Rejected("sources_changed_during_input_read")
            control, candidate = Experiment.from_spec(control_spec), Experiment.from_spec(candidate_spec)
            if control.to_spec() != control_spec or candidate.to_spec() != candidate_spec:
                raise Rejected("row_codec_noncanonical_frozen_experiment")
            # These operations inspect source/tokenizer metadata; no model tensors/providers.
            suite = study.matched_suite(control=control, candidate=candidate, build=BuildRef.capture(),
                max_output_tokens=args.max_output_tokens, tiny=args.tiny)
            admission = study.preflight(suite)
            if any(not row["admission"]["admitted"] for row in admission):
                atomic_json(plan_path, {"schema": "pllm.private_boundary_row_codec_plan.v1",
                     "suite": suite.to_spec(), "preflight": admission, "frozen_sources": frozen,
                     "verifier_sources": verifier,
                    "tiny": args.tiny, "max_output_tokens": args.max_output_tokens, "certificate": certificate})
                raise Rejected("topology_memory_limit")
            counts, source = public_input_counts(control.pipeline.model, control.pipeline.components.get("tokenizer"),
                {"tiny" if args.tiny else "150": {"prompt": study.public_prompt_context(prompt)}})
            if not counts or any(type(count) is not int or count <= 0 or count > control.budget.max_input_tokens
                    for count in counts.values()) or (not args.tiny and set(counts.values()) != {150}):
                raise Rejected("row_codec_public_prompt_token_count_invalid")
            plan = {"schema": "pllm.private_boundary_row_codec_plan.v1", "suite": suite.to_spec(),
                "control": control_spec, "candidate": candidate_spec, "preflight": admission,
                "frozen_sources": frozen, "tiny": args.tiny, "max_output_tokens": args.max_output_tokens,
                "verifier_sources": verifier,
                "public_input_counts": counts, "prompt_digest": prompt_digest,
                "public_prompt": {"path": str(args.prompt), "sha256": prompt_digest,
                    "size_bytes": len(raw[str(args.prompt)])}, "source_lock_digest": source,
                "certificate": certificate, "certificate_declaration": declaration,
                "native_head_commitments": native,
                "execution": "one private salt; six serial complete fresh SDK workers; native shared100/40",
                "schedule": [[study.CONTROL, 0], [study.CANDIDATE, 0], [study.CANDIDATE, 1],
                    [study.CONTROL, 1], [study.CANDIDATE, 2], [study.CONTROL, 2]],
                "pending": None if native is not None else "independent native commitment pins required to execute",
                "output_policy": "natural EOS; no prompt, stop, budget or component rewriting"}
            atomic_json(plan_path, plan)
            if study.verifier_sources() | hashes(sources) != frozen:
                raise Rejected("sources_changed_during_preflight")
            if args.preflight_only:
                progress["status"] = "admitted"
            else:
                progress["status"] = "running"
                atomic_json(progress_path, progress)
                # SDK atomically checkpoints every completed arm; failed attempts retained.
                run = benchmark_suite(suite, inputs={name: {"prompt": prompt} for name in study.NAMES}, output=args.output)
                document = study.bounded_json(args.output, 256 << 20)
                if document != run.to_dict():
                    raise Rejected("row_codec_persisted_report_mismatch")
                receipt = study.terminal_receipt(document, plan, sources)
                from scripts.summarize_private_boundary_row_codec import summarize
                summary = summarize(document, plan, receipt=receipt)
                # Only audited, post-checked complete runs get terminal provenance.
                atomic_json(receipt_path, receipt)
                atomic_json(summary_path, summary)
                progress.update(status="completed", measurement_complete=True, clients=summary["clients"],
                    maximum_swap_growth_bytes=summary["maximum_observed_swap_growth_bytes"])
            atomic_json(progress_path, progress)
            print(json.dumps(progress, indent=2))
            return 0
    except BaseException as exc:
        code = ("frontier_cohort_lock_busy" if isinstance(exc, Timeout) else str(exc) if isinstance(exc, Rejected)
            else "interrupted" if isinstance(exc, KeyboardInterrupt) else "preflight_or_execution_error")
        progress.update(status="rejected", failure_code=code, error_type=type(exc).__name__)
        atomic_json(progress_path, progress)
        print(json.dumps(progress, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
