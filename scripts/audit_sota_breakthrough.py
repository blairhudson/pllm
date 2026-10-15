#!/usr/bin/env python3
"""Bounded archive replay; no tensors, training targets, timing or serving admission.

Only fixed, SHA256-pinned repository snapshots execute. --evidence-dir selects
JSON inputs, never code. Default checks the installed native head codec; use
--skip-native for arithmetic/refit-only replay (explicitly not native evidence).
--skip-primary-sources checks pinned retained provenance metadata, not PDF/text
bytes. Both skips together need neither the original wheel nor ignored papers.
The original pilot's numeric dependency was observed AFTER launch, not pre-frozen.
Corpus/checkpoint bytes and historical process execution are not replayed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import re
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "docs/evidence/sota-breakthrough-source-snapshots"
PINS = {
    "checkpoint_tiny_model.py": "3fa854c35e361a0de71bc3690ada7ae390f8e8816e3ac6fb0a1aaa464e130daf",
    "train_checkpoint_tiny_model.py": "d7e9c076e7df318f020894c9d14f4ead4b3c8ff2b46f3b7148bc5038a268b595",
    "decoder_probe_support.py": "6b102bd674403a431e898ef1d6d411e74a663f063555e021ac56a1159a8a0d3e",
    "probe_compact_private_head.py": "8078bf03a2ad13a7e6cadd551ec51b4434ab3c9ab859adef54f4def6fc462f20",
    "audit_compact_private_head.py": "1f5290782a1a1844d513b742636829d89f6b156fd0e743d5a9fe1b1c28561d21",
    "probe_mlp_conveyor_transport.py": "10ee1a648df45ff40b93286a1981a474eb320c338356abadf7534b0b988122d2",
    "probe_mlp_conveyor.py": "bb59d3175337ed379fe8ab1be5ea84f50afafe6a9eff0eff89c3b96f8d7341fb",
    "frontier_probe_support.py": "6f060bf3403b5991f0f3728a54e02cc1080148b6ffc667948fbf675214779bc5",
    "probe_fixed_weight_correlation_gate.py": "7d85d4a744198b190590d295701283d99c5520c92883c954d883d600a19d885d",
    "test_fixed_weight_correlation_gate_v3.py": "f0e1dd9fd74ce0f172cdcacbb3ef2218c1c0f9a7dc8eb0234db86948c9a76710",
    "audit_sota_breakthrough_v1.py": "86a821e1010cc979e9c12d98b178be683287e11a4a0eb5a60e4dde159cc7e5f0",
    "test_sota_breakthrough_audit_v1.py": "63e04b7da78ec9bd58000a7e5997c2a23854f608f8c08e57b4d6eaa8853a4426",
}
PILOT_LOG = "checkpoint-tiny-model-qwen25-pilot-v1.log.txt"
# Frozen v3 metadata, deliberately NOT hashes of today's docs/tests. Original
# executable and test bytes are independently pinned snapshots above; the doc
# hash remains the original report's metadata commitment, not a live-doc check.
CORRELATION_IMPLEMENTATION = {
    "scripts/probe_fixed_weight_correlation_gate.py": PINS["probe_fixed_weight_correlation_gate.py"],
    "tests/test_fixed_weight_correlation_gate.py": PINS["test_fixed_weight_correlation_gate_v3.py"],
    "docs/research/fixed-weight-correlation-gate.md": "29b9d4d06bbae1fb1dde80a7e5bb6f6b322d0ef5534dd779732df662b8ecf8c3",
}
PROVENANCE_DIGEST = "a8e141258190b91c4490811881ba191ba6994e36cfbe886a16de56a0b752729f"
PRIMARY_MANIFEST_DIGEST = "4797390ba0ed1d42eebeccb5add20d829b3dc4938a05301dc57b6e99c555f38c"
FILES = {
    "pilot": "checkpoint-tiny-model-qwen25-pilot-v1.json",
    "manifest": "checkpoint-tiny-model-pilot-manifest-v1.json",
    "helper": "checkpoint-tiny-model-qwen25-helper-v1.json",
    "attempt": "checkpoint-tiny-model-attempt-status-v1.json",
    "head": "compact-private-head-qwen25-v1.json",
    "native": "compact-private-head-native-audit-v2.json",
    "conveyor": "mlp-conveyor-transport-qwen150-288-v2.json",
    "conveyor_lock": "mlp-conveyor-transport-tiny-screen-v2.json",
    "conveyor_original": "mlp-conveyor-transport-qwen150-288-v1.json",
    "conveyor_original_lock": "mlp-conveyor-transport-tiny-screen.json",
    "correlation": "fixed-weight-correlation-gate-bounded-v3.json",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return sha(encoded(value))


def bounded(path, limit=8 << 20):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, f"oversize audit input: {path}")
    return raw


def file_hash(path):
    # Installed native/NumPy binaries can exceed the JSON bound; stream, never map.
    require(Path(path).stat().st_size <= 256 << 20, "oversize source dependency")
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(name, execute=False):
    require(name in PINS, "untrusted snapshot name")
    path = SNAPSHOTS / name
    raw = bounded(path, 128 << 10)
    require(sha(raw) == PINS[name], f"snapshot source lock: {name}")
    if not execute:
        return PINS[name]
    key = "_sota_archive_" + path.stem
    module = types.ModuleType(key)
    module.__file__ = str(path)
    sys.modules[key] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def normalized_em(text, answers):
    def normalize(value):
        value = re.sub(r"[^\w\s]", "", value.lower())
        return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())
    return any(normalize(text) == normalize(answer) for answer in answers)


def pilot_record(record, *, forced=False):
    """Original Qwen bill: 24 layers, QKV/O/GU at 24 bits, down at 32.

    Validate that declared projection, not recompute source-conditioned widths
    from tensors. Neither local-clear timings nor these bytes are serving costs.
    """
    n, selected = record["input_tokens"], record["selected_tokens"]
    require(type(n) is int and 1 <= n <= 512 and type(selected) is int
            and 1 <= selected <= (16 if forced else 24), "pilot token bounds")
    rows = n + selected - 1
    require(record["executed_sequence_rows"] == rows and record["stage_rows"] == 96 * rows
            and record["stage_calls"] == 96 * selected, "pilot causal stage accounting")
    request = 24 * (3 * 896 * 3 + 4864 * 4)
    reply = 24 * ((1152 + 896 + 9728) * 3 + 896 * 4)
    require(record["projected_arithmetic_bytes"] == dict(client_to_inference=rows * request,
            inference_to_client=rows * reply, preparation_to_inference=(rows if forced else n + 23) * reply),
            "pilot original arithmetic projection")
    require(all(math.isfinite(record[k]) and record[k] >= 0 for k in ("cpu_seconds", "wall_seconds")), "pilot finite timings")


def audit_pilot(d, hashes, *, exact_refit=True):
    p, manifest, artifact, attempt = (d[k] for k in ("pilot", "manifest", "helper", "attempt"))
    identity = manifest["manifest_digest"]
    require(digest({k: v for k, v in manifest.items() if k != "manifest_digest"}) == identity
            == p["manifest_digest"], "pilot manifest digest")
    expected_code = {"scripts/train_checkpoint_tiny_model.py": PINS["train_checkpoint_tiny_model.py"],
                     "benchmarks/research/checkpoint_tiny_model.py": PINS["checkpoint_tiny_model.py"]}
    require(manifest["code"] == expected_code, "pilot original code/source lock")
    require(attempt["status"] == "complete" and attempt["exit_code"] == 0
            and attempt["report_sha256"] == hashes[FILES["pilot"]]
            and Path(attempt["pilot_output"]).name == FILES["pilot"], "pilot report/attempt mismatch")
    require(p["artifact_sha256"] == hashes[FILES["helper"]], "pilot artifact file mismatch")
    require(p["artifact_bytes"] == p["cold_additional_client_artifact_bytes"], "pilot artifact costs")
    require(attempt["external_dependency"]["sha256"] == PINS["decoder_probe_support.py"]
            and attempt["dependency_unchanged_at_completion"] is True
            and "after launch" in attempt["external_dependency"]["scope"], "pilot after-launch dependency caveat")
    require(digest({k: v for k, v in artifact.items() if k != "artifact_digest"})
            == artifact["artifact_digest"], "helper artifact digest")
    tiny = snapshot("checkpoint_tiny_model.py", execute=True)
    data = manifest["data"]
    require(data["policy"] == tiny.POLICY and data["git_blob"] == tiny.DATA_BLOB
            and data["url"] == tiny.DATA_URL and digest(data["rows"]) == data["split_digest"], "frozen data/policy lock")
    require(p["source"]["pilot"] == tiny.PILOT == manifest["preflight"]["pilot"]
            and p["source"]["files"] == manifest["preflight"]["files"], "pilot source/tokenizer lock")
    for name, value in tiny.PILOT["files"].items():
        require(p["source"]["files"][name]["sha256"] == value, "pilot checkpoint lock")
    require(set(p["source"]["files"]) == {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"},
            "pilot complete file inventory")
    rows = data["rows"]
    require(len(rows) == 20 and len({r["id"] for r in rows}) == 20, "pilot row inventory")
    by_id = {r["id"]: r for r in rows}
    for split, count in tiny.POLICY["counts"].items():
        cohort = [r for r in rows if r["split"] == split]
        require(sum(r["kind"] == "public_qa" for r in cohort) == count
                and [r for r in cohort if r["kind"] != "public_qa"] == tiny.stress_rows(split), "pilot split/stress inventory")
        require(all(tiny.split_for(r["group"]) == split for r in cohort if r["kind"] == "public_qa"), "pilot article split")
        for other in tiny.POLICY["counts"]:
            if other != split:
                require(not ({r["group"] for r in cohort} & {r["group"] for r in rows if r["split"] == other})
                        and not ({r["context"] for r in cohort} & {r["context"] for r in rows if r["split"] == other}), "pilot cross-split overlap")
    train = [r for r in rows if r["split"] == "train"]
    require([r["id"] for r in train] == [r["id"] for r in p["teacher_training"]], "pilot train-only teacher inventory")
    for row in p["teacher_training"]:
        require(len(row["candidates"]) == 3, "pilot teacher candidate count")
        for record in row["candidates"]:
            pilot_record(record, forced=True)
    losses = [[c["answer_nll"] for c in r["candidates"]] for r in p["teacher_training"]]
    model = tiny.fit(train, losses, digest(p["source"]), epochs=tiny.POLICY["epochs"])
    saved_weights = np.asarray(artifact["weights"])
    bit_exact = np.array_equal(model.weights, saved_weights)
    # Cross-platform BLAS reduction order may change final float64 rounding.
    # Original-build replay stays exact; portable replay declares its tolerance.
    tolerance = 0.0 if exact_refit else 1e-12
    require(saved_weights.shape == model.weights.shape
            and np.allclose(model.weights, saved_weights, rtol=0, atol=tolerance)
            and artifact["source_binding"] == model.source_binding
            and artifact["training_split_digest"] == model.training_split_digest
            and artifact["parameters"] == p["parameters"] == model.parameter_inventory
            and sum(p["parameters"].values()) == 264
            and artifact["role"] == "extractive_helper_not_standalone", "pilot original refit/artifact mismatch")
    evaluation = p["evaluation"]
    require([e["id"] for e in evaluation] == [r["id"] for r in rows if r["split"] != "train"], "pilot evaluation inventory/order")
    fallback_count = Counter()
    for e in evaluation:
        r = by_id[e["id"]]
        require((e["split"], e["kind"]) == (r["split"], r["kind"]), "pilot evaluation identity")
        choices = tiny.candidates(r)
        contexts = dict(full=r["context"], head=choices[0]["text"], tail=choices[1]["text"],
                        lexical=choices[2]["text"], learned=model.choose(r)["text"])
        require(set(e["results"]) == set(contexts) | {"learned_then_full"}, "pilot control inventory")
        for mode, context in contexts.items():
            rec = e["results"][mode]
            pilot_record(rec)
            require(rec["em"] == normalized_em(rec["text"], r["answers"])
                    and rec["literal_exact"] == (rec["text"].strip() in r["answers"])
                    and rec["answer_retention"] == any(a in context for a in r["answers"]), "pilot row grading")
        for mode, context in contexts.items():
            if context == contexts["learned"]:
                require(all(e["results"][mode][key] == e["results"]["learned"][key]
                            for key in ("input_digest", "plan_digest", "text", "eos", "selected_tokens")),
                        "pilot selected-context causal identity")
        learned, final = e["results"]["learned"], e["results"]["learned_then_full"]
        fallback = not learned["eos"] or not learned["text"].strip() or learned["text"].strip() not in contexts["learned"]
        retry = final["retry"]
        require(final["fallback"] == fallback and (retry is not None) == fallback, "pilot fallback policy")
        if retry is not None:
            pilot_record(retry)
            require(all(retry[key] == e["results"]["full"][key]
                        for key in ("input_digest", "plan_digest", "text", "eos", "selected_tokens")),
                    "pilot full-retry source input")
        actual = retry if fallback else learned
        require(final["em"] == normalized_em(actual["text"], r["answers"])
                and final["literal_exact"] == (actual["text"].strip() in r["answers"])
                and final["eos"] == actual["eos"], "pilot final grading")
        for k in ("cpu_seconds", "wall_seconds"):
            require(math.isclose(final[k], e["helper_work"][k] + learned[k] + (retry[k] if fallback else 0), abs_tol=1e-9), "pilot fallback CPU/wall cost")
        for key, value in final["projected_arithmetic_bytes"].items():
            require(value == learned["projected_arithmetic_bytes"][key] + (retry["projected_arithmetic_bytes"][key] if fallback else 0), "pilot fallback complete cost")
        fallback_count[e["split"]] += fallback
    summaries = {}
    for split in ("dev", "heldout"):
        cohort = [r for r in evaluation if r["split"] == split]
        summary = {}
        for mode in evaluation[0]["results"]:
            qa = [r["results"][mode] for r in cohort if r["kind"] == "public_qa"]
            stress = [r["results"][mode] for r in cohort if r["kind"] != "public_qa"]
            summary[mode] = dict(qa_em=sum(r["em"] for r in qa) / len(qa), qa_count=len(qa), qa_correct=sum(r["em"] for r in qa),
                                 stress_count=len(stress), stress_literal_correct=sum(r["literal_exact"] for r in stress),
                                 stress_literal_exact=sum(r["literal_exact"] for r in stress) / len(stress),
                                 all_eos=all(r["results"][mode]["eos"] for r in cohort),
                                 projected_arithmetic_bytes=sum(sum(r["results"][mode]["projected_arithmetic_bytes"].values()) for r in cohort))
        full, candidate, gates = summary["full"], summary["learned_then_full"], tiny.POLICY["gates"]
        ratio = candidate["projected_arithmetic_bytes"] / full["projected_arithmetic_bytes"]
        learned, lexical = summary["learned"], summary["lexical"]
        summary["gate"] = dict(arithmetic_ratio=ratio,
                               screen_pass=candidate["qa_em"] >= max(gates["minimum_qa_em"], full["qa_em"])
                               and candidate["stress_literal_exact"] >= gates["minimum_stress_em"] and candidate["all_eos"]
                               and ratio <= gates["max_projected_arithmetic_ratio"],
                               learned_adds_value_over_lexical=learned["qa_em"] > lexical["qa_em"] or
                               (learned["qa_em"] == lexical["qa_em"] and learned["projected_arithmetic_bytes"] < lexical["projected_arithmetic_bytes"]))
        require(summary == p["summary"][split], "pilot summary reconciliation")
        summaries[split] = dict(qa_correct=candidate["qa_correct"], qa_count=candidate["qa_count"], fallbacks=fallback_count[split],
                                full_projected_bytes=full["projected_arithmetic_bytes"], final_projected_bytes=candidate["projected_arithmetic_bytes"], **summary["gate"])
    require(p["maximum_swap_growth_bytes"] == 0, "pilot swap growth")
    return dict(parameters=264, refit_exact=bit_exact, refit_within_tolerance=True,
                refit_absolute_tolerance=tolerance, refit_max_abs_error=float(np.max(np.abs(model.weights - saved_weights))), splits=summaries,
                dependency_caveat="Numeric helper hash first observed after launch; unchanged through completion, not a prelaunch freeze.",
                scope="Original frozen manifest rows/teacher losses; corpus bytes, logits/KV and causal target execution not rerun.")


def audit_head(d, hashes, native):
    head, previous = d["head"], d["native"]
    require(previous["scan_sha256"] == hashes[FILES["head"]]
            and previous["auditor_sha256"] == PINS["audit_compact_private_head.py"], "head native report/source binding")
    scanner = snapshot("probe_compact_private_head.py", execute=True)
    cert = dict(metadata=encoded(head["certificate"]), widths=bytes.fromhex(head["widths_hex"]), bounds=bytes.fromhex(head["bounds_le_u64_hex"]))
    scanner.validate_pin(head["pin"])
    widths, bounds = cert["widths"], np.frombuffer(cert["bounds"], "<u8")
    require(len(widths) == len(bounds) == head["pin"]["vocabulary"]
            and len(widths) <= 262144 and digest(head["certificate"]) == head["certificate_sha256"]
            and sha(widths) == head["certificate"]["width_sha256"]
            and sha(cert["bounds"]) == head["certificate"]["bounds_sha256"], "head certificate arrays/digest")
    require(all(head["certificate"].get(k) == v for k, v in head["pin"].items())
            and widths == bytes(max(1, int(b).bit_length() + 1) for b in bounds)
            and all(int(b) % 127 == 0 and int(b) <= head["pin"]["hidden"] * 128 * 127 for b in bounds), "head source/bound certificate")
    require(head["verification"] == dict(full_passes=2, every_row_scalar_verified=True,
                                        installed_native_every_row_verified=True), "head source verification")
    computed = scanner.costs(cert, head["pin"]["hidden"], calls=head["costs"]["head_calls"])
    require(head["costs"]["zlib_runtime_version"] == "1.2.12", "head original compression runtime metadata")
    compared = json.loads(encoded(computed))
    if not native:
        # Version label is historical, not a claim about today's portable runtime.
        # All actual regenerated compressed lengths/costs still must match.
        compared["zlib_runtime_version"] = head["costs"]["zlib_runtime_version"]
    require(compared == head["costs"], "head complete cost ledger")
    # The fixed archived validator checks all bound/layout hashes and codec costs.
    auditor = snapshot("audit_compact_private_head.py", execute=True)
    if native:
        modules = original_native_modules(head)
        result = auditor.audit(head, native=modules[0])
    else:
        # Costs/certificate only: no mock codec and no claimed reconstruction.
        signed = bounds.astype(np.int64)
        vectors = [np.zeros(len(widths), dtype=np.int64), signed, -signed,
                   np.where(np.arange(len(widths)) % 2, signed, -signed),
                   np.random.default_rng(60217).integers(-signed, signed + 1, dtype=np.int64)]
        result = dict(certificate_sha256=head["certificate_sha256"], source_pin=head["pin"],
                      output_sha256=[sha(v.tobytes()) for v in vectors], checked_integer_outputs=5 * len(widths),
                      payload_bytes_both=computed["reply_payload_bytes_both"],
                      baseline_payload_bytes_both=computed["baseline_reply_payload_bytes_both"],
                      framed_saved_bytes_both_per_head_estimate=computed["saved_reply_bytes_both_per_head_estimate"],
                      cold_net_saved_bytes_estimate=computed["net_saved_bytes_at_head_calls_estimate"])
    require(all(previous[k] == value for k, value in result.items()), "head archived native audit mismatch")
    locked = {Path(k).name: v for k, v in head["source_locks"].items() if "/site-packages/" not in k}
    require(locked == {"probe_compact_private_head.py": PINS["probe_compact_private_head.py"],
                       "frontier_probe_support.py": PINS["frontier_probe_support.py"],
                       "offset_transport.rs": file_hash(ROOT / "crates/pllm-core/src/offset_transport.rs")}, "head scanner source lock")
    return dict(native_replayed=native, checked_integer_outputs=result["checked_integer_outputs"] if native else 0,
                compression_runtime_matches_original=computed["zlib_runtime_version"] == head["costs"]["zlib_runtime_version"],
                payload_bytes_both=result["payload_bytes_both"], saved_framed_bytes_per_head_estimate=result["framed_saved_bytes_both_per_head_estimate"],
                cold_net_saved_bytes_estimate=result["cold_net_saved_bytes_estimate"],
                scope="Per-row domain codec and modeled costs; no actual W*x or decoder rerun. Native/installed checks skipped." if not native else result["scope"])


def original_native_modules(head):
    """Strict original-build dependency check, shared with opt-in CI regression.

    Missing wheel, another OS/NumPy ABI or changed installed code is NOT a native
    replay failure: callers may explicitly skip that integration test. CLI's
    default remains strict and never silently downgrades its requested replay.
    """
    names = ("pllm._native", "pllm.runtime.private_boundary_loader", "pllm.runtime.private_boundary_head_codec",
             "pllm.runtime.quantization", "pllm.runtime.transformer_client", "pllm.runtime.transformer_engine",
             "pllm.model_loader", "numpy", "numpy._core._multiarray_umath", "numpy.random._generator", "numpy.random._pcg64")
    modules = [importlib.import_module(name) for name in names]
    recorded = {k.split("/site-packages/", 1)[1]: v for k, v in head["source_locks"].items() if "/site-packages/" in k}
    installed = {}
    for module in modules:
        path = str(Path(module.__file__))
        require("/site-packages/" in path, "head installed source lock: original wheel layout unavailable")
        installed[path.split("/site-packages/", 1)[1]] = file_hash(path)
    require(recorded == installed, "head installed source lock: original binary/NumPy/SDK dependencies differ")
    return modules


def audit_conveyor(d):
    report, lock = d["conveyor"], d["conveyor_lock"]
    require(report["source_sha256"] == lock["source_sha256"], "conveyor source lock")
    require(report["status"] == "complete" and report["phase"] == "closed" and report["failure"] is None
            and report["cohort_exact_and_reconciled"] and report["controls_equal_outputs"] and not report["serving_claim"], "conveyor cohort status")
    companion = snapshot("probe_mlp_conveyor.py", execute=True)
    sys.modules["probe_mlp_conveyor"] = companion
    validator = snapshot("probe_mlp_conveyor_transport.py", execute=True)
    expected_paths = set(validator.SOURCES)
    require(set(report["source_sha256"]) == expected_paths, "conveyor source inventory")
    actual = {p: snapshot(Path(p).name) if Path(p).name in PINS else file_hash(ROOT / p) for p in expected_paths}
    require(actual == report["source_sha256"], "conveyor archived source binding")
    validator.source_hashes = lambda: actual
    plan = [dict(rows=rows, hidden=896, intermediate=4864, tile=64, mode=mode) for rows in (150, 288) for mode in ("serial", "conveyor", "full")]
    require(report["plan"] == plan and len(report["cases"]) == len(report["attempts"]) == 6, "conveyor six-case inventory")
    accepted, summary = [], []
    for index, (case, payload, attempt) in enumerate(zip(report["cases"], plan, report["attempts"], strict=True)):
        require(attempt == dict(index=index, payload=payload, status="accepted") and case["acceptance_status"] == "accepted", "conveyor accepted attempt order")
        validator.validate_accepted_case(case, payload, actual, report["build"], case["compiler_eligibility"], accepted)
        descriptors = validator.descriptors(validator.geometry(**payload))
        intervals = {(e["event"], e["sequence"]): e["seconds"] for e in case["events"]}
        overlap = 0.0
        for seq, (stage, tile, _, _) in enumerate(descriptors):
            if stage != "G" or tile == 0:
                continue
            for prior, (other, previous, _, _) in enumerate(descriptors[:seq]):
                if other == "G" and previous == tile - 1:
                    a, b, p, q = [intervals[k] for k in (("client_send_start", seq), ("client_send_end", seq),
                                                        ("provider_reply_start", prior), ("provider_reply_end", prior))]
                    overlap += max(0., min(b, q) - max(a, p))
        require(overlap == case["next_gate_upload_vs_previous_gate_reply_write_overlap_seconds"], "conveyor narrow overlap counter")
        accepted.append(case)
        summary.append(dict(rows=payload["rows"], mode=payload["mode"], frames=case["frames_expected"],
                            directed_bytes=case["ledger"]["client_tx"] + case["ledger"]["server_tx"], narrow_overlap_seconds=overlap))
    guards = report["guard_observations"]
    require(len(guards) >= 7 and all(g["guard_error"] is None and g["maximum_swap_growth_bytes"] == 0
                                    and "sample_error" not in g for g in guards), "conveyor guard/swap growth")
    original = d["conveyor_original"]
    require(original["source_sha256"] == d["conveyor_original_lock"]["source_sha256"]
            and len(original["cases"]) == 6, "conveyor original report/source lock")
    return dict(cases=summary, maximum_swap_growth_bytes=0, initial_swap_used_bytes=guards[0]["initial_host"]["swap_used"],
                scope="Six fixed-order single observations; v2 owner/counter/hash replay, no numerical kernel rerun or speedup inference.",
                original_v1="Retained/hash-bound only; original pre-repair driver is not the v2 archived source.")


def retained_provenance(report, source):
    """Authenticate retained page metadata without reading cached PDF/text bytes."""
    provenance = report["source_provenance"]
    require(digest(provenance) == PROVENANCE_DIGEST, "correlation retained provenance metadata")
    raw = bounded(ROOT / "docs/evidence/compact-setup-fulltext-review.json")
    require(sha(raw) == PRIMARY_MANIFEST_DIGEST, "correlation primary manifest lock")
    papers = [p for p in json.loads(raw)["papers"] if p["id"] in source.SOURCE_PAGES]
    require([p["id"] for p in provenance["papers"]] == [p["id"] for p in papers], "correlation primary-source inventory")
    for record, paper in zip(provenance["papers"], papers, strict=True):
        identity = paper["id"]
        require(record["pdf_sha256"] == paper["sha256"] and record["pdf_bytes"] == paper["bytes"]
                and record["pdf"] == paper["path"] and record["url"] == paper["url"]
                and record["text_sha256"] == source.TEXT_HASHES[identity], "correlation retained paper identity")
        require([e["pdf_page"] for e in record["reviewed_excerpts"]] == list(source.SOURCE_PAGES[identity])
                and all(1 <= e["pdf_page"] <= paper["pages"]
                        and re.fullmatch(r"[0-9a-f]{64}", e["page_text_sha256"])
                        and source.SOURCE_PAGES[identity][e["pdf_page"]] in e["excerpt"]
                        for e in record["reviewed_excerpts"]), "correlation retained page metadata")
    return provenance


def replay_correlation(source, provenance):
    """Replay bounded arithmetic using pinned original functions, no live metadata.

    Original report() also hashes live docs/test files. Keep those incidental
    historical identities explicit instead of rewriting/ignoring their hashes.
    """
    rng = random.Random(610031)
    transcripts, outputs = {}, 0
    for mode in source.MODES:
        client, inference = source.setup(source.W, 7, mode, rng)
        authority, traces = source.Authority(client.context), []
        for index in range(7):
            x = tuple(rng.randrange(-source.MOD, source.MOD) for _ in source.W[0])
            got, trace = source.query(client, inference, authority, index, x)
            oracle = tuple(sum(w * v for w, v in zip(row, x, strict=True)) & 0xffffffff for row in source.W)
            require(got == oracle == source.lookup_mv(source.W, x), "correlation independent ring oracle")
            trace.update(chosen_input=list(x), reconstructed=list(got))
            traces.append(trace)
            outputs += len(got)
        transcripts[mode] = dict(public_fixture_only=True, rows=traces,
                                 client_key_fields=list(client.__dict__), inference_key_fields=list(inference.__dict__))
    costs = {}
    for label, shape, rows, cap, waste in (("small", (3, 5), 7, 4, 2), ("stage_analytic_only", (9728, 896), 288, 128, 1)):
        baseline = source.ledger(*shape, rows, cap, "prepared", waste)
        costs[label] = {}
        for mode in source.MODES:
            cost = source.ledger(*shape, rows, cap, mode, waste)
            costs[label][mode] = dict(cost, gate=source.admission(cost, baseline))
        table = source.lookup_stage_ledger(*shape, rows, cap, waste)
        costs[label]["zero_lookup"] = dict(table, gate=source.admission(table, baseline))
    return dict(schema="pllm.fixed_weight_correlation_gate.v1", ring_bits=32, matrix=source.W,
                checked_outputs=outputs, transcripts=transcripts, leakage=source.leakage_controls(), ledgers=costs,
                public_linear_mask_controls=dict(even_full_rational_rank=source.linear_mask_gate(((2, 0), (0, 2))),
                    public_rank_one=source.linear_mask_gate(((1,), (3,))), surjective_control=source.linear_mask_gate(((1, 2, 3), (0, 1, 4)))),
                lookup_control=dict(small=source.lookup_ledger(3, 5), stage_analytic_only=source.lookup_ledger(9728, 896)),
                source_provenance=provenance, implementation_hashes=CORRELATION_IMPLEMENTATION,
                conclusion="No cheap-client construction admitted. Zero correction can hide inputs but does not remove masked-image production cost.",
                production_crypto_implemented=False, serving_saving_measured=False)


def audit_correlation(d, *, primary_sources=True):
    report = d["correlation"]
    source = snapshot("probe_fixed_weight_correlation_gate.py", execute=True)
    require(report["implementation_hashes"] == CORRELATION_IMPLEMENTATION, "correlation original implementation metadata lock")
    provenance = retained_provenance(report, source)
    if primary_sources:
        for paper in provenance["papers"]:
            require((ROOT / paper["pdf"]).stat().st_size <= 16 << 20
                    and (ROOT / paper["text_path"]).stat().st_size <= 8 << 20, "correlation primary-source size bound")
        require(source.provenance(ROOT) == provenance, "correlation cached primary-source reproduction")
    require(json.loads(encoded(replay_correlation(source, provenance))) == report, "correlation normalized reproduction")
    return dict(checked_outputs=report["checked_outputs"], normalized_reproduction=True,
                primary_sources_reverified=primary_sources,
                primary_source_scope="Cached pinned PDF/text bytes and page excerpts rechecked; no latest-revision certification."
                if primary_sources else "Pinned retained paper/hash/page metadata only; original PDF/text bytes NOT reverified.",
                original_implementation_hashes=CORRELATION_IMPLEMENTATION,
                production_crypto_implemented=False, scope=report["conclusion"])


def audit(evidence_dir=ROOT / "docs/evidence", *, native=True, primary_sources=True):
    documents, hashes, sizes = {}, {}, {}
    for key, name in FILES.items():
        raw = bounded(Path(evidence_dir) / name)
        documents[key] = json.loads(raw)
        hashes[name] = sha(raw)
        sizes[key] = len(raw)
    for name in PINS:
        snapshot(name)
    require(documents["pilot"]["artifact_bytes"] == sizes["helper"], "pilot artifact measured length")
    require(documents["native"]["native_build"] == documents["head"]["installed_build"]
            == documents["pilot"]["build"] == documents["conveyor"]["build"], "cross-track installed build binding")
    require(documents["head"]["pin"]["checkpoint_sha256"] == documents["pilot"]["source"]["checkpoint_digest"]
            and documents["head"]["pin"]["source_lock_sha256"] == documents["pilot"]["source"]["source_lock_digest"], "cross-track checkpoint source identity")
    result = dict(schema="pllm.sota-breakthrough-independent-audit.v2", status="verified_bounded_evidence",
                  report_sha256=hashes, trusted_snapshot_sha256=PINS,
                  auditor_sha256=file_hash(Path(__file__)), serving_qualified=False, memory_tier_qualified=False)
    # The historical attempt retains its old target-path spelling. Resolve only
    # this fixed public log basename, never arbitrary paths supplied by reports.
    require(Path(documents["attempt"]["stderr_log"]).name == PILOT_LOG.removesuffix(".txt"), "pilot stderr log identity")
    log_raw = bounded(Path(evidence_dir) / PILOT_LOG)
    require(json.loads(log_raw) == documents["pilot"]["summary"], "pilot archived log/summary mismatch")
    result["pilot_log"] = dict(status="archived_summary_reconciled", basename=PILOT_LOG, sha256=sha(log_raw))
    result["pilot"] = audit_pilot(documents, hashes, exact_refit=native)
    result["head"] = audit_head(documents, hashes, native)
    result["conveyor"] = audit_conveyor(documents)
    result["correlation"] = audit_correlation(documents, primary_sources=primary_sources)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "docs/evidence", help="JSON archive directory; executable snapshots always fixed repository paths")
    parser.add_argument("--skip-native", action="store_true", help="certificate/cost/domain-hash checks only; no native codec or installed-source validation")
    parser.add_argument("--skip-primary-sources", action="store_true", help="pinned retained provenance metadata only; do not require or reverify cached PDF/text bytes")
    parser.add_argument("--output", type=Path, required=True, help="exclusive fresh JSON output; existing files never overwritten")
    args = parser.parse_args(argv)
    # Reserve exclusively before replay. Failure leaves an explicit incomplete
    # attempt, never a successful artifact or an overwritten prior report.
    with args.output.open("x") as stream:
        stream.write('{"status":"incomplete"}\n')
        stream.flush()
        result = audit(args.evidence_dir, native=not args.skip_native, primary_sources=not args.skip_primary_sources)
        stream.seek(0)
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.truncate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
