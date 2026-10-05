"""Screen five new all-link hypotheses and five 100x constructions.

Pinned real-checkpoint clear-kernel parity and bounded algebra/cost gates, not
an admitted private decoder. Never archive prompts, token IDs, logits or KV.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import msgpack
import numpy as np
import psutil

from aggregate_network_hypotheses import (
    anchor_output, century_budgets, integer_anchors, lifting_forest, packed_size,
    residue_widths, restore_lifted, rounding_debt_witness, snapshot_digest,
    symmetry_privacy_witness, terminal_stage_ids,
)
from decoder_probe_support import DecoderFixture, PROMPT, digest


class MeteredRemote:
    def __init__(self, fixture, selected=frozenset()):
        self.fixture, self.selected = fixture, selected
        self.counts = Counter()
        self.kernel_cpu = 0.0

    def __call__(self, key, activation):
        from pllm.runtime.quantization import quantize_activation_per_row, dequantize_matmul
        stage = self.fixture.body[key]
        value = np.asarray(activation)
        selected = key in self.selected and value.shape[-2] > 1
        work = value[..., -1:, :] if selected else value
        q = quantize_activation_per_row(work, bits=8)
        rows, bits = q.rows, stage.seeded_profile.wire_bits
        self.counts["online_input_bytes"] += rows * stage.spec.in_features * bits // 8
        self.counts["online_output_bytes"] += rows * stage.spec.out_features * bits // 8
        self.counts["offline_correction_bytes"] += rows * stage.spec.out_features * bits // 8
        self.counts["integer_macs_one_worker"] += rows * stage.spec.in_features * stage.spec.out_features
        self.counts["remote_rows"] += rows
        self.counts["rows_elided"] += value.size // stage.spec.in_features - rows
        started = time.process_time()
        integer = stage.compiled_weight.clear(q.values)
        self.kernel_cpu += time.process_time() - started
        result = dequantize_matmul(integer, q.scales, stage.weight.scales,
                                  output_shape=work.shape[:-1] + (stage.spec.out_features,))
        if stage.bias is not None:
            result = result + stage.bias
        if selected:
            # Non-demanded outputs have no path to logits or retained state.
            # Keep original local shapes in this deliberately isolated oracle.
            output = np.zeros(value.shape[:-1] + (stage.spec.out_features,), np.float32)
            output[..., -1:, :] = result
            return output
        return np.ascontiguousarray(result, dtype=np.float32)


def execute(fixture, compiled, ids, count, selected=frozenset(), snapshot=None, composition=None):
    remote = MeteredRemote(fixture, selected)
    runtime = compiled.runtime(remote)
    start = time.process_time()
    if snapshot is None:
        _, logits, _ = runtime.prepare_ids(ids)
    else:
        runtime.install_continuation(compiled._plan.continuation_schedule(composition))
        runtime.restore(snapshot)
        logits = runtime.continue_ids(ids)[-1]
    logit_hash = hashlib.sha256()
    outputs = []
    for index in range(count):
        logit_hash.update(logits.astype("<f4").tobytes())
        outputs.append(int(np.argmax(logits)))
        if index + 1 < count:
            logits = runtime.forward_ids([outputs[-1]])[-1]
    total_cpu = time.process_time() - start
    counts = dict(remote.counts)
    all_link = sum(counts.get(key, 0) for key in
                   ("online_input_bytes", "online_output_bytes", "offline_correction_bytes"))
    return {"generated_outputs": count, "output_digest": digest(outputs),
            "all_logit_bits_digest": logit_hash.hexdigest(), "kv_digest": snapshot_digest(runtime),
            "arithmetic_bodies": counts, "all_link_arithmetic_bytes": all_link,
            "all_link_arithmetic_MB_per_generated_token": all_link / count / 1e6,
            "local_client_algorithm_cpu_seconds": total_cpu - remote.kernel_cpu,
            "clear_provider_kernel_cpu_seconds": remote.kernel_cpu,
            "active_kv_bytes": sum(c.length * c.key[0].nbytes * 2 for c in runtime.caches)}, outputs


def check_parity(a, b):
    keys = ("generated_outputs", "output_digest", "all_logit_bits_digest", "kv_digest")
    if any(a[key] != b[key] for key in keys):
        raise RuntimeError("candidate changes generated outputs, logits or retained KV")


def demand_probe(fixture, outputs):
    stages = terminal_stage_ids(fixture.compiled)
    if not stages:
        raise RuntimeError("no conservatively demand-prunable stages were found")
    rows = []
    for text in (PROMPT, "Describe one benefit of keeping medical queries private.",
                 "Give a short explanation of anonymous model inference."):
        ids = fixture.tokens(text)
        baseline, _ = execute(fixture, fixture.compiled, ids, outputs)
        candidate, _ = execute(fixture, fixture.compiled, ids, outputs, stages)
        check_parity(baseline, candidate)
        rows.append({"input_tokens": len(ids), "control": baseline, "candidate": candidate})
    return {"candidate": "terminal-row demand slicing", "selected_stages": sorted(stages),
            "cohort": rows, "full_logit_and_KV_bits_equal": True,
            "scope": "exact remote-row elision oracle; original local shapes retained; arithmetic bodies only",
            "new_client_weights_bytes": 0, "new_client_state_bytes": 0,
            "decision": "numeric gate passed; needs compiler-owned row-demand and reservation schedule before promotion"}


def basis_probes(fixture, rows):
    rng = np.random.default_rng(61008)
    records = []
    for key, stage in fixture.body.items():
        weight = stage.weight.values
        widths = residue_widths(weight)
        ordinary = packed_size(widths, rows)
        entry = {"stage": key, "weight_digest": stage.weight_digest,
                 "raw_residue_output_bytes": ordinary, "anchors": []}
        for block in (32, 128):
            if weight.shape[1] % block:
                continue
            start = time.process_time()
            residual, anchor = integer_anchors(weight, block)
            count = min(16, len(weight))
            x = rng.integers(-127, 128, (3, weight.shape[1]), dtype=np.int64)
            correction = anchor_output(anchor[:count], x, block)
            actual = x @ residual[:count].astype(np.int64).T + correction
            expected = x @ weight[:count].astype(np.int64).T
            if not np.array_equal(actual, expected):
                raise RuntimeError("integer anchor identity failed")
            # Verify every transformed coefficient, independently of test inputs.
            restored = residual.astype(np.int16).reshape(weight.shape[0], -1, block) + anchor[..., None]
            if not np.array_equal(restored.reshape(weight.shape), weight):
                raise RuntimeError("anchor transform changed a source coefficient")
            new_size = packed_size(residue_widths(residual), rows)
            entry["anchors"].append({"block": block,
                "output_plus_correction_bytes_saved": 2 * (ordinary - new_size),
                "new_client_anchor_bytes": anchor.nbytes,
                "new_client_integer_macs": rows * anchor.size,
                "cold_net_bytes_saved_before_framing": 2 * (ordinary - new_size) - anchor.nbytes,
                "public_compilation_and_separate_oracle_cpu_seconds": time.process_time() - start})
        residual, edges = lifting_forest(weight)
        x = rng.integers(-127, 128, (1, weight.shape[1]), dtype=np.int64)
        if not np.array_equal(restore_lifted(x @ residual.astype(np.int64).T, edges),
                              x @ weight.astype(np.int64).T):
            raise RuntimeError("lifting identity failed")
        new_size = packed_size(residue_widths(residual), rows)
        entry["lifting"] = {"selected_edges": len(edges),
            "output_plus_correction_bytes_saved": 2 * (ordinary - new_size),
            "public_forest_bytes": len(msgpack.packb(edges)),
            "extra_client_additions": len(edges) * rows,
            "native_i8_residual_compatible": bool(np.all((residual >= -128) & (residual <= 127)))}
        records.append(entry)
    return {"candidates": ["exact integer anchor splitting", "lossless output-basis lifting"],
            "executed_rows": rows, "stages": records, "exact_integer_oracle": True,
            "privacy": "input masks stay full-width; public source weights alone determine residual layouts",
            "scope": "compared against existing row residues, not uncompressed responses; input and control bodies unchanged"}


def capsule_probe(fixture, outputs):
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.state import ClientPrefixReuse
    from pllm.runtime.model_binding import compile_runtime_model
    composition = MaskedLinearCpu(fixture.composition.model,
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
        cache=ClientPrefixReuse(fixed_input_tokens=288, max_bytes=16 << 20))
    compiled = compile_runtime_model(fixture.plan, fixture.bundle, composition=composition)
    # Fixed public service policy, before private user text. This disclosure must
    # be chosen in public configuration; never infer it by inspecting a prompt.
    tokenizer = fixture.bundle.tokenizer()
    public_ids = tokenizer.encode("Public service policy: explain concepts accurately and concisely. " * 30)[:256]
    suffixes = [tokenizer.encode(text) for text in
                (" Explain masks.", " What is a private query?", " Describe one privacy benefit.")]
    public_runtime = compiled.runtime(fixture.remote)
    start = time.process_time()
    public_runtime.prepare_ids(public_ids)
    snapshot = public_runtime.snapshot()
    public_cpu = time.process_time() - start
    capsule = msgpack.packb({"v": 1, "binding": compiled.digest,
        "public_token_digest": digest(public_ids), "position": snapshot.position,
        "states": [[c.key[:c.length].astype("<f4").tobytes(), c.value[:c.length].astype("<f4").tobytes()]
                    for c in snapshot.caches]}, use_bin_type=True)
    # Explicit zero-residency scenarios. These are public artifact lengths,
    # not measured HTTP transfers, and do not include full-wire/control costs.
    bundle_bytes = len(fixture.engine.client_bundle_document(fixture.source.manifest.id))
    checkpoint_bytes = sum(path.stat().st_size for path in fixture.source.path.glob("*.safetensors"))
    records = []
    for suffix in suffixes:
        baseline, _ = execute(fixture, compiled, public_ids + suffix, outputs)
        candidate, _ = execute(fixture, compiled, suffix, outputs, snapshot=snapshot, composition=composition)
        check_parity(baseline, candidate)
        first_bytes = candidate["all_link_arithmetic_bytes"] + len(capsule)
        records.append({"private_suffix_tokens": len(suffix), "control": baseline,
                         "candidate": candidate, "capsule_plus_arithmetic_bytes": first_bytes,
                         "arithmetic_plus_capsule_reduction_factor": baseline["all_link_arithmetic_bytes"] / first_bytes,
                         "cold_client_bundle_and_arithmetic_reduction_factor":
                             (baseline["all_link_arithmetic_bytes"] + bundle_bytes) / (first_bytes + bundle_bytes),
                         "cold_two_provider_checkpoint_plus_client_bundle_reduction_factor":
                             (baseline["all_link_arithmetic_bytes"] + bundle_bytes + 2 * checkpoint_bytes) /
                             (first_bytes + bundle_bytes + 2 * checkpoint_bytes)})
    return {"candidate": "public-prefix state capsules", "public_prefix_tokens": len(public_ids),
            "composition_digest": composition.digest(), "capsule_bytes": len(capsule),
            "capsule_sha256": hashlib.sha256(capsule).hexdigest(),
            "public_once_per_capsule_compilation_cpu_seconds": public_cpu,
            "raw_client_bundle_bytes": bundle_bytes,
            "source_safetensors_bytes_per_provider": checkpoint_bytes,
            "cohort": records, "full_logit_and_KV_bits_equal": True,
            "scope": "trusted in-process producer snapshot; cold capsule body charged once per independent request; no portable importer or role transport",
            "decision": "requires explicit public-prefix contract and authenticated source/numeric/state admission"}


def finite_response_probe(fixture, outputs):
    from pllm.metrics.private_pages import PrivatePageLookupProbe
    start = time.process_time()
    records, baseline_bytes, baseline_cpu = [], [], []
    # Public bounded task grammar. All 16 possible private selectors are compiled.
    # This cannot service arbitrary text or preserve an arbitrary earlier history.
    for left in range(4):
        for right in range(4):
            ids = fixture.tokens(f"Answer only with a number. What is {left} plus {right}?")
            report, tokens = execute(fixture, fixture.compiled, ids, outputs)
            records.append(np.asarray(tokens, dtype="<u4").tobytes())
            baseline_bytes.append(report["all_link_arithmetic_bytes"])
            baseline_cpu.append(report["local_client_algorithm_cpu_seconds"] + report["clear_provider_kernel_cpu_seconds"])
    compiler_cpu = time.process_time() - start
    table = b"".join(records)
    result = PrivatePageLookupProbe(records=16, record_bytes=4 * outputs, queries=3).run(table)
    cold = result["body_bytes_per_query"] + 2 * len(table)
    return {"candidate": "finite-task private response capsules", "task_domain_cardinality": 16,
            "generated_outputs": outputs, "native_private_lookup": result,
            "public_table_compilation_cpu_seconds": compiler_cpu,
            "public_table_delivery_both_workers_bytes": 2 * len(table),
            "cold_table_plus_query_reply_bytes": cold,
            "smallest_matched_arithmetic_control_bytes": min(baseline_bytes),
            "narrow_cold_body_reduction_factor_at_least": min(baseline_bytes) / cold,
            "offline_compilation_amortization_responses_vs_single_clear_execution": compiler_cpu / float(np.median(baseline_cpu)),
            "new_client_model_weights_bytes": 0,
            "client_local_public_table_control": {"cold_table_delivery_bytes": len(table),
                "online_bytes": 0, "client_table_bytes": len(table),
                "decision": "At this tiny public task domain, local lookup beats two-server PIR bodies and trust costs."},
            "scope": "native two-server research PIR over exact fixed-length W8A8 continuations; fixed greedy task grammar; model checkpoint distribution and full wire omitted",
            "decision": "scope-restricted 100x candidate; no arbitrary-prompt support, authenticated role transport or independent cryptographic review"}


def frame_probe(fixture, outputs):
    from pllm.runtime.stage_protocol import MaskedStageRequest, PreparedStageBatchRequest
    entries = []
    for key, stage in fixture.body.items():
        profile = stage.seeded_profile
        value = np.zeros((1, stage.spec.in_features), np.uint32)
        full = MaskedStageRequest(model=fixture.bundle.model_id, stage_id=key,
            correlation_id="ab" * 16, masked_input=value, activation_scales=1.0,
            modulus=profile.modulus, wire_bits=profile.wire_bits, ring=profile.ring,
            body_fingerprint=fixture.lock()["body_fingerprint"], weight_digest=stage.weight_digest,
            weight_bits=8, activation_bits=8, session_id="cd" * 16,
            out_features=stage.spec.out_features, signed_output_bound=profile.signed_output_bound).pack()
        # Fresh response nonce plus one-use ticket retained. Other commitments
        # come from the checked immutable inventory, as in existing prefill.
        compact = PreparedStageBatchRequest("ef" * 16, ("ab" * 16,), value, profile.wire_bits).pack()
        entries.append({"stage": key, "full_bytes": len(full), "compact_bytes": len(compact)})
    return {"candidate": "lease-relative protocol frames", "decode_rows": outputs - 1,
            "stages": entries, "decode_upload_bytes_saved": (outputs - 1) * sum(e["full_bytes"] - e["compact_bytes"] for e in entries),
            "new_client_weights_bytes": 0, "new_persistent_client_state_bytes": 0,
            "scope": "serialized single-row request proof; fresh tickets/nonces remain; full live-session admission required"}


def run(output, outputs, sections):
    if psutil.virtual_memory().available < 4 << 30:
        raise RuntimeError("research fixture requires 4 GiB physical headroom")
    before_swap = psutil.swap_memory().used
    fixture = DecoderFixture(inputs=288, outputs=outputs)
    result = {"schema": "pllm.aggregate_network_hypotheses.v1", "complete": False,
              "created_at": datetime.now(timezone.utc).isoformat(), "source": fixture.lock(),
              "platform": platform.platform(), "units": "decimal bytes/MB per authoritative generated output",
              "scope": "bounded isolated research; no live pipeline or full-wire claim",
              "whole_client_peak_rss_bytes": None, "full_wire_bytes": None}
    probes = {"demand": lambda: demand_probe(fixture, outputs),
              "bases": lambda: basis_probes(fixture, 39 + outputs - 1),
              "capsule": lambda: capsule_probe(fixture, outputs),
              "frames": lambda: frame_probe(fixture, outputs),
              "finite": lambda: finite_response_probe(fixture, outputs)}
    for name in sections:
        print(f"Checking {name}", flush=True)
        result[name] = probes[name]()
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        gc.collect()
        if psutil.swap_memory().used > before_swap + (64 << 20):
            raise RuntimeError("research probe stopped on new swap pressure")
    result["century_gates"] = {"public_basis": symmetry_privacy_witness(),
        "rounding_debt": rounding_debt_witness(),
        "budget": century_budgets(body_bytes=178_970_000, rows=70, layers=24, hidden=896, outputs=32),
        "packed_threshold_resident_decoder": {
            "hypothesis": "cross-request threshold-key SIMD, retain hidden state and selection within non-colluding workers",
            "gate": "a lane batch amortizes ciphertexts, not per-client secret-shared arithmetic; needs complete nonlinear/key-switch schedule and independent cryptographic review",
            "measured_reduction_factor": None},
        "one_island_distilled_decoder": {
            "hypothesis": "jointly train a public-affine recurrent trunk with one narrow private nonlinear settlement per output, instead of one cut per original layer",
            "gate": "trained source, language quality, exact private rescaling and selection must fit the all-link budget; cannot retrofit Qwen by suppressing its nonlinear operators",
            "measured_reduction_factor": None}}
    result.update(complete=True, new_swap_bytes=max(0, psutil.swap_memory().used - before_swap),
                  entire_research_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                  (1 if platform.system() == "Darwin" else 1024))
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"complete": True, "source": result["source"], "sections": sections,
                      "new_swap_bytes": result["new_swap_bytes"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outputs", type=int, choices=(8, 32), default=8)
    parser.add_argument("--sections", nargs="+", choices=("demand", "bases", "capsule", "frames", "finite"),
                        default=["demand", "bases", "capsule", "frames", "finite"])
    args = parser.parse_args()
    run(args.output, args.outputs, args.sections)
