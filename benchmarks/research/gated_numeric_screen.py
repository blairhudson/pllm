"""Reusable clear semantic-MLP screen with an independent fresh-runtime control."""
import hashlib
import json
import statistics

import numpy as np

from benchmarks.research.piecewise_gated_reference import snapshot, state_digest


def screen(fixture, bridges, prompts, check, *, positions=4, gate_bound=48, up_bound=128):
    trials = []
    layers = {op["layer"] for op in fixture.plan.prefill["operations"] if op["operator"] == "silu"}
    for text in prompts:
        check()
        ids = fixture.tokens(text)
        reference = []
        for name, bridge in [("reference", None), ("control", None), *bridges.items()]:
            runtime = fixture.compiled.runtime(fixture.remote)
            original = runtime._local
            pending, seen = {}, set()
            stats = {"elements": 0, "gate_outside": 0, "up_outside": 0, "tail_elements": 0,
                     "max_abs_gate": 0.0, "max_abs_up": 0.0, "worst_block_error": 0.0}

            def local(operation, values, *state):
                check()
                result = original(operation, values, *state)
                if operation["operator"] == "silu":
                    pending[operation["id"]] = np.array(values[operation["inputs"][0]], copy=True)
                elif operation["operator"] == "multiply":
                    keys = [key for key in operation["inputs"] if key in pending]
                    if len(keys) == 1:
                        gate = pending.pop(keys[0])
                        up = np.asarray(values[next(key for key in operation["inputs"] if key != keys[0])])
                        seen.add(operation["layer"])
                        stats["elements"] += gate.size
                        stats["gate_outside"] += int(np.count_nonzero(np.abs(gate) > gate_bound))
                        stats["up_outside"] += int(np.count_nonzero(np.abs(up) > up_bound))
                        stats["tail_elements"] += int(np.count_nonzero(np.abs(gate) >= 16))
                        stats["max_abs_gate"] = max(stats["max_abs_gate"], float(np.max(np.abs(gate))))
                        stats["max_abs_up"] = max(stats["max_abs_up"], float(np.max(np.abs(up))))
                        if bridge:
                            candidate = bridge.evaluate(gate, up)
                            stats["worst_block_error"] = max(stats["worst_block_error"], float(np.max(np.abs(result-candidate))))
                            return candidate
                return result

            runtime._local = local
            observations, rejected = [], None
            try:
                _, logits, _ = runtime.prepare_ids(ids)
                for position in range(positions):
                    if position:
                        logits = runtime.forward_ids([reference[position-1][2]])[-1]
                    states = snapshot(runtime)
                    selected = int(np.argmax(logits))
                    if name == "reference":
                        reference.append((logits.copy(), states, selected))
                        continue
                    expected, expected_states, expected_token = reference[position]
                    if len(states) != len(expected_states) or any(a.shape != b.shape for a, b in zip(states, expected_states, strict=True)):
                        raise ValueError("full-KV layout differs")
                    observations.append({"position": position, "selection_match": selected == expected_token,
                        "all_logits_bit_exact": np.array_equal(logits.view(np.uint32), expected.view(np.uint32)),
                        "worst_abs_logit_error": float(np.max(np.abs(logits-expected))),
                        "top5_recall": len(set(np.argsort(logits)[-5:]) & set(np.argsort(expected)[-5:]))/5,
                        "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(), "kv_sha256": state_digest(states),
                        "kv_bit_exact": all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(states, expected_states, strict=True)),
                        "kv_worst_abs_error": max(float(np.max(np.abs(a-b))) for a, b in zip(states, expected_states, strict=True))})
                if pending or seen != layers:
                    raise ValueError("joint override lacks full semantic MLP coverage")
            except ValueError as error:
                if str(error) != "profile domain rejection" or bridge is None:
                    raise
                rejected = str(error)
            finally:
                del runtime._local
            if name != "reference":
                trials.append({"candidate": name, "input_tokens": len(ids),
                    "public_tokens_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
                    "positions": observations, "domain": stats, "rejection": rejected})
            original = None
            del runtime, local
    summary = {}
    for name in ["control", *bridges]:
        rows = [r for r in trials if r["candidate"] == name]
        checked = [p for row in rows for p in row["positions"]]
        summary[name] = {"attempted_prompts": len(rows), "rejected_prompts": sum(r["rejection"] is not None for r in rows),
            "prefill_matches": sum(p["selection_match"] for p in checked if p["position"] == 0),
            "decode_matches": sum(p["selection_match"] for p in checked if p["position"] > 0),
            "completed_positions": len(checked), "bit_exact_positions": sum(p["all_logits_bit_exact"] for p in checked),
            "exact_kv_positions": sum(p["kv_bit_exact"] for p in checked),
            "worst_abs_logit_error": max((p["worst_abs_logit_error"] for p in checked), default=None),
            "mean_top5_recall": statistics.mean(p["top5_recall"] for p in checked) if checked else None,
            "screen_pass": len(checked) == positions*len(prompts) and all(p["selection_match"] for p in checked),
            "native_oracle_checked_elements": bridges[name].checked if name in bridges else 0}
    return {"source": fixture.lock(), "public_heldout_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
            "reference": "pinned compiled W8A8; separate fresh-runtime control; same-token decode, not upstream FP32 or broad quality",
            "summary": summary, "trials": trials, "protected_checkpoint_execution": False}
