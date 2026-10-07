"""Clear free-running semantic gated-block trajectories, with a fresh control."""
from __future__ import annotations

import hashlib

import numpy as np


def generation(fixture, bridges, prompts, check, *, outputs=8):
    trials = []
    layers = {op["layer"] for op in fixture.plan.prefill["operations"] if op["operator"] == "silu"}
    for text in prompts:
        ids = fixture.tokens(text)
        reference = []
        for name, bridge in [("reference", None), ("control", None), *bridges.items()]:
            runtime = fixture.compiled.runtime(fixture.remote)
            original = runtime._local
            pending, seen = {}, set()

            def local(operation, values, *state):
                check()
                result = original(operation, values, *state)
                if operation["operator"] == "silu":
                    pending[operation["id"]] = values[operation["inputs"][0]]
                elif operation["operator"] == "multiply":
                    keys = [key for key in operation["inputs"] if key in pending]
                    if len(keys) == 1:
                        gate = pending.pop(keys[0])
                        up = values[next(key for key in operation["inputs"] if key != keys[0])]
                        seen.add(operation["layer"])
                        if bridge is not None:
                            return bridge.evaluate(gate, up)
                return result

            runtime._local = local
            selected, rejection = [], None
            try:
                _, logits, _ = runtime.prepare_ids(ids)
                for index in range(outputs):
                    check()
                    if index:
                        logits = runtime.forward_ids([selected[-1]])[-1]
                    selected.append(int(np.argmax(logits)))
                    if selected[-1] == int(fixture.bundle.cfg["eos_token_id"]):
                        break
                if pending or seen != layers:
                    raise ValueError("generation does not cover every semantic gated region")
            except ValueError as error:
                if str(error) != "profile domain rejection" or bridge is None:
                    raise
                rejection = str(error)
            finally:
                del runtime._local
            if name == "reference":
                reference = selected
            else:
                prefix = 0
                for a, b in zip(reference, selected):
                    if a != b:
                        break
                    prefix += 1
                trials.append({
                    "candidate": name, "input_tokens": len(ids),
                    "public_input_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
                    "reference_output_tokens": len(reference), "generated_tokens": len(selected),
                    "matching_prefix_tokens": prefix,
                    "matching_positions": sum(a == b for a, b in zip(reference, selected)),
                    "sequence_match": rejection is None and selected == reference,
                    "output_sha256": hashlib.sha256(np.asarray(selected, dtype="<i8").tobytes()).hexdigest(),
                    "rejection": rejection,
                })
            original = None
            del runtime, local
    summary = {}
    for name in ["control", *bridges]:
        rows = [row for row in trials if row["candidate"] == name]
        summary[name] = {
            "attempted_prompts": len(rows),
            "rejected_prompts": sum(row["rejection"] is not None for row in rows),
            "exact_trajectories": sum(row["sequence_match"] for row in rows),
            "generated_tokens": sum(row["generated_tokens"] for row in rows),
            "reference_output_tokens": sum(row["reference_output_tokens"] for row in rows),
            "matching_positions": sum(row["matching_positions"] for row in rows),
            "matching_prefix_tokens": sum(row["matching_prefix_tokens"] for row in rows),
        }
    return {"summary": summary, "trials": trials, "maximum_generated_tokens": outputs,
            "sampling": "greedy; each candidate feeds back its own selected token; stop on EOS"}
