"""Bounded clear grammar boundary oracle and source-priced PIR accounting.

No SDK imports, checkpoint tensors, network execution, or privacy simulation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["scripts/probe_cold_grammar_boundary.py", "tests/test_cold_grammar_boundary.py",
           "python/pllm/runtime/transformer_engine.py",
           "python/pllm/runtime/transformer_client.py", "python/pllm/protocols/__init__.py",
           "crates/pllm-garble/src/private_pages.rs",
           "docs/research/cold-grammar-boundary-reference.md"]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(rows, scales):
    return hashlib.sha256(canonical({"rows": rows, "scales": scales})).hexdigest()


def specialize(rows, scales, grammar, expected_identity):
    if not rows or not rows[0] or len(rows) > 256 or len(rows[0]) > 64:
        raise ValueError("synthetic geometry exceeds bound")
    if len(scales) != len(rows) or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("shape differs")
    if any(type(x) is not int or not -128 <= x <= 127 for r in rows for x in r):
        raise ValueError("not signed i8")
    if any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("invalid scale")
    if identity(rows, scales) != expected_identity:
        raise ValueError("source identity differs")
    if grammar.get("visibility") != "PUBLIC":
        raise ValueError("private prompt-derived union cannot be delivered")
    states = grammar["states"]
    if not states or any(not allowed for allowed in states):
        raise ValueError("empty allowed state")
    ids = sorted(set(t for state in states for t in state))
    if any(type(t) is not int or not 0 <= t < len(rows) for t in ids):
        raise ValueError("missing token")
    if grammar["eos"] not in ids or grammar["eos"] not in states[-1]:
        raise ValueError("missing EOS")
    return {"ids": ids, "rows": [rows[t] for t in ids], "scales": [scales[t] for t in ids],
            "source_identity": expected_identity, "grammar": copy.deepcopy(grammar)}


def logits(rows, scales, hidden):
    if any(len(r) != len(hidden) for r in rows):
        raise ValueError("hidden shape differs")
    # Declared clear contract: signed integer dot; multiply row f32 scale in
    # binary64. Not evidence of SDK float32 reduction/dequantization parity.
    return [sum(a * b for a, b in zip(r, hidden)) * s for r, s in zip(rows, scales)]


def embedding_record(row, scale):
    return struct.pack(f"<{len(row)}bf", *row, scale)


def recover_embedding(record, width):
    if width < 1 or len(record) != width + 4:
        raise ValueError("embedding response shape differs")
    values = struct.unpack(f"<{width}bf", record)
    scale = values[-1]
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("embedding response scale differs")
    return [struct.unpack("<f", struct.pack("<f", x * scale))[0] for x in values[:-1]]


def distribution(values, ids, allowed, temperature, top_p):
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("invalid sampling parameters")
    if len(values) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("logit identity/shape differs")
    if not set(allowed) <= set(ids) or not allowed:
        raise ValueError("missing token")
    scores = {t: v for t, v in zip(ids, values) if t in allowed}
    if any(not math.isfinite(v) for v in scores.values()):
        raise ValueError("nonfinite logit")
    ranked = sorted(scores, key=lambda t: (-scores[t], t))
    if temperature == 0:
        return {t: float(t == ranked[0]) for t in sorted(scores)}
    peak = scores[ranked[0]]
    weights = {t: math.exp((scores[t] - peak) / temperature) for t in ranked}
    total = sum(weights.values())
    # Match current SDK's cumulative<=top_p convention (always keep maximum),
    # NOT common nucleus convention that also keeps the crossing token.
    # Global token-ID tie break is declared here; SDK argsort ties unspecified.
    keep, mass = [], 0.0
    for t in ranked:
        mass += weights[t] / total
        if mass <= top_p or not keep or top_p == 1:
            keep.append(t)
    norm = sum(weights[t] for t in keep)
    return {t: weights[t] / norm if t in keep else 0.0 for t in sorted(scores)}


def select(probabilities, uniform):
    if not 0 <= uniform < 1:
        raise ValueError("uniform outside [0,1)")
    mass = 0.0
    for t, p in sorted(probabilities.items()):
        mass += p
        if uniform < mass:
            return t
    return max(t for t, p in probabilities.items() if p > 0)


def accounting(vocab=151936, hidden=896, union=256, prompt_slots=16,
               output_slots=8, extra_public_bytes=0, worker_cold=False):
    if any(type(n) is not int or n < 0 for n in
           (union, prompt_slots, output_slots, extra_public_bytes)):
        raise ValueError("invalid accounting geometry")
    if not 1 <= union <= vocab <= 262144 or not 1 <= hidden + 4 <= 65536:
        raise ValueError("unsupported private-page geometry")
    record = hidden + 4
    if vocab * record > 512 * 1024 * 1024 or prompt_slots + output_slots > 4096:
        raise ValueError("private-page admission exceeded")
    depth = (vocab - 1).bit_length()
    # Exact native format: 65-byte header + 16 root + 17/depth + final bit.
    query_each, reply_each = 82 + 17 * depth, 65 + record
    lookups = prompt_slots + output_slots  # conservative fixed dummy final slot
    online = lookups * 2 * (query_each + reply_each)
    full = vocab * record
    # Proposed two public 48-byte descriptors (digest32, count u64, width u64).
    # No native descriptor transport exists. No per-client key setup in DPF.
    descriptor_setup = 96
    head = union * record
    token_map = union * 4
    client = head + token_map + extra_public_bytes + descriptor_setup + online
    worker_setup = 2 * full if worker_cold else 0
    saved = full - client - worker_setup
    per_request = 2 * (query_each + reply_each)
    remaining = full - head - token_map - extra_public_bytes - descriptor_setup - worker_setup
    return {"vocabulary": vocab, "hidden": hidden, "union": union,
            "full_tied_raw_i8_and_f32_bytes": full, "head_payload_bytes": head,
            "token_id_map_bytes": token_map, "additional_public_bytes": extra_public_bytes,
            "descriptor_setup_bytes": descriptor_setup, "cryptographic_key_setup_bytes": 0,
            "query_bytes_each_worker_each_slot": query_each,
            "reply_bytes_each_worker_each_slot": reply_each,
            "fixed_lookup_slots": lookups, "lookup_online_bytes": online,
            "client_boundary_total_bytes": client, "worker_cold_distribution_bytes": worker_setup,
            "global_total_bytes": client + worker_setup, "net_saved_bytes": saved,
            "strict_saving_max_lookup_slots": max(-1, (remaining - 1) // per_request),
            "replicated_table_bytes_each_worker": full,
            "two_worker_masked_xor_byte_operations": 2 * full * lookups,
            "two_worker_unbatched_table_traversal_bytes": 2 * full * lookups,
            "two_worker_prompt_batched_plus_serial_decode_table_bytes":
                2 * full * ((prompt_slots + 63) // 64 + output_slots),
            "prompt_batch_limit": 64,
            "depth": depth, "worker_cold": worker_cold,
            "excluded_common_costs": ["tokenizer/chat-template", "body bundle/setup/transport",
                                      "TLS/framing/authentication", "hidden/cache transport"],
            "status": "source-priced proposal; no private provider consumer"}


def fixture():
    rows = [[((t * 7 + d * 3) % 17) - 8 for d in range(4)] for t in range(12)]
    scales = [struct.unpack("<f", struct.pack("<f", (t + 1) / 16))[0] for t in range(12)]
    grammar = {"visibility": "PUBLIC", "eos": 11, "states": [[2, 5], [7, 9], [11]]}
    return rows, scales, grammar


def clear_generation(prompt_ids, restricted, uniform=0.5):
    """Tiny clear body recurrence, NOT transformer/provider execution.

    Arbitrary prompt embeddings seed cache. Each emitted non-EOS token adds its
    embedding; EOS never enters cache. Public state masks advance locally.
    """
    rows, scales, grammar = fixture()
    if not prompt_ids or any(type(t) is not int or not 0 <= t < len(rows) for t in prompt_ids):
        raise ValueError("invalid prompt token")
    shard = specialize(rows, scales, grammar, identity(rows, scales))
    cache = list(prompt_ids)
    hidden = [0.0] * 4

    def advance(token):
        embedding = recover_embedding(embedding_record(rows[token], scales[token]), 4)
        for d in range(4):
            hidden[d] += embedding[d]

    for t in prompt_ids:
        advance(t)
    emitted, steps = [], []
    for state, allowed in enumerate(grammar["states"]):
        # Explicit toy activation quantizer; no upstream execution.
        activation = [max(-127, min(127, round(v))) for v in hidden]
        values = logits(shard["rows"], shard["scales"], activation) if restricted else logits(rows, scales, activation)
        ids = shard["ids"] if restricted else list(range(len(rows)))
        probabilities = distribution(values, ids, allowed, 1.3, 1)
        token = select(probabilities, uniform)
        steps.append({"state": state, "activation": activation,
                      "distribution": probabilities, "selected": token, "position": len(cache)})
        if token == grammar["eos"]:
            break
        emitted.append(token)
        cache.append(token)
        advance(token)
    return {"prompt_ids": list(prompt_ids), "emitted": emitted, "cache_ids": cache,
            "hidden": hidden, "steps": steps, "eos_selected": steps[-1]["selected"] == grammar["eos"]}


def evidence():
    rows, scales, grammar = fixture()
    shard = specialize(rows, scales, grammar, identity(rows, scales))
    records = b"".join(embedding_record(r, s) for r, s in zip(rows, scales))
    for r, s in zip(rows, scales):
        if recover_embedding(embedding_record(r, s), 4) != [x * s for x in r]:
            raise AssertionError("embedding bytes differ")
    cases = []
    for hidden in ([1, -2, 3, 4], [0, 0, 0, 0]):
        full = logits(rows, scales, hidden)
        restricted = logits(shard["rows"], shard["scales"], hidden)
        for state, allowed in enumerate(grammar["states"]):
            for temperature, top_p in ((0, 1), (0.7, 1), (1.3, 0.8)):
                a = distribution(full, list(range(len(rows))), allowed, temperature, top_p)
                b = distribution(restricted, shard["ids"], allowed, temperature, top_p)
                draws = [0, 0.125, 0.5, 0.999999]
                if a != b or [select(a, u) for u in draws] != [select(b, u) for u in draws]:
                    raise AssertionError("masked/shard distribution differs")
                cases.append({"hidden": hidden, "state": state, "temperature": temperature,
                              "top_p": top_p, "distribution": a,
                              "selections": [select(a, u) for u in draws]})
    generations = []
    for prompt in ([0, 1, 10], [4, 6, 8]):  # every prompt ID outside output union
        for uniform in (0.125, 0.875):
            full = clear_generation(prompt, False, uniform)
            if full != clear_generation(prompt, True, uniform):
                raise AssertionError("clear toy generation/cache differs")
            generations.append(full)
    return {"schema": "cold-grammar-boundary-synthetic/v1",
            "scope": "clear synthetic oracle + source-format accounting, not private execution",
            "sampling_contract": "mask first; SDK cumulative<=top_p; global-ID ascending ties and inverse CDF; binary64 oracle, not SDK RNG/float32 parity",
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES},
            "fixture_identity": identity(rows, scales), "numeric_cases": cases,
            "embedding_record_sha256": hashlib.sha256(records).hexdigest(),
            "embedding_clear_recovered_rows": len(rows),
            "clear_toy_generation_not_transformer": generations,
            "qwen_geometry_note": "896 width is Qwen2.5-0.5B, not Qwen3-0.6B (1024)",
            "warm_workers": accounting(), "cold_workers": accounting(worker_cold=True),
            "negative_full_union": accounting(union=151936),
            "negative_private_union": {"visibility": "PRIVATE", "ids": [2, 5],
                                       "decision": "reject publication; reveals prompt-derived task data"},
            "negative_json": "arbitrary JSON strings: lexical union may approach full vocabulary; no 256-row claim",
            "privacy_execution": False}


def replay(document):
    """Check known fields exactly; retain unknown extension fields recursively."""
    fresh = json.loads(canonical(evidence()))

    def check(expected, actual, path):
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                raise ValueError(f"replay differs: {path}")
            for key, value in expected.items():
                if key not in actual:
                    raise ValueError(f"replay differs: {path}/{key}")
                check(value, actual[key], f"{path}/{key}")
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(expected) != len(actual):
                raise ValueError(f"replay differs: {path}")
            for index, (a, b) in enumerate(zip(expected, actual)):
                check(a, b, f"{path}/{index}")
        elif canonical(expected) != canonical(actual):
            raise ValueError(f"replay differs: {path}")

    check(fresh, json.loads(canonical(document)), "")
    return copy.deepcopy(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    result = replay(json.loads(args.replay.read_text())) if args.replay else evidence()
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
