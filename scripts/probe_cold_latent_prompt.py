#!/usr/bin/env python3
"""Bounded synthetic latent-prompt falsification; not checkpoint inference or wire measurement."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MIB = 1 << 20


def samples(seed: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique unrelated length-eight prompts, with unseen token IDs in rare split."""
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, ...]] = set()
    rows = []
    while len(rows) < 384:
        row = tuple(int(x) for x in rng.integers(0, 8, size=8))
        if row not in seen:
            seen.add(row)
            rows.append(row)
    train, test = np.asarray(rows[:256]), np.asarray(rows[256:])
    rare = test.copy()
    rare[np.arange(128), np.arange(128) % 8] = 8 + np.arange(128) % 8
    return train, test, rare


def pooled(tokens: np.ndarray, slots: int) -> np.ndarray:
    if tokens.ndim != 2 or tokens.shape[1] != 8 or slots not in (1, 2, 4, 8):
        raise ValueError("length eight; slots must divide length and be 1, 2, 4, or 8")
    if np.any(tokens < 0) or np.any(tokens >= 16):
        raise ValueError("token outside vocabulary")
    return np.eye(16)[tokens].reshape(len(tokens), slots, 8 // slots, 16).mean(axis=2)


def fit(train: np.ndarray, slots: int) -> np.ndarray:
    """Learn width-eight linear autoencoder via uncentered PCA of chunk histograms.

    Basis signs canonicalized. Decoder tied to transpose; no held-out fitting.
    Learned embedding table has 16*8 doubles, rather than an LLM encoder.
    """
    h = pooled(train, slots).reshape(-1, 16)
    _, _, vt = np.linalg.svd(h, full_matrices=False)
    basis = vt[:8].T.copy()
    for col in range(8):
        pivot = int(np.argmax(np.abs(basis[:, col])))
        if basis[pivot, col] < 0:
            basis[:, col] *= -1
    return basis


def reconstruct(tokens: np.ndarray, slots: int, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    latent = pooled(tokens, slots) @ basis
    # 16-bit signed fixed-point boundary; not a tested production quantizer.
    latent = np.clip(np.rint(latent * 4096), -32768, 32767).astype(np.int16)
    hist = latent.astype(np.float64) / 4096 @ basis.T
    # Toy decoder knows chunk-count grid. Snap after fixed-point transport;
    # ties then select lowest token ID. Not a language-model decoder trick.
    count = 8 // slots
    hist = np.rint(hist * count) / count
    copy = np.repeat(hist.argmax(axis=2), 8 // slots, axis=1)
    return copy, hist


def quality(seed: int = 4) -> dict:
    train, test, rare = samples(seed)
    entries = []
    max_arrays = 0
    for slots in (1, 2, 4, 8):
        basis = fit(train, slots)
        splits = {}
        for name, data in (("train", train), ("heldout_common", test), ("heldout_rare", rare)):
            pred, hist = reconstruct(data, slots, basis)
            truth_hist = np.eye(16)[data].sum(axis=1)
            splits[name] = {
                "prompts": len(data),
                "exact_copy": int(np.all(pred == data, axis=1).sum()),
                "token_correct": int((pred == data).sum()),
                "token_total": int(data.size),
                "majority_correct": int((hist.sum(axis=1).argmax(axis=1)
                                         == truth_hist.argmax(axis=1)).sum()),
                "first_position_correct": int((pred[:, 0] == data[:, 0]).sum()),
                "rare_token_correct": int(((pred == data) & (data >= 8)).sum()),
                "rare_token_total": int((data >= 8).sum()),
            }
        # Explicit reversal within each chunk: same histogram, different exact answer.
        left = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7]])
        right = left.reshape(1, slots, 8 // slots)[:, :, ::-1].reshape(1, 8)
        collision = bool(np.array_equal(pooled(left, slots), pooled(right, slots)))
        differing = bool(not np.array_equal(left, right))
        entries.append({"slots": slots, "width": 8, "latent_boundary_bytes": slots * 8 * 2,
                        "learned_embedding_parameters": int(basis.size),
                        "splits": splits, "collision_same_representation": collision,
                        "collision_different_copy_answer": differing,
                        "collision_pair": [left.tolist()[0], right.tolist()[0]]})
        # Conservative live-array ceiling includes dense one-hot/PCA workspace allowance.
        max_arrays = max(max_arrays, 32 * (256 * 8 * 16 * 8 + 16 * 16 * 8))
    assert max_arrays <= 128 * MIB
    return {"seed": seed, "train_test_overlap": 0, "train_prompts": 256,
            "heldout_common_prompts": 128, "heldout_rare_prompts": 128,
            "full_prompt_oracle_exact_copy": 128, "results": entries,
            "added_arrays_conservative_ceiling_bytes": max_arrays,
            "capacity_note": "finite precision only: k*d*b bits; pooling loses order even when bit capacity is ample",
            "exact_copy_input_bits": 8 * 4,
            "smallest_latent_bits": 1 * 8 * 16,
            "training": "uncentered PCA; tied linear decoder; common token IDs 0..7 only; no checkpoint"}


def ledger(prompt: int, slots: int, outputs: int, encoder_bytes: int,
           requests: int = 1) -> dict:
    """Illustrative dense Qwen-shaped all-directed-body ledger, not runtime counters.

    Every request has unrelated prompt and empty KV. Terminal pruning, actual
    codecs and framing differ; fixed row byte assumptions explicit, not measured.
    """
    if min(prompt, slots, outputs, requests) < 1 or slots > prompt or encoder_bytes < 0:
        raise ValueError("positive lengths/horizon; slots <= prompt; encoder nonnegative")
    layers, hidden, intermediate, vocab = 24, 896, 4864, 151936
    # qkv (896 -> 1152), o (896 -> 896), gate/up (896 -> 9728), down (4864 -> 896)
    input_words = layers * (3 * hidden + intermediate)
    output_words = layers * (1152 + 2 * hidden + 2 * intermediate)
    word = 4  # hypothetical 32-bit field words, not packed/residue production frames
    decode_rows = outputs - 1
    common_delivery = vocab * hidden + vocab * 4 + 4 * MIB + 2 * MIB
    extra_fit = 12 * MIB
    extra_adapter = 2 * MIB  # changed provider matrices also sent to Preparation
    setup = 1 * MIB

    def bodies(rows: int, latent: bool) -> dict[str, int]:
        total_rows = requests * (rows + decode_rows)
        return {
            "Inference->Client:tokenizer_tied_embedding_head_boundary": common_delivery,
            "Inference->Client:encoder_weights": encoder_bytes if latent else 0,
            "Inference->Client:fitting_scales_manifest": extra_fit if latent else 0,
            "Inference->Preparation:decoder_adapter": extra_adapter if latent else 0,
            "Client->Preparation:setup_seeds_control": requests * setup,
            "Preparation->Inference:setup_control": requests * setup,
            "Client->Inference:masked_stage_inputs": total_rows * input_words * word,
            "Inference->Client:masked_stage_outputs": total_rows * output_words * word,
            "Preparation->Inference:fresh_stage_corrections": total_rows * output_words * word,
            "Client->Inference:admission_position_controls": requests * (4096 + rows * 8),
            "Inference->Client:final_control": requests * 4096,
        }

    base, latent = bodies(prompt, False), bodies(slots, True)
    saved_before_encoder = sum(base.values()) - (sum(latent.values()) - encoder_bytes)
    row_saving = (prompt - slots) * (input_words + 2 * output_words) * word + (prompt - slots) * 8
    fixed_extra = extra_fit + extra_adapter + encoder_bytes
    break_even = fixed_extra // row_saving + 1 if row_saving > 0 else None
    return {"prompt_tokens": prompt, "soft_slots": slots, "output_tokens": outputs,
            "unrelated_requests": requests, "encoder_weight_bytes": encoder_bytes,
            "baseline_links": base, "latent_links": latent,
            "baseline_bytes": sum(base.values()), "latent_bytes": sum(latent.values()),
            "net_saved_bytes": sum(base.values()) - sum(latent.values()),
            "encoder_bytes_strict_break_even_ceiling": max(-1, saved_before_encoder - 1),
            "strict_positive_saving_requests": break_even,
            "stage_input_words_per_row": input_words, "stage_output_words_per_row": output_words,
            "field_word_bytes": word, "dense_schedule_projection_only": True,
            "byte_scope": "directed application bodies; excludes TCP/TLS framing; no timing claim"}


def owners(encoder_bytes: int, slots: int = 32, outputs: int = 8,
           boundary_word_bytes: int = 1) -> dict:
    if encoder_bytes < 0 or slots < 1 or outputs < 1 or boundary_word_bytes not in (1, 2, 4):
        raise ValueError("invalid logical-owner configuration")
    items = {
        "client_encoder_weights": encoder_bytes,
        "client_tokenizer": 4 * MIB,
        "client_tied_embedding_head": 151936 * 896 * boundary_word_bytes,
        "client_boundary_scales_and_metadata": 151936 * 4 + 2 * MIB,
        "client_encoder_and_decoder_workspace_allowance": 32 * MIB,
        "client_one_window_masks_and_corrections_allowance": 4 * MIB,
        "client_active_KV_float32": 2 * 24 * 2 * 64 * (slots + outputs - 1) * 4,
        "client_logits_float32": 151936 * 4,
        "client_soft_boundary_int32": slots * 896 * 4,
    }
    total = sum(items.values())
    return {"logical_owners": items, "logical_total_bytes": total,
            "tiers": {name: {"budget_bytes": budget, "admitted_logical": total <= budget,
                              "remaining_bytes": budget - total}
                      for name, budget in (("Low256MiB", 256 * MIB), ("Medium2GiB", 2048 * MIB))},
            "not_RSS": True, "excludes": ["Python/runtime/native allocator", "mapping/page copies",
                                              "transport buffers", "training optimizer", "disk artifacts"],
            "fitting_artifacts_disk_bytes": 12 * MIB,
            "provider_decoder_adapter_bytes": 2 * MIB,
            "preparation_adapter_copy_bytes": 2 * MIB}


def report() -> dict:
    sources = ["scripts/probe_cold_latent_prompt.py", "tests/test_cold_latent_prompt.py",
               "docs/research/cold-latent-prompt-reference.md",
               "python/pllm/runtime/masked_runtime.py", "crates/pllm-core/src/kv_cache.rs",
               "crates/pllm-core/src/rms_norm.rs", "crates/pllm-python/src/packed_embedding.rs",
               "crates/pllm-core/src/prepared_stage.rs",
               "crates/pllm-models/src/dense_gated_source.rs"]
    return {"schema": "cold-latent-prompt-synthetic.v1", "synthetic_only": True,
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
            "quality": quality(),
            "cold_capacity_sweep": [ledger(n, k, 8, e * MIB)
                                    for n in (32, 150, 512) for k in (16, 32) if k <= n
                                    for e in (16, 128, 1024)],
            "memory_sweep": [owners(e * MIB, boundary_word_bytes=b)
                             for e in (16, 128, 1024) for b in (1, 2)],
            "claim": "hypothesis screen only; no protected runtime integration, real model quality, RSS or timings"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(report(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
