"""Bounded causal hierarchy oracle and cold-request accounting, not a trained LM.

Only stdlib; no checkpoints, SDK modifications, timing, or large allocations.
Shares simulate two noncolluding providers, with deliberately public toy weights
and deterministic masks. This is NOT a secure private-model implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
MODULUS = 257
EOS = 3
SOURCES = (
    ("MegaByte", "https://arxiv.org/html/2305.07185v2", 326809,
     "2fbcb5d488a04c9cdc523fb15fc72188566d69175cfe3e75e0186a62042a9141"),
    ("Multi-token prediction", "https://arxiv.org/html/2404.19737v1", 457004,
     "b35a404107542155ee9b6ca7ecc9c54a6ac72f0d7cb58255b292cb1b2a7a4ee7"),
    ("Byte Latent Transformer", "https://arxiv.org/html/2412.09871v1", 379818,
     "6af016f8ddd245a01bada92842839ea4e9407f05e9eca94d12d99c14a0ce9c3d"),
    ("DeepSeek-V3", "https://arxiv.org/html/2412.19437v1", 550040,
     "4af6fcd13cccdf8d39e12efd22f09158d6034d6f341f72685d6f8c971e42dce5"),
)


def encode(tokens: Sequence[int]) -> tuple[int, int]:
    """Order- and length-sensitive private block feedback (not lossless)."""
    a, b = 0, len(tokens)
    for token in tokens:
        if token not in range(4):
            raise ValueError("toy token outside vocabulary")
        a = (5 * a + token + 1) % MODULUS
        b = (7 * b + token + 1) % MODULUS
    return a, b


def clear_commit(state: tuple[int, int], feedback: tuple[int, int]) -> tuple[int, int]:
    return tuple((3 * s + f) % MODULUS for s, f in zip(state, feedback))


def clear_forward(state: tuple[int, int]) -> tuple[int, int]:
    a, b = state
    return (2 * a + b + 11) % MODULUS, (a + 3 * b + 19) % MODULUS


class SharedGlobal:
    """Simulated party-local additive state; no remote clear feedback/latent/KV.

    Trusted test harness can inspect both parties. Public affine weights bypass
    all private-weight multiplication and nonlinear transformer difficulties.
    """

    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)  # Reproducibility only, never security.
        self.states = [(0, 0), (0, 0)]
        self.kv: list[list[tuple[int, int]]] = [[], []]
        self.pending = False
        self.forwards = 0
        self.feedbacks = 0
        self.input_shares: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def commit(self, feedback: tuple[int, int]) -> None:
        first = tuple(self.rng.randrange(MODULUS) for _ in feedback)
        second = tuple((f - r) % MODULUS for f, r in zip(feedback, first))
        self.input_shares.append((first, second))
        for party, share in enumerate((first, second)):
            old_a, old_b = self.states[party]
            self.states[party] = ((3 * old_a + share[0]) % MODULUS,
                                  (3 * old_b + share[1]) % MODULUS)
            self.kv[party].append(self.states[party])
        self.pending = False
        self.feedbacks += 1

    def forward(self) -> tuple[int, int]:
        if self.pending:
            raise ValueError("next block requires sampled-token feedback")
        outputs = []
        for party, (a, b) in enumerate(self.states):
            # Add public bias once, not to both shares.
            outputs.append(((2 * a + b + (11 if party == 0 else 0)) % MODULUS,
                            (a + 3 * b + (19 if party == 0 else 0)) % MODULUS))
        self.pending = True
        self.forwards += 1
        # Reconstruction occurs ONLY at trusted client in this oracle.
        return tuple(sum(o[j] for o in outputs) % MODULUS for j in range(2))

    def client_state(self) -> tuple[int, int]:
        return tuple(sum(s[j] for s in self.states) % MODULUS for j in range(2))


def probabilities(latent: tuple[int, int], history: Sequence[int], position: int,
                  *, allow_eos: bool = False) -> tuple[float, ...]:
    """Small causal local decoder: previous selected token changes next law."""
    previous = history[-1] if history else 0
    center = (latent[0] + 2 * latent[1] + previous + position) % 3
    weights = [1.0, 1.0, 1.0, 1.0 if allow_eos else 0.0]
    weights[center] = 6.0
    total = sum(weights)
    return tuple(w / total for w in weights)


def select(probs: Sequence[float], uniform: float, *, greedy: bool = False) -> int:
    if not 0 <= uniform < 1:
        raise ValueError("sampling uniform must be in [0,1)")
    if greedy:
        return max(range(len(probs)), key=lambda i: probs[i])
    cumulative = 0.0
    for i, probability in enumerate(probs):
        cumulative += probability
        if uniform < cumulative:
            return i
    raise ValueError("invalid categorical distribution")


def run_fixture(*, block_size: int = 4, max_tokens: int = 10,
                prompt: Sequence[int] = (0, 1, 2, 0, 1), seed: int = 17,
                greedy: bool = False, allow_eos: bool = False,
                eos_at: int | None = None, cancel_after: int | None = None,
                unused_prepared_blocks: int = 0) -> dict:
    if not 1 <= block_size <= 16 or not 0 <= max_tokens <= 64 or len(prompt) > 64:
        raise ValueError("fixture exceeds bounded dimensions")
    if unused_prepared_blocks not in range(65):
        raise ValueError("unbounded preparation")
    if eos_at is not None and not 0 <= eos_at < max_tokens:
        raise ValueError("EOS index outside request")
    if cancel_after is not None and not 0 <= cancel_after <= max_tokens:
        raise ValueError("cancel index outside request")
    remote = SharedGlobal()
    oracle = (0, 0)
    prefill_rows = 0
    for start in range(0, len(prompt), block_size):
        feedback = encode(prompt[start:start + block_size])
        remote.commit(feedback)
        oracle = clear_commit(oracle, feedback)
        prefill_rows += 1
    rng = random.Random(seed)
    emitted: list[int] = []
    blocks = []
    history: list[int] = list(prompt[-block_size:])
    stopped = "length"
    while len(emitted) < max_tokens:
        latent = remote.forward()
        assert latent == clear_forward(oracle)
        block: list[int] = []
        rows = []
        capacity = min(block_size, max_tokens - len(emitted))
        for position in range(capacity):
            # Eager forward precedes cancellation polling: explicit overissue.
            if cancel_after is not None and len(emitted) >= cancel_after:
                stopped = "cancel"
                break
            uniform = rng.random()
            probs = probabilities(latent, history, position, allow_eos=allow_eos)
            token = select(probs, uniform, greedy=greedy)
            if eos_at == len(emitted):
                # Controlled hazard injection, NOT learned EOS quality evidence.
                probs = (0.0, 0.0, 0.0, 1.0)
                token = select(probs, uniform, greedy=greedy)
            rows.append({"position": position, "previous": history[-1] if history else None,
                         "probabilities": probs, "uniform": uniform, "token": token})
            emitted.append(token)
            block.append(token)
            history = (history + [token])[-block_size:]
            if token == EOS:
                stopped = "eos"
                break
        if block:
            feedback = encode(block)
            remote.commit(feedback)  # Includes final partial/EOS block.
            oracle = clear_commit(oracle, feedback)
        blocks.append({"latent": latent, "capacity": capacity, "tokens": block,
                       "rows": rows, "committed": bool(block),
                       "client_state_after": remote.client_state()})
        if stopped != "length":
            break
    assert remote.client_state() == oracle
    reserved = (max_tokens + block_size - 1) // block_size + unused_prepared_blocks
    return {"block_size": block_size, "requested_original_tokens": max_tokens,
            "emitted_original_tokens": len(emitted), "tokens": emitted,
            "stop": stopped, "prefill_global_rows": prefill_rows,
            "remote_decode_steps": remote.forwards, "local_decoder_rows": len(emitted),
            "feedback_commits": remote.feedbacks - prefill_rows,
            "reserved_preparation_blocks": reserved,
            "unused_preparation_blocks": reserved - remote.forwards,
            "unused_issued_forwards": sum(not b["tokens"] for b in blocks),
            "unused_capacity_tokens": sum(b["capacity"] - len(b["tokens"]) for b in blocks),
            "provider_kv_rows_per_party": len(remote.kv[0]),
            "final_state": remote.client_state(), "blocks": blocks,
            "scope": "public-weight affine share simulation; no trained quality or security claim"}


@dataclass(frozen=True)
class CostSpec:
    """Illustrative complete ledger inputs; unknown real protocol stays unknown."""
    decoder_weights: int = 4 * 1024 * 1024
    encoder_weights: int = 1024 * 1024
    vocab_embedding_and_head: int = 128 * 1024
    tokenizer: int = 32 * 1024
    numeric_tables: int = 16 * 1024
    manifest: int = 4096
    setup_up: int = 1024
    setup_down: int = 2048
    provider_provisioning: int = 0  # Existing serving model, not new client setup.
    global_row_up: int = 8192
    global_row_down: int = 8192  # Includes latent shares, not extra clear latent.
    global_row_peer: int = 16384
    preparation_per_global_row: int = 65536
    feedback_payload_per_commit: int = 8192
    feedback_peer_per_commit: int = 4096
    frame_header: int = 64
    close_up: int = 256
    close_down: int = 256
    global_macs_per_row: int = 1_000_000
    commit_macs_per_row: int = 4096
    local_decoder_macs_per_token: int = 100_000
    local_encoder_macs_per_token: int = 20_000
    kv_bytes_per_row_per_party: int = 8192
    link_bytes_per_second: int = 12_500_000
    round_trip_ms: int = 40
    dependent_rounds_per_global_row: int = 8
    setup_rounds: int = 2
    local_macs_per_second: int = 100_000_000
    provider_macs_per_second: int = 500_000_000


def cold_cost(run: dict, spec: CostSpec = CostSpec(), *, prompt_tokens: int = 5) -> dict:
    if type(prompt_tokens) is not int or not 0 <= prompt_tokens <= 64:
        raise ValueError("prompt token count outside bounded request")
    values = asdict(spec)
    if any(type(v) is not int or v < 0 for v in values.values()):
        raise ValueError("cost inputs must be nonnegative integers")
    if min(spec.link_bytes_per_second, spec.local_macs_per_second,
           spec.provider_macs_per_second) <= 0:
        raise ValueError("rates must be positive")
    prefill = run["prefill_global_rows"]
    steps = run["remote_decode_steps"]
    rows = prefill + steps
    commits = prefill + run["feedback_commits"]
    prepared = prefill + run["reserved_preparation_blocks"]
    artifacts = sum(values[k] for k in ("decoder_weights", "encoder_weights",
                    "vocab_embedding_and_head", "tokenizer", "numeric_tables", "manifest"))
    ledger = {
        "client_artifact_download": artifacts,
        "setup_up": spec.setup_up, "setup_down": spec.setup_down,
        "provider_provisioning_all_links": spec.provider_provisioning,
        "global_forward_up": rows * spec.global_row_up,
        "global_forward_down": rows * spec.global_row_down,
        "global_forward_peer": rows * spec.global_row_peer,
        "preparation_download_including_unused": prepared * spec.preparation_per_global_row,
        "feedback_up_including_prefill_and_final": commits * spec.feedback_payload_per_commit,
        "feedback_peer": commits * spec.feedback_peer_per_commit,
        # Two provider directions for input/output, two feedbacks plus two ACKs,
        # and two preparation deliveries. Geometry is explicit, not measured TLS.
        "frames_and_ack_headers": (6 * rows + 4 * commits + 2 * prepared + 4) * spec.frame_header,
        "close_up": spec.close_up, "close_down": spec.close_down,
    }
    total = sum(ledger.values())
    local_macs = (run["local_decoder_rows"] * spec.local_decoder_macs_per_token
                  + (prompt_tokens + run["emitted_original_tokens"]) * spec.local_encoder_macs_per_token)
    provider_macs = rows * spec.global_macs_per_row + commits * spec.commit_macs_per_row
    rounds = spec.setup_rounds + rows * spec.dependent_rounds_per_global_row + commits + 1
    transfer_seconds = total / spec.link_bytes_per_second
    dependency_seconds = rounds * spec.round_trip_ms / 1000
    compute_seconds = local_macs / spec.local_macs_per_second + provider_macs / spec.provider_macs_per_second
    return {"inputs": values, "ledger_bytes": ledger, "all_link_bytes": total,
            "bytes_per_emitted_original_token": total / run["emitted_original_tokens"] if run["emitted_original_tokens"] else None,
            "unused_preparation_bytes": run["unused_preparation_blocks"] * spec.preparation_per_global_row,
            "unused_forward_body_bytes": run["unused_issued_forwards"] * (spec.global_row_up + spec.global_row_down + spec.global_row_peer),
            "global_rows_including_prefill_and_overissue": rows,
            "local_macs": local_macs, "provider_macs": provider_macs,
            "provider_kv_bytes_two_parties": 2 * run["provider_kv_rows_per_party"] * spec.kv_bytes_per_row_per_party,
            "dependent_rounds": rounds,
            "serial_no_overlap_estimate_seconds": transfer_seconds + dependency_seconds + compute_seconds,
            "measured_wire_bytes": None, "measured_latency_seconds": None,
            "real_masked_transformer_compatibility": False,
            "scope": "hypothetical all-link geometry, one cold request; not a WAN measurement"}


def independent_head_failure() -> dict:
    # AR: first bit uniform, second bit copies first. Independent exact marginals
    # assign probability 1/4 to impossible 01 and 10.
    ar = {"00": 0.5, "01": 0.0, "10": 0.0, "11": 0.5}
    independent = {key: 0.25 for key in ar}
    return {"autoregressive_joint": ar, "independent_exact_marginals_joint": independent,
            "total_variation": sum(abs(ar[k] - independent[k]) for k in ar) / 2,
            "impossible_sequence_mass": independent["01"] + independent["10"]}


def public_prepared_screen(*, prompt_tokens: int = 39, output_tokens: int = 32,
                           block_size: int = 4, unused_prepared_rows: int = 0) -> dict:
    """Shape-only public-weight global Transformer under client-state placement.

    No SDK execution or direct BLT support implied. A prepared stage body projects
    one ring input, one ring output and one offline output correction. Real stage
    issuance/framing/setup must replace these hypothetical geometry inputs.
    """
    if (not 0 <= prompt_tokens <= 64 or not 0 <= output_tokens <= 64
            or not 1 <= block_size <= 16 or not 0 <= unused_prepared_rows <= 64):
        raise ValueError("prepared screen exceeds bounded dimensions")
    layers, hidden, kv_heads, head_dim, element_bytes = 2, 32, 2, 8, 4
    # Public toy shape: unfused q/k/v/o and gated MLP, not a loaded checkpoint.
    stages = [("q", 32, 32), ("k", 32, 16), ("v", 32, 16), ("o", 32, 32),
              ("gate", 32, 64), ("up", 32, 64), ("down", 64, 32)]
    prefill = (prompt_tokens + block_size - 1) // block_size
    decode = (output_tokens + block_size - 1) // block_size
    rows = prefill + decode  # Conservative schedule input, not target equivalence.
    reserved = rows + unused_prepared_rows
    ring_bytes, header = 4, 64
    input_width = layers * sum(a for _, a, _ in stages)
    output_width = layers * sum(b for _, _, b in stages)
    artifacts = CostSpec()
    artifact_bytes = sum(getattr(artifacts, k) for k in (
        "decoder_weights", "encoder_weights", "vocab_embedding_and_head",
        "tokenizer", "numeric_tables", "manifest"))
    ledger = {
        "public_local_artifact_download": artifact_bytes,
        "setup_up": artifacts.setup_up, "setup_down": artifacts.setup_down,
        "online_masked_stage_inputs": rows * ring_bytes * input_width,
        "online_masked_stage_outputs": rows * ring_bytes * output_width,
        "offline_output_corrections_including_unused": reserved * ring_bytes * output_width,
        "stage_headers": (2 * rows + reserved) * layers * len(stages) * header,
        "setup_close_headers": 4 * header,
        "close_up": artifacts.close_up, "close_down": artifacts.close_down,
    }
    global_row_kv = 2 * layers * kv_heads * head_dim * element_bytes
    # Matched GLOBAL geometry only. Local encoder+decoder window KV is extra.
    token_slots = prompt_tokens + output_tokens
    patch_slots = prefill + decode
    window = 8
    local_each = 2 * 2 * 2 * 8 * element_bytes * min(window, token_slots)
    remote_matrices = layers * sum(a * b for _, a, b in stages) * 4
    return {
        "composition": "public_weights_prepared_linears_client_global_kv_and_nonlinear",
        "client_owns": ["global_patch_kv", "global_attention_softmax_norm_activation_rescale",
                        "local_encoder_decoder_weights_and_state", "decoded_latents", "sampling"],
        "provider_owns": ["global_body_matrices", "masked_linear_execution"],
        "preparation_owns": ["offline_stage_correlations_and_output_corrections"],
        "latent_release_permission_required": False,
        "direct_blt_or_megabyte_support": False,
        "measured_wire_bytes": None, "measured_peak_rss_bytes": None,
        "shape": {"layers": layers, "hidden": hidden, "kv_heads": kv_heads,
                  "head_dim": head_dim, "kv_element_bytes": element_bytes,
                  "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
                  "block_size": block_size, "local_window": window},
        "global_rows": rows, "prefill_rows": prefill, "decode_rows": decode,
        "reserved_preparation_rows": reserved,
        "stage_shapes_per_layer": [{"name": n, "in": a, "out": b} for n, a, b in stages],
        "ledger_bytes": ledger, "projected_cold_bytes": sum(ledger.values()),
        "unused_preparation_correction_bytes": unused_prepared_rows * ring_bytes * output_width,
        "provider_resident_matrix_bytes_not_client_download": remote_matrices,
        "kv_geometry": {
            "global_kv_bytes_per_slot": global_row_kv,
            "full_token_global_kv_slots": token_slots,
            "patch_global_kv_slots": patch_slots,
            "full_token_global_kv_bytes": token_slots * global_row_kv,
            "patch_global_kv_bytes": patch_slots * global_row_kv,
            "local_encoder_window_kv_bytes": local_each,
            "local_decoder_window_kv_bytes": local_each,
            "patch_global_plus_local_window_kv_bytes": patch_slots * global_row_kv + 2 * local_each,
            "current_client_latent_bytes": hidden * element_bytes,
            "not_included": ["weights", "attention_scratch", "cross_attention_cache",
                             "allocator_overhead", "prepared_masks_and_correlations", "startup_peak"],
        },
        "unmeasured_local_work": ["patch_global_attention_and_nonlinear_cpu",
                                  "local_encoder_decoder_cpu", "cold_import_load_and_issuance"],
        "scope": "bounded hypothetical prepared-stage body/header and KV geometry; no measured admission or trained quality",
    }


def report() -> dict:
    scenarios = {}
    for name, kwargs in (
        ("serial_new_model_k1", {"block_size": 1}),
        ("hierarchy_k4", {"block_size": 4}),
        ("final_partial_k4", {"block_size": 4, "max_tokens": 10}),
        ("eos_first", {"eos_at": 0}),
        ("eos_middle", {"eos_at": 2}),
        ("cancel_before_local", {"cancel_after": 0}),
        ("cancel_boundary_overissue", {"cancel_after": 4}),
        ("unused_extra_preparation", {"unused_prepared_blocks": 3}),
        ("sampled_eos", {"allow_eos": True, "seed": 1}),
    ):
        run = run_fixture(**{"max_tokens": 32, "prompt": (0, 1, 2) * 13, **kwargs})
        scenarios[name] = {"execution": run, "cost": cold_cost(run, prompt_tokens=39)}
    sensitivities = []
    base = scenarios["hierarchy_k4"]["execution"]
    for decoder in (1024 * 1024, 32 * 1024 * 1024, 128 * 1024 * 1024):
        cost = cold_cost(base, CostSpec(decoder_weights=decoder), prompt_tokens=39)
        sensitivities.append({"decoder_download_bytes": decoder, "all_link_bytes": cost["all_link_bytes"]})
    paths = ("scripts/probe_cold_block_decoder.py", "tests/test_cold_block_decoder.py",
             "docs/research/cold-block-decoder-reference.md", "docs/research/checkpoint-drafter-reference.md")
    return {"schema": "pllm.cold_block_decoder.synthetic.v1", "trained": False,
            "same_original_checkpoint_execution": False, "protected_serving_ready": False,
            "resource_contract": {"max_tokens": 64, "max_prompt_tokens": 64,
                                  "max_block_size": 16, "additional_numeric_arrays_bytes": 0,
                                  "checkpoint_loads": 0, "timing_runs": 0},
            "sources": [{"title": title, "url": url, "raw_html_bytes": size, "sha256": sha}
                        for title, url, size, sha in SOURCES],
            "source_files_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
            "independent_head_failure": independent_head_failure(),
            "public_prepared_placement": public_prepared_screen(),
            "provider_resident_alternative": {
                "fixture": "public_weight_affine_additive_share_oracle",
                "real_transformer_nonlinear_bytes": None,
                "real_transformer_attention_and_kv_protocol_bytes": None,
                "nonlinear_cost_in_old_scenarios": "absent for affine fixture; unknown and must be added for Transformer alternative",
                "promotion_blocked_by_missing_costs": True,
            },
            "scenarios": scenarios, "decoder_download_sensitivity": sensitivities,
            "conclusion": "Public hierarchy can use prepared masked global linears with client-owned global nonlinear/KV and decoded latents; no extra latent permission flow. Trained architecture, pinned patch/numeric schedule, quality, full bytes and admission remain missing. Provider-resident secure nonlinear alternative is separate and uncosted."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path, help="Offline deterministic evidence replay")
    args = parser.parse_args()
    result = report()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.check and json.loads(args.check.read_text()) != json.loads(text):
        raise SystemExit("evidence replay differs")
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
