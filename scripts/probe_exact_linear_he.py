"""Bounded offline E3 BFV public down-projection screen; never a selectable API.

Run with the existing environment, no dependency synchronization:
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/probe_exact_linear_he.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time

sys.dont_write_bytecode = True
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "PLLM_NATIVE_THREADS",
):
    os.environ[_name] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEGREE = 8192
CHUNK = DEGREE // 2
PLAIN = 268_369_921
COEFF_BITS = [55, 54, 54, 55]  # SEAL tc128 maximum total for degree 8192: 218.
MAX_RSS = 1 << 30
MAX_CPU = 120.0
GROUP = 32
SCREEN = ROOT / "docs/evidence/exact-linear-he-screen-2026-10-01.json"
REVIEW_JSON = ROOT / "docs/evidence/exact-linear-he-review-2026-10-01.json"
REVIEW_MD = ROOT / "docs/evidence/exact-linear-he-review-2026-10-01.md"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def public_bounds(weight: np.ndarray) -> dict:
    w = np.asarray(weight)
    if w.ndim != 2 or min(w.shape) == 0 or w.dtype.kind not in "iu":
        raise ValueError("nonempty integer public matrix required")
    if int(w.min()) < -127 or int(w.max()) > 127:
        raise ValueError("symmetric W8 domain is [-127,127]")
    bounds = np.abs(w.astype(np.int64)).sum(axis=1) * 127
    maximum = int(bounds.max())
    uniform = int(w.shape[1]) * 127 * 127
    if uniform >= 2**31 or uniform >= 2**63:
        raise ValueError("exact signed int32/int64 oracle would overflow")
    return {
        "per_output_abs_bound": bounds.tolist(),
        "actual_max_abs_bound": maximum,
        "uniform_w8a8_abs_bound": uniform,
        "activation_domain": [-127, 127],
        "weight_domain": [-127, 127],
    }


def prove_lift(bound: int, plain: int = PLAIN, ring_bits: int = 32) -> dict:
    if bound < 0 or plain <= 2 * bound or bound >= 2 ** (ring_bits - 1):
        raise ValueError("centered BFV/ring lift cannot represent entire signed domain")
    if (plain - 1) % (2 * DEGREE):
        raise ValueError("plaintext modulus incompatible with degree-8192 batching")
    # Trial division proves this particular small (28-bit) modulus prime.
    if plain < 2 or any(plain % d == 0 for d in range(2, math.isqrt(plain) + 1)):
        raise ValueError("plaintext modulus must be prime")
    return {
        "plain_modulus": plain,
        "plain_modulus_prime_checked": True,
        "batching_congruence": (plain - 1) % (2 * DEGREE),
        "ring_bits": ring_bits,
        "bound": bound,
        "center_domain": [-(plain // 2), plain // 2],
        "proof": "Every partial signed sum has abs <= 127*sum(abs(W_row)); "
        "t>2B makes centered BFV output unique. Mapping that integer mod 2^w "
        "equals prepared ring output; B<2^(w-1) permits exact centered inverse. "
        "No lift of arbitrary full-ring masks into BFV is claimed.",
    }


def new_contexts():
    import tenseal as ts

    start = time.process_time()
    client = ts.context(
        ts.SCHEME_TYPE.BFV, DEGREE, PLAIN, coeff_mod_bit_sizes=COEFF_BITS, n_threads=1
    )
    context_cpu = time.process_time() - start
    start = time.process_time()
    client.generate_galois_keys()
    galois_cpu = time.process_time() - start
    start = time.process_time()
    base = client.serialize(save_secret_key=False, save_galois_keys=False, save_relin_keys=False)
    public = client.serialize(save_secret_key=False, save_galois_keys=True, save_relin_keys=False)
    serialize_cpu = time.process_time() - start
    start = time.process_time()
    provider = ts.context_from(public, n_threads=1)
    import_cpu = time.process_time() - start
    if provider.has_secret_key() or provider.has_relin_keys() or not provider.has_galois_keys():
        raise RuntimeError("provider must contain public/Galois keys only")
    seal = client.seal_context().data
    qualifiers = seal.key_context_data().qualifiers()
    if not seal.parameters_set() or not qualifiers.using_batching:
        raise RuntimeError("SEAL rejected BFV batching parameters")
    return (
        client,
        provider,
        {
            "poly_modulus_degree": DEGREE,
            "batch_slots": DEGREE,
            "used_row_slots": CHUNK,
            "plain_modulus": PLAIN,
            "coeff_modulus_bit_sizes": COEFF_BITS,
            "coeff_modulus_total_bits": seal.key_context_data().total_coeff_modulus_bit_count(),
            "security_level": None,
            "security_level_contract": "SEAL constructor default tc128; degree8192 coefficient bit bound 218",
            "security_level_binding_limitation": "TenSEAL cannot convert seal::sec_level_type",
            "key_parms_id": list(seal.key_context_data().parms_id()),
            "coeff_modulus_values": None,
            "coeff_modulus_values_unavailable_reason": "TenSEAL binding cannot convert seal::Modulus; parms_id pins identity",
            "provider_has_secret_key": provider.has_secret_key(),
            "provider_has_relin_keys": provider.has_relin_keys(),
            "client_default_relin_keys_retained_but_unused": client.has_relin_keys(),
            "public_base_context_bytes": len(base),
            "public_context_and_galois_bytes": len(public),
            "galois_key_increment_bytes": len(public) - len(base),
            "client_context_cpu_seconds": context_cpu,
            "client_galois_cpu_seconds": galois_cpu,
            "client_key_serialization_cpu_seconds": serialize_cpu,
            "provider_context_import_cpu_seconds": import_cpu,
            "setup_cpu_seconds": context_cpu + galois_cpu + serialize_cpu + import_cpu,
            "galois_keys_generated": client.galois_keys().data.size(),
            "key_distribution_count": 1,
        },
    )


def encrypt_row(client, row: np.ndarray):
    import tenseal as ts

    x = np.asarray(row)
    if x.ndim != 1 or x.dtype.kind not in "iu" or int(x.min()) < -127 or int(x.max()) > 127:
        raise ValueError("activation must be a signed symmetric A8 integer row")
    start = time.process_time()
    requests = []
    for offset in range(0, len(x), CHUNK):
        padded = np.zeros(CHUNK, dtype=np.int64)
        part = x[offset : offset + CHUNK]
        padded[: len(part)] = part
        requests.append(ts.bfv_vector(client, padded.tolist()).serialize())
    return requests, time.process_time() - start


def evaluate(provider, requests: list[bytes], weight: np.ndarray, *, progress=None):
    """Packed-input public dots, then two-level packing; no decryption capability.

    Each chunk occupies exactly 4096 slots, including zero padding. Dot then
    replicates scalar across BFV's first batching row, as required by pack_vectors.
    Keeping at most GROUP scalars plus ceil(outputs/GROUP) packed groups avoids
    retaining every scalar ciphertext. Backend plaintext encoding is transient.
    """
    import tenseal as ts

    if provider.has_secret_key():
        raise ValueError("provider evaluation refuses a secret key")
    public_bounds(weight)
    if len(requests) != math.ceil(weight.shape[1] / CHUNK):
        raise ValueError("input ciphertext chunk count differs from public matrix")
    start = time.process_time()
    encrypted = [ts.bfv_vector_from(provider, body) for body in requests]
    if any(vector.size() != CHUNK for vector in encrypted):
        raise ValueError("every BFV chunk must be padded to exactly 4096")
    import_cpu = time.process_time() - start
    dot_cpu = packing_cpu = conversion_cpu = 0.0
    groups, pending = [], []
    for output, w in enumerate(weight):
        partials = []
        for index, vector in enumerate(encrypted):
            start = time.process_time()
            padded = np.zeros(CHUNK, dtype=np.int64)
            part = w[index * CHUNK : (index + 1) * CHUNK]
            padded[: len(part)] = part
            plaintext = ts.plain_tensor(padded.tolist(), dtype="int")
            conversion_cpu += time.process_time() - start
            start = time.process_time()
            partials.append(vector.dot(plaintext))
            dot_cpu += time.process_time() - start
        start = time.process_time()
        scalar = partials[0]
        for part in partials[1:]:
            scalar += part
        dot_cpu += time.process_time() - start
        pending.append(scalar)
        if len(pending) == GROUP or output == len(weight) - 1:
            start = time.process_time()
            groups.append(ts.BFVVector.pack_vectors(pending))
            packing_cpu += time.process_time() - start
            pending.clear()
            if progress:
                progress(output + 1)
    start = time.process_time()
    packed = ts.BFVVector.pack_vectors(groups) if len(groups) > 1 else groups[0]
    packing_cpu += time.process_time() - start
    start = time.process_time()
    response = packed.serialize()
    serialization_cpu = time.process_time() - start
    return response, {
        "provider_input_import_cpu_seconds": import_cpu,
        "provider_plaintext_conversion_cpu_seconds": conversion_cpu,
        "provider_dot_and_add_cpu_seconds": dot_cpu,
        "provider_output_packing_cpu_seconds": packing_cpu,
        "provider_output_serialization_cpu_seconds": serialization_cpu,
        "provider_total_cpu_seconds": import_cpu
        + conversion_cpu
        + dot_cpu
        + packing_cpu
        + serialization_cpu,
        "public_plaintext_dots": len(weight) * len(encrypted),
        "dot_rotate_add_steps": len(weight) * len(encrypted) * 12,
        "chunk_sum_ciphertext_additions": len(weight) * (len(encrypted) - 1),
        "output_pack_groups": len(groups),
        "output_ciphertexts": len(packed.ciphertext()),
    }


def decrypt_and_check(client, response: bytes, weight: np.ndarray, row: np.ndarray) -> dict:
    import tenseal as ts

    start = time.process_time()
    received = ts.bfv_vector_from(client, response)
    actual = np.asarray(received.decrypt(), dtype=np.int64)
    cpu = time.process_time() - start
    expected = weight.astype(np.int64) @ row.astype(np.int64)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise RuntimeError("BFV packed output fails all-output exact signed parity")
    return {
        "client_import_decrypt_cpu_seconds": cpu,
        "all_outputs_exact": True,
        "verified_outputs": len(expected),
        "output_i64_sha256": sha(actual.tobytes()),
        "output_signed_min_max": [int(actual.min()), int(actual.max())],
        "client_measured_noise_budget_bits": [
            client.decryptor().data.invariant_noise_budget(ct) for ct in received.ciphertext()
        ],
    }


def load_weight() -> tuple[np.ndarray, dict]:
    from pllm import Model
    from pllm.model_loader import resolve_model
    from pllm.runtime.quantization import quantize_weight_per_row
    from pllm.runtime.safetensors_store import SafeTensorStore

    source = resolve_model(Model.hf(MODEL, revision=REVISION))
    if source.path is None:
        raise RuntimeError("pinned cached checkpoint unavailable")
    config = json.loads((source.path / "config.json").read_text())
    if (config["hidden_size"], config["intermediate_size"], config["num_hidden_layers"]) != (
        896,
        4864,
        24,
    ):
        raise RuntimeError("pinned model configuration shape mismatch")
    store = SafeTensorStore(source.path)
    key = "model.layers.0.mlp.down_proj.weight"
    raw = store.get_slice(key, (slice(None), slice(None)), dtype=np.float32)
    if raw.shape != (896, 4864):
        raise RuntimeError("real down weight orientation/shape mismatch")
    quantized = quantize_weight_per_row(raw, bits=8)
    reference_path = ROOT / "docs/evidence/integer-structure-screen-2026-10-01.json"
    reference = json.loads(reference_path.read_text())
    prior = next(r for r in reference["matrices"] if r["layer"] == 0 and r["role"] == "mlp_down")
    if (
        sha(quantized.values.tobytes()) != prior["weight_i8_sha256"]
        or sha(quantized.scales.tobytes()) != prior["per_output_row_scales_f32_sha256"]
        or source.checkpoint_digest != reference["checkpoint_digest"]
    ):
        raise RuntimeError("actual fixed-profile weight/scales differ from pinned prepared stage")
    lock = {
        "model": MODEL,
        "revision": REVISION,
        "layer": 0,
        "source_key": key,
        "checkpoint_digest": source.checkpoint_digest,
        "source_lock_digest": source.source_lock_digest,
        "body_fingerprint": "5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974",
        "prepared_stage_reference_sha256": file_sha(reference_path),
        "same_prepared_weight_and_scales_verified": True,
        "raw_shape_output_by_input": list(raw.shape),
        "raw_loaded_dtype": str(raw.dtype),
        "raw_float32_sha256": sha(raw.tobytes()),
        "weight_i8_sha256": sha(quantized.values.tobytes()),
        "scales_float32_sha256": sha(quantized.scales.tobytes()),
        "raw_float32_copy_bytes": raw.nbytes,
        "weight_i8_bytes": quantized.values.nbytes,
        "client_output_scale_bytes": quantized.scales.nbytes,
        "quantization": "existing SymmetricPerRow weight_bits=8, activation_bits=8; no transpose",
        "bias": None,
        "source_raw_tensor_dtype": store._tensor_metadata(key)[1],
        "quantizer_source_sha256": file_sha(
            Path(sys.modules[quantize_weight_per_row.__module__].__file__)
        ),
    }
    return quantized.values.copy(), lock


def prepared_control(weight: np.ndarray, row: np.ndarray, lock: dict, bound: int) -> dict:
    """Actual existing single-thread native kernel, masks and protocol body codecs.

    Local stage-only calculation; no live admission/session claims. Timers charge
    arithmetic, mask expansion and body codecs, not transport or orchestration.
    """
    from pllm.runtime.native import MaskedGEMM
    from pllm.runtime.preparation_protocol import (
        CorrectionPush,
        PreparationRequest,
        expand_output_mask,
        expand_preparation_mask,
        seeded_ring_profile,
    )
    from pllm.runtime.stage_protocol import MaskedStageRequest, MaskedStageResponse

    kernel = MaskedGEMM(threads=1)
    if not kernel.native:
        raise RuntimeError("matched prepared native kernel required")
    start = time.process_time()
    matrix = kernel.compile(weight)
    compile_cpu = time.process_time() - start
    profile = seeded_ring_profile(bound)
    start = time.process_time()
    attempt, session = os.urandom(16).hex(), os.urandom(16).hex()
    request = PreparationRequest(
        attempt_id=attempt,
        session_id=session,
        model=MODEL,
        body_fingerprint=lock["body_fingerprint"],
        stage_id="layer.0.mlp_down",
        weight_digest=lock["weight_i8_sha256"],
        rows=1,
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        weight_bits=8,
        activation_bits=8,
        signed_output_bound=bound,
        ring=profile.ring,
        modulus=profile.modulus,
        wire_bits=profile.wire_bits,
        seed=os.urandom(32),
    )
    mask, output_mask = expand_preparation_mask(request), expand_output_mask(request)
    request_body = request.pack()
    client_mask_cpu = time.process_time() - start
    start = time.process_time()
    # Preparation role independently expands same seeded material.
    prep_mask, prep_output = expand_preparation_mask(request), expand_output_mask(request)
    transformed = (
        matrix.wrap32(prep_mask)
        if profile.wire_bits == 32
        else matrix.modular(prep_mask, profile.modulus)
    )
    correction = ((transformed.astype(np.int64) - prep_output) % profile.modulus).astype(np.uint32)
    correction_body = CorrectionPush(
        **{k: v for k, v in vars_like(request).items() if k != "seed"}, correction=correction
    ).pack()
    preparation_cpu = time.process_time() - start
    start = time.process_time()
    masked = ((row.astype(np.int64)[None, :] - mask) % profile.modulus).astype(np.uint32)
    online_request = MaskedStageRequest(
        MODEL,
        request.stage_id,
        attempt,
        masked,
        1.0,
        profile.modulus,
        profile.wire_bits,
        ring=profile.ring,
        body_fingerprint=request.body_fingerprint,
        weight_digest=request.weight_digest,
        weight_bits=8,
        activation_bits=8,
        session_id=session,
        out_features=weight.shape[0],
        signed_output_bound=bound,
    ).pack()
    client_online_cpu = time.process_time() - start
    start = time.process_time()
    masked_output = (
        matrix.wrap32(masked)
        if profile.wire_bits == 32
        else matrix.modular(masked, profile.modulus)
    )
    corrected = ((masked_output.astype(np.int64) + correction) % profile.modulus).astype(np.uint32)
    online_response = MaskedStageResponse(
        attempt,
        corrected,
        profile.modulus,
        profile.wire_bits,
        stage_id=request.stage_id,
        ring=profile.ring,
    ).pack()
    inference_cpu = time.process_time() - start
    start = time.process_time()
    unmasked = (corrected.astype(np.int64) + output_mask) % profile.modulus
    centered = np.where(unmasked > profile.modulus // 2, unmasked - profile.modulus, unmasked)
    if not np.array_equal(centered[0], weight.astype(np.int64) @ row.astype(np.int64)):
        raise RuntimeError("prepared control exact parity failed")
    client_finish_cpu = time.process_time() - start
    # Two independent offsets: same ring, two native masked products. Arithmetic
    # comparator only, not an implementation of the complete offset protocol.
    start = time.process_time()
    a = (
        np.frombuffer(os.urandom(row.size * 4), dtype=np.uint32).astype(np.int64).reshape(1, -1)
        % profile.modulus
    ).astype(np.uint32)
    b = ((row.astype(np.int64)[None, :] - a) % profile.modulus).astype(np.uint32)
    offset_a = matrix.wrap32(a) if profile.wire_bits == 32 else matrix.modular(a, profile.modulus)
    offset_b = matrix.wrap32(b) if profile.wire_bits == 32 else matrix.modular(b, profile.modulus)
    offsets = (offset_a.astype(np.int64) + offset_b) % profile.modulus
    offsets = np.where(offsets > profile.modulus // 2, offsets - profile.modulus, offsets)
    if not np.array_equal(offsets, centered):
        raise RuntimeError("two-offset native arithmetic parity failed")
    offset_cpu = time.process_time() - start
    return {
        "scope": "actual one-row stage-only native operations and existing serialized bodies; no live protocol",
        "ring": profile.to_dict(),
        "weight_compile_cpu_seconds": compile_cpu,
        "client_mask_expansion_cpu_seconds": client_mask_cpu,
        "client_online_body_cpu_seconds": client_online_cpu,
        "client_finish_and_oracle_cpu_seconds": client_finish_cpu,
        "preparation_cpu_seconds": preparation_cpu,
        "inference_cpu_seconds": inference_cpu,
        "aggregate_stage_cpu_seconds": client_mask_cpu
        + client_online_cpu
        + client_finish_cpu
        + preparation_cpu
        + inference_cpu,
        "two_offset_native_arithmetic_cpu_seconds": offset_cpu,
        "two_offset_complete_protocol_cpu_seconds": None,
        "preparation_request_body_bytes": len(request_body),
        "offline_correction_body_bytes": len(correction_body),
        "online_input_body_bytes": len(online_request),
        "online_output_body_bytes": len(online_response),
        "covered_measured_stage_body_bytes": sum(
            map(len, [request_body, correction_body, online_request, online_response])
        ),
        "offline_control_ack_admission_bytes": None,
        "client_dense_weight_bytes": 0,
        "client_output_scale_bytes": lock["client_output_scale_bytes"],
        "preparation_and_inference_each_weight_i8_bytes": weight.nbytes,
        "exact_parity": True,
        "native_source_binary_sha256": file_sha(kernel.library_path),
    }


def vars_like(record) -> dict:
    from dataclasses import fields

    return {f.name: getattr(record, f.name) for f in fields(record)}


def memory_preflight(outputs: int, inputs: int) -> dict:
    ciphertext = DEGREE * 2 * 3 * 8
    # Full default Galois set: at most 26 keys, 3 decompositions, 2 polys, 4 limbs.
    one_key_set = 26 * 3 * 2 * 4 * DEGREE * 8
    key_copies = 4 * one_key_set  # client, serialized, provider, import scratch.
    live_ciphertexts = GROUP + math.ceil(outputs / GROUP) + 16
    scratch = 128 << 20
    baseline = psutil.Process().memory_info().rss
    transient_encoded = 2 * DEGREE * 8  # one coefficient plain plus NTT working copy.
    expanded_all = outputs * math.ceil(inputs / CHUNK) * transient_encoded
    total = baseline + key_copies + ciphertext * live_ciphertexts + transient_encoded + scratch
    return {
        "max_resident_bytes": MAX_RSS,
        "baseline_worker_rss_bytes": baseline,
        "conservative_key_copies_bytes": key_copies,
        "ciphertext_uncompressed_bytes_each": ciphertext,
        "max_live_ciphertexts_bound": live_ciphertexts,
        "transient_encoded_weight_copies_bytes": transient_encoded,
        "all_encoded_weight_copies_if_retained_bytes": expanded_all,
        "all_encoded_weights_retained": False,
        "native_allocator_scratch_allowance_bytes": scratch,
        "planned_worker_resident_bytes": total,
        "admitted": total < MAX_RSS - (64 << 20),
    }


def worker(send):
    def event(name, data):
        send.send((name, data))

    try:
        begin = time.process_time()
        weight, lock = load_weight()
        bounds = public_bounds(weight)
        from pllm.runtime.preparation_protocol import seeded_ring_profile

        profile = seeded_ring_profile(bounds["actual_max_abs_bound"])
        lift = prove_lift(bounds["uniform_w8a8_abs_bound"], ring_bits=profile.wire_bits)
        event("source", lock)
        event("bounds", bounds)
        event("lift", lift)
        memory = memory_preflight(*weight.shape)
        event("memory_preflight", memory)
        if not memory["admitted"]:
            event("stop_reason", "resident_preflight_veto")
            return
        client, provider, context = new_contexts()
        event("context", context)
        # Full public domain boundary witness, not a prompt-derived activation.
        largest = int(np.argmax(bounds["per_output_abs_bound"]))
        row = (127 * np.sign(weight[largest].astype(np.int16))).astype(np.int8)
        event(
            "input",
            {
                "scope": "one synthetic adversarial A8 row, actual layer-0 weights",
                "rows": 1,
                "features": len(row),
                "sha256": sha(row.tobytes()),
                "construction": "127*sign(W[argmax public row L1 bound]); zero weight -> zero",
                "saturates_output_index": largest,
                "saturated_positive_signed_output": bounds["actual_max_abs_bound"],
            },
        )
        event(
            "prepared_control", prepared_control(weight, row, lock, bounds["actual_max_abs_bound"])
        )
        requests, encrypt_cpu = encrypt_row(client, row)
        event(
            "input_encryption",
            {
                "client_cpu_seconds": encrypt_cpu,
                "ciphertext_bytes": [len(x) for x in requests],
                "ciphertext_count": len(requests),
                "padded_slots_per_ciphertext": CHUNK,
            },
        )
        start = time.process_time()
        response, costs = evaluate(provider, requests, weight[:GROUP])
        small_cpu = time.process_time() - start
        small = decrypt_and_check(client, response, weight[:GROUP], row)
        small.update(costs)
        small["shape_output_by_input"] = [GROUP, weight.shape[1]]
        small["scope"] = "actual full-input-width bounded substage; not full projection"
        small["output_bytes"] = len(response)
        event("bounded_substage", small)
        predicted = small_cpu * (weight.shape[0] / GROUP) * 1.4 + 3.0
        event(
            "cpu_preflight",
            {
                "cap_cpu_seconds": MAX_CPU,
                "already_spent_cpu_seconds": time.process_time() - begin,
                "full_evaluation_forecast_cpu_seconds": predicted,
                "forecast_method": "measured 32x4864 dot/pack/serialize *28 *1.4 +3s final packing allowance",
                "unknown_extra_cpu_seconds": None,
            },
        )
        if predicted + time.process_time() - begin >= MAX_CPU - 2:
            event("stop_reason", "full_shape_cpu_preflight_veto")
            return
        response, costs = evaluate(
            provider, requests, weight, progress=lambda n: event("progress_outputs", n)
        )
        exact = decrypt_and_check(client, response, weight, row)
        exact.update(costs)
        exact.update(
            {
                "shape_input_by_output": [weight.shape[1], weight.shape[0]],
                "input_rows": 1,
                "actual_full_shape_execution": True,
                "output_ciphertext_bytes": len(response),
                "online_ciphertext_bytes": sum(map(len, requests)) + len(response),
                "client_online_cpu_seconds": encrypt_cpu
                + exact["client_import_decrypt_cpu_seconds"],
                "whole_projection_cpu_seconds": encrypt_cpu
                + exact["client_import_decrypt_cpu_seconds"]
                + costs["provider_total_cpu_seconds"],
            }
        )
        event("full_projection", exact)
        event("worker_cpu_seconds", time.process_time() - begin)
        event("stop_reason", "full_projection_exact_but_region_body_gate_failed")
    except Exception as exc:
        event("error", {"type": type(exc).__name__, "message": str(exc)})
        event("stop_reason", "native_or_source_failure")
    finally:
        send.close()


def comparisons(evidence: dict) -> dict:
    full = evidence.get("full_projection")
    control = evidence.get("prepared_control")
    if not full or not control:
        return {
            "region_25_percent_gate_pass": False,
            "complete_cost_bytes": None,
            "reason": "full shape not executed; unknown terms fail closed",
        }
    setup = evidence["context"]["public_context_and_galois_bytes"]
    he = full["online_ciphertext_bytes"]
    baseline = control["covered_measured_stage_body_bytes"]
    ledger_path = ROOT / "docs/evidence/prepared-stage-attribution-qwen25-2026-09-29.json"
    ledger = json.loads(ledger_path.read_text())
    cohort = ledger["cohorts"]["32"]
    projections = []
    for horizon in (1, 10, 100):
        for rows, phase in (
            (1, "decode"),
            (39, "prefill39_rowwise_projection"),
            (70, "39_prefill_plus_31_decode_response_projection"),
        ):
            candidate = rows * he + setup / horizon
            prepared = rows * baseline
            projections.append(
                {
                    "horizon_responses": horizon,
                    "phase": phase,
                    "rows_one_layer": rows,
                    "scope": "projection; independent rowwise HE, no cross-row batching",
                    "he_measured_body_terms_projected_bytes": candidate,
                    "prepared_measured_stage_terms_projected_bytes": prepared,
                    "he_to_prepared_body_ratio": candidate / prepared,
                    "region_25_percent_gate_pass": candidate <= 0.75 * prepared,
                    "client_online_cpu_projected_seconds": rows * full["client_online_cpu_seconds"],
                    "client_setup_cpu_amortized_seconds": (
                        evidence["context"]["client_context_cpu_seconds"]
                        + evidence["context"]["client_galois_cpu_seconds"]
                        + evidence["context"]["client_key_serialization_cpu_seconds"]
                    )
                    / horizon,
                    "aggregate_he_cpu_projected_seconds": rows
                    * full["whole_projection_cpu_seconds"]
                    + evidence["context"]["setup_cpu_seconds"] / horizon,
                    "complete_body_bytes": None,
                    "complete_aggregate_cpu_seconds": None,
                }
            )
    whole = []
    for horizon in (1, 10, 100):
        for layers in (1, 24):
            # Same setup once per client/provider, across stages. Measured weights
            # only layer 0; other-layer numerical/CPU costs remain unmeasured.
            replacement = layers * 70 * he + setup / horizon
            removed = cohort["stage_body_bytes_by_semantic_role"]["mlp_down"] * layers / 24
            projected = cohort["covered_all_link_body_bytes"] - removed + replacement
            whole.append(
                {
                    "scope": "whole 39+32 response BODY projection, not decoder execution",
                    "horizon_responses": horizon,
                    "down_layers_replaced": layers,
                    "prepared_covered_all_link_body_bytes": cohort["covered_all_link_body_bytes"],
                    "removed_down_region_bytes": removed,
                    "projected_covered_known_terms_bytes": projected,
                    "projected_ratio": projected / cohort["covered_all_link_body_bytes"],
                    "whole_decoder_complete_cpu_seconds": None,
                    "complete_wire_bytes": None,
                }
            )
    return {
        "region_25_percent_gate_pass": he <= 0.75 * baseline,
        "one_row_online_he_to_prepared_covered_body_ratio": he / baseline,
        "one_row_75_percent_prepared_budget_bytes": 0.75 * baseline,
        "horizon_phase_projections": projections,
        "whole_response_projections": whole,
        "whole_response_control_path": str(ledger_path.relative_to(ROOT)),
        "whole_response_control_sha256": file_sha(ledger_path),
        "prepared_matched_stage_compute_measured": True,
        "he_to_prepared_stage_cpu_ratio": full["whole_projection_cpu_seconds"]
        / control["aggregate_stage_cpu_seconds"],
        "two_offset_complete_matched_compute_cap_admitted": False,
        "client_work_cap_seconds_one_row": 1.0,
        "measured_one_row_client_online_cap_pass": full["client_online_cpu_seconds"] <= 1.0,
        "client_complete_work_cap_admitted": False,
        "client_dense_weight_bytes_he_and_prepared": 0,
        "client_output_scales_bytes_he_and_prepared": evidence["source"][
            "client_output_scale_bytes"
        ],
        "complete_cost_bytes": None,
        "selection_admitted": False,
    }


def run() -> dict:
    ctx = mp.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    child = ctx.Process(target=worker, args=(send,))
    child.start()
    send.close()
    process, parent = psutil.Process(child.pid), psutil.Process()
    evidence = {
        "schema": "pllm.exact_linear_he_screen.v1",
        "date": "2026-10-01",
        "selectable": False,
        "live_protocol_executed": False,
        "limits": {
            "max_aggregate_resident_bytes": MAX_RSS,
            "max_worker_cpu_seconds": MAX_CPU,
            "native_threads": 1,
            "resident_model_layers": 1,
            "offline": True,
            "monitor_interval_seconds": 0.02,
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "script_source_sha256": file_sha(Path(__file__)),
        "unknown_costs": {
            "transport_framing_tls_wire_bytes": None,
            "admission_authentication_control_bytes": None,
            "provider_cold_weight_distribution_bytes": None,
            "complete_two_offset_stage_cpu_seconds": None,
            "pinned_prompt_activation_and_decode_parity": None,
            "prefill39_he_execution_cpu_seconds": None,
            "whole_decoder_execution_cpu_seconds": None,
            "malicious_output_integrity_cost": None,
        },
        "threat_model": "honest-but-curious, client-key BFV, provider sees public weights and ciphertexts; no decryption oracle",
        "prefill_executed": False,
        "whole_decoder_executed": False,
    }
    peak = cpu = 0.0
    wall_start = time.monotonic()
    while child.is_alive() or receive.poll():
        try:
            rss = process.memory_info().rss + parent.memory_info().rss
            peak = max(peak, rss)
            times = process.cpu_times()
            cpu = max(cpu, times.user + times.system)
            if rss >= MAX_RSS or cpu >= MAX_CPU or time.monotonic() - wall_start >= 180:
                child.terminate()
                evidence["stop_reason"] = "supervisor_resource_veto"
                break
        except psutil.NoSuchProcess:
            pass
        if receive.poll(0.02):
            try:
                key, value = receive.recv()
            except EOFError:
                break
            evidence[key] = value
    child.join(timeout=3)
    if child.is_alive():
        child.kill()
        child.join()
    receive.close()
    evidence["supervisor"] = {
        "peak_sampled_parent_plus_worker_rss_bytes": peak,
        "observed_worker_cpu_seconds": cpu,
        "wall_seconds": time.monotonic() - wall_start,
        "worker_exitcode": child.exitcode,
        "peak_memory_is_sampled_not_allocator_exact": True,
    }
    import tenseal as ts

    evidence["environment"]["tenseal_version"] = ts.__version__
    evidence["environment"]["tenseal_native_binary_sha256"] = file_sha(Path(ts._ts_cpp.__file__))
    evidence["environment"]["tenseal_bfv_wrapper_sha256"] = file_sha(
        Path(sys.modules[ts.BFVVector.__module__].__file__)
    )
    evidence["comparison"] = comparisons(evidence)
    evidence["selection_admitted"] = False
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-evidence", action="store_true")
    args = parser.parse_args()
    result = run()
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.write_evidence:
        SCREEN.write_text(text)
        write_reviews(result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("stop_reason", "supervisor", "full_projection", "comparison")
                if k in result
            },
            sort_keys=True,
            indent=2,
        )
    )


def write_reviews(result):
    full = result.get("full_projection", {})
    context = result.get("context", {})
    control = result.get("prepared_control", {})
    review = {
        "schema": "pllm.exact_linear_he_review.v1",
        "date": "2026-10-01",
        "screen_sha256": file_sha(SCREEN),
        "script_source_sha256": result["script_source_sha256"],
        "decision": "veto; research nonselectable",
        "stop_reason": result.get("stop_reason"),
        "actual_full_projection_executed": full.get("actual_full_shape_execution", False),
        "all_896_outputs_exact": full.get("verified_outputs") == 896,
        "region_25_percent_gate_pass": result["comparison"]["region_25_percent_gate_pass"],
        "unknown_costs": result["unknown_costs"],
        "comparison": result["comparison"],
        "reproduce": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/probe_exact_linear_he.py --write-evidence",
        "test": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_exact_linear_he.py",
        "lint": ".venv/bin/ruff check scripts/probe_exact_linear_he.py tests/test_exact_linear_he.py",
        "format": ".venv/bin/ruff format --check scripts/probe_exact_linear_he.py tests/test_exact_linear_he.py",
    }
    REVIEW_JSON.write_text(json.dumps(review, sort_keys=True, indent=2) + "\n")
    md = f"""# E3 exact packed public-linear HE review — 2026-10-01

**Decision: veto; research nonselectable.** Stop reason: `{result.get("stop_reason")}`.

## Actual execution

Pinned `{MODEL}@{REVISION}`, layer 0 `model.layers.0.mlp.down_proj.weight`,
existing SymmetricPerRow W8A8. Raw BF16 shape `[896,4864]`, output-by-input;
loaded float32 source, quantized int8 weights and original output scales are
SHA-pinned and match prior prepared-stage evidence.

One **full 4864-input row → 896 ciphertext outputs**, packed into one output
ciphertext: actual full-shape executed = **{full.get("actual_full_shape_execution", False)}**;
exact verified outputs = **{full.get("verified_outputs", 0)}**.
Input is an adversarial synthetic A8 row `127*sign(W[max-L1-row])`, not a
prompt-derived Qwen activation. It saturates actual signed bound **18,993,866**.
Full symmetric W8A8 shape bound is **78,451,456**. No prompt/full-decoder parity claim.

| Term | Actual result |
| --- | ---: |
| Two input BFV ciphertext bytes | {sum(result.get("input_encryption", {}).get("ciphertext_bytes", [])):,} |
| Packed output BFV ciphertext bytes | {full.get("output_ciphertext_bytes", 0):,} |
| Total online ciphertext bodies | {full.get("online_ciphertext_bytes", 0):,} |
| One-client public context + Galois-key setup bytes | {context.get("public_context_and_galois_bytes", 0):,} |
| HE client online CPU seconds | {full.get("client_online_cpu_seconds", 0):.6f} |
| HE provider CPU seconds | {full.get("provider_total_cpu_seconds", 0):.6f} |
| HE complete stage-operation CPU seconds | {full.get("whole_projection_cpu_seconds", 0):.6f} |
| Client/provider setup CPU seconds | {context.get("setup_cpu_seconds", 0):.6f} |
| Prepared same-stage measured covered body bytes | {control.get("covered_measured_stage_body_bytes", 0):,} |
| Prepared native stage-operation aggregate CPU seconds | {control.get("aggregate_stage_cpu_seconds", 0):.6f} |
| Two-offset native arithmetic CPU seconds (partial comparator) | {control.get("two_offset_native_arithmetic_cpu_seconds", 0):.6f} |
| Sampled parent + worker peak RSS bytes | {int(result["supervisor"]["peak_sampled_parent_plus_worker_rss_bytes"]):,} |
| Observed worker CPU seconds | {result["supervisor"]["observed_worker_cpu_seconds"]:.6f} |

## Arithmetic, packing and resources

TenSEAL 0.3.17 / SEAL BFV, degree 8192, plaintext prime **268369921**
(`1 mod 16384`), coefficient bit sizes `[55,54,54,55]`, total **218**:
SEAL default tc128 degree-8192 coefficient bound. Parameter validity and batching
are checked natively; actual `parms_id`, binary/source SHA pins are in screen JSON.
TenSEAL cannot expose coefficient Modulus values or security enum through its
bindings; those getters are null, not fabricated measurements.

`t > 2*78,451,456` proves centered BFV output uniquely represents every public
domain dot, including intermediate partial sums. Bound is below signed int32
half-ring; converting centered output mod `2^32` reproduces prepared semantics.
Int64 oracle cannot overflow. Original client activation/weight scales and bias
semantics remain outside encrypted integer evaluation; no modular lift of random
prepared masks into BFV is assumed.

Scheme uses two padded 4096-slot inputs. Each public output uses two plaintext
dots (12 rotate/add steps each), then adds partials. Groups of 32 outputs pack,
then 28 groups pack to final 896-slot output. Full-width non-power-of-two input
cannot use direct TenSEAL dot (`step count too large`); an unpadded 768-slot tail
also corrupts later packed outputs. Padding to full BFV row is compulsory.
Fresh-context positive/negative boundary tests cover this regression and nested
packing. Provider receives public/Galois context only; no secret or relin keys.
Client-created default relin keys remain resident but are unused and not sent.

One CPU thread per native kernel/context. Preflight prices key/context copies,
transient encoded weights, ciphertext groups and allocator allowance. Encoding
is streamed, not all public plaintext weights retained. Full retained encoded
copies would cost 234,881,024 bytes for this layout. Supervisor samples aggregate
parent/worker RSS and worker CPU every 20 ms, terminates at 1 GiB / 120 CPU seconds,
and preserves partial evidence. Sampled peak is not an exact allocator high-water mark.
32×4864 native substage is preflight evidence only; full 4864→896 result is separate.

## Gate and projection scope

25% targeted-region body reduction **fails**, even excluding key setup. Matching
control executes fresh full-ring seeded masks, native offline correction and online
masked multiplication with existing request/response/correction codecs. Control
does not include live authorization/acks/transport. Client retains **zero dense
stage weights** in both designs, plus **3584 bytes** of output scales. Public stage
weights cost 4,358,144 bytes at each weight-owning provider; HE streaming int64/
plaintext copies, padded zero lanes and evaluation key expansion are charged separately.

JSON contains 1/10/100-response horizons for decode first, prefill39 and 39+32
(39 prefill + 31 executed decode rows). Prefill uses **rowwise projection**, no
measured batched HE prefill or cross-client key sharing. Whole-response known-body
projections replace one or all 24 down stages against pinned measured prepared
178,970,558-byte cohort. Only layer 0 weights execute; other layers, full decoder,
sampling/quality and transport remain unmeasured. Shared setup charged once per
client/provider, not once per stage. Actual per-key serialized sizes vary with fresh keys.

One-row client online screen cap is 1 CPU second; complete client cap and aggregate
compute comparison against complete two-offset protocol remain unadmitted. Native
two-offset arithmetic is explicitly partial. Missing admission/control, full wire,
cold weight distribution, prompt activation parity, prefill execution, complete
two-offset and full decoder CPU costs stay **null**. No missing term priced as zero.
Honest-but-curious research contract; malicious integrity cost unknown. No live
protocol, SDK/runtime/compiler integration or selectable component.

## Reproduction

```sh
{review["reproduce"]}
{review["test"]}
{review["lint"]}
{review["format"]}
```

Screen SHA256: `{review["screen_sha256"]}`.
"""
    REVIEW_MD.write_text(md)
    (ROOT / "docs/evidence/exact-linear-he-screen-2026-10-01.md").write_text(
        "# E3 exact linear HE screen — 2026-10-01\n\n"
        + "Actual source-locked 4864→896 packed BFV projection and fail-closed cost gate.\n\n"
        + f"Stop: `{review['stop_reason']}`. Full shape: {review['actual_full_projection_executed']}. "
        + f"All 896 outputs exact: {review['all_896_outputs_exact']}. Region gate: failed.\n\n"
        + "Full measurement: [screen JSON](exact-linear-he-screen-2026-10-01.json).\n\n"
        + "Method, boundaries, projected horizons and reproduction: "
        + "[review](exact-linear-he-review-2026-10-01.md).\n"
    )


if __name__ == "__main__":
    main()
