"""Complete bounded coefficient-repacked HE lookup and party-input conversion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import secrets
import struct
import tempfile
import time

from pllm.configuration import ConfigurationError
from pllm.runtime import coefficient_repacking as packing
from .packed_embedding import (
    _context, _save_frame, _load_frame, _FRAME_BYTES, _PARAMETER_FRAME_BYTES,
    _DEGREE, _WIDTHS, _SOURCE,
)
from .operator_benchmarks import (
    _sdk_digest, OperatorBenchmarkQuality, OperatorBenchmarkReport,
    OperatorBenchmarkResources, OperatorBenchmarkSample, OperatorCitation,
)

_PACKING_SOURCE = OperatorCitation(
    "bumblebee", "BumbleBee: Secure Two-party Inference Framework for Large Transformers",
    "https://eprint.iacr.org/2023/1678", "15583a86593998079d42b8b134dc4f4f5be9ca72a7ced8f31bba05f68893516a",
    "Section IV coefficient trace and Figure 6 FFT-style interleaving; independently adapted to BFV one-hot embedding outputs",
)
_FASTQUERY_SOURCE = OperatorCitation(_SOURCE.paper_id, _SOURCE.title, _SOURCE.url, _SOURCE.source_sha256,
    "Sections 2.2 and 4.1–4.2 one-hot, mixed-width and extracted-LWE output packing; explicit BFV trace/interleave and party-local half-gates adaptations, not SiRNN")


def _packing_geometry(stride, count, groups, mode):
    if mode == "pack_lwes":
        return packing.geometry(_DEGREE, groups)
    if mode == "interleave":
        return packing.geometry(stride, count)
    return 1, (), 0


def _convert(native, precision, client_share, provider_share, plan, session, index, party_local):
    if not party_local:
        output, raw = native.packed_embedding_convert(precision, client_share, provider_share)
        r = json.loads(raw)
        return output, {"keys": [r["issuer_encoding_bytes"], r["evaluator_material_bytes"]],
                        "metadata": r["public_metadata_bytes_estimate"], "boundary": r["boundary_label_bytes"],
                        "phases": [], "allocation": r["protocol_allocation_estimate_bytes"]}, r["issuance_seconds"], r["online_seconds"]
    start = time.perf_counter()
    provider, client, raw = native.packed_embedding_issue_parties(precision, plan, session, index)
    issuance = time.perf_counter() - start
    r = json.loads(raw)
    start = time.perf_counter()
    try:
        a, b = provider.start(struct.pack("<I", provider_share)), client.start(struct.pack("<I", client_share))
        phases = [(len(b), len(a))]
        done_a, next_a = provider.advance(b)
        done_b, next_b = client.advance(a)
        if done_a or done_b:
            raise RuntimeError("packed conversion completed before OT response")
        phases.append((len(next_b), len(next_a)))
        done_a, a = provider.advance(next_b)
        done_b, b = client.advance(next_a)
        if not done_a or not done_b or len(a) != len(b):
            raise RuntimeError("packed conversion parties disagree on completion")
        if phases != [tuple(reversed(p)) for p in r["phase_bytes"]]:
            raise RuntimeError("packed conversion frame conservation differs")
        # Trusted benchmark oracle only. Each party receives only its output share.
        count = len(a) // 4
        words = [((x + y) & 0xffffffff) for x, y in zip(struct.unpack(f"<{count}I", a), struct.unpack(f"<{count}I", b))]
        output = struct.pack(f"<{count}I", *words)
    finally:
        provider.cancel()
        client.cancel()
    return output, {"keys": list(reversed(r["key_bytes"])), "metadata": r["public_metadata_bytes"],
                    "boundary": 0, "phases": phases, "allocation": r["allocation_bytes"]}, issuance, time.perf_counter() - start


def _sample(seal, native, texts, shape, precision, token, directory, repacking, party_local, plan):
    started = time.perf_counter()
    stride, count = shape["padded_rows"], shape["polynomials"]
    _, elements, operations = _packing_geometry(stride, count, len(shape["positions"]), repacking)
    keyswitch = bool(elements)
    # 54+54 remains within SEAL's 109-bit TC128 N=4096 bound; one data prime
    # survives key switching and each reply checks its remaining noise budget.
    bits = 54 if keyswitch else 50
    parameters, local = _context(seal, key_switching=keyswitch, coefficient_bits=bits)
    kg = seal.KeyGenerator(local)
    public = seal.PublicKey()
    kg.create_public_key(public)
    pk_capacity = 2 * _DEGREE * 8 * (2 if keyswitch else 1) + 4096
    parameter_frame = _save_frame(parameters, directory / "params", _PARAMETER_FRAME_BYTES)
    public_frame = _save_frame(public, directory / "public", pk_capacity)
    remote_parameters = _load_frame(seal.EncryptionParameters(seal.SCHEME_TYPE.BFV), None,
                                   parameter_frame, directory / "remote-params", _PARAMETER_FRAME_BYTES)
    remote = seal.SEALContext(remote_parameters, True, seal.SEC_LEVEL_TYPE.TC128)
    if not remote.parameters_set():
        raise RuntimeError("provider SEAL parameters rejected")
    remote_public = _load_frame(seal.PublicKey(), remote, public_frame, directory / "remote-pk", pk_capacity)
    key_frame_bytes, remote_keys = 0, None
    if keyswitch:
        keys = seal.GaloisKeys()
        kg.create_galois_keys(list(elements), keys)
        key_frame_bytes = len(elements) * (2 * _DEGREE * 2 * 8 + 4096) + 4096
        frame = _save_frame(keys, directory / "galois", key_frame_bytes)
        remote_keys = _load_frame(seal.GaloisKeys(), remote, frame, directory / "remote-galois", key_frame_bytes)
    encryptor = seal.Encryptor(local, public)
    decryptor = seal.Decryptor(local, kg.secret_key())
    provider_encryptor = seal.Encryptor(remote, remote_public)
    evaluator = seal.Evaluator(remote)
    issuance = time.perf_counter() - started
    query = seal.Ciphertext()
    exponent = stride - 1 - token
    encryptor.encrypt(seal.Plaintext(f"1x^{exponent}" if exponent else "1"), query)
    request = _save_frame(query, directory / "query", _FRAME_BYTES)
    remote_query = _load_frame(seal.Ciphertext(), remote, request, directory / "remote-query", _FRAME_BYTES)
    responses = []
    for text in texts:
        response = seal.Ciphertext()
        if text == "0":
            provider_encryptor.encrypt_zero(response)
        else:
            evaluator.multiply_plain(remote_query, seal.Plaintext(text), response)
        responses.append(response)
    positions = shape["positions"]
    if repacking == "pack_lwes":
        response, positions, actual_operations = packing.pack_lwes(
            seal, native, remote, evaluator, provider_encryptor, remote_keys, responses,
            positions, request[8:24], directory / "rns")
        if operations != actual_operations:
            raise RuntimeError("LWE packing operation budget differs")
        responses = [response]
    elif repacking == "interleave":
        response, position, actual_operations = packing.interleave(
            seal, native, remote, evaluator, provider_encryptor, remote_keys, responses,
            stride, request[8:24], directory / "rns")
        if operations != actual_operations:
            raise RuntimeError("ciphertext repacking operation budget differs")
        positions = [(0, position(owner, coefficient)) for owner, coefficient in positions]
        responses = [response]
    # Independent client/provider secret share owners. The issuer receives no
    # plaintext input. Only the final independent benchmark oracle reconstructs.
    output = [None] * len(positions)
    phases = [(_FRAME_BYTES, len(responses) * _FRAME_BYTES)]
    conversion, minimum_noise = None, None
    session = secrets.token_bytes(32)
    for index, response in enumerate(responses):
        mask_text, mask_bytes = native.packed_embedding_mask()
        evaluator.sub_plain_inplace(response, seal.Plaintext(mask_text))
        reply = _save_frame(response, directory / "reply", _FRAME_BYTES)
        received = _load_frame(seal.Ciphertext(), local, reply, directory / "received", _FRAME_BYTES)
        noise = decryptor.invariant_noise_budget(received)
        if noise <= 0:
            raise RuntimeError("repacked response exhausted BFV noise budget")
        minimum_noise = noise if minimum_noise is None else min(noise, minimum_noise)
        decoded = seal.Plaintext()
        decryptor.decrypt(received, decoded)
        for group, (owner, coefficient) in enumerate(positions):
            if owner != index:
                continue
            client_share = decoded.data(coefficient) if coefficient < decoded.coeff_count() else 0
            provider_share = struct.unpack_from("<H", mask_bytes, 2 * coefficient)[0]
            values, stats, issue_seconds, _ = _convert(native, precision, client_share, provider_share, plan, session, group, party_local)
            issuance += issue_seconds
            if conversion is not None and stats != conversion:
                raise RuntimeError("private values changed conversion resources")
            conversion = stats
            phases.extend(tuple(pair) for pair in stats["phases"])
            output[group] = struct.unpack(f"<{len(values)//4}i", values)
        del decoded, mask_bytes, mask_text, received, reply
    context_keys = (_DEGREE * 8 * (2 if keyswitch else 1) + pk_capacity + _PARAMETER_FRAME_BYTES,
                    pk_capacity + _PARAMETER_FRAME_BYTES)
    stats = {"conversion": conversion, "phases": phases, "context_keys": context_keys,
             "galois_keys": (key_frame_bytes, key_frame_bytes), "automorphisms": operations,
             "setup_delivery": pk_capacity + _PARAMETER_FRAME_BYTES + key_frame_bytes,
             "rns_primes": 2 if keyswitch else 1, "coefficient_bits": bits}
    return [v for group in output for v in group][:shape["columns"]], stats, issuance, time.perf_counter() - started - issuance, minimum_noise


def allocation_bound(allocation, stride, count, groups, repacking):
    _, elements, _ = _packing_geometry(stride, count, groups, repacking)
    allocation += len(elements) * (2 * _DEGREE * 2 * 8 + 4096) * 8 + 8 * (1 << 20)
    if repacking == "pack_lwes":
        allocation += 4 * groups * 2 * _DEGREE * 8
    if allocation > 128 * (1 << 20):
        raise ConfigurationError("repacked embedding exceeds the 128 MiB allocation estimate")
    return allocation


def run_variant(benchmark, table, token, source, texts, shape, public_setup, allocation):
    from pllm import _native, __version__
    import _sealapi_cpp as seal

    params = benchmark.component.get_params()
    repacking, party_local = params["repacking"], params["conversion"] == "party_local"
    # Both resident Galois-key copies, generation/serialization scratch, all
    # intermediate ciphertexts and one serial conversion were preflighted before
    # native table packing, public text construction or any HE/OT issuance.
    samples, reference, minimum_noise = [], None, None
    for index in range(benchmark.samples):
        cpu = time.process_time()
        start = time.perf_counter()
        expected = list(table[token * benchmark.lanes:(token + 1) * benchmark.lanes])
        clear = time.perf_counter() - start
        with tempfile.TemporaryDirectory(prefix="pllm-repacked-embedding-") as directory:
            output, stats, issuance, online, noise = _sample(seal, _native, texts, shape, params["precision"], token,
                Path(directory), repacking, party_local, bytes.fromhex(benchmark.digest))
        if output != expected:
            raise RuntimeError("repacked embedding differs from independent signed row oracle")
        if reference is not None and reference != stats:
            raise RuntimeError("repacked embedding geometry changes across samples")
        reference = stats
        minimum_noise = noise if minimum_noise is None else min(minimum_noise, noise)
        samples.append(OperatorBenchmarkSample(issuance, online, clear, time.process_time() - cpu,
            tuple(stats["phases"]), public_setup_seconds=public_setup if index == 0 else 0.0))
    r = reference
    groups = len(shape["positions"])
    conversion_keys = tuple(groups * k for k in r["conversion"]["keys"])
    keys = tuple(sum(x) for x in zip(conversion_keys, r["context_keys"], r["galois_keys"]))
    frames = tuple(sum(p[i] for p in r["phases"]) for i in (0, 1))
    numeric = {"schema": "pllm.embedding.signed_integer.v1", "widths": _WIDTHS[params["precision"]],
               "output": "sign-extended selected row in ring32", "vocabulary_size": params["vocabulary_size"]}
    numeric_digest = "sha256:" + hashlib.sha256(json.dumps(numeric, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return OperatorBenchmarkReport(
        method="private-packed-embedding", configuration_json=json.dumps(benchmark.to_dict(), sort_keys=True, separators=(",", ":")),
        configuration_digest=benchmark.digest, numeric_contract_digest=numeric_digest,
        native_implementation_digest=shape["native_implementation_digest"], sdk_implementation_digest=_sdk_digest("repacked_embedding.py"),
        input_source=source, resources=OperatorBenchmarkResources(
            keys, tuple(k*8 for k in keys), frames, sum(frames)*8, len(r["phases"]), 0, allocation,
            2*_DEGREE + 4*len(_WIDTHS[params["precision"]]), groups*r["conversion"]["boundary"],
            role_names=("client-decryptor-and-conversion-evaluator", "provider-and-conversion-garbler") if party_local else
                       ("client-and-local-issuer", "provider-and-conversion-evaluator"),
            public_metadata_bytes_estimate=groups*r["conversion"]["metadata"],
            key_payload_bytes_by_operation=(("bfv-context-secret-and-public", *r["context_keys"]),
                ("repacking-galois-key-copies", *r["galois_keys"]), ("packed-share-conversion", *conversion_keys)),
            public_table_bytes_by_party=(0, len(table) + shape["coefficient_bytes"])),
        quality=OperatorBenchmarkQuality(benchmark.lanes*benchmark.samples, benchmark.lanes*benchmark.samples,
            "Independent Python signed-integer row oracle; fixed precision and numeric identity", 0.0, 0.0),
        samples=tuple(samples), citations=(_FASTQUERY_SOURCE, _PACKING_SOURCE) if repacking != "none" else (_FASTQUERY_SOURCE,),
        environment=(("platform",platform.platform()), ("python",platform.python_version()), ("pllm",__version__),
            ("native",shape["native_version"]), ("minimum_bfv_noise_budget_bits",str(minimum_noise))),
        limitations=(
            f"N=4096, t=8192; {r['rns_primes']} {r['coefficient_bits']}-bit key-context primes with SEAL TC128 validation; one {r['coefficient_bits']}-bit online data prime",
            f"Fresh setup delivery is {r['setup_delivery']} padded bytes per sample, separate from online frames; both public-key copies and client secret polynomial are charged",
            f"Complete trace/interleave uses {r['automorphisms']} public-geometry automorphisms per query; all other response coefficients receive uniform fresh masks",
            "Extracted-LWE PackLWEs uses native sample extraction, constant-term embedding and trace/interleave; FastQuery's full system and SiRNN are not reproduced" if repacking == "pack_lwes" else
            "Direct interleaving is a BumbleBee-derived alternative to extracted-LWE packing; repacking choice is explicit" if repacking == "interleave" else
            "Unrepacked control sends every full BFV reply ciphertext",
            "Party-local input/output shares use local-dealer random-OT correlations; distributed OT setup, SiRNN and authenticated role transport are absent" if party_local else
            "Local conversion issuer sees both shares; no distributed SiRNN conversion",
            "Every sample creates fresh HE keys and retires serial one-use conversion state; benchmark oracle alone reconstructs final words",
            "Honest-client one-hot assumption; no malicious-query proof, circuit privacy proof or independent cryptographic review",
            "Frames include HE query/reply plus actual conversion phases; material, setup delivery, boundary estimates and public metadata remain distinct",
            "Caller-owned integer codes; quantizer fitting, fine-tuning and full-checkpoint quality are separate",
            "Tied client output head still retains the complete matrix: embedding relocation releases zero client weight bytes",
            "No decoder placement, independent providers, physical-wire or measured whole-client RSS claim",
        ),
    )
