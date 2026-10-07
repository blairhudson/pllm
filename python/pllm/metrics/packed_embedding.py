"""FastQuery-inspired BFV lookup with explicitly charged half-gates conversion.

SEAL owns encryption. Rust owns bounded coefficient packing, uniform masks and
one-use conversion. Python orchestrates separate client/provider HE contexts and
uses an independent row oracle only at the trusted benchmark boundary.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import platform
import struct
import tempfile
import time

from pllm.configuration import ConfigurationError
from .operator_benchmarks import (
    _OperatorBenchmark, _sdk_digest, OperatorBenchmarkQuality, OperatorBenchmarkReport,
    OperatorBenchmarkResources, OperatorBenchmarkSample, OperatorCitation,
)

_SOURCE = OperatorCitation(
    "fastquery", "FastQuery: Communication-efficient Embedding Table Query for Private LLM Inference",
    "https://arxiv.org/abs/2405.16241", "5a242b1a65ceaaaa256a3f02fd0d3cc6eb40c249ca39c0e645502176520711be",
    "Sections 4.1–4.2 one-hot coefficient and 13-bit mixed-width packing; full ciphertext replies and local half-gates conversion replace PackLWEs and SiRNN",
)
_DEGREE, _MODULUS = 4096, 8192
_FRAME_BYTES = 2 * _DEGREE * 8 + 4096
_PARAMETER_FRAME_BYTES = 512
_WIDTHS = {"int8": (8,), "int4": (4, 4), "mixed_4_3_3": (4, 3, 3)}


def _save_frame(obj, path: Path, capacity: int) -> bytes:
    """SEAL's installed binding exposes file serialization only; scratch is private."""
    obj.save(str(path))
    try:
        length = path.stat().st_size
        if not 0 < length <= capacity - 8:
            raise ValueError("SEAL serialization exceeds its public frame bound")
        raw = path.read_bytes()
        if len(raw) != length:
            raise ValueError("SEAL serialization changed during receipt")
        return struct.pack("<Q", length) + raw + bytes(capacity - 8 - length)
    finally:
        path.unlink(missing_ok=True)


def _load_frame(obj, context, frame: bytes, path: Path, capacity: int):
    if len(frame) != capacity:
        raise ValueError("SEAL frame length differs from its public geometry")
    length = struct.unpack_from("<Q", frame)[0]
    if not 0 < length <= capacity - 8 or any(frame[8 + length:]):
        raise ValueError("invalid bounded SEAL frame")
    path.write_bytes(frame[8:8 + length])
    try:
        if context is None:
            obj.load(str(path))
        else:
            obj.load(context, str(path))
    finally:
        path.unlink(missing_ok=True)
    return obj


def _context(seal, *, key_switching=False, coefficient_bits=50):
    parameters = seal.EncryptionParameters(seal.SCHEME_TYPE.BFV)
    parameters.set_poly_modulus_degree(_DEGREE)
    parameters.set_coeff_modulus(seal.CoeffModulus.Create(_DEGREE,
        [coefficient_bits, coefficient_bits] if key_switching else [coefficient_bits]))
    parameters.set_plain_modulus(_MODULUS)
    context = seal.SEALContext(parameters, True, seal.SEC_LEVEL_TYPE.TC128)
    if not context.parameters_set():
        raise RuntimeError("SEAL rejected the fixed 128-bit-security parameter set")
    return parameters, context


def _sample(seal, native, texts, geometry, precision, token, directory):
    """One fresh HE keypair; each response and conversion is consumed serially."""
    sample_start = time.perf_counter()
    start = time.perf_counter()
    parameters, client_context = _context(seal)
    keygen = seal.KeyGenerator(client_context)
    secret = keygen.secret_key()
    public = seal.PublicKey()
    keygen.create_public_key(public)
    parameters_frame = _save_frame(parameters, directory / "parameters", _PARAMETER_FRAME_BYTES)
    public_frame = _save_frame(public, directory / "public", _FRAME_BYTES)
    remote_parameters = _load_frame(seal.EncryptionParameters(seal.SCHEME_TYPE.BFV), None,
                                   parameters_frame, directory / "remote-parameters", _PARAMETER_FRAME_BYTES)
    provider_context = seal.SEALContext(remote_parameters, True, seal.SEC_LEVEL_TYPE.TC128)
    remote_public = _load_frame(seal.PublicKey(), provider_context, public_frame,
                               directory / "remote-public", _FRAME_BYTES)
    encryptor = seal.Encryptor(client_context, public)
    decryptor = seal.Decryptor(client_context, secret)
    provider_encryptor = seal.Encryptor(provider_context, remote_public)
    evaluator = seal.Evaluator(provider_context)
    issuance = time.perf_counter() - start

    start = time.perf_counter()
    query = seal.Ciphertext()
    exponent = geometry["padded_rows"] - 1 - token
    encryptor.encrypt(seal.Plaintext(f"1x^{exponent}" if exponent else "1"), query)
    request = _save_frame(query, directory / "query", _FRAME_BYTES)
    remote_query = _load_frame(seal.Ciphertext(), provider_context, request,
                               directory / "remote-query", _FRAME_BYTES)
    online = time.perf_counter() - start
    output, conversion = [], None
    conversion_count = 0
    minimum_noise = None
    for polynomial_index, text in enumerate(texts):
        start = time.perf_counter()
        response = seal.Ciphertext()
        if text == "0":
            provider_encryptor.encrypt_zero(response)
        else:
            evaluator.multiply_plain(remote_query, seal.Plaintext(text), response)
        # Mask EVERY coefficient: decrypting an unmasked convolution would
        # otherwise reveal unselected table rows. Masks are independent per reply.
        mask_text, mask_bytes = native.packed_embedding_mask()
        evaluator.sub_plain_inplace(response, seal.Plaintext(mask_text))
        reply = _save_frame(response, directory / "reply", _FRAME_BYTES)
        received = _load_frame(seal.Ciphertext(), client_context, reply,
                               directory / "received", _FRAME_BYTES)
        noise = decryptor.invariant_noise_budget(received)
        if noise <= 0:
            raise RuntimeError("embedding response exhausted its BFV noise budget")
        minimum_noise = noise if minimum_noise is None else min(minimum_noise, noise)
        decoded = seal.Plaintext()
        decryptor.decrypt(received, decoded)
        online += time.perf_counter() - start
        for owner, coefficient in geometry["positions"]:
            if owner != polynomial_index:
                continue
            client_share = decoded.data(coefficient) if coefficient < decoded.coeff_count() else 0
            provider_share = struct.unpack_from("<H", mask_bytes, 2 * coefficient)[0]
            values, raw = native.packed_embedding_convert(precision, client_share, provider_share)
            details = json.loads(raw)
            issuance += details.pop("issuance_seconds")
            online += details.pop("online_seconds")
            if conversion is not None and details != conversion:
                raise RuntimeError("conversion geometry changed within one query")
            conversion = details
            conversion_count += 1
            output.extend(struct.unpack(f"<{len(values) // 4}i", values))
        # Scope retirement precedes the next one-use ciphertext/mask/conversion.
        del mask_text, mask_bytes, decoded, response, received, reply
    # Include conversion marshaling, frame extraction and orchestration in the
    # online wall time; aggregate CPU is independently sampled by the caller.
    online = time.perf_counter() - sample_start - issuance
    return output[:geometry["columns"]], conversion, conversion_count, issuance, online, minimum_noise


class _PackedEmbeddingBenchmark(_OperatorBenchmark):
    __slots__ = ()

    def run(self, *, table: Sequence[int] | None = None,
            token: Sequence[int] | None = None) -> OperatorBenchmarkReport:
        from pllm import _native, __version__

        params = self.component.get_params()
        rows, precision = params["vocabulary_size"], params["precision"]
        widths = _WIDTHS[precision]
        padded_rows = 1 << (rows - 1).bit_length()
        packed_columns = (self.lanes + len(widths) - 1) // len(widths)
        columns_per_polynomial = _DEGREE // padded_rows
        polynomials = (packed_columns + columns_per_polynomial - 1) // columns_per_polynomial
        allocation = 64 * (1 << 20) + 4 * rows * self.lanes + 32 * polynomials * _DEGREE * 2
        if allocation > 128 * (1 << 20):
            raise ConfigurationError("embedding reference exceeds the 128 MiB allocation estimate")
        variant = params.get("repacking", "none") != "none" or params.get("conversion", "local") != "local"
        if variant:
            from .repacked_embedding import allocation_bound
            allocation = allocation_bound(allocation, padded_rows, polynomials, packed_columns, params["repacking"])
        source = "public-fixture" if table is None and token is None else "caller-owned"
        if source == "public-fixture":
            table = [((i * 19 + 3) % (1 << widths[i % self.lanes % len(widths)]))
                     - (1 << (widths[i % self.lanes % len(widths)] - 1))
                     for i in range(rows * self.lanes)]
            token = [rows // 2]
        if (not isinstance(table, Sequence) or not isinstance(token, Sequence)
                or isinstance(table, (str, bytes)) or isinstance(token, (str, bytes))
                or len(table) != rows * self.lanes or len(token) != 1
                or type(token[0]) is not int or not 0 <= token[0] < rows
                or any(type(v) is not int or not -(1 << (widths[i % self.lanes % len(widths)] - 1))
                       <= v < (1 << (widths[i % self.lanes % len(widths)] - 1))
                       for i, v in enumerate(table))):
            raise ConfigurationError("embedding requires bounded row-major signed codes and one private row index")
        try:
            import _sealapi_cpp as seal
        except ImportError as exc:
            raise ConfigurationError("embedding benchmark requires the supported TenSEAL/SEAL optional dependency") from exc
        packed = bytes(v & 255 for v in table)
        setup_start = time.perf_counter()
        texts, raw_geometry = _native.packed_embedding_table(rows, self.lanes, precision, packed)
        public_setup = time.perf_counter() - setup_start
        geometry = json.loads(raw_geometry)
        # Native packing preflights <=1 MiB codes. This conservative live-owner
        # estimate includes Python text, SEAL pools, two contexts, coefficient
        # buffers, one response/mask and one <=4 MiB conversion circuit.
        if geometry["polynomials"] != polynomials or geometry["padded_rows"] != padded_rows:
            raise RuntimeError("native embedding geometry differs from preflight")
        samples, conversion, groups = [], None, None
        if variant:
            from .repacked_embedding import run_variant
            return run_variant(self, table, token[0], source, texts, geometry, public_setup, allocation)
        for index in range(self.samples):
            start = time.process_time()
            clear_start = time.perf_counter()
            expected = list(table[token[0] * self.lanes:(token[0] + 1) * self.lanes])
            clear_seconds = time.perf_counter() - clear_start
            with tempfile.TemporaryDirectory(prefix="pllm-embedding-") as directory:
                actual, measured, count, issuance, online, _ = _sample(
                    seal, _native, texts, geometry, precision, token[0], Path(directory))
            if actual != expected:
                raise RuntimeError("private embedding result differs from the independent row oracle")
            if conversion is not None and (measured != conversion or count != groups):
                raise RuntimeError("embedding resources changed across fresh queries")
            conversion, groups = measured, count
            samples.append(OperatorBenchmarkSample(issuance, online, clear_seconds,
                time.process_time() - start, ((_FRAME_BYTES, _FRAME_BYTES * geometry["polynomials"]),),
                public_setup_seconds=public_setup if index == 0 else 0.0))
        assert conversion is not None and groups is not None
        conversion_keys = (groups * conversion["issuer_encoding_bytes"], groups * conversion["evaluator_material_bytes"])
        he_keys = (_DEGREE * 8, _FRAME_BYTES + _PARAMETER_FRAME_BYTES)
        keys = tuple(a + b for a, b in zip(conversion_keys, he_keys))
        frames = (_FRAME_BYTES, _FRAME_BYTES * geometry["polynomials"])
        numeric = {"schema": "pllm.embedding.signed_integer.v1", "widths": widths,
                   "output": "sign-extended selected row in ring32", "vocabulary_size": rows}
        numeric_digest = "sha256:" + hashlib.sha256(json.dumps(numeric, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return OperatorBenchmarkReport(
            method="private-packed-embedding", configuration_json=json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")),
            configuration_digest=self.digest, numeric_contract_digest=numeric_digest,
            native_implementation_digest=geometry["native_implementation_digest"],
            sdk_implementation_digest=_sdk_digest("packed_embedding.py"), input_source=source,
            resources=OperatorBenchmarkResources(
                keys, tuple(k * 8 for k in keys), frames, sum(frames) * 8, 1,
                0, allocation,
                2 * _DEGREE + 4 * len(widths), groups * conversion["boundary_label_bytes"],
                role_names=("trusted-client-and-local-issuer", "provider-and-conversion-evaluator"),
                public_metadata_bytes_estimate=groups * conversion["public_metadata_bytes_estimate"],
                key_payload_bytes_by_operation=(("bfv-context", *he_keys), ("packed-share-conversion", *conversion_keys)),
                public_table_bytes_by_party=(0, len(packed) + geometry["coefficient_bytes"]),
            ),
            quality=OperatorBenchmarkQuality(self.lanes * self.samples, self.lanes * self.samples,
                "Independent exact Python row lookup; integer table is supplied already quantized", 0.0, 0.0),
            samples=tuple(samples), citations=(_SOURCE,),
            environment=(("platform", platform.platform()), ("python", platform.python_version()),
                         ("pllm", __version__), ("native", geometry["native_version"])),
            limitations=(
                "Bounded N=4096, plaintext modulus 8192, one 50-bit coefficient prime; SEAL TC128 admission",
                "HE public context is transferred to a distinct context without the client secret key; all response coefficients have fresh uniform masks",
                "Every sample creates a fresh HE key and serial one-use conversion circuits; no reusable secret handle escapes",
                "Honest-client one-hot input assumption; no malicious-query proof, circuit privacy proof or independent cryptographic review",
                "Full padded SEAL ciphertext replies: FastQuery PackLWEs is absent, so paper communication claims do not transfer",
                "Half-gates reconstruction/sign extension/re-sharing uses a local issuer, not SiRNN OT; it sees both input shares",
                "Measured frames count only HE query/replies; conversion labels are boundary estimates, separate from material and public instruction metadata",
                "HE material counts the raw secret polynomial and padded public key/parameters; SEAL native pool peaks remain unmeasured",
                "Public table bytes count provider codes plus coefficients; actual provider string/pool storage and checkpoint distribution are separate",
                "Caller supplies already-quantized codes; per-channel fitting, FastQuery fine-tuning and checkpoint quality are separate",
                "Current Qwen tied client head retains the same full matrix; embedding-only relocation does not release it",
                "No compiler-bound embedding placement, provider service, full-response memory saving or whole-decoder execution",
            ),
        )
