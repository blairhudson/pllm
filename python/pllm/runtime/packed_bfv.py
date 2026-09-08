from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


class PackedBFVUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PackedBFVResult:
    clear: np.ndarray
    context_ms: float
    key_generation_ms: float
    encode_encrypt_ms: float
    server_ms: float
    decrypt_decode_ms: float
    input_bytes: int
    output_bytes: int
    rotations: int
    plaintext_products: int
    streams: int
    input_width: int
    output_width: int
    segment_width: int
    noise_budget_bits: int

    @property
    def total_ms(self) -> float:
        return self.encode_encrypt_ms + self.server_ms + self.decrypt_decode_ms


class PackedBFVLinearFactory:
    """Reference packed BFV matrix preparation with BSGS rotations.

    Several independent random masks are placed into guarded regions of the two
    BFV batching rows. A single encrypted matrix program returns all output
    masks. The implementation uses the low level Microsoft SEAL bindings shipped
    with TenSEAL. It is intended as a reproducible CPU reference for preparation
    research rather than a production GPU backend.
    """

    def __init__(
        self,
        *,
        poly_modulus_degree: int = 4096,
        plain_modulus: int = 65537,
        threads: int = 1,
        tenseal_path: str | Path | None = None,
    ) -> None:
        if threads < 1:
            raise ValueError("threads must be positive")
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        if tenseal_path is not None:
            import sys

            sys.path.insert(0, str(tenseal_path))
        try:
            import _sealapi_cpp as seal
        except ImportError as exc:
            raise PackedBFVUnavailable("TenSEAL low level bindings are required") from exc
        self.seal = seal
        self.poly_modulus_degree = int(poly_modulus_degree)
        self.plain_modulus = int(plain_modulus)

        started = time.perf_counter_ns()
        parms = seal.EncryptionParameters(seal.SCHEME_TYPE.BFV)
        parms.set_poly_modulus_degree(self.poly_modulus_degree)
        parms.set_coeff_modulus(
            seal.CoeffModulus.BFVDefault(
                self.poly_modulus_degree, seal.SEC_LEVEL_TYPE.TC128
            )
        )
        parms.set_plain_modulus(seal.Modulus(self.plain_modulus))
        self.context = seal.SEALContext(parms, True, seal.SEC_LEVEL_TYPE.TC128)
        if not self.context.parameters_set():
            raise ValueError(self.context.parameters_error_message())
        self.encoder = seal.BatchEncoder(self.context)
        self.evaluator = seal.Evaluator(self.context)
        self.context_ms = (time.perf_counter_ns() - started) / 1e6

        started = time.perf_counter_ns()
        generator = seal.KeyGenerator(self.context)
        self.secret_key = generator.secret_key()
        self.galois_keys = seal.GaloisKeys()
        generator.create_galois_keys(self.galois_keys)
        self.encryptor = seal.Encryptor(self.context, self.secret_key)
        self.decryptor = seal.Decryptor(self.context, self.secret_key)
        self.key_generation_ms = (time.perf_counter_ns() - started) / 1e6
        self.slot_count = int(self.encoder.slot_count())
        self.row_size = self.slot_count // 2

    @staticmethod
    def _choose_baby_steps(diagonal_count: int) -> int:
        root = max(1, int(round(math.sqrt(diagonal_count))))
        candidates = range(max(1, root - 8), root + 9)
        return min(
            candidates,
            key=lambda value: value + math.ceil(diagonal_count / value),
        )

    def capacity(self, input_width: int, output_width: int) -> int:
        segment = int(input_width) + int(output_width) - 1
        if segment > self.row_size:
            return 0
        return 2 * (self.row_size // segment)

    def _save_size(self, value) -> int:
        with tempfile.NamedTemporaryFile(suffix=".seal", delete=False) as handle:
            path = handle.name
        try:
            value.save(path)
            return Path(path).stat().st_size
        finally:
            Path(path).unlink(missing_ok=True)

    def evaluate(
        self,
        masks: np.ndarray | Sequence[Sequence[int]],
        weight: np.ndarray | Sequence[Sequence[int]],
    ) -> PackedBFVResult:
        mask_array = np.asarray(masks, dtype=np.int64)
        weight_array = np.asarray(weight, dtype=np.int64)
        if mask_array.ndim != 2 or weight_array.ndim != 2:
            raise ValueError("masks and weight must be matrices")
        streams, input_width = mask_array.shape
        output_width, weight_input = weight_array.shape
        if input_width != weight_input:
            raise ValueError("input widths differ")
        segment = input_width + output_width - 1
        capacity = self.capacity(input_width, output_width)
        if capacity < streams:
            raise ValueError(
                f"shape supports {capacity} packed streams, received {streams}"
            )
        modulus = self.plain_modulus
        per_row = self.row_size // segment

        encoded = np.zeros(self.slot_count, dtype=np.int64)
        for stream_index in range(streams):
            row_index, segment_index = divmod(stream_index, per_row)
            base = row_index * self.row_size + segment_index * segment
            start = base + output_width - 1
            encoded[start : start + input_width] = mask_array[stream_index] % modulus

        started = time.perf_counter_ns()
        plain = self.seal.Plaintext()
        self.encoder.encode(encoded.tolist(), plain)
        encrypted = self.seal.Ciphertext()
        self.encryptor.encrypt_symmetric(plain, encrypted)
        encode_encrypt_ms = (time.perf_counter_ns() - started) / 1e6
        input_bytes = self._save_size(encrypted)

        diagonal_count = segment
        baby_count = self._choose_baby_steps(diagonal_count)
        giant_count = math.ceil(diagonal_count / baby_count)
        started = time.perf_counter_ns()
        baby_rotations = [encrypted]
        for baby_index in range(1, baby_count):
            rotated = self.seal.Ciphertext()
            self.evaluator.rotate_rows(
                encrypted, baby_index, self.galois_keys, rotated
            )
            baby_rotations.append(rotated)

        giant_terms = []
        plaintext_products = 0
        giant_rotations = 0
        for giant_index in range(giant_count):
            giant_shift = giant_index * baby_count
            inner_terms = []
            for baby_index in range(baby_count):
                diagonal_index = giant_shift + baby_index
                if diagonal_index >= diagonal_count:
                    break
                diagonal = np.zeros(self.slot_count, dtype=np.int64)
                any_nonzero = False
                for stream_index in range(streams):
                    row_index, segment_index = divmod(stream_index, per_row)
                    base = row_index * self.row_size + segment_index * segment
                    for output_index in range(output_width):
                        input_index = (
                            output_index
                            + diagonal_index
                            - (output_width - 1)
                        )
                        if 0 <= input_index < input_width:
                            value = int(weight_array[output_index, input_index])
                            diagonal[base + output_index] = value
                            any_nonzero = any_nonzero or value != 0
                if not any_nonzero:
                    continue
                # A giant rotation is applied after the inner sum. Rotate each
                # plaintext in the opposite direction before encoding.
                if giant_shift:
                    first = np.roll(diagonal[: self.row_size], giant_shift)
                    second = np.roll(diagonal[self.row_size :], giant_shift)
                    diagonal = np.concatenate((first, second))
                diagonal_plain = self.seal.Plaintext()
                self.encoder.encode(diagonal.tolist(), diagonal_plain)
                product = self.seal.Ciphertext()
                self.evaluator.multiply_plain(
                    baby_rotations[baby_index], diagonal_plain, product
                )
                inner_terms.append(product)
                plaintext_products += 1
            if not inner_terms:
                continue
            if len(inner_terms) == 1:
                inner = inner_terms[0]
            else:
                inner = self.seal.Ciphertext()
                self.evaluator.add_many(inner_terms, inner)
            if giant_shift:
                outer = self.seal.Ciphertext()
                self.evaluator.rotate_rows(
                    inner, giant_shift, self.galois_keys, outer
                )
                giant_terms.append(outer)
                giant_rotations += 1
            else:
                giant_terms.append(inner)
        if not giant_terms:
            raise RuntimeError("matrix produced no encrypted terms")
        if len(giant_terms) == 1:
            result = giant_terms[0]
        else:
            result = self.seal.Ciphertext()
            self.evaluator.add_many(giant_terms, result)
        server_ms = (time.perf_counter_ns() - started) / 1e6
        output_bytes = self._save_size(result)

        started = time.perf_counter_ns()
        output_plain = self.seal.Plaintext()
        self.decryptor.decrypt(result, output_plain)
        decoded = np.asarray(self.encoder.decode_int64(output_plain), dtype=np.int64)
        clear = np.empty((streams, output_width), dtype=np.int64)
        for stream_index in range(streams):
            row_index, segment_index = divmod(stream_index, per_row)
            base = row_index * self.row_size + segment_index * segment
            clear[stream_index] = decoded[base : base + output_width] % modulus
        decrypt_decode_ms = (time.perf_counter_ns() - started) / 1e6
        expected = (mask_array % modulus @ weight_array.T) % modulus
        if not np.array_equal(clear, expected):
            raise AssertionError("packed BFV result does not match clear arithmetic")
        noise = int(self.decryptor.invariant_noise_budget(result))
        return PackedBFVResult(
            clear=clear,
            context_ms=self.context_ms,
            key_generation_ms=self.key_generation_ms,
            encode_encrypt_ms=encode_encrypt_ms,
            server_ms=server_ms,
            decrypt_decode_ms=decrypt_decode_ms,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            rotations=(baby_count - 1) + giant_rotations,
            plaintext_products=plaintext_products,
            streams=streams,
            input_width=input_width,
            output_width=output_width,
            segment_width=segment,
            noise_budget_bits=noise,
        )
