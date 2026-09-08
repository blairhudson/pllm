from __future__ import annotations

import importlib
import math
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Sequence

import msgpack
import numpy as np


POLY_MODULUS_DEGREE = 8192
SEGMENT_TARGET = 2048
MAX_STREAMS = 4
_EVALUATION_GROUP_CHUNK = 4
_MAGIC = b"PLLM-TILED-BFV"
_VERSION = 1
_BACKEND = "tiled-bfv"


class TiledBFVUnavailable(RuntimeError):
    pass


class TiledBFVError(ValueError):
    pass


def import_tenseal(tenseal_path: str | Path | None = None):
    """Load TenSEAL before its low-level SEAL extension."""
    if tenseal_path is not None:
        path = str(tenseal_path)
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        importlib.import_module("tenseal")
        return importlib.import_module("_sealapi_cpp")
    except ImportError as exc:
        raise TiledBFVUnavailable(
            "Install the HE extra or provide a directory containing TenSEAL"
        ) from exc


def _save_seal(value: Any) -> bytes:
    with tempfile.TemporaryDirectory(prefix="pllm-tiled-bfv-") as directory:
        path = Path(directory) / "value.seal"
        value.save(str(path))
        return path.read_bytes()


def _load_parameters(seal: Any, payload: bytes):
    value = seal.EncryptionParameters(seal.SCHEME_TYPE.BFV)
    with tempfile.TemporaryDirectory(prefix="pllm-tiled-bfv-") as directory:
        path = Path(directory) / "parameters.seal"
        path.write_bytes(payload)
        value.load(str(path))
    return value


def _load_with_context(seal: Any, cls: Any, context: Any, payload: bytes, label: str):
    value = cls()
    try:
        with tempfile.TemporaryDirectory(prefix="pllm-tiled-bfv-") as directory:
            path = Path(directory) / "value.seal"
            path.write_bytes(payload)
            value.load(context, str(path))
    except Exception as exc:
        raise TiledBFVError(f"invalid {label}") from exc
    return value


def _require_int(value: Any, label: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise TiledBFVError(f"{label} must be an integer >= {minimum}")
    return value


def _unpack(payload: bytes, kind: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise TiledBFVError(f"{kind} payload must be bytes")
    try:
        value = msgpack.unpackb(payload, raw=False, strict_map_key=True)
    except Exception as exc:
        raise TiledBFVError(f"invalid {kind} envelope") from exc
    if not isinstance(value, dict) or set(value) != keys:
        raise TiledBFVError(f"invalid {kind} envelope fields")
    if value["magic"] != _MAGIC:
        raise TiledBFVError(f"invalid {kind} magic")
    if type(value["version"]) is not int or value["version"] != _VERSION:
        raise TiledBFVError(f"unsupported {kind} version")
    if value["backend"] != _BACKEND:
        raise TiledBFVError(f"invalid {kind} backend")
    if value["kind"] != kind:
        raise TiledBFVError(f"invalid envelope kind, expected {kind}")
    return value


_CONTEXT_KEYS = {
    "magic",
    "version",
    "backend",
    "kind",
    "poly_modulus_degree",
    "plain_modulus",
    "parameters",
    "galois_keys",
}
_REQUEST_KEYS = {
    "magic",
    "version",
    "backend",
    "kind",
    "input_width",
    "output_width",
    "streams",
    "input_tiles",
    "ciphertexts",
}
_RESPONSE_KEYS = {
    "magic",
    "version",
    "backend",
    "kind",
    "input_width",
    "output_width",
    "streams",
    "output_tiles",
    "ciphertexts",
}


def _pack(kind: str, **values: Any) -> bytes:
    return msgpack.packb(
        {
            "magic": _MAGIC,
            "version": _VERSION,
            "backend": _BACKEND,
            "kind": kind,
            **values,
        },
        use_bin_type=True,
    )


def tiled_context_modulus(payload: bytes) -> int | None:
    """Return the modulus for a valid tiled context envelope, else None."""
    try:
        envelope = _unpack(payload, "context", _CONTEXT_KEYS)
        return _validate_modulus(envelope["plain_modulus"])
    except TiledBFVError:
        return None


def _tile_widths(input_width: int, output_width: int) -> tuple[int, int]:
    if input_width + output_width - 1 <= SEGMENT_TARGET:
        return input_width, output_width
    if input_width <= SEGMENT_TARGET and (
        input_width <= output_width or output_width > SEGMENT_TARGET
    ):
        return input_width, SEGMENT_TARGET - input_width + 1
    if output_width <= SEGMENT_TARGET:
        return SEGMENT_TARGET - output_width + 1, output_width
    input_tile = (SEGMENT_TARGET + 1) // 2
    return input_tile, SEGMENT_TARGET + 1 - input_tile


def _baby_steps(diagonal_count: int) -> int:
    root = max(1, int(round(math.sqrt(diagonal_count))))
    return min(
        range(max(1, root - 8), root + 9),
        key=lambda value: value + math.ceil(diagonal_count / value),
    )


def _validate_modulus(plain_modulus: int) -> int:
    modulus = _require_int(plain_modulus, "plain_modulus", minimum=3)
    if (modulus - 1) % (2 * POLY_MODULUS_DEGREE):
        raise TiledBFVError(
            f"plain_modulus must be batching-compatible: 1 mod {2 * POLY_MODULUS_DEGREE}"
        )
    return modulus


def _new_context(seal: Any, plain_modulus: int):
    parameters = seal.EncryptionParameters(seal.SCHEME_TYPE.BFV)
    parameters.set_poly_modulus_degree(POLY_MODULUS_DEGREE)
    parameters.set_coeff_modulus(
        seal.CoeffModulus.BFVDefault(POLY_MODULUS_DEGREE, seal.SEC_LEVEL_TYPE.TC128)
    )
    parameters.set_plain_modulus(seal.Modulus(plain_modulus))
    context = seal.SEALContext(parameters, True, seal.SEC_LEVEL_TYPE.TC128)
    if not context.parameters_set():
        raise TiledBFVError(context.parameters_error_message())
    return parameters, context


class TiledBFVClient:
    """Client-owned symmetric BFV encryption and decryption state."""

    def __init__(
        self,
        input_width: int,
        output_width: int,
        *,
        plain_modulus: int,
        threads: int = 1,
        tenseal_path: str | Path | None = None,
        _shared: TiledBFVClient | None = None,
    ) -> None:
        self.input_width = _require_int(input_width, "input_width")
        self.output_width = _require_int(output_width, "output_width")
        self.plain_modulus = _validate_modulus(plain_modulus)
        threads = _require_int(threads, "threads")
        self.threads = threads
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        if _shared is None:
            self.seal = import_tenseal(tenseal_path)
            parameters, self.context = _new_context(self.seal, self.plain_modulus)
            generator = self.seal.KeyGenerator(self.context)
            self._secret_key = generator.secret_key()
            self._encryptor = self.seal.Encryptor(self.context, self._secret_key)
            self._decryptor = self.seal.Decryptor(self.context, self._secret_key)
            galois_keys = self.seal.GaloisKeys()
            baby_count = _baby_steps(SEGMENT_TARGET)
            galois_elements = [pow(3, step, 2 * POLY_MODULUS_DEGREE) for step in (1, baby_count)]
            generator.create_galois_keys(galois_elements, galois_keys)
            self.public_context = _pack(
                "context",
                poly_modulus_degree=POLY_MODULUS_DEGREE,
                plain_modulus=self.plain_modulus,
                parameters=_save_seal(parameters),
                galois_keys=_save_seal(galois_keys),
            )
        else:
            if self.plain_modulus != _shared.plain_modulus:
                raise TiledBFVError("shared context has a different plain modulus")
            self.seal = _shared.seal
            self.context = _shared.context
            self._secret_key = _shared._secret_key
            self._encryptor = _shared._encryptor
            self._decryptor = _shared._decryptor
            self.public_context = _shared.public_context
        self.encoder = self.seal.BatchEncoder(self.context)
        self.slot_count = int(self.encoder.slot_count())
        self.row_size = self.slot_count // 2
        self.input_tile_width, self.output_tile_width = _tile_widths(
            self.input_width, self.output_width
        )
        self.segment_width = self.input_tile_width + self.output_tile_width - 1
        if self.segment_width > min(SEGMENT_TARGET, self.row_size // 2):
            raise TiledBFVError("tile segment exceeds guarded BFV row capacity")
        self.capacity = min(MAX_STREAMS, 2 * (self.row_size // self.segment_width))
        self.input_tiles = math.ceil(self.input_width / self.input_tile_width)
        self.output_tiles = math.ceil(self.output_width / self.output_tile_width)
        self._per_row = self.row_size // self.segment_width
        self._baby_count = min(_baby_steps(SEGMENT_TARGET), self.segment_width)

    def for_shape(self, input_width: int, output_width: int) -> TiledBFVClient:
        return TiledBFVClient(
            input_width,
            output_width,
            plain_modulus=self.plain_modulus,
            _shared=self,
        )

    def group_sizes(self, stream_count: int) -> list[int]:
        count = _require_int(stream_count, "stream_count")
        full, remainder = divmod(count, self.capacity)
        return [self.capacity] * full + ([remainder] if remainder else [])

    def _bases(self, streams: int) -> list[int]:
        return [
            (index // self._per_row) * self.row_size + (index % self._per_row) * self.segment_width
            for index in range(streams)
        ]

    def encrypt_many(self, masks: np.ndarray | Sequence[Sequence[int]]) -> list[bytes]:
        array = np.asarray(masks)
        if array.ndim != 2 or array.shape[1] != self.input_width:
            raise TiledBFVError(f"masks must have shape (streams, {self.input_width})")
        if array.dtype.kind not in "iu":
            raise TiledBFVError("masks must contain integers")
        if array.shape[0] < 1:
            raise TiledBFVError("at least one mask is required")
        modular = (array % self.plain_modulus).astype(np.int64, copy=False)
        payloads: list[bytes] = []
        offset = 0
        for streams in self.group_sizes(int(array.shape[0])):
            group = modular[offset : offset + streams]
            bases = self._bases(streams)
            ciphertexts: list[bytes] = []
            for input_start in range(0, self.input_width, self.input_tile_width):
                width = min(self.input_tile_width, self.input_width - input_start)
                slots = np.zeros(self.slot_count, dtype=np.int64)
                for row, base in zip(group, bases, strict=True):
                    start = base + self.output_tile_width - 1
                    slots[start : start + width] = row[input_start : input_start + width]
                plaintext = self.seal.Plaintext()
                self.encoder.encode(slots.tolist(), plaintext)
                ciphertext = self.seal.Ciphertext()
                self._encryptor.encrypt_symmetric(plaintext, ciphertext)
                ciphertexts.append(_save_seal(ciphertext))
            payloads.append(
                _pack(
                    "request",
                    input_width=self.input_width,
                    output_width=self.output_width,
                    streams=streams,
                    input_tiles=self.input_tiles,
                    ciphertexts=ciphertexts,
                )
            )
            offset += streams
        return payloads

    def decrypt_many(self, responses: Sequence[bytes], group_sizes: Sequence[int]) -> np.ndarray:
        if isinstance(responses, (bytes, bytearray)) or isinstance(group_sizes, (bytes, bytearray)):
            raise TiledBFVError("responses and group_sizes must be sequences")
        response_list = list(responses)
        sizes = [_require_int(value, "group size") for value in group_sizes]
        if len(response_list) != len(sizes) or not sizes:
            raise TiledBFVError("response and group-size counts differ")
        if any(size > self.capacity for size in sizes):
            raise TiledBFVError("group size exceeds ciphertext capacity")

        groups: list[np.ndarray] = []
        for payload, expected_streams in zip(response_list, sizes, strict=True):
            envelope = _unpack(payload, "response", _RESPONSE_KEYS)
            self._validate_shape(envelope, expected_streams, "output_tiles", self.output_tiles)
            ciphertexts = envelope["ciphertexts"]
            if not isinstance(ciphertexts, list) or len(ciphertexts) != self.output_tiles:
                raise TiledBFVError("response ciphertext count does not match output tiles")
            output = np.empty((expected_streams, self.output_width), dtype=np.int64)
            bases = self._bases(expected_streams)
            for tile_index, ciphertext_payload in enumerate(ciphertexts):
                if not isinstance(ciphertext_payload, bytes):
                    raise TiledBFVError("response ciphertext must be bytes")
                ciphertext = _load_with_context(
                    self.seal,
                    self.seal.Ciphertext,
                    self.context,
                    ciphertext_payload,
                    "response ciphertext",
                )
                plaintext = self.seal.Plaintext()
                try:
                    self._decryptor.decrypt(ciphertext, plaintext)
                    decoded = np.asarray(self.encoder.decode_int64(plaintext), dtype=np.int64)
                except Exception as exc:
                    raise TiledBFVError("response ciphertext cannot be decrypted") from exc
                output_start = tile_index * self.output_tile_width
                width = min(self.output_tile_width, self.output_width - output_start)
                for stream, base in enumerate(bases):
                    output[stream, output_start : output_start + width] = (
                        decoded[base : base + width] % self.plain_modulus
                    )
            groups.append(output)
        return np.concatenate(groups, axis=0)

    def _validate_shape(
        self,
        envelope: dict[str, Any],
        streams: int,
        tile_field: str,
        tile_count: int,
    ) -> None:
        if (
            type(envelope["input_width"]) is not int
            or envelope["input_width"] != self.input_width
            or type(envelope["output_width"]) is not int
            or envelope["output_width"] != self.output_width
            or type(envelope["streams"]) is not int
            or envelope["streams"] != streams
            or type(envelope[tile_field]) is not int
            or envelope[tile_field] != tile_count
        ):
            raise TiledBFVError("envelope dimensions do not match client context")


class TiledBFVServer:
    """Public evaluation context plus server-only signed W8 matrix."""

    def __init__(
        self,
        public_context: bytes,
        weight: np.ndarray | Sequence[Sequence[int]],
        *,
        threads: int = 1,
        tenseal_path: str | Path | None = None,
    ) -> None:
        threads = _require_int(threads, "threads")
        self.threads = threads
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        self.seal = import_tenseal(tenseal_path)
        envelope = _unpack(public_context, "context", _CONTEXT_KEYS)
        if (
            type(envelope["poly_modulus_degree"]) is not int
            or envelope["poly_modulus_degree"] != POLY_MODULUS_DEGREE
        ):
            raise TiledBFVError("context has unsupported polynomial modulus degree")
        self.plain_modulus = _validate_modulus(envelope["plain_modulus"])
        matrix = np.asarray(weight)
        if matrix.ndim != 2:
            raise TiledBFVError("weight must be rank two")
        self.output_width, self.input_width = matrix.shape
        expected_tiles = _tile_widths(self.input_width, self.output_width)
        self.input_tile_width, self.output_tile_width = expected_tiles
        self.segment_width = self.input_tile_width + self.output_tile_width - 1
        parameters_payload = envelope["parameters"]
        galois_payload = envelope["galois_keys"]
        if not isinstance(parameters_payload, bytes) or not isinstance(galois_payload, bytes):
            raise TiledBFVError("context key material must be bytes")
        try:
            parameters = _load_parameters(self.seal, parameters_payload)
            self.context = self.seal.SEALContext(parameters, True, self.seal.SEC_LEVEL_TYPE.TC128)
        except Exception as exc:
            raise TiledBFVError("invalid BFV parameters") from exc
        if not self.context.parameters_set():
            raise TiledBFVError(self.context.parameters_error_message())
        if (
            int(parameters.poly_modulus_degree()) != POLY_MODULUS_DEGREE
            or int(parameters.plain_modulus().value()) != self.plain_modulus
        ):
            raise TiledBFVError("serialized BFV parameters do not match context envelope")

        self.encoder = self.seal.BatchEncoder(self.context)
        self.slot_count = int(self.encoder.slot_count())
        self.row_size = self.slot_count // 2
        self.capacity = min(MAX_STREAMS, 2 * (self.row_size // self.segment_width))
        if self.segment_width > self.row_size // 2:
            raise TiledBFVError("context capacity does not match tile dimensions")
        self.input_tiles = math.ceil(self.input_width / self.input_tile_width)
        self.output_tiles = math.ceil(self.output_width / self.output_tile_width)
        self._per_row = self.row_size // self.segment_width
        self._bases = [
            (index // self._per_row) * self.row_size + (index % self._per_row) * self.segment_width
            for index in range(self.capacity)
        ]
        self._baby_count = min(_baby_steps(SEGMENT_TARGET), self.segment_width)
        self.galois_keys = _load_with_context(
            self.seal,
            self.seal.GaloisKeys,
            self.context,
            galois_payload,
            "Galois keys",
        )
        self.evaluator = self.seal.Evaluator(self.context)
        self._lock = threading.Lock()

        if matrix.ndim != 2 or matrix.shape != (self.output_width, self.input_width):
            raise TiledBFVError(f"weight must have shape ({self.output_width}, {self.input_width})")
        if matrix.dtype.kind not in "iu":
            raise TiledBFVError("weight must contain integers")
        if np.any(matrix < -128) or np.any(matrix > 127):
            raise TiledBFVError("weight values must fit signed W8 range [-128, 127]")
        self.weight = matrix.astype(np.int8, copy=True)

    @property
    def has_secret_key(self) -> bool:
        return False

    def evaluate_many(
        self,
        payloads: Sequence[bytes],
        *,
        cancel_event: threading.Event | None = None,
    ) -> list[bytes]:
        if isinstance(payloads, (bytes, bytearray)):
            raise TiledBFVError("payloads must be a sequence of request envelopes")
        requests = [self._parse_request(payload) for payload in payloads]
        if not requests:
            raise TiledBFVError("at least one request payload is required")
        with self._lock:
            return self._evaluate_locked(requests, cancel_event)

    def _parse_request(self, payload: bytes) -> dict[str, Any]:
        envelope = _unpack(payload, "request", _REQUEST_KEYS)
        streams = _require_int(envelope["streams"], "streams")
        if streams > self.capacity:
            raise TiledBFVError("request stream count exceeds ciphertext capacity")
        if (
            type(envelope["input_width"]) is not int
            or envelope["input_width"] != self.input_width
            or type(envelope["output_width"]) is not int
            or envelope["output_width"] != self.output_width
            or type(envelope["input_tiles"]) is not int
            or envelope["input_tiles"] != self.input_tiles
        ):
            raise TiledBFVError("request dimensions do not match server context")
        ciphertexts = envelope["ciphertexts"]
        if not isinstance(ciphertexts, list) or len(ciphertexts) != self.input_tiles:
            raise TiledBFVError("request ciphertext count does not match input tiles")
        if any(not isinstance(value, bytes) for value in ciphertexts):
            raise TiledBFVError("request ciphertext must be bytes")
        return envelope

    def _load_ciphertext(self, payload: bytes):
        return _load_with_context(
            self.seal,
            self.seal.Ciphertext,
            self.context,
            payload,
            "request ciphertext",
        )

    def _evaluate_locked(
        self,
        requests: list[dict[str, Any]],
        cancel_event: threading.Event | None,
    ) -> list[bytes]:
        outputs: list[list[bytes]] = [[] for _ in requests]
        for output_start in range(0, self.output_width, self.output_tile_width):
            self._check_cancelled(cancel_event)
            values = self._evaluate_output(
                requests, output_start, self.evaluator, self.encoder, cancel_event
            )
            for output, value in zip(outputs, values, strict=True):
                output.append(value)
        return self._pack_responses(requests, outputs)

    def _evaluate_output(
        self,
        requests: list[dict[str, Any]],
        output_start: int,
        evaluator: Any,
        encoder: Any,
        cancel_event: threading.Event | None,
    ) -> list[bytes]:
        output_width = min(self.output_tile_width, self.output_width - output_start)
        accumulators: list[Any | None] = [None] * len(requests)
        for input_tile_index, input_start in enumerate(
            range(0, self.input_width, self.input_tile_width)
        ):
            self._check_cancelled(cancel_event)
            input_width = min(self.input_tile_width, self.input_width - input_start)
            weight_tile = self.weight[
                output_start : output_start + output_width,
                input_start : input_start + input_width,
            ]
            if not np.any(weight_tile):
                continue
            for group_start in range(0, len(requests), _EVALUATION_GROUP_CHUNK):
                group_end = group_start + _EVALUATION_GROUP_CHUNK
                program_results = self._evaluate_program(
                    requests[group_start:group_end],
                    input_tile_index,
                    weight_tile,
                    evaluator,
                    encoder,
                    cancel_event,
                )
                for relative_index, result in enumerate(program_results):
                    index = group_start + relative_index
                    if accumulators[index] is None:
                        accumulators[index] = result
                    else:
                        evaluator.add_inplace(accumulators[index], result)

        zero_plaintext = None
        for index, accumulator in enumerate(accumulators):
            if accumulator is None:
                if zero_plaintext is None:
                    zero_plaintext = self.seal.Plaintext()
                    encoder.encode([0] * self.slot_count, zero_plaintext)
                source = self._load_ciphertext(requests[index]["ciphertexts"][0])
                accumulator = self.seal.Ciphertext()
                evaluator.multiply_plain(source, zero_plaintext, accumulator)
            accumulators[index] = accumulator
        return [_save_seal(accumulator) for accumulator in accumulators]

    @staticmethod
    def _check_cancelled(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise TiledBFVError("BFV evaluation cancelled")

    def _pack_responses(
        self, requests: list[dict[str, Any]], outputs: list[list[bytes]]
    ) -> list[bytes]:
        return [
            _pack(
                "response",
                input_width=self.input_width,
                output_width=self.output_width,
                streams=request["streams"],
                output_tiles=self.output_tiles,
                ciphertexts=ciphertexts,
            )
            for request, ciphertexts in zip(requests, outputs, strict=True)
        ]

    def _evaluate_program(
        self,
        requests: list[dict[str, Any]],
        input_tile_index: int,
        weight_tile: np.ndarray,
        evaluator: Any,
        encoder: Any,
        cancel_event: threading.Event | None,
    ) -> list[Any]:
        sources = [
            self._load_ciphertext(request["ciphertexts"][input_tile_index]) for request in requests
        ]
        baby_rotations: list[list[Any]] = []
        for source in sources:
            coefficient_rotations = [source]
            for _ in range(1, self._baby_count):
                rotated = self.seal.Ciphertext()
                evaluator.rotate_rows(coefficient_rotations[-1], 1, self.galois_keys, rotated)
                coefficient_rotations.append(rotated)
            ntt_rotations = []
            for rotated in coefficient_rotations:
                transformed = self.seal.Ciphertext()
                evaluator.transform_to_ntt(rotated, transformed)
                ntt_rotations.append(transformed)
            baby_rotations.append(ntt_rotations)
        output_width, input_width = weight_tile.shape
        offset = self.output_tile_width - 1
        first_diagonal = max(0, offset - (output_width - 1))
        last_diagonal = min(self.segment_width - 1, offset + input_width - 1)
        first_giant = first_diagonal // self._baby_count
        last_giant = last_diagonal // self._baby_count
        giant_inners: list[list[Any | None]] = [[None] * first_giant for _ in requests]

        output_indices = np.arange(output_width, dtype=np.int64)
        for giant_index in range(first_giant, last_giant + 1):
            self._check_cancelled(cancel_event)
            giant_shift = giant_index * self._baby_count
            inners: list[Any | None] = [None] * len(requests)
            for baby_index in range(self._baby_count):
                diagonal_index = giant_shift + baby_index
                if diagonal_index < first_diagonal:
                    continue
                if diagonal_index > last_diagonal:
                    break
                input_indices = output_indices + diagonal_index - offset
                valid = (input_indices >= 0) & (input_indices < input_width)
                selected_outputs = output_indices[valid]
                selected_inputs = input_indices[valid]
                values = weight_tile[selected_outputs, selected_inputs].astype(np.int64, copy=False)
                nonzero = values != 0
                if not np.any(nonzero):
                    continue
                selected_outputs = selected_outputs[nonzero]
                values = values[nonzero]
                diagonal = np.zeros(self.slot_count, dtype=np.int64)
                for base in self._bases:
                    diagonal[base + selected_outputs] = values
                if giant_shift:
                    diagonal = np.concatenate(
                        (
                            np.roll(diagonal[: self.row_size], giant_shift),
                            np.roll(diagonal[self.row_size :], giant_shift),
                        )
                    )
                plaintext = self.seal.Plaintext()
                encoder.encode(diagonal.tolist(), plaintext)
                ntt_plaintext = self.seal.Plaintext()
                evaluator.transform_to_ntt(
                    plaintext, self.context.first_parms_id(), ntt_plaintext
                )
                for group_index in range(len(requests)):
                    product = self.seal.Ciphertext()
                    evaluator.multiply_plain(
                        baby_rotations[group_index][baby_index], ntt_plaintext, product
                    )
                    if inners[group_index] is None:
                        inners[group_index] = product
                    else:
                        evaluator.add_inplace(inners[group_index], product)

            for group_index, inner in enumerate(inners):
                if inner is not None:
                    evaluator.transform_from_ntt_inplace(inner)
                giant_inners[group_index].append(inner)

        results: list[Any | None] = []
        for inners in giant_inners:
            result = inners[-1]
            for inner in reversed(inners[:-1]):
                if result is not None:
                    evaluator.rotate_rows_inplace(result, self._baby_count, self.galois_keys)
                if inner is not None:
                    if result is None:
                        result = inner
                    else:
                        evaluator.add_inplace(result, inner)
            results.append(result)

        if any(result is None for result in results):
            raise RuntimeError("nonzero matrix tile produced no BFV terms")
        return [result for result in results if result is not None]
