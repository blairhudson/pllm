"""Bounded two-worker offset comparator for compiled public decoders.

Each worker evaluates one fresh modular input share with the existing native
stage kernel. References may run in-process or over co-located loopback HTTP;
their body counts and stage times are not full wire or independent-operator
measurements.
"""

from __future__ import annotations

import json
import secrets
import threading
import contextvars
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from math import prod
from typing import Callable
from urllib.parse import quote

import httpx
import numpy as np

from pllm.configuration import Pipeline
from pllm.modeling import DecoderContinuationSchedule
from pllm.roles import two_online_reference_graph
from pllm.runtime.model_binding import CompiledRuntimeModel
from pllm.runtime.stage_protocol import MaskedStageRequest, MaskedStageResponse
from pllm.runtime.transformer_client import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.transformer_engine import MaskedTransformerEngine

_MAX_TENSOR_ELEMENTS = 4_000_000


class OffsetReferenceError(ValueError):
    """A comparator role, stage, or response differs from its compiled contract."""


@dataclass(frozen=True, slots=True)
class OffsetReferenceCosts:
    stages: int
    rows: int
    client_to_worker_a_bytes: int
    worker_a_to_client_bytes: int
    client_to_worker_b_bytes: int
    worker_b_to_client_bytes: int
    worker_a_stage_ns: int
    worker_b_stage_ns: int
    worker_a_integer_macs: int
    worker_b_integer_macs: int

    @property
    def total_stage_body_bytes(self) -> int:
        return (
            self.client_to_worker_a_bytes
            + self.worker_a_to_client_bytes
            + self.client_to_worker_b_bytes
            + self.worker_b_to_client_bytes
        )

    @property
    def total_integer_macs(self) -> int:
        return self.worker_a_integer_macs + self.worker_b_integer_macs


class _TwoOnlineShareEvaluator:
    """Client-side additive split, independent of how the two bound workers run."""

    def __init__(
        self, compiled: CompiledRuntimeModel, *, model_id: str,
        exchange_a: Callable[[str, list[bytes]], list[bytes]],
        exchange_b: Callable[[str, list[bytes]], list[bytes]],
        session_a: str | None = None,
        session_b: str | None = None,
    ) -> None:
        if (
            type(compiled) is not CompiledRuntimeModel
            or exchange_a is exchange_b
            or not callable(exchange_a)
            or not callable(exchange_b)
        ):
            raise OffsetReferenceError("two distinct offset worker exchanges are required")
        sessions = (
            session_a if session_a is not None else secrets.token_hex(16),
            session_b if session_b is not None else secrets.token_hex(16),
        )
        if any(
            type(session) is not str or len(session) != 32
            or any(char not in "0123456789abcdef" for char in session)
            for session in sessions
        ) or sessions[0] == sessions[1]:
            raise OffsetReferenceError("offset worker sessions must be distinct opaque IDs")
        compiled.validate()
        manifest = compiled._bundle.manifest
        fingerprint = manifest["metadata"]["body_fingerprint"]
        bindings = {stage.stage_id: stage for stage in compiled._stages}
        self._compiled = compiled
        self._model_id = model_id
        self._fingerprint = fingerprint
        self._bindings = bindings
        self._exchange_a = exchange_a
        self._exchange_b = exchange_b
        self._session_a, self._session_b = sessions
        self._costs = OffsetReferenceCosts(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self._maximum_rows = int(compiled._plan.to_dict()["prefill"]["query_sequence"])
        composition = Pipeline.from_spec(json.loads(compiled._canonical_composition))
        self._composition_digest = composition.digest()
        self._input_encoding = composition.components["linear"].params.get("input_encoding", "raw")
        self._output_encoding = composition.components["linear"].params.get("output_encoding", "raw")
        if self._output_encoding == "row_residues":
            from .offset_codec import compiled_row_layout
            compiled_row_layout(compiled)

    @property
    def costs(self) -> OffsetReferenceCosts:
        return self._costs

    def _begin_seed(self, stage_id: str, payload: bytes) -> Future[list[bytes]] | None:
        return None

    def __call__(self, stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = self._compiled._bundle.stages.get(stage_id)
        binding = self._bindings.get(stage_id)
        if (
            stage is None
            or binding is None
            or binding.client_weight_layout is not None
            or stage.seeded_profile is None
        ):
            raise OffsetReferenceError("stage is not a compiled remote offset stage")
        value = np.asarray(activation)
        if (
            value.dtype != np.float32
            or value.ndim not in (2, 3)
            or value.shape[-1] != stage.in_features
        ):
            raise OffsetReferenceError("offset stage input is malformed")
        rows = prod(value.shape[:-1])
        if (
            rows < 1
            or rows > self._maximum_rows
            or rows * max(stage.in_features, stage.out_features) > _MAX_TENSOR_ELEMENTS
        ):
            raise OffsetReferenceError("offset stage exceeds the bounded tensor policy")
        if not np.all(np.isfinite(value)):
            raise OffsetReferenceError("offset stage input is not finite")
        quantized = quantize_activation_per_row(value, bits=stage.activation_bits)
        profile = stage.seeded_profile
        if profile.ring not in {"u16", "u24", "u32"}:
            raise OffsetReferenceError("offset stages require an exact wrapping ring")
        modulus = profile.modulus
        mask = None
        right_packet = None
        pending_seed = None
        if self._input_encoding == "seeded":
            from pllm import _native
            from .offset_codec import context, pack_seed, seed_header

            seed = secrets.token_bytes(32)
            right_ticket = secrets.token_hex(16)
            header = seed_header(model=self._model_id, stage=stage_id,
                body=self._fingerprint, weight=stage.weight_digest, plan=self._compiled._plan.digest,
                composition=self._composition_digest, session=self._session_b, ticket=right_ticket,
                rows=rows, columns=stage.in_features, bits=profile.wire_bits)
            right_packet = (right_ticket, pack_seed(header, seed))
            # Public schedule and both sessions are already admitted. The seed
            # goes only to worker B; A's fresh complement is still built natively.
            pending_seed = self._begin_seed(stage_id, right_packet[1])
            left = np.frombuffer(bytearray(_native.offset_seeded_share(seed, context(header),
                rows * stage.in_features, profile.wire_bits, quantized.values.tobytes())), dtype="<u4").reshape(rows, stage.in_features)
        else:
            clear = quantized.values.reshape(rows, stage.in_features).astype(np.int64) % modulus
            raw_mask = bytearray(secrets.token_bytes(rows * stage.in_features * 4))
            mask = np.frombuffer(raw_mask, dtype="<u4").reshape(rows, stage.in_features)
            left = np.asarray((clear - mask.astype(np.int64)) % modulus, dtype=np.uint32)
        try:
            # Every uint32 value is already a valid residue modulo 2**32;
            # NumPy cannot cast that modulus itself to a uint32 scalar.
            if mask is not None and modulus != 1 << 32:
                mask %= modulus
            def request(share: np.ndarray, session_id: str) -> tuple[str, bytes]:
                ticket = secrets.token_hex(16)
                return ticket, MaskedStageRequest(
                    model=self._model_id,
                    stage_id=stage_id,
                    correlation_id=ticket,
                    masked_input=share,
                    activation_scales=np.ones(1, dtype=np.float32),
                    modulus=modulus,
                    wire_bits=profile.wire_bits,
                    ring=profile.ring,
                    body_fingerprint=self._fingerprint,
                    weight_digest=stage.weight_digest,
                    weight_bits=stage.weight_bits,
                    activation_bits=stage.activation_bits,
                    session_id=session_id,
                    out_features=stage.out_features,
                    signed_output_bound=profile.signed_output_bound,
                ).pack()

            left_ticket, left_request = request(left, self._session_a)
            if mask is not None:
                right_packet = request(mask, self._session_b)
            if right_packet is None:
                raise OffsetReferenceError("offset share encoding did not produce a request")
            right_ticket, right_request = right_packet
            left_result = self._exchange_a(stage_id, [left_request])
            right_result = (pending_seed.result() if pending_seed is not None
                            else self._exchange_b(stage_id, [right_request]))
            if (
                type(left_result) is not list
                or type(right_result) is not list
                or len(left_result) != 1
                or len(right_result) != 1
                or type(left_result[0]) is not bytes
                or type(right_result[0]) is not bytes
            ):
                raise OffsetReferenceError("offset worker returned an invalid response batch")
            if self._output_encoding == "row_residues":
                from pllm import _native
                from .offset_codec import unpack_row_response

                widths = stage.output_residue_bits
                a, first_ns = unpack_row_response(left_result[0], ticket=left_ticket,
                    stage=stage_id, widths=widths, rows=rows, bits=profile.wire_bits)
                b, second_ns = unpack_row_response(right_result[0], ticket=right_ticket,
                    stage=stage_id, widths=widths, rows=rows, bits=profile.wire_bits)
                centered = np.frombuffer(_native.offset_reconstruct_rows(a, b, widths, rows), dtype="<i8").reshape(rows, stage.out_features)
            else:
                first = MaskedStageResponse.unpack(left_result[0])
                second = MaskedStageResponse.unpack(right_result[0])
                for result, ticket in ((first, left_ticket), (second, right_ticket)):
                    if (
                        result.correlation_id != ticket
                        or result.stage_id != stage_id
                        or result.modulus != modulus
                        or result.wire_bits != profile.wire_bits
                        or result.ring != profile.ring
                        or result.masked_output.shape != (rows, stage.out_features)
                        or result.server_ns < 0
                    ):
                        raise OffsetReferenceError("offset worker result differs from its stage contract")
                combined = (
                    first.masked_output.astype(np.int64)
                    + second.masked_output.astype(np.int64)
                ) % modulus
                centered = np.where(combined >= modulus // 2, combined - modulus, combined)
                first_ns, second_ns = first.server_ns, second.server_ns
            if np.any(np.abs(centered) > profile.signed_output_bound):
                raise OffsetReferenceError("offset result exceeds the declared signed stage bound")
            output = dequantize_matmul(
                centered, quantized.scales, stage.weight_scales,
                output_shape=quantized.original_shape[:-1] + (stage.out_features,),
            )
            if stage.bias is not None:
                output = output + stage.bias
            previous = self._costs
            self._costs = OffsetReferenceCosts(
                previous.stages + 1,
                previous.rows + rows,
                previous.client_to_worker_a_bytes + len(left_request),
                previous.worker_a_to_client_bytes + len(left_result[0]),
                previous.client_to_worker_b_bytes + len(right_request),
                previous.worker_b_to_client_bytes + len(right_result[0]),
                previous.worker_a_stage_ns + first_ns,
                previous.worker_b_stage_ns + second_ns,
                previous.worker_a_integer_macs + rows * stage.in_features * stage.out_features,
                previous.worker_b_integer_macs + rows * stage.in_features * stage.out_features,
            )
            return np.ascontiguousarray(output, dtype=np.float32)
        finally:
            left.fill(0)
            if mask is not None:
                mask.fill(0)


class TwoOnlineOffsetReference(_TwoOnlineShareEvaluator):
    """A non-deployable reference with two separately loaded provider engines."""

    def __init__(
        self, compiled: CompiledRuntimeModel, worker_a: MaskedTransformerEngine,
        worker_b: MaskedTransformerEngine,
        *, model_id: str,
        exchange_a: Callable[[str, list[bytes]], list[bytes]],
        exchange_b: Callable[[str, list[bytes]], list[bytes]],
    ) -> None:
        params = Pipeline.from_spec(json.loads(compiled._canonical_composition)).components["linear"].params
        if params.get("input_encoding", "raw") != "raw" or params.get("output_encoding", "raw") != "raw":
            raise OffsetReferenceError("encoded offset execution requires authenticated worker transport")
        if (
            type(worker_a) is not MaskedTransformerEngine
            or type(worker_b) is not MaskedTransformerEngine
            or worker_a is worker_b
        ):
            raise OffsetReferenceError("two distinct loaded offset workers are required")
        super().__init__(
            compiled, model_id=model_id, exchange_a=exchange_a, exchange_b=exchange_b,
        )
        for worker in (worker_a, worker_b):
            model = worker._model(model_id)
            if model.manifest.metadata.get("body_fingerprint") != self._fingerprint:
                raise OffsetReferenceError("offset worker body differs from the compiled decoder")
            for stage_id, binding in self._bindings.items():
                if binding.client_weight_layout is not None:
                    continue
                stage = compiled._bundle.stages[stage_id]
                runtime = model.stages.get(stage_id)
                if runtime is None:
                    raise OffsetReferenceError("offset worker stage is absent")
                if (
                    runtime.spec.in_features != binding.in_features
                    or runtime.spec.out_features != binding.out_features
                    or runtime.spec.weight_bits != binding.weight_bits
                    or runtime.spec.activation_bits != binding.activation_bits
                ):
                    raise OffsetReferenceError("offset worker stage geometry differs")
                if runtime.weight_digest != binding.weight_digest:
                    raise OffsetReferenceError("offset worker stage weight differs")
                if runtime.seeded_profile != stage.seeded_profile:
                    raise OffsetReferenceError("offset worker stage ring differs")
                if not np.array_equal(runtime.weight.scales, stage.weight_scales):
                    raise OffsetReferenceError("offset worker stage scales differ")
                if not np.array_equal(runtime.bias, stage.bias):
                    raise OffsetReferenceError("offset worker stage bias differs")


class TwoOnlineOffsetTransport(_TwoOnlineShareEvaluator):
    """Bounded research session with two authenticated HTTP worker exchanges.

    Its caller owns both HTTP clients. Compiled Experiments can select this
    transport, while local co-located workers do not meet the independent-
    operator privacy condition; whole-response resource gates remain open.
    """

    def __init__(
        self, compiled: CompiledRuntimeModel, *, model_id: str,
        worker_a: httpx.Client, worker_b: httpx.Client,
        api_key_a: str, api_key_b: str,
        continuation: DecoderContinuationSchedule | None = None,
    ) -> None:
        if (
            type(worker_a) is not httpx.Client or type(worker_b) is not httpx.Client
            or worker_a is worker_b or type(api_key_a) is not str
            or type(api_key_b) is not str or len(api_key_a) < 16
            or len(api_key_b) < 16 or api_key_a == api_key_b
        ):
            raise OffsetReferenceError("offset transport requires distinct authenticated workers")
        compiled.validate()
        self.continuation = None
        if continuation is not None:
            if type(continuation) is not DecoderContinuationSchedule:
                raise OffsetReferenceError("offset continuation requires a native contract")
            expected_contract = compiled._plan.continuation_schedule(
                Pipeline.from_spec(json.loads(compiled._canonical_composition)))
            if continuation.digest != expected_contract.digest:
                raise OffsetReferenceError("offset continuation differs from compiled decoder")
            self.continuation = continuation.handshake_spec()
        metadata = compiled._bundle.manifest["metadata"]
        plan = compiled._plan.to_dict()
        max_input = plan["prefill"]["query_sequence"]
        max_new = plan["decode"]["maximum_key_sequence"] - max_input + 1
        graph_digest = two_online_reference_graph().digest()
        clients = (worker_a, worker_b)
        keys = (api_key_a, api_key_b)
        session_ids: list[str] = []
        self._closed = True
        self._seed_executor = None
        self._call_lock = threading.Lock()
        self._finish_lock = threading.Lock()
        self._http_bodies: dict[str, dict[str, int]] = {
            role: {
                "setup_upload_bytes": 0, "setup_download_bytes": 0,
                "online_upload_bytes": 0, "online_download_bytes": 0,
                "teardown_upload_bytes": 0, "teardown_download_bytes": 0,
            }
            for role in ("worker_a", "worker_b")
        }

        def post(
            index: int, path: str, *, phase: str, limit: int = 4_096, **kwargs,
        ) -> tuple[int, bytes]:
            chunks: list[bytes] = []
            size = 0
            with clients[index].stream(
                "POST", path, headers={"authorization": f"Bearer {keys[index]}"}, **kwargs,
            ) as response:
                request_size = len(response.request.content)
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > limit:
                        raise OffsetReferenceError("offset worker response exceeds its body limit")
                    chunks.append(chunk)
                values = self._http_bodies[("worker_a", "worker_b")[index]]
                values[f"{phase}_upload_bytes"] += request_size
                values[f"{phase}_download_bytes"] += size
                return response.status_code, b"".join(chunks)

        try:
            composition = Pipeline.from_spec(json.loads(compiled._canonical_composition))
            row_layout = None
            if composition.components["linear"].params.get("output_encoding", "raw") == "row_residues":
                from .offset_codec import compiled_row_layout
                row_layout = compiled_row_layout(compiled)
            for index, role_id in enumerate(("worker_a", "worker_b")):
                request = {
                    "schema": "pllm.offset_worker_session.v1", "model": model_id,
                    "role": role_id, "topology_digest": graph_digest,
                    "decoder_plan": compiled._plan.digest,
                    "max_input_tokens": max_input, "max_new_tokens": max_new,
                    "body_fingerprint": metadata["body_fingerprint"],
                    "stage_commitment": metadata["seeded_stage_commitment"],
                    "runtime_config_digest": metadata["runtime_config_digest"],
                    "composition_digest": Pipeline.from_spec(
                        json.loads(compiled._canonical_composition)
                    ).digest(),
                }
                if row_layout is not None:
                    request["residue_layout_digest"] = row_layout
                if self.continuation is not None:
                    request["decoder_continuation"] = {
                        "composition": composition.to_spec(), "contract": self.continuation,
                    }
                status, payload = post(index, "/v1/offset-reference/sessions", phase="setup",
                                       json=request)
                if status != 200:
                    raise OffsetReferenceError("offset worker declined the bound decoder session")

                def distinct_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
                    value: dict[str, object] = {}
                    for field, item in pairs:
                        if field in value:
                            raise OffsetReferenceError("offset worker admission has duplicate fields")
                        value[field] = item
                    return value

                value = json.loads(payload, object_pairs_hook=distinct_fields)
                expected = {
                    "schema": request["schema"], "role": role_id,
                    "topology_digest": graph_digest, "decoder_plan": compiled._plan.digest,
                    "body_fingerprint": metadata["body_fingerprint"],
                    "stage_commitment": metadata["seeded_stage_commitment"],
                }
                if row_layout is not None:
                    expected["residue_layout_digest"] = row_layout
                if self.continuation is not None:
                    expected["decoder_continuation"] = self.continuation
                # Remember a valid issued ID before checking all acknowledgement
                # fields, so a refused/malformed second acknowledgement burns both.
                valid_id = (type(value) is dict and type(value.get("id")) is str
                            and len(value["id"]) == 32
                            and all(char in "0123456789abcdef" for char in value["id"]))
                if valid_id:
                    session_ids.append(value["id"])
                if (
                    type(value) is not dict or set(value) != set(expected) | {"id"}
                    or any(value.get(field) != item for field, item in expected.items())
                    or not valid_id
                ):
                    raise OffsetReferenceError("offset worker admission response differs")
            if session_ids[0] == session_ids[1]:
                raise OffsetReferenceError("offset workers issued an identical session ID")

            def exchange(index: int) -> Callable[[str, list[bytes]], list[bytes]]:
                def invoke(stage_id: str, payloads: list[bytes]) -> list[bytes]:
                    if type(payloads) is not list or len(payloads) != 1 or type(payloads[0]) is not bytes:
                        raise OffsetReferenceError("offset stage requires one packed share")
                    stage_path = quote(stage_id, safe="")
                    path = f"/v1/offset-reference/sessions/{session_ids[index]}/stages/{stage_path}"
                    status, response_body = post(
                        index, path, phase="online", content=payloads[0],
                        limit=16 * 1024 * 1024 + 8_192,
                    )
                    if status != 200:
                        raise OffsetReferenceError("offset worker refused its committed stage")
                    return [response_body]
                return invoke

            super().__init__(
                compiled, model_id=model_id, exchange_a=exchange(0), exchange_b=exchange(1),
                session_a=session_ids[0], session_b=session_ids[1],
            )
            self._clients = clients
            self._keys = keys
            self._sessions = (session_ids[0], session_ids[1])
            if composition.components["linear"].params.get("dispatch") == "seed_first":
                self._seed_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pllm-offset-seed")
            self._closed = False
        except BaseException:
            for index, session_id in enumerate(session_ids):
                try:
                    post(index, f"/v1/offset-reference/sessions/{session_id}/cancel",
                         phase="teardown")
                except (httpx.HTTPError, OffsetReferenceError):
                    pass
            raise

    @property
    def http_body_costs(self) -> dict[str, dict[str, int]]:
        """Actual HTTP application bodies by role/phase; excludes headers/TLS."""
        return {role: dict(values) for role, values in self._http_bodies.items()}

    def __call__(self, stage_id: str, activation: np.ndarray) -> np.ndarray:
        if self._closed:
            raise OffsetReferenceError("offset worker session is already terminal")
        if not self._call_lock.acquire(blocking=False):
            raise OffsetReferenceError("offset session already has an active stage")
        try:
            result = super().__call__(stage_id, activation)
            if self._closed:
                raise OffsetReferenceError("offset session cancelled during its stage")
            return result
        except BaseException:
            self.abort()
            raise
        finally:
            self._call_lock.release()

    def _begin_seed(self, stage_id: str, payload: bytes):
        if self._seed_executor is None:
            return None
        if self._closed:
            raise OffsetReferenceError("offset session closed before seed dispatch")
        return self._seed_executor.submit(contextvars.copy_context().run, self._exchange_b, stage_id, [payload])

    def _finish(self, action: str) -> None:
        with self._finish_lock:
            if self._closed:
                return
            self._closed = True
        failures = False
        for index, (client, key, session_id) in enumerate(zip(
            self._clients, self._keys, self._sessions, strict=True,
        )):
            try:
                with client.stream(
                    "POST", f"/v1/offset-reference/sessions/{session_id}/{action}",
                    headers={"authorization": f"Bearer {key}"},
                ) as response:
                    failures |= response.status_code != 200
                    response_size = 0
                    for chunk in response.iter_bytes():
                        response_size += len(chunk)
                        if response_size > 4_096:
                            failures = True
                            break
                    if response_size <= 4_096:
                        values = self._http_bodies[("worker_a", "worker_b")[index]]
                        values["teardown_upload_bytes"] += len(response.request.content)
                        values["teardown_download_bytes"] += response_size
            except httpx.HTTPError:
                failures = True
        if self._seed_executor is not None:
            self._seed_executor.shutdown(wait=True, cancel_futures=True)
        if failures and action == "complete":
            raise OffsetReferenceError("offset worker completion failed")

    def complete(self) -> None:
        """Finish after the compiled decoder completes successfully."""
        self._finish("complete")

    def abort(self) -> None:
        """Burn both sessions on cancellation, transport error, or validation failure."""
        self._finish("cancel")

    def __enter__(self) -> TwoOnlineOffsetTransport:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.abort()
