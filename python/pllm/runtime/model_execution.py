from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from pllm.runtime.model_binding import CompiledRuntimeModel, RuntimeBindingError
from pllm.runtime.transformer_client import MaskedTransformerClientRuntime


class RuntimeExecutionError(RuntimeError):
    pass


class CompiledRuntimeSession:
    __slots__ = (
        "_compiled",
        "_generated",
        "_last_logits",
        "_layers",
        "_lock",
        "_logrow",
        "_logrow_operations",
        "_logrow_phase",
        "_logrow_step",
        "_research_execution_digest",
        "_max_input",
        "_max_new",
        "_pending_token",
        "_position",
        "_runtime",
        "_status",
        "_vocab",
    )

    def __init__(self) -> None:
        raise RuntimeExecutionError(
            "CompiledRuntimeSession must be created by CompiledRuntimeModel.session"
        )

    @classmethod
    def _create(
        cls,
        compiled: CompiledRuntimeModel,
        remote: Callable[[str, np.ndarray], np.ndarray],
        *,
        research_logrow_material: object | None = None,
    ) -> CompiledRuntimeSession:
        if type(compiled) is not CompiledRuntimeModel:
            raise TypeError("compiled must be a CompiledRuntimeModel")
        if not callable(remote):
            raise TypeError("remote must be callable")
        compiled.validate()
        document = compiled._plan.to_dict()
        prefill = document["prefill"]
        decode = document["decode"]
        max_input = int(prefill["query_sequence"])
        max_new = int(decode["maximum_key_sequence"]) - max_input + 1
        if (
            int(prefill["batch"]) != 1
            or int(decode["batch"]) != 1
            or int(decode["query_sequence"]) != 1
            or max_input <= 0
            or max_new <= 0
        ):
            raise RuntimeExecutionError(
                "compiled runtime session requires a batch-one decoder plan"
            )
        bundle = compiled._bundle

        logrow = None
        operations: dict[str, dict[int, str]] = {}
        research_digest = None
        logrow_claimed = False
        if research_logrow_material is not None:
            from pllm import _native

            if type(research_logrow_material) is not _native.LogRowQ7SessionReference:
                raise TypeError("research LogRow material must be a native session handle")
            logrow = research_logrow_material
            try:
                estimate = json.loads(logrow.estimate())
                if (
                    estimate.get("schema_version") != "pllm.logrow_q7_session_estimate.v1"
                    or estimate.get("plan_digest") != compiled.model_plan_digest
                    or estimate.get("max_decode_steps") != max_new - 1
                    or len(logrow.issuance_digest) != 64
                ):
                    raise RuntimeExecutionError(
                        "research LogRow material differs from decoder plan"
                    )
                for phase, graph, width in (
                    ("prefill", prefill, max_input),
                    ("decode", decode, 1),
                ):
                    by_layer: dict[int, str] = {}
                    for operation in graph["operations"]:
                        if operation["operator"] != "silu":
                            continue
                        layer = operation["layer"]
                        if (
                            type(layer) is not int
                            or layer in by_layer
                            or operation["output_shape"]
                            != [1, width, bundle.cfg["intermediate_size"]]
                        ):
                            raise RuntimeExecutionError(
                                "research LogRow semantic SiLU shape differs"
                            )
                        by_layer[layer] = operation["id"]
                    if set(by_layer) != set(range(int(bundle.cfg["num_hidden_layers"]))):
                        raise RuntimeExecutionError("research LogRow misses a semantic SiLU layer")
                    operations[phase] = by_layer
                logrow.bind_decoder(compiled.model_plan_digest, max_new - 1)
                logrow_claimed = True
                research_digest = hashlib.sha256(
                    b"pllm.research.logrow_q7.local_execution.v1\0"
                    + compiled.digest.encode("ascii")
                    + logrow.issuance_digest.encode("ascii")
                    + estimate["estimate_digest"].encode("ascii")
                ).hexdigest()
            except Exception:
                if logrow_claimed:
                    logrow.abort()
                raise

        def bound_remote(stage_id: str, activation: np.ndarray) -> np.ndarray:
            stage = bundle.stages.get(stage_id)
            if stage is None or stage.id != stage_id:
                raise RuntimeExecutionError(f"remote stage {stage_id!r} is not bound")
            value = np.asarray(activation)
            if (
                value.dtype != np.float32
                or value.ndim < 1
                or value.shape[-1] != stage.in_features
                or not np.all(np.isfinite(value))
            ):
                raise RuntimeExecutionError(f"remote stage {stage_id!r} input is malformed")
            output = np.asarray(remote(stage_id, value))
            expected = value.shape[:-1] + (stage.out_features,)
            if (
                output.dtype != np.float32
                or output.shape != expected
                or not np.all(np.isfinite(output))
            ):
                raise RuntimeExecutionError(f"remote stage {stage_id!r} output is malformed")
            return np.ascontiguousarray(output, dtype=np.float32)

        self = object.__new__(cls)
        self._compiled = compiled
        self._logrow = logrow
        self._logrow_operations = operations
        self._logrow_phase = None
        self._logrow_step = 0
        self._research_execution_digest = research_digest
        try:
            self._runtime = (
                compiled.runtime(bound_remote)
                if logrow is None
                else compiled._runtime_with_nonlinear(bound_remote, self._evaluate_logrow)
            )
        except Exception:
            if logrow is not None:
                logrow.abort()
            raise
        self._max_input = max_input
        self._max_new = max_new
        self._vocab = int(compiled._bundle.cfg["vocab_size"])
        self._layers = int(compiled._bundle.cfg["num_hidden_layers"])
        self._lock = threading.RLock()
        self._status = "new"
        self._position = 0
        self._generated = 0
        self._pending_token: int | None = None
        self._last_logits: np.ndarray | None = None
        return self

    @property
    def complete(self) -> bool:
        return self._logrow is None

    @property
    def completeness_scope(self) -> str:
        return (
            "whole_decoder_runtime" if self._logrow is None else "bounded_local_research_reference"
        )

    @property
    def binding_digest(self) -> str:
        return self._research_execution_digest or self._compiled.digest

    @property
    def nonlinear_method(self) -> str | None:
        return None if self._logrow is None else "pllm/logrow-q7-local-reference/v1"

    def _evaluate_logrow(self, layer: int, value: np.ndarray) -> np.ndarray:
        if self._logrow is None or self._logrow_phase not in self._logrow_operations:
            raise RuntimeExecutionError("research LogRow phase is not bound")
        operation = self._logrow_operations[self._logrow_phase].get(layer)
        if operation is None or value.dtype != np.float32 or value.ndim != 2:
            self._logrow.abort()
            raise RuntimeExecutionError("research LogRow activation is not a semantic SiLU tensor")
        try:
            output = self._logrow.evaluate_float32(
                self._logrow_phase,
                self._logrow_step,
                operation,
                np.ascontiguousarray(value, dtype="<f4").tobytes(),
            )
            return np.frombuffer(output, dtype="<f4").reshape(value.shape).copy()
        except Exception as exc:
            self._logrow.abort()
            raise RuntimeExecutionError("research LogRow nonlinear execution failed") from exc

    @property
    def runtime_schedule_digest(self) -> str:
        return self._compiled.runtime_schedule_digest

    @property
    def status(self) -> str:
        return self._status

    @property
    def position(self) -> int:
        return self._position

    @property
    def generated_tokens(self) -> int:
        return self._generated

    @property
    def remaining_tokens(self) -> int:
        return self._max_new - self._generated

    @property
    def logits(self) -> np.ndarray:
        with self._lock:
            if self._last_logits is None:
                raise RuntimeExecutionError("session has no available logits")
            output = self._last_logits.copy()
            output.flags.writeable = False
            return output

    def _validate_token_ids(self, token_ids: Sequence[int] | np.ndarray) -> list[int]:
        value = np.asarray(token_ids)
        if value.ndim != 1 or value.dtype.kind not in {"i", "u"} or value.dtype == np.bool_:
            raise RuntimeExecutionError("token ids must be a one-dimensional integer sequence")
        if value.size == 0 or value.size > self._max_input:
            raise RuntimeExecutionError(
                f"prefill requires between 1 and {self._max_input} token ids"
            )
        if np.any(value < 0) or np.any(value >= self._vocab):
            raise RuntimeExecutionError("token id is outside the bound vocabulary")
        return [int(item) for item in value]

    def _validated_logits(self, value: Any) -> np.ndarray:
        logits = np.asarray(value)
        if logits.dtype != np.float32 or logits.shape != (self._vocab,):
            raise RuntimeExecutionError("runtime logits do not match the bound vocabulary")
        if not np.all(np.isfinite(logits)):
            raise RuntimeExecutionError("runtime logits must be finite")
        return np.ascontiguousarray(logits, dtype=np.float32).copy()

    def _validate_runtime_state(self, expected_position: int) -> None:
        if self._runtime.position != expected_position or len(self._runtime.caches) != self._layers:
            raise RuntimeExecutionError("runtime state does not match the bound decoder plan")
        if any(cache.length != expected_position for cache in self._runtime.caches):
            raise RuntimeExecutionError("runtime KV state lengths differ from the bound position")

    def _zero_runtime(self) -> None:
        if self._logrow is not None:
            self._logrow.abort()
        if self._last_logits is not None:
            self._last_logits.fill(0)
            self._last_logits = None
        for cache in self._runtime.caches:
            if cache.key is not None:
                cache.key.fill(0)
            if cache.value is not None:
                cache.value.fill(0)
        for key, value in self._runtime.shared_kv.values():
            key.fill(0)
            value.fill(0)
        self._runtime.reset()
        self._pending_token = None

    def _poison(self) -> None:
        self._zero_runtime()
        self._status = "poisoned"

    def prefill_ids(self, token_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        with self._lock:
            if self._status != "new":
                raise RuntimeExecutionError("session prefill may execute only once")
            try:
                validated = self._validate_token_ids(token_ids)
            except Exception:
                if self._logrow is not None:
                    self._logrow.abort()
                raise
            if self._logrow is not None and len(validated) != self._max_input:
                self._logrow.abort()
                raise RuntimeExecutionError(
                    "research LogRow prefill needs the exact planned token count"
                )
            self._status = "poisoned"
            try:
                self._compiled.validate()
                if self._logrow is not None:
                    self._logrow_phase = "prefill"
                    self._logrow_step = 0
                returned, logits, caches = self._runtime.prepare_ids(validated)
                if returned != validated or caches is not self._runtime.caches:
                    raise RuntimeExecutionError("runtime prefill returned unbound state")
                self._validate_runtime_state(len(validated))
                self._last_logits = self._validated_logits(logits)
            except Exception as exc:
                self._poison()
                if isinstance(exc, RuntimeExecutionError):
                    raise
                raise RuntimeExecutionError("compiled runtime prefill failed") from exc
            self._position = len(validated)
            self._generated = 0
            self._pending_token = None
            self._status = "ready"
            return self.logits

    def select_next(self) -> int:
        with self._lock:
            if self._status != "ready" or self._last_logits is None:
                raise RuntimeExecutionError("session is not ready for greedy token selection")
            token = int(np.argmax(self._last_logits))
            self._generated += 1
            if self._generated == self._max_new:
                self._zero_runtime()
                self._status = "exhausted"
            else:
                self._pending_token = token
                self._status = "selected"
            return token

    def decode_selected(self) -> np.ndarray:
        with self._lock:
            if self._status != "selected" or self._pending_token is None:
                raise RuntimeExecutionError("session has no selected token to decode")
            token = self._pending_token
            expected_position = self._position + 1
            self._status = "poisoned"
            try:
                self._compiled.validate()
                if self._logrow is not None:
                    self._logrow_phase = "decode"
                    self._logrow_step = self._generated - 1
                logits, caches = self._runtime.decode_step(
                    token, self._runtime.caches, self._position
                )
                if caches is not self._runtime.caches:
                    raise RuntimeExecutionError("runtime decode returned unbound state")
                self._validate_runtime_state(expected_position)
                self._last_logits = self._validated_logits(logits)
            except Exception as exc:
                self._poison()
                if isinstance(exc, (RuntimeExecutionError, RuntimeBindingError)):
                    raise RuntimeExecutionError("compiled runtime decode failed") from exc
                raise RuntimeExecutionError("compiled runtime decode failed") from exc
            self._position = expected_position
            self._pending_token = None
            self._status = "ready"
            return self.logits

    def finish(self) -> None:
        with self._lock:
            if self._status not in {"ready", "selected", "exhausted"}:
                raise RuntimeExecutionError("session cannot finish from its current state")
            self._zero_runtime()
            self._status = "exhausted"

    def generate_ids(
        self,
        token_ids: Sequence[int] | np.ndarray,
        *,
        max_new_tokens: int | None = None,
    ) -> tuple[int, ...]:
        with self._lock:
            if max_new_tokens is None:
                limit = self._max_new
            elif isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
                raise RuntimeExecutionError("max_new_tokens must be an integer")
            else:
                limit = max_new_tokens
            if not 1 <= limit <= self._max_new:
                raise RuntimeExecutionError(f"max_new_tokens must be in [1, {self._max_new}]")
            self.prefill_ids(token_ids)
            output = []
            for index in range(limit):
                output.append(self.select_next())
                if index + 1 < limit:
                    self.decode_selected()
            if self._status != "exhausted":
                self.finish()
            return tuple(output)

    def close(self) -> None:
        with self._lock:
            self._zero_runtime()
            self._status = "closed"


__all__ = ["CompiledRuntimeSession", "RuntimeExecutionError"]
