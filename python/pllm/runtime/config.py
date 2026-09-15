from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .privacy import PrivacyMode, ProprietaryProtocol


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    api_keys: tuple[str, ...] = ("pllm-local",)
    privacy_mode: str = "public"
    proprietary_protocol: str = "guarded"
    backends: tuple[dict[str, Any], ...] = ()
    engine_models: tuple[dict[str, Any], ...] = ()
    max_batch_size: int = 32
    max_batch_wait_ms: float = 0.25
    adaptive_batching: bool = True
    response_retention_seconds: float = 3600.0
    provider_push_api_key: str | None = None
    rendezvous_timeout_seconds: float = 30.0
    rendezvous_capacity: int = 32768
    rendezvous_max_bytes: int = 268_435_456
    rendezvous_max_attempts_per_session: int = 33_554_432
    prepared_session_capacity: int = 4096
    prepared_session_idle_seconds: float = 300.0
    preparation_request_max_bytes: int = 16_384
    prepared_payload_max_bytes: int = 268_435_456
    prepared_tensor_max_elements: int = 67_108_864
    prepared_stage_batch_rows: int = 4096
    preparation_inference_url: str | None = None
    preparation_push_api_key: str | None = None
    preparation_push_timeout_seconds: float = 10.0
    allow_insecure_local_correlations: bool = False
    tenseal_path: str | None = None
    reference_model_id: str = "pllm-bigram-demo"
    reference_phrase: str = "private\n"
    reference_alphabet: str = "private\\n"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GatewayConfig":
        data = dict(value)
        if "api_keys" in data:
            data["api_keys"] = tuple(data["api_keys"])
        data["privacy_mode"] = PrivacyMode.parse(data.get("privacy_mode", "public")).value
        data["proprietary_protocol"] = ProprietaryProtocol.parse(
            data.get("proprietary_protocol", "guarded")
        ).value
        if "backends" in data:
            data["backends"] = tuple(data["backends"])
        if "engine_models" in data:
            data["engine_models"] = tuple(data["engine_models"])
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> "GatewayConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_keys": list(self.api_keys),
            "privacy_mode": self.privacy_mode,
            "proprietary_protocol": self.proprietary_protocol,
            "backends": list(self.backends),
            "engine_models": list(self.engine_models),
            "max_batch_size": self.max_batch_size,
            "max_batch_wait_ms": self.max_batch_wait_ms,
            "adaptive_batching": self.adaptive_batching,
            "response_retention_seconds": self.response_retention_seconds,
            "provider_push_api_key": self.provider_push_api_key,
            "rendezvous_timeout_seconds": self.rendezvous_timeout_seconds,
            "rendezvous_capacity": self.rendezvous_capacity,
            "rendezvous_max_bytes": self.rendezvous_max_bytes,
            "rendezvous_max_attempts_per_session": self.rendezvous_max_attempts_per_session,
            "prepared_session_capacity": self.prepared_session_capacity,
            "prepared_session_idle_seconds": self.prepared_session_idle_seconds,
            "preparation_request_max_bytes": self.preparation_request_max_bytes,
            "prepared_payload_max_bytes": self.prepared_payload_max_bytes,
            "prepared_tensor_max_elements": self.prepared_tensor_max_elements,
            "prepared_stage_batch_rows": self.prepared_stage_batch_rows,
            "preparation_inference_url": self.preparation_inference_url,
            "preparation_push_api_key": self.preparation_push_api_key,
            "preparation_push_timeout_seconds": self.preparation_push_timeout_seconds,
            "allow_insecure_local_correlations": self.allow_insecure_local_correlations,
            "tenseal_path": self.tenseal_path,
            "reference_model_id": self.reference_model_id,
            "reference_phrase": self.reference_phrase,
            "reference_alphabet": self.reference_alphabet,
        }
