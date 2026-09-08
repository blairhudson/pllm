from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .privacy import PrivacyMode, ProprietaryProtocol


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    api_keys: tuple[str, ...] = ("he-local",)
    privacy_mode: str = "public"
    proprietary_protocol: str = "guarded"
    backends: tuple[dict[str, Any], ...] = ()
    engine_models: tuple[dict[str, Any], ...] = ()
    max_batch_size: int = 32
    max_batch_wait_ms: float = 0.25
    adaptive_batching: bool = True
    response_retention_seconds: float = 3600.0
    allow_insecure_local_correlations: bool = False
    tenseal_path: str | None = None
    reference_model_id: str = "he-bigram-demo"
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
            "allow_insecure_local_correlations": self.allow_insecure_local_correlations,
            "tenseal_path": self.tenseal_path,
            "reference_model_id": self.reference_model_id,
            "reference_phrase": self.reference_phrase,
            "reference_alphabet": self.reference_alphabet,
        }
