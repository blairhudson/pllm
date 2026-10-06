"""Explicitly public, trusted-publisher prefill artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError

if TYPE_CHECKING:
    import numpy as np
    from pllm.runtime.model_binding import CompiledRuntimeModel
    from pllm.runtime.transformer_client import RuntimeSnapshot


class PublicPrefixCapsule(ComponentRef):
    """Import a pre-positioned public prefix under a trusted SHA-256 commitment.

    Requires full-KV ``prefix_f32`` and ``ClientPrefixReuse``. Publishing is an
    explicit disclosure of every supplied token and KV tensor. Only publish
    already-public material; ordinary response caches are never exported.
    """
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/public-prefix-capsule/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/cache-optimization", category_version="1", lifecycle_phase="offline",
        parameter_schema={"type": "object", "required": ["path", "digest", "size_bytes"], "additionalProperties": False,
            "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 4096},
                "digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "size_bytes": {"type": "integer", "minimum": 1, "maximum": 67108864}}},
        capabilities=("public-prefill-import", "source-numeric-state-binding"), role_eligibility=("client",),
    )

    def __init__(self, path: str, *, digest: str, size_bytes: int) -> None:
        if type(path) is not str or not 0 < len(path) <= 4096:
            raise ConfigurationError("public prefix artifact path must be bounded and nonempty")
        if type(digest) is not str or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            raise ConfigurationError("public prefix requires a trusted SHA-256 digest")
        if type(size_bytes) is not int or not 0 < size_bytes <= 64 << 20:
            raise ConfigurationError("public prefix artifact must fit 64 MiB")
        super().__init__(self.descriptor.component, {"path": path, "digest": digest, "size_bytes": size_bytes})

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor

    @classmethod
    def publish(
        cls, compiled: CompiledRuntimeModel, *, public_token_ids: list[int] | tuple[int, ...],
        snapshot: RuntimeSnapshot, logits: np.ndarray, path: str | Path,
    ) -> PublicPrefixCapsule:
        """Publish explicitly public completed-prefill state from a trusted producer.

        The publisher vouches for computation and the token/state association.
        Consumers pin the returned digest independently of serving providers.
        """
        from pllm.runtime.public_prefix import publish
        return publish(compiled, public_token_ids, snapshot, logits, path)
