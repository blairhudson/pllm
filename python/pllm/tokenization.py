"""Source-bound, public offline tokenizer compilation with client-local queries."""

from __future__ import annotations

from pathlib import Path

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError

__all__ = ["IndexedTokenizer"]


class IndexedTokenizer(ComponentRef):
    """Opt-in bounded byte-level BPE index; never sends text or token lookups remotely.

    ``digest`` must come from a trusted compiler. ``path`` is a pre-positioned
    public artifact directory. Its distribution and compilation are separate
    from response transport counters. Inputs are bounded to 4096 UTF-8 bytes.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/indexed-tokenizer/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/tokenization", category_version="1", lifecycle_phase="offline",
        parameter_schema={"type": "object", "required": ["path", "digest"], "additionalProperties": False,
            "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 4096},
                           "digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}},
        capabilities=("public-bpe-compilation", "client-local-tokenization"), role_eligibility=("client",),
    )

    def __init__(self, path: str, *, digest: str) -> None:
        if type(path) is not str or not 0 < len(path) <= 4096:
            raise ConfigurationError("tokenizer artifact path must be bounded and nonempty")
        if type(digest) is not str or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            raise ConfigurationError("tokenizer artifact requires a trusted SHA-256 digest")
        super().__init__(self.descriptor.component, {"path": path, "digest": digest})

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor

    @classmethod
    def compile(cls, source: str | Path, directory: str | Path) -> IndexedTokenizer:
        """Compile a public tokenizer.json offline, returning its pinned selection."""
        from .runtime.indexed_tokenizer import compile_index
        return compile_index(source, directory)
