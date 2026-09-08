"""Compatibility exports for the consolidated client Transformer runtime."""

from .transformer_client import (
    ClientBundle,
    CorrelationInventory,
    GemmaNumpyRuntime,
    LayerCache,
    MaskedStageClient,
    MaskedTransformerClientRuntime,
    ModelByteTokenizer as ByteTokenizer,
    RemoteLinear,
    StageClientStats,
    StageMetadata,
    TransformerClientBundle,
    TransformerClientError,
)

# Historical aliases.
GemmaRuntimeError = TransformerClientError
RemoteW4A4Linear = RemoteLinear

__all__ = [
    "ClientBundle",
    "TransformerClientBundle",
    "CorrelationInventory",
    "GemmaNumpyRuntime",
    "MaskedTransformerClientRuntime",
    "RemoteLinear",
    "RemoteW4A4Linear",
    "MaskedStageClient",
    "StageClientStats",
    "StageMetadata",
    "LayerCache",
    "ByteTokenizer",
    "TransformerClientError",
    "GemmaRuntimeError",
]
