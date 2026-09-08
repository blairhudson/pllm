"""Compatibility exports for the Round 8 masked Transformer engine.

The implementation lives in :mod:`pllm.runtime.transformer_engine`.  This module
is kept so integrations built against the early Round 8 prototype do not end
up loading a second, incompatible protocol implementation.
"""

from .transformer_engine import (
    MaskedTransformerEngine,
    SafetensorsW4A4Engine,
    StageMetadata,
    TransformerEngineError,
)

# Historical name used by the first prototype.
SafetensorsMaskedTransformerEngine = MaskedTransformerEngine
EngineLoadError = TransformerEngineError

__all__ = [
    "MaskedTransformerEngine",
    "SafetensorsW4A4Engine",
    "SafetensorsMaskedTransformerEngine",
    "StageMetadata",
    "TransformerEngineError",
    "EngineLoadError",
]
