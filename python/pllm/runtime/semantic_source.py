"""Preserve a checkpoint's source vocabulary across runtime transport."""

from __future__ import annotations

from typing import Any


class SemanticSourceError(ValueError):
    """A runtime view cannot reproduce the locked source."""


def semantic_source_config(cfg: dict[str, Any]) -> dict[str, Any]:
    if "semantic_source_config" not in cfg:
        return cfg
    source = cfg["semantic_source_config"]
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("model_type"), str)
    ):
        raise SemanticSourceError("semantic source configuration is malformed")
    view = source
    if "text_config" in source:
        view = source["text_config"]
        if not isinstance(view, dict) or not isinstance(view.get("model_type"), str):
            raise SemanticSourceError("nested semantic source configuration is malformed")
    # Admission still belongs to native lowering of the entire source, including
    # unknown extensions. Runtime-only additions are never reinterpreted as source.
    for key, value in view.items():
        if key not in cfg or cfg[key] != value:
            raise SemanticSourceError(f"flattened runtime field {key!r} disagrees with source")
    return source


__all__ = ["SemanticSourceError", "semantic_source_config"]
