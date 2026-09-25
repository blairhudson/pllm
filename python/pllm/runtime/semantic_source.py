"""Preserve a nested checkpoint's source vocabulary across runtime transport."""

from __future__ import annotations

from typing import Any


class SemanticSourceError(ValueError):
    """A flattened runtime view cannot reproduce the locked source."""


def semantic_source_config(cfg: dict[str, Any]) -> dict[str, Any]:
    source = cfg.get("semantic_source_config")
    if source is None:
        return cfg
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("model_type"), str)
        or not isinstance(source.get("text_config"), dict)
        or not isinstance(source["text_config"].get("model_type"), str)
    ):
        raise SemanticSourceError("nested semantic source configuration is malformed")
    for key, value in source["text_config"].items():
        if key not in cfg or cfg[key] != value:
            raise SemanticSourceError(f"flattened runtime field {key!r} disagrees with source")
    return source


__all__ = ["SemanticSourceError", "semantic_source_config"]
