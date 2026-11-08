"""Small, model-neutral writer for generated source checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_checkpoint(
    directory: Path,
    config: dict,
    tensors: dict[str, np.ndarray],
    *,
    tokenizer_descriptor: dict | None = None,
) -> Path:
    """Serialize source data unchanged; special token IDs are never inferred."""
    from safetensors.numpy import save_file

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if tokenizer_descriptor is not None:
        (root / "pllm_tokenizer.json").write_text(
            json.dumps(tokenizer_descriptor), encoding="utf-8",
        )
    save_file(tensors, root / "model.safetensors")
    return root
