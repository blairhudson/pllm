"""Explicitly relocate the public artifacts used by saved research factories."""
from __future__ import annotations

import os
from pathlib import Path

from pllm import Experiment


def relocate_public_artifacts(experiment: Experiment) -> Experiment:
    """Retain content commitments; a changed path creates a new configuration ID."""
    directory = os.environ.get("PLLM_RESEARCH_ARTIFACTS")
    if not directory:
        return experiment
    root = Path(directory).expanduser().resolve()
    parameters = experiment.get_params(deep=True)
    changes = {}
    for slot, identity in (
        ("tokenizer", "pllm/indexed-tokenizer/v1"),
        ("public_prefix", "pllm/public-prefix-capsule/v1"),
    ):
        component = experiment.pipeline.components.get(slot)
        if component is not None and component.component == identity:
            path = f"pipeline__{slot}__path"
            if path not in parameters:
                path = f"pipeline__components__{slot}__path"
            changes[path] = str(root / Path(component.params["path"]).name)
    return experiment.with_params(**changes) if changes else experiment
