"""Pure admission checks for decoder-only entry points.

These checks inspect explicit intent only: no resolution, source access, or
normalization. Imports stay lazy because configuration also calls these guards.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pllm.configuration import Experiment, ExperimentProfile, Pipeline


_FORWARD_ONLY = "SourceFloat32 is forward-only, not a decoder numeric mode"


def _reject() -> None:
    from pllm.configuration import ConfigurationError

    raise ConfigurationError(_FORWARD_ONLY)


def require_decoder_pipeline(pipeline: Pipeline) -> None:
    # Same explicit criterion as resolve_decoder_runtime_composition, without
    # invoking resolution (unsupported decoder compositions keep their errors).
    if "numeric" in pipeline.components or any(
        value.component == "pllm/source-float32/v1"
        for value in pipeline.components.values()
    ):
        _reject()


def require_decoder_experiment(experiment: Experiment) -> None:
    from pllm.configuration import ForwardBudget

    if isinstance(experiment.budget, ForwardBudget):
        _reject()
    require_decoder_pipeline(experiment.pipeline)


def require_decoder_profile(profile: ExperimentProfile) -> None:
    # Runtime-kind labels are not authority for the underlying composition.
    require_decoder_source_record(json.loads(profile.canonical_composition))


def _require_decoder_pipeline_record(record: Mapping[str, Any]) -> None:
    components = record.get("components")
    if isinstance(components, Mapping) and (
        "numeric" in components or any(
            isinstance(value, Mapping)
            and value.get("component") == "pllm/source-float32/v1"
            for value in components.values()
        )
    ):
        _reject()


def require_decoder_source_record(record: Any) -> None:
    """Inspect only known schema, experiment, and pipeline envelopes.

    A bare checkpoint source (including a family/model label) declares no
    numeric execution intent. Do not walk arbitrary nested metadata.
    """
    from pllm.configuration import Experiment, ExperimentProfile, Pipeline

    if isinstance(record, Experiment):
        require_decoder_experiment(record)
        return
    if isinstance(record, ExperimentProfile):
        require_decoder_profile(record)
        return
    if isinstance(record, Pipeline):
        require_decoder_pipeline(record)
        return
    if not isinstance(record, Mapping):
        return
    if record.get("schema") == "pllm.experiment.forward.v1":
        _reject()
    _require_decoder_pipeline_record(record)
    experiment = record.get("experiment")
    if isinstance(experiment, Experiment):
        require_decoder_experiment(experiment)
    elif isinstance(experiment, Mapping):
        if experiment.get("schema") == "pllm.experiment.forward.v1":
            _reject()
        pipeline = experiment.get("pipeline")
        if isinstance(pipeline, Pipeline):
            require_decoder_pipeline(pipeline)
        elif isinstance(pipeline, Mapping):
            _require_decoder_pipeline_record(pipeline)
    pipeline = record.get("pipeline")
    if isinstance(pipeline, Pipeline):
        require_decoder_pipeline(pipeline)
    elif isinstance(pipeline, Mapping):
        _require_decoder_pipeline_record(pipeline)
