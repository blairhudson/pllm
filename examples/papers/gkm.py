"""Gupta–Katz–Miers: fixed-length full/seeded CPU compositions."""

from pllm import Deployment, Experiment, Model, Pipeline
from pllm.configuration import ConfigurationError
from pllm.kernels import Cpu
from pllm.numeric import Q11I16Ring64Boundary
from pllm.protocols import TwoShareInput, VerifiedPublicF32Lookup, ClientF32DecoderRegions
from pllm.protocols import OpenWeightVerifiableOutsourcing
from pllm.roles import TwoOnlineDelegationRoles
from pllm.state import ClientLocalKv
from pllm.sources import ModelSource
from pllm.verification import PublishedLinearVerify

from examples.papers.common import BUDGET, MODEL


def _factory_model(value: ModelSource) -> Model:
    """Preserve the former typed profile's structural source override contract."""
    if not isinstance(value, ModelSource):
        raise ConfigurationError("model must satisfy the ModelSource contract")
    return value if isinstance(value, Model) else Model.from_spec(value.to_spec())


def experiment(model=MODEL, budget=BUDGET):
    return Experiment(
        "gkm",
        Pipeline(
            _factory_model(model),
            components={
                "linear": OpenWeightVerifiableOutsourcing(capacity="decoder"),
                "kernels": Cpu(threads=1),
                "verification": PublishedLinearVerify(),
                "topology": TwoOnlineDelegationRoles(),
                "state": ClientLocalKv(),
            },
            profile="research.open_weight_verifiable_cpu",
        ),
        Deployment.local(root=".pllm/papers/gkm"),
        budget,
    )


def seeded(model=MODEL, budget=BUDGET):
    """Same source arithmetic with fresh role-local AES share expansion."""
    return experiment(model, budget).with_params(
        name="gkm-seeded", pipeline__components__linear__sharing="seeded",
    )


def smoke(model=MODEL):
    """Pinned short-request functionality cell; separate from publication workload."""
    return experiment(model).with_params(
        name="gkm-smoke", budget__max_input_tokens=8, budget__max_new_tokens=2,
    )


def seeded_smoke(model=MODEL):
    return smoke(model).with_params(
        name="gkm-seeded-smoke", pipeline__components__linear__sharing="seeded",
    )


def bounded_response():
    """Single-request declaration; source admission remains a separate gate."""
    return experiment().with_params(
        name="gkm-bounded-response", budget__requests=1,
        budget__max_input_tokens=32, budget__max_new_tokens=2,
    )


def composed_bounded_response(model=MODEL, budget=None):
    """Opt-in explicit bindings; declaration only, not qualified run evidence."""
    base = bounded_response()
    return Experiment(
        "gkm-composed-bounded-response",
        Pipeline(_factory_model(model), components={
            **base.pipeline.components,
            "input": TwoShareInput(),
            "lookup": VerifiedPublicF32Lookup(),
            "regions": ClientF32DecoderRegions(),
            "numeric": Q11I16Ring64Boundary(),
        }),
        base.deployment,
        base.budget if budget is None else budget,
    )
