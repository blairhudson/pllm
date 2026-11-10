"""PUMA source-Q18 composition; native admission retains its bounded envelope."""

from pllm import ConfigurationError, Deployment, Experiment, Model, Pipeline
from pllm.kernels import Cpu
from pllm.protocols import PumaThreePartyDecoder
from pllm.roles import ThreePartyReplicatedRoles
from pllm.sources import ModelSource
from pllm.state import ReplicatedPrivatePrefix

from examples.papers.common import BUDGET, MODEL


def experiment(model=MODEL, budget=BUDGET):
    """Compose components; generic overrides are unqualified configurations.

    Use the component map and generic parameter paths, not typed properties.
    """
    if not isinstance(model, ModelSource):
        raise ConfigurationError("model must satisfy the ModelSource contract")
    normalized_model = model if isinstance(model, Model) else Model.from_spec(model.to_spec())
    return Experiment(
        "puma",
        Pipeline(
            model=normalized_model,
            profile="research.puma_three_party_cpu",
            components={
                "linear": PumaThreePartyDecoder(),
                "topology": ThreePartyReplicatedRoles(),
                "state": ReplicatedPrivatePrefix(),
                "kernels": Cpu(),
            },
        ),
        Deployment.local(root=".pllm/papers/puma"),
        budget,
    )
