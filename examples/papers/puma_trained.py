"""Bounded trained-response CANDIDATE, not accepted execution or 64+8 capacity.

Proposed first functionality cohort: one ``hello`` request through the genuine
source ChatML renderer, admitted only if its actual input is <=32 tokens, plus
two output tokens. No measured token count, source-template override, or legacy
four-token assumption is implied. Full head, actual norm, and attention stay
unchanged; ordinary source, geometry, numeric, and resource admission still apply.

Construction loads no checkpoint. Actual execution needs user confirmation and
the existing norm2048, backing/original-root, and fresh exact native-slot rules.
Model/budget overrides are unqualified configurations, not trained candidates.
"""

from pllm import ConfigurationError, Deployment, ExecutionBudget, Experiment, Model, Pipeline
from pllm.kernels import Cpu
from pllm.protocols import PumaThreePartyDecoder
from pllm.roles import ThreePartyReplicatedRoles
from pllm.sources import ModelSource
from pllm.state import ReplicatedPrivatePrefix

from examples.papers.common import MODEL


BUDGET = ExecutionBudget(requests=1, max_input_tokens=32, max_new_tokens=2)


def experiment(model=MODEL, budget=BUDGET):
    """Compose existing policy; generic component overrides are unqualified.

    Use the component map and generic parameter paths, not typed properties.
    Overrides retain admission requirements, not qualification.
    """
    if not isinstance(model, ModelSource):
        raise ConfigurationError("model must satisfy the ModelSource contract")
    normalized_model = model if isinstance(model, Model) else Model.from_spec(model.to_spec())
    return Experiment(
        "puma-trained-candidate" if model == MODEL and budget == BUDGET else "puma-override",
        Pipeline(
            model=normalized_model,
            profile="research.puma_three_party_cpu",
            components={
                "linear": PumaThreePartyDecoder(
                    weights="public", range_policy="checked", storage="compact"
                ),
                "topology": ThreePartyReplicatedRoles(),
                "state": ReplicatedPrivatePrefix(),
                "kernels": Cpu(),
            },
        ),
        Deployment.local(root=".pllm/papers/puma-trained"),
        budget,
    )
