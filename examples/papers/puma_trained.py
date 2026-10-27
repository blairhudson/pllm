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

from pllm import Deployment, ExecutionBudget, Experiment
from pllm.profiles import PumaThreePartyCpu
from pllm.protocols import PumaThreePartyDecoder

from examples.papers.common import MODEL


BUDGET = ExecutionBudget(requests=1, max_input_tokens=32, max_new_tokens=2)


def experiment(model=MODEL, budget=BUDGET):
    """Compose existing policy; overrides retain admission, not qualification."""
    return Experiment(
        "puma-trained-candidate" if model == MODEL and budget == BUDGET else "puma-override",
        PumaThreePartyCpu(
            model,
            linear=PumaThreePartyDecoder(
                weights="public", range_policy="checked", storage="compact"
            ),
        ),
        Deployment.local(root=".pllm/papers/puma-trained"),
        budget,
    )
