"""Public deployment declarations."""

from pllm.configuration import Deployment
from pllm.components._planned import PendingComponent as PendingMethod, planned as pending


@pending("bifrost")
class TeeFheTopology(PendingMethod):
    pass

__all__ = ["Deployment", "TeeFheTopology"]
