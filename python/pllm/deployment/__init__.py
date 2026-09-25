"""Public deployment declarations."""

from pllm.configuration import Deployment
from pllm.components._planned import PendingComponent as PendingMethod, planned as pending
from pllm.deployment.role_placement import (
    AttestationPolicy, PlacementAssessment, RoleDeployment, RolePlacement,
)


@pending("bifrost")
class TeeFheTopology(PendingMethod):
    pass

__all__ = [
    "AttestationPolicy", "Deployment", "PlacementAssessment", "RoleDeployment",
    "RolePlacement", "TeeFheTopology",
]
