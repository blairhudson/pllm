"""Public deployment declarations."""

from pllm.configuration import Deployment
from pllm.components._planned import (
    PendingComponent as PendingMethod,
    install_planned_components,
    planned as pending,
)
from pllm.deployment.role_placement import (
    AttestationPolicy,
    PlacementAssessment,
    RoleDeployment,
    RolePlacement,
)
from pllm.deployment.network import (
    LinkConditions,
    LinkObservation,
    NetworkError,
    NetworkSnapshot,
    NetworkSpec,
    PartyOffer,
    LivePartyOffer,
    PartySpec,
    PartyTrust,
    discover,
    async_discover,
)
from pllm.deployment.execution import (
    ExecutionBinding, LiveExecutionBinding, ExecutionLease, open_execution, async_open_execution,
)


@pending("bifrost")
class TeeFheTopology(PendingMethod):
    pass


__all__ = [
    "AttestationPolicy",
    "Deployment",
    "PlacementAssessment",
    "RoleDeployment",
    "RolePlacement",
    "TeeFheTopology",
    "LinkObservation",
    "LinkConditions",
    "NetworkError",
    "NetworkSnapshot",
    "NetworkSpec",
    "PartyOffer",
    "LivePartyOffer",
    "PartySpec",
    "PartyTrust",
    "discover",
    "async_discover",
    "ExecutionBinding",
    "LiveExecutionBinding",
    "ExecutionLease",
    "open_execution",
    "async_open_execution",
]

install_planned_components(globals())
