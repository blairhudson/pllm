from pllm.configuration import Deployment as Deployment
from pllm.components._planned import PendingComponent as PendingMethod
from pllm.deployment.role_placement import (
    AttestationPolicy as AttestationPolicy,
    PlacementAssessment as PlacementAssessment,
    RoleDeployment as RoleDeployment,
    RolePlacement as RolePlacement,
)

class TeeFheTopology(PendingMethod): ...
class MosaicMaskedAcceleratorRoles(PendingMethod): ...
