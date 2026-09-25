from pllm.configuration import ComponentDescriptor, ComponentRef
from pllm.roles.topology import (
    Channel as Channel,
    Role as Role,
    RoleGraph as RoleGraph,
    client_only_reference_graph as client_only_reference_graph,
    two_online_reference_graph as two_online_reference_graph,
)

class InferenceRole(ComponentRef): ...

class Inference(InferenceRole):
    descriptor: ComponentDescriptor
    def __init__(self) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class RoleTopology(ComponentRef): ...

class PreparedProviderRoles(RoleTopology):
    descriptor: ComponentDescriptor
    def __init__(self) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...
