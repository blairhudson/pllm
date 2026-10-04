"""Public runtime role component declarations."""

from abc import ABC, abstractmethod

from pllm.configuration import ComponentDescriptor, ComponentRef
from pllm.roles.topology import (
    Channel,
    Role,
    RoleGraph,
    client_only_reference_graph,
    two_online_reference_graph,
)


class InferenceRole(ComponentRef, ABC):
    __slots__ = ()

    @classmethod
    @abstractmethod
    def describe(cls) -> ComponentDescriptor: ...


class Inference(InferenceRole):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/inference",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/inference-role",
        category_version="1",
        lifecycle_phase="online",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("remote-linear-stage-execution",),
        role_eligibility=("inference",),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class RoleTopology(ComponentRef, ABC):
    """One compatible placement of logical roles, not a privacy proof."""

    __slots__ = ()

    @classmethod
    @abstractmethod
    def describe(cls) -> ComponentDescriptor: ...


class PreparedProviderRoles(RoleTopology):
    """Client, offline trusted Preparation, and one online Inference provider."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/one-online-provider-offline-preparation/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/role-topology",
        category_version="1",
        lifecycle_phase="offline+online",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("prepared-public-linear", "single-online-provider"),
        role_eligibility=("client", "preparation", "inference"),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class ClientOnlyRoles(RoleTopology):
    """Client owns the full checkpoint and executes every decoder operation."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/client-only/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/role-topology",
        category_version="1",
        lifecycle_phase="offline+online",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("client-owned-weights", "zero-online-provider-traffic"),
        role_eligibility=("client",),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class TwoOnlineOffsetRoles(RoleTopology):
    """Two online public-weight workers whose operator independence needs evidence."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/two-online-offset-workers/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/role-topology",
        category_version="1",
        lifecycle_phase="online",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("two-online-public-linear", "separate-worker-operators-required"),
        role_eligibility=("client", "worker_a", "worker_b"),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class OutputHeadAtInference(ComponentRef):
    """Move an untied public output head to the prepared Inference role."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/output-head-at-inference/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/boundary-placement",
        category_version="1",
        lifecycle_phase="offline+online",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("remote-untied-output-head", "prepared-output-head"),
        role_eligibility=("client", "preparation", "inference"),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class ClientPlacement(ComponentRef, ABC):
    """Compiler-bound ownership of public decoder linear stages."""

    __slots__ = ()

    @classmethod
    @abstractmethod
    def describe(cls) -> ComponentDescriptor: ...


CLIENT_LINEAR_ROLES = ("attention_output", "mlp_down", "mlp_gate_up", "qkv_projection")


class ClientLinearRoles(ClientPlacement):
    """Own the union of semantic linear roles and an optional complete prefix."""

    __slots__ = ("roles", "prefix_layers")
    descriptor = ComponentDescriptor(
        component="pllm/client-owned-linear-roles/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/layer-placement",
        category_version="1",
        lifecycle_phase="offline+online",
        parameter_schema={
            "type": "object",
            "required": ["roles"],
            "additionalProperties": False,
            "properties": {
                "roles": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "uniqueItems": True,
                    "items": {"type": "string", "enum": list(CLIENT_LINEAR_ROLES)},
                },
                "prefix_layers": {"type": "integer", "minimum": 1, "maximum": 8},
            },
        },
        capabilities=("client-owned-semantic-weights", "prepared-remote-body"),
        role_eligibility=("client", "preparation", "inference"),
    )

    def __init__(self, roles: tuple[str, ...] | list[str], *, prefix_layers: int = 0) -> None:
        if (
            not isinstance(roles, (tuple, list))
            or not 1 <= len(roles) <= 4
            or any(type(role) is not str or role not in CLIENT_LINEAR_ROLES for role in roles)
            or len(set(roles)) != len(roles)
        ):
            raise ValueError("client linear roles require distinct supported semantic roles")
        if type(prefix_layers) is not int or not 0 <= prefix_layers <= 8:
            raise ValueError("client prefix_layers must be in [0, 8]")
        canonical = tuple(sorted(roles))
        super().__init__(self.descriptor.component, {
            "roles": list(canonical), **({"prefix_layers": prefix_layers} if prefix_layers else {}),
        })
        object.__setattr__(self, "roles", canonical)
        object.__setattr__(self, "prefix_layers", prefix_layers)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"roles": self.roles, **({"prefix_layers": self.prefix_layers} if self.prefix_layers else {})}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class ClientPrefixLayers(ClientPlacement):
    """Own the first bounded semantic decoder layers at the trusted client."""

    __slots__ = ("layers",)
    descriptor = ComponentDescriptor(
        component="pllm/client-owned-prefix-layers/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/layer-placement",
        category_version="1",
        lifecycle_phase="offline+online",
        parameter_schema={
            "type": "object",
            "required": ["layers"],
            "additionalProperties": False,
            "properties": {"layers": {"type": "integer", "minimum": 1, "maximum": 8}},
        },
        capabilities=("client-owned-prefix-weights", "prepared-remote-suffix"),
        role_eligibility=("client", "preparation", "inference"),
    )

    def __init__(self, layers: int) -> None:
        if type(layers) is not int or not 1 <= layers <= 8:
            raise ValueError("client-owned prefix requires one to eight layers")
        super().__init__(self.descriptor.component, {"layers": layers})
        object.__setattr__(self, "layers", layers)

    def get_params(self, deep: bool = True) -> dict[str, int]:
        return {"layers": self.layers}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = [
    "Channel",
    "ClientOnlyRoles",
    "Inference",
    "InferenceRole",
    "PreparedProviderRoles",
    "Role",
    "RoleGraph",
    "RoleTopology",
    "TwoOnlineOffsetRoles",
    "OutputHeadAtInference",
    "ClientPlacement",
    "ClientLinearRoles",
    "ClientPrefixLayers",
    "client_only_reference_graph",
    "two_online_reference_graph",
]
