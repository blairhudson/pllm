"""Public runtime role component declarations."""

from abc import ABC, abstractmethod

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError
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


class BoundaryPlacement(ComponentRef, ABC):
    """Explicit ownership and protocol selection for decoder boundaries."""

    __slots__ = ()

    @classmethod
    @abstractmethod
    def describe(cls) -> ComponentDescriptor: ...


class PrivateLookupAndFullHead(BoundaryPlacement):
    """Couple private token lookup with an exact two-party full-vocabulary head."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/private-lookup-full-head/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/boundary-placement", category_version="1",
        lifecycle_phase="offline+online",
        parameter_schema={"type": "object", "additionalProperties": False, "properties": {
            "head_dispatch": {"type": "string", "enum": ["serial", "parallel"]},
            "head_output_encoding": {"type": "string", "enum": ["u32", "uniform_residues"]},
            "head_kernel": {"type": "string", "enum": ["existing", "byte_dot"]},
        }},
        capabilities=("private-packed128-token-lookup", "two-party-full-vocabulary-head"),
        role_eligibility=("client", "boundary_a", "boundary_b"),
    )

    def __init__(self, head_dispatch: str = "serial", head_output_encoding: str = "u32",
                 *, head_kernel: str = "existing") -> None:
        params = {}
        for name, value, default, allowed in (
            ("head_dispatch", head_dispatch, "serial", ("serial", "parallel")),
            ("head_output_encoding", head_output_encoding, "u32", ("u32", "uniform_residues")),
            ("head_kernel", head_kernel, "existing", ("existing", "byte_dot")),
        ):
            if type(value) is not str or value not in allowed:
                raise ConfigurationError(f"{name} must be one of {allowed}")
            if value != default:
                params[name] = value
        super().__init__(self.descriptor.component, params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class PreparedBoundaryRoles(RoleTopology):
    """Offline Preparation, prepared body provider and two online boundary parties."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/prepared-body-two-online-boundary/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/role-topology", category_version="1",
        lifecycle_phase="offline+online",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("prepared-public-linear", "two-online-private-boundary"),
        role_eligibility=("client", "preparation", "inference", "boundary_a", "boundary_b"),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class OutputHeadAtInference(BoundaryPlacement):
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
    """Own selected semantic roles, layer/role pairs and an optional prefix."""

    __slots__ = ("roles", "prefix_layers", "layer_roles")
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
                    "minItems": 0,
                    "maxItems": 4,
                    "uniqueItems": True,
                    "items": {"type": "string", "enum": list(CLIENT_LINEAR_ROLES)},
                },
                "layer_roles": {
                    "type": "array", "minItems": 1, "maxItems": 1024,
                    "uniqueItems": True,
                    "items": {
                        "type": "array", "minItems": 2, "maxItems": 2,
                        "prefixItems": [
                            {"type": "integer", "minimum": 0, "maximum": 255},
                            {"type": "string", "enum": list(CLIENT_LINEAR_ROLES)},
                        ],
                    },
                },
                "prefix_layers": {"type": "integer", "minimum": 1, "maximum": 8},
            },
        },
        capabilities=("client-owned-semantic-weights", "prepared-remote-body"),
        role_eligibility=("client", "preparation", "inference"),
    )

    def __init__(self, roles: tuple[str, ...] | list[str] = (), *, prefix_layers: int = 0,
                 layer_roles: tuple[tuple[int, str], ...] | list = ()) -> None:
        if (
            not isinstance(roles, (tuple, list))
            or len(roles) > 4 or (not roles and not layer_roles)
            or any(type(role) is not str or role not in CLIENT_LINEAR_ROLES for role in roles)
            or len(set(roles)) != len(roles)
        ):
            raise ValueError("client linear roles require distinct supported semantic roles")
        if (not isinstance(layer_roles, (tuple, list)) or len(layer_roles) > 1024
                or any(not isinstance(pair, (tuple, list)) or len(pair) != 2
                       or type(pair[0]) is not int or not 0 <= pair[0] <= 255
                       or type(pair[1]) is not str or pair[1] not in CLIENT_LINEAR_ROLES
                       for pair in layer_roles)):
            raise ValueError("layer roles require bounded (layer, semantic role) pairs")
        pairs = tuple(sorted(tuple(pair) for pair in layer_roles))
        if len(set(pairs)) != len(pairs):
            raise ValueError("layer roles must be distinct")
        if type(prefix_layers) is not int or not 0 <= prefix_layers <= 8:
            raise ValueError("client prefix_layers must be in [0, 8]")
        canonical = tuple(sorted(roles))
        super().__init__(self.descriptor.component, {
            "roles": list(canonical), **({"prefix_layers": prefix_layers} if prefix_layers else {}),
            **({"layer_roles": [list(pair) for pair in pairs]} if pairs else {}),
        })
        object.__setattr__(self, "roles", canonical)
        object.__setattr__(self, "layer_roles", pairs)
        object.__setattr__(self, "prefix_layers", prefix_layers)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"roles": self.roles, **({"prefix_layers": self.prefix_layers} if self.prefix_layers else {}),
                **({"layer_roles": self.layer_roles} if self.layer_roles else {})}

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
    "BoundaryPlacement",
    "PrivateLookupAndFullHead",
    "PreparedBoundaryRoles",
    "ClientPlacement",
    "ClientLinearRoles",
    "ClientPrefixLayers",
    "client_only_reference_graph",
    "two_online_reference_graph",
]
