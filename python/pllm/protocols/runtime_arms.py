from pllm.configuration import ComponentDescriptor, ConfigurationError
from pllm.protocols.base import ProtocolMethod


class GuardedLinear(ProtocolMethod):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/guarded-linear/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/protocol-method",
        category_version="1",
        lifecycle_phase="online",
        parameter_schema={
            "type": "object",
            "properties": {
                "max_rows_per_request": {"type": "integer", "minimum": 1},
                "max_rows_per_owner_stage": {"type": "integer", "minimum": 1},
                "max_requests_per_minute": {"type": "integer", "minimum": 1},
                "output_dither_bound": {"type": "integer", "minimum": 0},
            },
            "required": [
                "max_rows_per_request",
                "max_rows_per_owner_stage",
                "max_requests_per_minute",
                "output_dither_bound",
            ],
            "additionalProperties": False,
        },
        capabilities=("guarded-proprietary-linear",),
        role_eligibility=("client", "inference"),
    )

    def __init__(
        self,
        *,
        max_rows_per_request: int = 4096,
        max_rows_per_owner_stage: int = 16384,
        max_requests_per_minute: int = 4096,
        output_dither_bound: int = 0,
    ) -> None:
        values = {
            "max_rows_per_request": max_rows_per_request,
            "max_rows_per_owner_stage": max_rows_per_owner_stage,
            "max_requests_per_minute": max_requests_per_minute,
        }
        if any(type(value) is not int or value < 1 for value in values.values()):
            raise ConfigurationError("guarded linear row and rate limits must be positive integers")
        if type(output_dither_bound) is not int or output_dither_bound < 0:
            raise ConfigurationError("output_dither_bound must be a non-negative integer")
        super().__init__(
            self.descriptor.component,
            {**values, "output_dither_bound": output_dither_bound},
        )

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class BlindedLinear(ProtocolMethod):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/blinded-linear/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/protocol-method",
        category_version="1",
        lifecycle_phase="online",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("blinded-proprietary-linear",),
        role_eligibility=("client", "inference"),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class SecureLinear(ProtocolMethod):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/secure-linear/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/protocol-method",
        category_version="1",
        lifecycle_phase="online",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("bounded-authenticated-linear-preview",),
        required_host_features=("synthetic-fixed-shape-only",),
        role_eligibility=("client", "inference"),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class DirectFHE(ProtocolMethod):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/direct-fhe",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/protocol-method",
        category_version="1",
        lifecycle_phase="online",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("direct-bfv-transformer-linear",),
        required_host_features=("tenseal-backend",),
        role_eligibility=("client", "inference"),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class CleartextLinear(ProtocolMethod):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/cleartext-linear",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/protocol-method",
        category_version="1",
        lifecycle_phase="online",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("plaintext-reference-linear",),
        role_eligibility=("inference",),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = [
    "BlindedLinear",
    "CleartextLinear",
    "DirectFHE",
    "GuardedLinear",
    "SecureLinear",
]
