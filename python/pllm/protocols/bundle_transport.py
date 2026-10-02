"""Immutable public client-bundle transport selection."""

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


class ClientBundleTransport(ComponentRef):
    """Select raw, bounded zlib frames, or public content-addressed artifacts."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/client-bundle-transport/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/bundle-transport",
        category_version="1",
        lifecycle_phase="offline",
        parameter_schema={
            "type": "object",
            "required": ["encoding"],
            "additionalProperties": False,
            "properties": {"encoding": {"type": "string", "enum": ["none", "zlib", "artifacts"]}},
        },
        capabilities=("bounded-bundle-frames", "raw-bundle-digest-verification"),
        role_eligibility=("client", "inference"),
    )

    def __init__(self, encoding: str = "zlib") -> None:
        if encoding not in {"none", "zlib", "artifacts"}:
            raise ConfigurationError("bundle encoding must be none, zlib, or artifacts")
        super().__init__(self.descriptor.component, {"encoding": encoding})

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor
