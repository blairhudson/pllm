"""Immutable public client-bundle transport selection."""

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


class ClientBundleTransport(ComponentRef):
    """Select raw/framed bundles or independently compressed public artifacts."""

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
            "properties": {
                "encoding": {"type": "string", "enum": ["none", "zlib", "artifacts"]},
                "compression": {"type": "string", "enum": ["zlib"]},
                "batch_objects": {"type": "integer", "minimum": 1, "maximum": 64},
            },
        },
        capabilities=("bounded-bundle-frames", "raw-bundle-digest-verification"),
        role_eligibility=("client", "inference"),
    )

    def __init__(self, encoding: str = "zlib", *, compression: str = "none", batch_objects: int = 1) -> None:
        if encoding not in {"none", "zlib", "artifacts"}:
            raise ConfigurationError("bundle encoding must be none, zlib, or artifacts")
        if compression not in {"none", "zlib"} or (compression != "none" and encoding != "artifacts"):
            raise ConfigurationError("object compression requires artifacts encoding and none or zlib")
        if (type(batch_objects) is not int or not 1 <= batch_objects <= 64
                or batch_objects != 1 and encoding != "artifacts"):
            raise ConfigurationError("object batching requires artifacts encoding and a count in [1, 64]")
        params: dict[str, object] = {"encoding": encoding}
        if compression != "none":
            params["compression"] = compression
        if batch_objects != 1:
            params["batch_objects"] = batch_objects
        super().__init__(self.descriptor.component, params)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor
