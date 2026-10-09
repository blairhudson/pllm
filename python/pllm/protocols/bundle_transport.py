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
                "storage": {"type": "string", "enum": ["paged"]},
                "provider_cache_bytes": {"type": "integer", "minimum": 1048576, "maximum": 2147483648},
            },
        },
        capabilities=("bounded-bundle-frames", "raw-bundle-digest-verification"),
        role_eligibility=("client", "inference"),
    )

    def __init__(self, encoding: str = "zlib", *, compression: str = "none", batch_objects: int = 1,
                 storage: str = "memory", provider_cache_bytes: int = 0) -> None:
        if encoding not in {"none", "zlib", "artifacts"}:
            raise ConfigurationError("bundle encoding must be none, zlib, or artifacts")
        if compression not in {"none", "zlib"} or (compression != "none" and encoding != "artifacts"):
            raise ConfigurationError("object compression requires artifacts encoding and none or zlib")
        if (type(batch_objects) is not int or not 1 <= batch_objects <= 64
                or batch_objects != 1 and encoding != "artifacts"):
            raise ConfigurationError("object batching requires artifacts encoding and a count in [1, 64]")
        if storage not in {"memory", "paged"} or storage == "paged" and encoding != "artifacts":
            raise ConfigurationError("paged client weight storage requires artifacts encoding")
        if (type(provider_cache_bytes) is not int or provider_cache_bytes != 0 and (
                not 1 << 20 <= provider_cache_bytes <= 2 << 30
                or encoding != "artifacts" or compression != "zlib")):
            raise ConfigurationError("provider cache requires zlib artifacts and a 1 MiB–2 GiB byte bound")
        params: dict[str, object] = {"encoding": encoding}
        if compression != "none":
            params["compression"] = compression
        if batch_objects != 1:
            params["batch_objects"] = batch_objects
        if storage != "memory":
            params["storage"] = storage
        if provider_cache_bytes:
            params["provider_cache_bytes"] = provider_cache_bytes
        super().__init__(self.descriptor.component, params)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"encoding": self.params["encoding"],
                "compression": self.params.get("compression", "none"),
                "batch_objects": self.params.get("batch_objects", 1),
                "storage": self.params.get("storage", "memory"),
                "provider_cache_bytes": self.params.get("provider_cache_bytes", 0)}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor
