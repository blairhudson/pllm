"""Explicit source arithmetic, separate from quantization choices."""

from pllm.configuration import ComponentDescriptor, ComponentRef


class SourceFloat32(ComponentRef):
    """Public source FP32 forward arithmetic; never an integer decoder mode."""

    __slots__ = ()
    NUMERIC_PROFILE = "pllm.numeric.source_float32.v1"
    descriptor = ComponentDescriptor(
        component="pllm/source-float32/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/numeric",
        category_version="1",
        lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class Q11I16Ring64Boundary(ComponentRef):
    """Existing nearest-even Q11 i16 weights and bounded Q11 activation encoding.

    Signed ring64 reconstruction uses float32 division by 2**22 and source bias.
    This declaration does not select sharing or introduce another arithmetic mode.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/q11-i16-ring64-boundary/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/numeric", category_version="1", lifecycle_phase="compile",
        parameter_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = ["SourceFloat32", "Q11I16Ring64Boundary"]
