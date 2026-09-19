from collections.abc import Mapping, Sequence

from pllm.configuration import ComponentDescriptor, ComponentRef

class PlanPass(ComponentRef): ...

class KvCacheEviction(PlanPass):
    descriptor: ComponentDescriptor
    def __init__(
        self,
        *,
        implementation: str = "pllm/mpcache/v1",
        observation_window: tuple[int, int] | Mapping[str, int] = (1, 5),
        static_keep: tuple[int, int] | Mapping[str, int] = (3, 10),
        dynamic_keep: tuple[int, int] | Mapping[str, int] = (1, 4),
        alpha: tuple[int, int] | Mapping[str, int] = (3, 5),
        cluster_sizes: Sequence[int] = (32, 16),
        share_adjacent_layers: bool = True,
    ) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...
