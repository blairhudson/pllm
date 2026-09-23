from collections.abc import Callable
from pllm.configuration import ConfigurationError

class PlannedComponent:
    paper_id: str
    module: str
    name: str
    gate: str
    kind: str
    slug: str
    status: str
    @property
    def identity(self) -> str: ...
    @property
    def paper_route(self) -> str: ...
    @property
    def paper_url(self) -> str: ...
    @property
    def sdk_route(self) -> str: ...

class NotYetImplementedError(ConfigurationError, NotImplementedError):
    paper_id: str
    paper_url: str
    sdk_route: str
    identity: str
    kind: str
    gate: str
    def __init__(self, stub: PlannedComponent) -> None: ...

class PendingComponent:
    paper_id: str
    paper_url: str
    paper_route: str
    planned_identity: str
    next_gate: str
    kind: str
    def __init__(self, *args: object, **kwargs: object) -> None: ...

def planned_components() -> tuple[PlannedComponent, ...]: ...
def planned_component(paper_id: str) -> PlannedComponent: ...
def planned_identity(identity: str) -> PlannedComponent | None: ...
def require_implemented_identity(identity: str) -> None: ...
def planned(paper_id: str) -> Callable[[type[PendingComponent]], type[PendingComponent]]: ...
