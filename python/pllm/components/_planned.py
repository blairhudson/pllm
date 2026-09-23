"""Importable, fail-closed component placeholders and their packaged inventory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, TypeVar

from pllm.configuration import ConfigurationError

_CATALOG = Path(__file__).with_name("planned_methods.json")
_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_SYMBOL = re.compile(r"[A-Z][A-Za-z0-9]*\Z")
_MODULES = frozenset(
    {
        "pllm.assurance",
        "pllm.correlation",
        "pllm.deployment",
        "pllm.kernels",
        "pllm.nonlinear",
        "pllm.passes",
        "pllm.protocols",
        "pllm.protocols.masked_linear",
        "pllm.state",
        "pllm.verification",
    }
)
_KINDS = frozenset({"candidate", "security_control", "blocked", "survey", "unverified"})
_STATUSES = frozenset({"pending", "implemented"})


@dataclass(frozen=True, slots=True)
class PlannedComponent:
    """A paper-linked component declaration, not an executable implementation."""

    paper_id: str
    module: str
    name: str
    gate: str
    kind: str = "candidate"
    slug: str = ""
    status: str = "pending"

    @property
    def identity(self) -> str:
        return f"pllm/planned/{self.paper_id}/v1"

    @property
    def paper_route(self) -> str:
        return f"/research/papers/{self.slug or self.paper_id}/"

    @property
    def paper_url(self) -> str:
        return f"https://pllm.run{self.paper_route}"

    @property
    def sdk_route(self) -> str:
        module = self.module.removeprefix("pllm.").replace(".", "-").replace("_", "-")
        return f"/sdk/reference/python/pllm/{module}/#{self.name.lower()}"


@lru_cache(maxsize=1)
def planned_components() -> tuple[PlannedComponent, ...]:
    """Return the packaged, immutable 84-source component roadmap."""
    document = json.loads(_CATALOG.read_text(encoding="utf-8"))
    if document.get("schema") != "pllm.component_plans.v1":
        raise RuntimeError("unsupported packaged component plan schema")
    entries = document.get("entries")
    if not isinstance(entries, list) or len(entries) != 84:
        raise RuntimeError("component roadmap must cover all 84 mapped entries")
    records: list[PlannedComponent] = []
    for raw in entries:
        if not isinstance(raw, dict) or set(raw) - {"paper", "module", "name", "gate", "kind", "slug", "status"}:
            raise RuntimeError("invalid packaged component plan entry")
        values: dict[str, Any] = raw
        record = PlannedComponent(
            paper_id=values["paper"],
            module=values["module"],
            name=values["name"],
            gate=values["gate"],
            kind=values.get("kind", "candidate"),
            slug=values.get("slug", ""),
            status=values.get("status", "pending"),
        )
        if (
            not isinstance(record.paper_id, str)
            or not _SLUG.fullmatch(record.paper_id)
            or record.module not in _MODULES
            or not isinstance(record.name, str)
            or not _SYMBOL.fullmatch(record.name)
            or not isinstance(record.gate, str)
            or not record.gate.strip()
            or "\n" in record.gate
            or record.kind not in _KINDS
            or record.status not in _STATUSES
            or not isinstance(record.slug, str)
            or (record.slug and not _SLUG.fullmatch(record.slug))
        ):
            raise RuntimeError(f"invalid component plan contract: {record.paper_id!r}")
        records.append(record)
    if (
        len({item.paper_id for item in records}) != len(records)
        or len({(item.module, item.name) for item in records}) != len(records)
        or len({item.paper_route for item in records}) != len(records)
    ):
        raise RuntimeError("planned component IDs, Python symbols, and research routes must be unique")
    return tuple(records)


@lru_cache(maxsize=1)
def _by_paper() -> dict[str, PlannedComponent]:
    return {item.paper_id: item for item in planned_components()}


def planned_component(paper_id: str) -> PlannedComponent:
    """Look up one non-executable component by its mapped source ID."""
    return _by_paper()[paper_id]


def planned_identity(identity: str) -> PlannedComponent | None:
    """Match only reserved, exact identities; never widen the active component registry."""
    if type(identity) is not str or not identity.startswith("pllm/planned/"):
        return None
    paper = identity.removeprefix("pllm/planned/").removesuffix("/v1")
    record = _by_paper().get(paper)
    return record if record is not None and record.identity == identity else None


def require_implemented_identity(identity: str) -> None:
    """Keep both known and forged planned IDs outside all executable paths."""
    if identity.startswith("pllm/planned/"):
        stub = planned_identity(identity)
        if stub is None:
            raise ConfigurationError(f"unknown reserved planned component identity: {identity}")
        if stub.status == "implemented":
            raise ConfigurationError(
                f"{identity} is a retired planned identity; use the implemented component identity"
            )
        raise NotYetImplementedError(stub)


class NotYetImplementedError(ConfigurationError, NotImplementedError):
    """A paper-linked method has no admitted implementation or executable contract."""

    def __init__(self, stub: PlannedComponent) -> None:
        self.paper_id = stub.paper_id
        self.paper_url = stub.paper_url
        self.sdk_route = stub.sdk_route
        self.identity = stub.identity
        self.kind = stub.kind
        self.gate = stub.gate
        status = "not yet implemented" if stub.kind == "candidate" else f"not executable ({stub.kind})"
        super().__init__(
            f"{stub.module}.{stub.name} is {status}. "
            f"Paper: {stub.paper_url}. Next gate: {stub.gate}."
        )


class PendingComponent:
    """Base for public paper-linked symbols that cannot be instantiated or selected."""

    __slots__ = ()
    paper_id: str
    paper_url: str
    paper_route: str
    planned_identity: str
    next_gate: str
    kind: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotYetImplementedError(planned_component(type(self).paper_id))


_T = TypeVar("_T", bound=type[PendingComponent])


def planned(paper_id: str) -> Callable[[_T], _T]:
    """Attach checked paper provenance to an explicitly declared public class."""
    stub = planned_component(paper_id)

    def bind(component: _T) -> _T:
        if stub.status != "pending":
            raise RuntimeError(f"implemented component {paper_id} must replace its placeholder")
        if component.__module__ != stub.module or component.__name__ != stub.name:
            raise RuntimeError(f"paper {paper_id!r} does not match its declared Python class")
        component.paper_id = stub.paper_id
        component.paper_url = stub.paper_url
        component.paper_route = stub.paper_route
        component.planned_identity = stub.identity
        component.next_gate = stub.gate
        component.kind = stub.kind
        component.__doc__ = (
            f"Planned {stub.kind.replace('_', ' ')} from {stub.paper_id}; not executable. "
            f"Paper: {stub.paper_url}. Next gate: {stub.gate}."
        )
        return component

    return bind
