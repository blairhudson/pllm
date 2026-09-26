"""Source-neutral model capability placeholders, never executable components."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, TypeVar

from pllm.components._planned import PendingComponent
from pllm.configuration import ConfigurationError

_CATALOG = Path(__file__).with_name("model_capabilities.json")
_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_SYMBOL = re.compile(r"[A-Z][A-Za-z0-9]*\Z")
_MODULES = frozenset({"pllm.kernels", "pllm.passes", "pllm.quantization", "pllm.state"})
_PREFIX = "pllm/planned-model-capability/"


@dataclass(frozen=True, slots=True)
class ModelCapabilityStub:
    """A typed, versioned missing shared contract; no family-specific runtime."""

    slug: str
    capability: str
    module: str
    name: str
    gate: str
    status: str = "pending"

    @property
    def identity(self) -> str:
        return f"{_PREFIX}{self.slug}/v1"

    @property
    def sdk_route(self) -> str:
        return f"/sdk/models/capabilities/{self.capability}/"


@lru_cache(maxsize=1)
def model_capabilities() -> tuple[ModelCapabilityStub, ...]:
    """Return all packaged pending shared model-operator/numeric contracts."""
    document = json.loads(_CATALOG.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != "pllm.model_capability_plans.v1":
        raise RuntimeError("unsupported model capability plan schema")
    entries = document.get("entries")
    if not isinstance(entries, list) or len(entries) < 9:
        raise RuntimeError("model capability roadmap must include the declared gaps")
    stubs: list[ModelCapabilityStub] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {"slug", "capability", "module", "name", "gate", "status"} or not {"slug", "capability", "module", "name", "gate"} <= set(entry):
            raise RuntimeError("invalid model capability plan entry")
        raw: dict[str, Any] = entry
        stub = ModelCapabilityStub(**raw)
        if (
            not isinstance(stub.slug, str) or not _SLUG.fullmatch(stub.slug)
            or not isinstance(stub.capability, str) or not _SLUG.fullmatch(stub.capability)
            or stub.module not in _MODULES
            or not isinstance(stub.name, str) or not _SYMBOL.fullmatch(stub.name)
            or not isinstance(stub.gate, str) or not stub.gate.strip() or "\n" in stub.gate
            or stub.status not in {"pending", "implemented"}
        ):
            raise RuntimeError(f"invalid model capability plan: {stub.slug!r}")
        stubs.append(stub)
    if len({stub.slug for stub in stubs}) != len(stubs) or len({(stub.module, stub.name) for stub in stubs}) != len(stubs):
        raise RuntimeError("duplicate model capability identity or Python symbol")
    return tuple(stubs)


@lru_cache(maxsize=1)
def _by_slug() -> dict[str, ModelCapabilityStub]:
    return {stub.slug: stub for stub in model_capabilities()}


def model_capability(slug: str) -> ModelCapabilityStub:
    """Inspect one missing reusable capability by its stable slug."""
    return _by_slug()[slug]


class ModelCapabilityUnavailable(ConfigurationError, NotImplementedError):
    """The declared shared model contract has no admitted execution path."""

    def __init__(self, stub: ModelCapabilityStub) -> None:
        self.identity = stub.identity
        self.capability = stub.capability
        self.sdk_route = stub.sdk_route
        self.gate = stub.gate
        super().__init__(
            f"{stub.module}.{stub.name} is not yet implemented. "
            f"Capability: {stub.sdk_route}. Next gate: {stub.gate}."
        )


class PendingModelCapability(PendingComponent):
    """Class interface for a missing shared contract, excluded from discovery."""

    __slots__ = ()
    slug: str
    capability: str
    sdk_route: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise ModelCapabilityUnavailable(model_capability(type(self).slug))


_T = TypeVar("_T", bound=type[PendingModelCapability])


def model_capability_stub(slug: str) -> Callable[[_T], _T]:
    """Attach checked model-neutral provenance to a public pending class."""
    stub = model_capability(slug)

    def bind(component: _T) -> _T:
        if stub.status != "pending":
            raise RuntimeError(f"implemented model capability {slug!r} must replace its placeholder")
        if component.__module__ != stub.module or component.__name__ != stub.name:
            raise RuntimeError(f"model capability {slug!r} does not match its declared Python class")
        component.slug = stub.slug
        component.capability = stub.capability
        component.planned_identity = stub.identity
        component.sdk_route = stub.sdk_route
        component.next_gate = stub.gate
        component.__doc__ = (
            f"Planned reusable {stub.capability} capability; no executable implementation. "
            f"See {stub.sdk_route}. Next gate: {stub.gate}."
        )
        return component

    return bind


def require_model_capability_identity(identity: str) -> None:
    """Reject recognized and forged model capability IDs before provider admission."""
    if not identity.startswith(_PREFIX):
        return
    slug = identity.removeprefix(_PREFIX).removesuffix("/v1")
    stub = _by_slug().get(slug)
    if stub is None or stub.identity != identity:
        raise ConfigurationError(f"unknown reserved model capability identity: {identity}")
    if stub.status == "implemented":
        raise ConfigurationError(f"{identity} is a retired planned identity; use the implemented capability")
    raise ModelCapabilityUnavailable(stub)
