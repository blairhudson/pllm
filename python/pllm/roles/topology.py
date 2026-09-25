"""Immutable role and channel graph for inspected runtime compositions.

Graphs describe traffic and separation requirements. Their presence does not
establish a privacy proof, a physical deployment, or executable coverage.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from pllm.configuration import ConfigurationError

if TYPE_CHECKING:
    from pllm.profiles import RuntimeComposition

_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_PHASES = frozenset({"offline", "online"})
_DIGEST_DOMAIN = b"pllm.role_topology.v1\0"


def _identifier(value: object, path: str) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ConfigurationError(f"{path} must be a bounded lowercase identifier")
    return value


def _record(value: object, keys: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ConfigurationError(f"{path} must contain exactly {', '.join(sorted(keys))}")
    return value


@dataclass(frozen=True, slots=True)
class Role:
    id: str
    capability: str

    def __post_init__(self) -> None:
        _identifier(self.id, "role.id")
        _identifier(self.capability, "role.capability")

    def to_spec(self) -> dict[str, str]:
        return {"id": self.id, "capability": self.capability}


@dataclass(frozen=True, slots=True)
class Channel:
    source: str
    target: str
    phase: str
    payload: str

    def __post_init__(self) -> None:
        _identifier(self.source, "channel.source")
        _identifier(self.target, "channel.target")
        _identifier(self.payload, "channel.payload")
        if type(self.phase) is not str or self.phase not in _PHASES:
            raise ConfigurationError("channel.phase must be 'offline' or 'online'")

    def to_spec(self) -> dict[str, str]:
        return {
            "source": self.source,
            "target": self.target,
            "phase": self.phase,
            "payload": self.payload,
        }


@dataclass(frozen=True, slots=True)
class RoleGraph:
    roles: tuple[Role, ...]
    channels: tuple[Channel, ...]
    separate_operators: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.roles) is not tuple or not 1 <= len(self.roles) <= 16:
            raise ConfigurationError("role graph requires 1 to 16 roles")
        if type(self.channels) is not tuple or not 0 <= len(self.channels) <= 64:
            raise ConfigurationError("role graph requires at most 64 channels")
        if (len(self.roles) == 1) != (len(self.channels) == 0):
            raise ConfigurationError("one-client topology has no channels; multi-role graphs require channels")
        if type(self.separate_operators) is not tuple or len(self.separate_operators) > 120:
            raise ConfigurationError("role graph has invalid separation requirements")
        if any(type(role) is not Role for role in self.roles):
            raise ConfigurationError("role graph roles must be Role values")
        if any(type(channel) is not Channel for channel in self.channels):
            raise ConfigurationError("role graph channels must be Channel values")
        role_ids = {role.id for role in self.roles}
        if len(role_ids) != len(self.roles) or "client" not in role_ids:
            raise ConfigurationError("role graph requires a unique client and distinct role IDs")
        channel_keys = {
            (channel.source, channel.target, channel.phase, channel.payload)
            for channel in self.channels
        }
        if len(channel_keys) != len(self.channels) or any(
            channel.source not in role_ids
            or channel.target not in role_ids
            or channel.source == channel.target
            for channel in self.channels
        ):
            raise ConfigurationError("role graph has duplicate or unbound channels")
        pairs: set[tuple[str, str]] = set()
        for pair in self.separate_operators:
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or not all(type(role) is str and role in role_ids for role in pair)
                or pair[0] == pair[1]
            ):
                raise ConfigurationError("role separation must name two distinct graph roles")
            left, right = sorted(pair)
            canonical = (left, right)
            if canonical in pairs:
                raise ConfigurationError("role separation must not repeat a role pair")
            pairs.add(canonical)
        object.__setattr__(self, "roles", tuple(sorted(self.roles, key=lambda role: role.id)))
        object.__setattr__(
            self,
            "channels",
            tuple(sorted(self.channels, key=lambda channel: (
                channel.source, channel.target, channel.phase, channel.payload
            ))),
        )
        object.__setattr__(self, "separate_operators", tuple(sorted(pairs)))

    def to_spec(self) -> dict[str, Any]:
        return {
            "schema": "pllm.role_topology.v1",
            "roles": [role.to_spec() for role in self.roles],
            "channels": [channel.to_spec() for channel in self.channels],
            "separate_operators": [list(pair) for pair in self.separate_operators],
        }

    @classmethod
    def from_spec(cls, value: object) -> RoleGraph:
        data = _record(value, {"schema", "roles", "channels", "separate_operators"}, "role graph")
        if data["schema"] != "pllm.role_topology.v1":
            raise ConfigurationError("unsupported role graph schema")
        for key in ("roles", "channels", "separate_operators"):
            if type(data[key]) is not list:
                raise ConfigurationError(f"role graph {key} must be a list")
        roles = tuple(
            Role(**_record(item, {"id", "capability"}, "role")) for item in data["roles"]
        )
        channels = tuple(
            Channel(**_record(item, {"source", "target", "phase", "payload"}, "channel"))
            for item in data["channels"]
        )
        if any(type(pair) is not list for pair in data["separate_operators"]):
            raise ConfigurationError("role separation pairs must be lists")
        pairs = tuple(tuple(pair) for pair in data["separate_operators"])
        return cls(roles=roles, channels=channels, separate_operators=pairs)

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_spec(), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(_DIGEST_DOMAIN + self.canonical_bytes()).hexdigest()

    def separation_violations(self, operators: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
        """Check operator ownership declarations, not real-world independence."""
        if set(operators) != {role.id for role in self.roles} or any(
            type(value) is not str or not value for value in operators.values()
        ):
            raise ConfigurationError("operator placement must name every role exactly once")
        return tuple(
            (left, right)
            for left, right in self.separate_operators
            if operators[left] == operators[right]
        )


def graph_for_runtime(composition: RuntimeComposition) -> RoleGraph:
    """Describe installed executor traffic; never enable execution from a graph."""
    if composition.privacy_mode == "client_only" and not composition.requires_preparation:
        return client_only_reference_graph()
    if composition.privacy_mode == "offset_public" and not composition.requires_preparation:
        return two_online_reference_graph()
    if composition.privacy_mode == "public" and composition.requires_preparation:
        channels = [
            Channel("client", "inference", "offline", "inventory_request"),
            Channel("inference", "client", "offline", "inventory_status"),
            Channel("client", "preparation", "offline", "inventory_authorization"),
            Channel("client", "preparation", "offline", "stage_seed"),
            Channel("preparation", "inference", "offline", "masked_correction"),
            Channel("inference", "preparation", "offline", "correction_acknowledgement"),
            Channel("client", "inference", "online", "session_reservation"),
            Channel("client", "inference", "online", "masked_stage_input"),
            Channel("inference", "client", "online", "masked_stage_output"),
            Channel("inference", "client", "offline", "public_boundary_bundle"),
        ]
        if composition.verification_component is not None:
            channels.append(Channel("preparation", "client", "offline", "verification_projection"))
        return RoleGraph(
            roles=(
                Role("client", "trusted_client"),
                Role("preparation", "trusted_preparation"),
                Role("inference", "masked_linear_provider"),
            ),
            channels=tuple(channels),
            separate_operators=(("preparation", "inference"),),
        )
    if composition.privacy_mode == "proprietary" and not composition.requires_preparation:
        return RoleGraph(
            roles=(Role("client", "trusted_client"), Role("inference", "protected_provider")),
            channels=(
                Channel("client", "inference", "online", "protected_request"),
                Channel("inference", "client", "online", "protected_response"),
            ),
        )
    raise ConfigurationError("composition has no admitted role topology")


def client_only_reference_graph() -> RoleGraph:
    """The local-clear comparator has no provider, online link, or separation claim."""
    return RoleGraph(roles=(Role("client", "trusted_client"),), channels=())


def two_online_reference_graph() -> RoleGraph:
    """Two public linear workers with a declared independent-operator requirement."""
    return RoleGraph(
        roles=(
            Role("client", "trusted_client"),
            Role("worker_a", "public_linear_provider"),
            Role("worker_b", "public_linear_provider"),
        ),
        channels=(
            Channel("client", "worker_a", "online", "input_share"),
            Channel("worker_a", "client", "online", "output_share"),
            Channel("client", "worker_b", "online", "input_share"),
            Channel("worker_b", "client", "online", "output_share"),
            Channel("worker_a", "client", "offline", "public_boundary_bundle"),
        ),
        separate_operators=(("worker_a", "worker_b"),),
    )


__all__ = [
    "Channel", "Role", "RoleGraph", "client_only_reference_graph",
    "graph_for_runtime", "two_online_reference_graph",
]
