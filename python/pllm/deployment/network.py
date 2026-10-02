"""Strict static/live network records and authenticated discovery; never independence proof."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Mapping

MAX_DOCUMENT_BYTES = 1 << 20
CAPABILITIES = frozenset(
    {
        "trusted_client",
        "trusted_preparation",
        "masked_linear_provider",
        "public_linear_provider",
    }
)


class NetworkError(ValueError):
    """Invalid public network declaration or unresolved admission requirement."""


def canonical(value: Any) -> bytes:
    try:
        result = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    except (ValueError, TypeError) as exc:
        raise NetworkError("document must contain finite JSON values") from exc
    if len(result) > MAX_DOCUMENT_BYTES:
        raise NetworkError("document exceeds 1 MiB")
    return result


def strict_load(value: bytes | str) -> dict[str, Any]:
    if type(value) not in {bytes, str}:
        raise TypeError("JSON input must be bytes or str")
    if len(value if isinstance(value, bytes) else value.encode()) > MAX_DOCUMENT_BYTES:
        raise NetworkError("document exceeds 1 MiB")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, item in items:
            if key in result:
                raise NetworkError(f"duplicate JSON field: {key}")
            result[key] = item
        return result

    def invalid(value: str) -> None:
        raise NetworkError(f"nonfinite JSON value: {value}")

    try:
        result = json.loads(value, object_pairs_hook=pairs, parse_constant=invalid)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise NetworkError(f"invalid JSON: {exc}") from exc
    if type(result) is not dict:
        raise NetworkError("document must be a JSON object")
    canonical(result)  # Also rejects overflowing exponent literals (1e999).
    return result


def exact(value: Mapping[str, Any], names: set[str], schema: str | None = None) -> None:
    if not isinstance(value, Mapping) or set(value) != names:
        raise NetworkError(f"unexpected or missing fields; expected {sorted(names)}")
    if schema is not None and value["schema"] != schema:
        raise NetworkError(f"expected schema {schema}")
    canonical(dict(value))


def identity(value: str, name: str) -> None:
    if (
        type(value) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:+-]{0,255}", value) is None
    ):
        raise NetworkError(f"invalid {name}")


def digest_value(value: str, name: str, *, nonzero: bool = False) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise NetworkError(f"invalid {name} digest")
    if nonzero and value == "0" * 64:
        raise NetworkError(f"{name} must be nonzero")


def integer(value: int, name: str, *, minimum: int = 0, maximum: int = (1 << 64) - 1) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise NetworkError(f"{name} must be an integer in [{minimum}, {maximum}]")


def finite(value: float, name: str, *, minimum: float = 0) -> None:
    if type(value) not in {int, float} or not math.isfinite(value) or value < minimum:
        raise NetworkError(f"{name} must be finite and >= {minimum}")


def strings(value: tuple[str, ...], name: str, *, choices: frozenset[str] | None = None) -> None:
    if type(value) is not tuple or not value or len(value) > 128:
        raise NetworkError(f"{name} must be a nonempty bounded tuple")
    if any(type(item) is not str for item in value) or len(set(value)) != len(value):
        raise NetworkError(f"{name} must contain unique strings")
    if choices is not None and not set(value) <= choices:
        raise NetworkError(f"unsupported {name}")


def _public(value: Any) -> Any:
    if hasattr(value, "to_spec"):
        return value.to_spec()
    if type(value) is tuple:
        return [_public(item) for item in value]
    return value


class PublicRecord:
    __slots__ = ()
    SCHEMA: ClassVar[str]

    def to_spec(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            **{item.name: _public(getattr(self, item.name)) for item in fields(self)},
        }

    def canonical_bytes(self) -> bytes:
        return canonical(self.to_spec())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.SCHEMA.encode() + b"\0" + self.canonical_bytes()).hexdigest()

    @classmethod
    def from_file(cls, path: str | Path) -> Any:
        with Path(path).expanduser().open("rb") as stream:
            return cls.from_spec(strict_load(stream.read(MAX_DOCUMENT_BYTES + 1)))

    @classmethod
    def _fields(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        exact(value, {"schema", *(item.name for item in fields(cls))}, cls.SCHEMA)
        return {key: item for key, item in value.items() if key != "schema"}


@dataclass(frozen=True, slots=True)
class PartyOffer(PublicRecord):
    SCHEMA = "pllm.party_offer.v1"
    party_id: str
    operator_id: str
    instance_epoch: str
    expires_at_ms: int
    capabilities: tuple[str, ...]
    max_memory_bytes: int
    max_weight_bytes: int
    max_sessions: int
    active_sessions: int
    allowed_models: tuple[str, ...]
    allowed_compositions: tuple[str, ...]
    devices: tuple[str, ...] = ("cpu",)
    authenticated: bool = False

    def __post_init__(self) -> None:
        identity(self.party_id, "party_id")
        identity(self.operator_id, "operator_id")
        digest_value(self.instance_epoch, "instance_epoch", nonzero=True)
        integer(self.expires_at_ms, "expires_at_ms", minimum=1)
        strings(self.capabilities, "capabilities", choices=CAPABILITIES)
        strings(self.devices, "devices", choices=frozenset({"cpu", "metal"}))
        for name in ("max_memory_bytes", "max_weight_bytes"):
            integer(getattr(self, name), name)
        integer(self.max_sessions, "max_sessions", minimum=1, maximum=(1 << 32) - 1)
        integer(self.active_sessions, "active_sessions", maximum=self.max_sessions)
        for name in ("allowed_models", "allowed_compositions"):
            strings(getattr(self, name), name)
            for value in getattr(self, name):
                if value != "*":
                    digest_value(value, name)
        if self.authenticated is not False:
            raise NetworkError("static offers must declare authenticated=false")

    def native_spec(self) -> dict[str, Any]:
        spec = self.to_spec()
        for key in ("schema", "allowed_compositions", "devices", "authenticated"):
            del spec[key]
        return spec

    @classmethod
    def from_spec(cls, value: Mapping[str, Any]) -> PartyOffer:
        data = cls._fields(value)
        for key in ("capabilities", "allowed_models", "allowed_compositions", "devices"):
            if type(data[key]) is not list:
                raise NetworkError(f"{key} must be an array")
            data[key] = tuple(data[key])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class LivePartyOffer(PartyOffer):
    """Public authenticated observation; import alone grants no live authority."""

    SCHEMA = "pllm.party_offer.v2"
    origin: str = ""
    descriptor_digest: str = ""
    source_lock_digest: str = ""
    role_ids: tuple[str, ...] = ()
    observed_at_ms: int = 0
    accepting: bool = True
    independence_verified: bool = False

    def __post_init__(self):
        base = {item.name: getattr(self, item.name) for item in fields(PartyOffer)}
        if base.pop("authenticated") is not True:
            raise NetworkError("live offers require authenticated=true")
        PartyOffer(**base)
        origin_value(self.origin)
        digest_value(self.descriptor_digest, "descriptor", nonzero=True)
        digest_value(self.source_lock_digest, "source_lock", nonzero=True)
        strings(self.role_ids, "role_ids", choices=frozenset({"worker_a", "worker_b", "inference", "preparation"}))
        integer(self.observed_at_ms, "observed_at_ms")
        if not self.observed_at_ms < self.expires_at_ms <= self.observed_at_ms + 300_000:
            raise NetworkError("live offer lifetime must be bounded by 300s")
        if type(self.accepting) is not bool or self.independence_verified is not False:
            raise NetworkError("operator declaration cannot assert verified independence")

    def native_spec(self):
        return {key: getattr(self, key) if type(getattr(self, key)) is not tuple
                else list(getattr(self, key)) for key in (
                    "party_id", "operator_id", "instance_epoch", "expires_at_ms",
                    "capabilities", "max_memory_bytes", "max_weight_bytes", "max_sessions",
                    "active_sessions", "allowed_models")}

    @classmethod
    def from_spec(cls, value):
        data = cls._fields(value)
        for key in ("capabilities", "allowed_models", "allowed_compositions", "devices", "role_ids"):
            if type(data[key]) is not list:
                raise NetworkError(f"{key} must be an array")
            data[key] = tuple(data[key])
        return cls(**data)


def origin_value(value):
    """HTTPS outside loopback; never userinfo, paths, fragments or redirects."""
    import ipaddress
    from urllib.parse import urlsplit

    if type(value) is not str:
        raise NetworkError("origin must be a string")
    parsed = urlsplit(value)
    try:
        loopback = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        loopback = False
    try:
        port = parsed.port
    except ValueError as exc:
        raise NetworkError("invalid origin port") from exc
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
        or (parsed.scheme == "http" and not loopback)
        or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)):
        raise NetworkError("origin must be HTTPS (HTTP allowed only on loopback), without credentials/path")


def credential_reference(value):
    if type(value) is not str or re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", value) is None:
        raise NetworkError("credential reference must name an environment variable")


@dataclass(frozen=True, slots=True)
class PartyTrust(PublicRecord):
    SCHEMA = "pllm.party_trust.v1"
    party_id: str
    operator_id: str
    origin: str
    credential_env: str

    def __post_init__(self):
        identity(self.party_id, "party_id")
        identity(self.operator_id, "operator_id")
        origin_value(self.origin)
        credential_reference(self.credential_env)

    @classmethod
    def from_spec(cls, value):
        return cls(**cls._fields(value))


@dataclass(frozen=True, slots=True)
class PartySpec(PublicRecord):
    """Local installed allowlist. One exact pipeline, source and bounded workload."""

    SCHEMA = "pllm.party_spec.v1"
    party_id: str
    experiment: Any
    source_lock_digest: str
    role_ids: tuple[str, ...]
    max_memory_bytes: int
    max_weight_bytes: int
    max_sessions: int = 1
    max_lease_seconds: int = 300
    offer_seconds: int = 60
    peer_party_ids: tuple[str, ...] = ()

    def __post_init__(self):
        from pllm.configuration import Experiment

        identity(self.party_id, "party_id")
        if not isinstance(self.experiment, Experiment):
            raise NetworkError("party experiment must be an immutable Experiment")
        self.experiment.resolve()
        digest_value(self.source_lock_digest, "source_lock", nonzero=True)
        strings(self.role_ids, "role_ids", choices=frozenset({"worker_a", "worker_b", "inference", "preparation"}))
        roles = {role.id for role in self.experiment.resolve().role_graph.roles}
        if not set(self.role_ids) <= roles:
            raise NetworkError("UNSUPPORTED_NETWORK_GRAPH: party role not in installed topology")
        for key in ("max_memory_bytes", "max_weight_bytes"):
            integer(getattr(self, key), key, minimum=1)
        integer(self.max_sessions, "max_sessions", minimum=1, maximum=8)
        integer(self.max_lease_seconds, "max_lease_seconds", minimum=5, maximum=300)
        integer(self.offer_seconds, "offer_seconds", minimum=5, maximum=300)
        strings(self.peer_party_ids, "peer_party_ids")
        for peer in self.peer_party_ids:
            identity(peer, "peer_party_id")
        kernels = self.experiment.pipeline.components.get("kernels")
        if kernels is None or kernels.component != "pllm/cpu":
            raise NetworkError("UNSUPPORTED_NETWORK_GRAPH: network host currently requires exact CPU kernel")
        if self.experiment.budget.requests != 1:
            raise NetworkError("party admission supports one bounded response attempt")
        integer(self.experiment.budget.max_input_tokens, "max_input_tokens", minimum=1, maximum=64)
        integer(self.experiment.budget.max_new_tokens, "max_new_tokens", minimum=1, maximum=32)

    @classmethod
    def from_spec(cls, value):
        from pllm.configuration import Experiment

        data = cls._fields(value)
        data["experiment"] = Experiment.from_spec(data["experiment"])
        for key in ("role_ids", "peer_party_ids"):
            if type(data[key]) is not list:
                raise NetworkError(f"{key} must be an array")
            data[key] = tuple(data[key])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class LinkObservation(PublicRecord):
    SCHEMA = "pllm.directed_link_observation.v1"
    source_party_id: str
    target_party_id: str
    source: str
    observed_at_ms: int
    expires_at_ms: int
    bytes_per_second: float
    latency_ms: float
    uncertainty_fraction: float = 0.0
    origin: str = "estimate"
    scope: str = "directed-link"

    def __post_init__(self) -> None:
        for name in ("source_party_id", "target_party_id", "source"):
            identity(getattr(self, name), name)
        if self.source_party_id == self.target_party_id:
            raise NetworkError("link must join distinct parties")
        integer(self.observed_at_ms, "observed_at_ms")
        integer(self.expires_at_ms, "expires_at_ms", minimum=self.observed_at_ms + 1)
        finite(self.bytes_per_second, "bytes_per_second", minimum=1)
        finite(self.latency_ms, "latency_ms")
        finite(self.uncertainty_fraction, "uncertainty_fraction")
        if (
            self.uncertainty_fraction > 1
            or self.origin != "estimate"
            or self.scope != "directed-link"
        ):
            raise NetworkError("static link supports scoped estimates only, not measured evidence")

    @classmethod
    def from_spec(cls, value: Mapping[str, Any]) -> LinkObservation:
        return cls(**cls._fields(value))


@dataclass(frozen=True, slots=True)
class NetworkSnapshot(PublicRecord):
    SCHEMA = "pllm.network_snapshot.v1"
    network_id: str
    offers: tuple[PartyOffer, ...]
    links: tuple[LinkObservation, ...]
    source: str
    observed_at_ms: int
    expires_at_ms: int

    def __post_init__(self) -> None:
        identity(self.network_id, "network_id")
        identity(self.source, "source")
        integer(self.observed_at_ms, "observed_at_ms")
        integer(self.expires_at_ms, "expires_at_ms", minimum=self.observed_at_ms + 1)
        if type(self.offers) is not tuple or not 1 <= len(self.offers) <= 64:
            raise NetworkError("snapshot requires 1..64 offers")
        if any(not isinstance(item, PartyOffer) for item in self.offers):
            raise TypeError("offers must contain PartyOffer records")
        ids = {item.party_id for item in self.offers}
        if len(ids) != len(self.offers):
            raise NetworkError("duplicate party_id")
        if type(self.links) is not tuple or len(self.links) > 4096:
            raise NetworkError("links must be a bounded tuple")
        pairs = set()
        for item in self.links:
            if not isinstance(item, LinkObservation):
                raise TypeError("links must contain LinkObservation records")
            pair = (item.source_party_id, item.target_party_id)
            if not set(pair) <= ids or pair in pairs:
                raise NetworkError("unknown or repeated directed link")
            pairs.add(pair)
        object.__setattr__(
            self, "offers", tuple(sorted(self.offers, key=lambda item: item.party_id))
        )
        object.__setattr__(
            self,
            "links",
            tuple(
                sorted(self.links, key=lambda item: (item.source_party_id, item.target_party_id))
            ),
        )
        self.canonical_bytes()

    @classmethod
    def from_spec(cls, value: Mapping[str, Any]) -> NetworkSnapshot:
        data = cls._fields(value)
        if type(data["offers"]) is not list or type(data["links"]) is not list:
            raise NetworkError("offers and links must be arrays")
        data["offers"] = tuple(
            (LivePartyOffer if item.get("schema") == LivePartyOffer.SCHEMA else PartyOffer).from_spec(item)
            for item in data["offers"])
        data["links"] = tuple(LinkObservation.from_spec(item) for item in data["links"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class NetworkSpec(PublicRecord):
    SCHEMA = "pllm.network_spec.v1"
    network_id: str
    snapshot: NetworkSnapshot
    backend: str = "local"
    allowed_operators: tuple[str, ...] = ("*",)
    parties: tuple[PartyTrust, ...] = ()
    directory_origin: str | None = None
    directory_credential_env: str | None = None

    def __post_init__(self) -> None:
        identity(self.network_id, "network_id")
        if (
            not isinstance(self.snapshot, NetworkSnapshot)
            or self.snapshot.network_id != self.network_id
        ):
            raise NetworkError("snapshot network identity mismatch")
        if self.backend not in {"local", "http"}:
            raise NetworkError("only static local or authenticated http backend is supported")
        if self.backend == "local":
            if self.parties or self.directory_origin or self.directory_credential_env:
                raise NetworkError("local network cannot contain live trust roots")
        else:
            if type(self.parties) is not tuple or not 1 <= len(self.parties) <= 63:
                raise NetworkError("live network requires 1..63 party trust roots")
            if any(not isinstance(item, PartyTrust) for item in self.parties):
                raise NetworkError("parties must contain PartyTrust")
            if len({item.party_id for item in self.parties}) != len(self.parties):
                raise NetworkError("duplicate party trust root")
            if self.directory_origin is not None:
                origin_value(self.directory_origin)
                credential_reference(self.directory_credential_env)
            elif self.directory_credential_env is not None:
                raise NetworkError("directory credential needs directory origin")
        strings(self.allowed_operators, "allowed_operators")
        for operator in self.allowed_operators:
            if operator != "*":
                identity(operator, "operator")
        if "*" not in self.allowed_operators and any(
            item.operator_id not in self.allowed_operators for item in self.snapshot.offers
        ):
            raise NetworkError("snapshot includes disallowed operator")
        if self.backend == "http" and ("*" in self.allowed_operators or any(
            item.operator_id not in self.allowed_operators for item in self.parties
        )):
            raise NetworkError("live trust roots require explicitly approved operators")

    def to_spec(self):
        if self.backend == "local":
            return {"schema": self.SCHEMA, "network_id": self.network_id,
                    "snapshot": self.snapshot.to_spec(), "backend": self.backend,
                    "allowed_operators": list(self.allowed_operators)}
        return {"schema": "pllm.network_spec.v2", "network_id": self.network_id,
                "snapshot": self.snapshot.to_spec(), "backend": self.backend,
                "allowed_operators": list(self.allowed_operators),
                "parties": [item.to_spec() for item in self.parties],
                "directory_origin": self.directory_origin,
                "directory_credential_env": self.directory_credential_env}

    @property
    def digest(self):
        schema = self.to_spec()["schema"]
        return hashlib.sha256(schema.encode() + b"\0" + self.canonical_bytes()).hexdigest()

    @classmethod
    def from_spec(cls, value: Mapping[str, Any]) -> NetworkSpec:
        if value.get("schema") == "pllm.network_spec.v2":
            exact(value, {"schema", *(item.name for item in fields(cls))}, "pllm.network_spec.v2")
            data = {key: item for key, item in value.items() if key != "schema"}
            if data["backend"] != "http" or type(data["parties"]) is not list:
                raise NetworkError("v2 network requires http backend and party array")
            data["parties"] = tuple(PartyTrust.from_spec(item) for item in data["parties"])
        else:
            exact(value, {"schema", "network_id", "snapshot", "backend", "allowed_operators"}, cls.SCHEMA)
            data = {key: item for key, item in value.items() if key != "schema"}
            if data["backend"] != "local":
                raise NetworkError("v1 network requires local backend")
        data["snapshot"] = NetworkSnapshot.from_spec(data["snapshot"])
        if type(data["allowed_operators"]) is not list:
            raise NetworkError("allowed_operators must be an array")
        data["allowed_operators"] = tuple(data["allowed_operators"])
        return cls(**data)


def credential(env: str, credentials=None):
    import os

    credential_reference(env)
    if credentials is not None:
        if not isinstance(credentials, Mapping):
            raise NetworkError("credentials must map environment reference names to local values")
        for name in credentials:
            credential_reference(name)
    value = os.environ.get(env) if credentials is None else credentials.get(env)
    if type(value) is not str or len(value) < 16 or any(ord(c) < 33 or ord(c) > 126 for c in value):
        raise NetworkError(f"missing or invalid credential reference: {env}")
    return value


def control_request(origin, env, path, *, body=None, credentials=None):
    import httpx

    origin_value(origin)
    try:
        with httpx.Client(timeout=20, follow_redirects=False, trust_env=False) as client:
            with client.stream("GET" if body is None else "POST", origin + path,
                               headers={"authorization": f"Bearer {credential(env, credentials)}",
                                        "content-type": "application/json"},
                               content=None if body is None else canonical(body)) as response:
                if response.status_code != 200:
                    raise NetworkError(f"NETWORK_CONTROL_REJECTED: HTTP {response.status_code}")
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > MAX_DOCUMENT_BYTES:
                        raise NetworkError("network response exceeds 1 MiB")
                    chunks.append(chunk)
                return strict_load(b"".join(chunks))
    except httpx.HTTPError as exc:
        raise NetworkError("NETWORK_CONTROL_UNAVAILABLE") from exc


def checked_offer(value, trust, now):
    offer = LivePartyOffer.from_spec(value)
    if (offer.party_id, offer.operator_id, offer.origin) != (trust.party_id, trust.operator_id, trust.origin):
        raise NetworkError("CAPABILITY_MISMATCH: offer differs from configured trust root")
    if offer.observed_at_ms > now + 1000 or offer.expires_at_ms <= now:
        raise NetworkError("STALE_OFFER")
    return offer


def discover(spec: NetworkSpec, *, credentials=None, clock=None) -> NetworkSnapshot:
    """Read static declarations or authenticated installed offers, without probes/reserves."""
    if not isinstance(spec, NetworkSpec):
        raise TypeError("spec must be NetworkSpec")
    if spec.backend == "local":
        return spec.snapshot
    import time

    now = (lambda: time.time_ns() // 1_000_000) if clock is None else clock
    approved = {item.party_id for item in spec.parties}
    if spec.directory_origin is not None:
        directory = control_request(spec.directory_origin, spec.directory_credential_env,
                                    "/v1/directory/offers", credentials=credentials)
        exact(directory, {"schema", "network_id", "offers"}, "pllm.directory_offers.v1")
        if directory["network_id"] != spec.network_id or type(directory["offers"]) is not list:
            raise NetworkError("directory network mismatch")
        roots = {item.party_id: item for item in spec.parties}
        membership = set()
        for value in directory["offers"]:
            party = value.get("party_id")
            if party not in roots or party in membership:
                raise NetworkError("directory contains unapproved or duplicate membership")
            checked_offer(value, roots[party], now())
            membership.add(party)
        approved = membership
    offered = []
    for trust in sorted(spec.parties, key=lambda item: item.party_id):
        if trust.party_id not in approved:
            continue
        value = control_request(trust.origin, trust.credential_env, "/v1/party/offer", credentials=credentials)
        offered.append(checked_offer(value, trust, now()))
    # Client authority stays local. Never replace it with a directory declaration.
    roots = {item.party_id for item in spec.parties}
    clients = tuple(item for item in spec.snapshot.offers if item.party_id not in roots)
    stamp = now()
    expiry = min(item.expires_at_ms for item in (*clients, *offered))
    ids = {item.party_id for item in (*clients, *offered)}
    links = tuple(link for link in spec.snapshot.links
                  if {link.source_party_id, link.target_party_id} <= ids)
    return NetworkSnapshot(spec.network_id, (*clients, *offered), links,
                           "authenticated-party-http", stamp, expiry)


async def async_discover(spec: NetworkSpec, *, credentials=None) -> NetworkSnapshot:
    """Read-only discovery without blocking an application's event loop."""
    import asyncio

    return await asyncio.to_thread(discover, spec, credentials=credentials)


__all__ = [
    "NetworkError",
    "PartyOffer",
    "LivePartyOffer",
    "PartySpec",
    "PartyTrust",
    "LinkObservation",
    "NetworkSnapshot",
    "NetworkSpec",
    "discover",
    "async_discover",
]
