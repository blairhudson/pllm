"""Immutable inspection declarations for logical role placement.

These declarations neither verify operator independence nor authorize a TEE.
The active Experiment deployment remains the sole live runtime input.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from pllm.configuration import ConfigurationError
from pllm.roles.topology import RoleGraph

ROLE_DEPLOYMENT_SCHEMA = "pllm.role_deployment.v1"
_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_TECHNOLOGIES = frozenset({"amd-sev-snp", "intel-tdx", "arm-cca", "nvidia-confidential-gpu"})
_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})


def _record(value: object, expected: set[str], *, optional: set[str] | None = None) -> dict:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ConfigurationError("role deployment record must be an object")
    if not expected <= value.keys() or value.keys() - expected - (optional or set()):
        raise ConfigurationError("role deployment record has missing or unknown fields")
    return value


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None or value == "0" * 64:
        raise ConfigurationError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True, slots=True)
class AttestationPolicy:
    """Pinned policy identifiers, never evidence that a quote was verified."""

    technology: str
    measurement_sha256: str
    verifier_root_sha256: str
    tcb_policy_sha256: str

    def __post_init__(self) -> None:
        if type(self.technology) is not str or self.technology not in _TECHNOLOGIES:
            raise ConfigurationError("unknown TEE technology")
        for field in ("measurement_sha256", "verifier_root_sha256", "tcb_policy_sha256"):
            _digest(getattr(self, field), field)

    def to_spec(self) -> dict[str, str]:
        return {
            "technology": self.technology,
            "measurement_sha256": self.measurement_sha256,
            "verifier_root_sha256": self.verifier_root_sha256,
            "tcb_policy_sha256": self.tcb_policy_sha256,
        }

    @classmethod
    def from_spec(cls, spec: object) -> AttestationPolicy:
        fields = {"technology", "measurement_sha256", "verifier_root_sha256", "tcb_policy_sha256"}
        value = _record(spec, fields)
        return cls(**{field: value[field] for field in fields})


@dataclass(frozen=True, slots=True)
class RolePlacement:
    role_id: str
    operator: str
    origin: str | None
    hardware: str = "standard"
    attestation_policy: AttestationPolicy | None = None

    def __post_init__(self) -> None:
        for value in (self.role_id, self.operator):
            if type(value) is not str or _ID.fullmatch(value) is None:
                raise ConfigurationError("role and operator IDs must be bounded identifiers")
        if type(self.hardware) is not str or self.hardware not in {"standard", "tee"}:
            raise ConfigurationError("role hardware must be standard or tee")
        if self.role_id == "client":
            if self.origin is not None or self.hardware != "standard":
                raise ConfigurationError("the trusted client has no provider origin or remote TEE")
        else:
            if type(self.origin) is not str:
                raise ConfigurationError("provider role requires an HTTP(S) origin")
            try:
                parsed = urlsplit(self.origin)
                port = parsed.port
            except ValueError as exc:
                raise ConfigurationError("provider origin has invalid URL syntax or port") from exc
            if (
                parsed.scheme not in {"http", "https"} or not parsed.hostname
                or not parsed.netloc or "@" in parsed.netloc or "%" in parsed.netloc
                or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
                or (port is not None and port == 0)
                or (parsed.scheme == "http" and parsed.hostname not in _LOOPBACK)
                or len(self.origin) > 255
            ):
                raise ConfigurationError("provider origin must be an HTTPS origin or loopback HTTP")
        if self.hardware == "tee":
            if type(self.attestation_policy) is not AttestationPolicy:
                raise ConfigurationError("TEE role requires a pinned attestation policy")
        elif self.attestation_policy is not None:
            raise ConfigurationError("standard hardware cannot claim TEE attestation")

    def to_spec(self) -> dict[str, object]:
        value: dict[str, object] = {
            "role_id": self.role_id,
            "operator": self.operator,
            "origin": self.origin,
            "hardware": self.hardware,
        }
        if self.attestation_policy is not None:
            value["attestation_policy"] = self.attestation_policy.to_spec()
        return value

    @classmethod
    def from_spec(cls, spec: object) -> RolePlacement:
        value = _record(
            spec, {"role_id", "operator", "origin", "hardware"},
            optional={"attestation_policy"},
        )
        return cls(
            role_id=value["role_id"], operator=value["operator"],
            origin=value["origin"], hardware=value["hardware"],
            attestation_policy=(
                AttestationPolicy.from_spec(value["attestation_policy"])
                if "attestation_policy" in value else None
            ),
        )


@dataclass(frozen=True, slots=True)
class PlacementAssessment:
    declared_separation_violations: tuple[tuple[str, str], ...]
    roles_requiring_unverified_attestation: tuple[str, ...]
    operator_independence_verified: bool = False
    attestation_verified: bool = False
    runtime_admission_supported: bool = False

    def to_spec(self) -> dict[str, object]:
        return {
            "declared_separation_violations": [list(pair) for pair in self.declared_separation_violations],
            "roles_requiring_unverified_attestation": list(self.roles_requiring_unverified_attestation),
            "operator_independence_verified": self.operator_independence_verified,
            "attestation_verified": self.attestation_verified,
            "runtime_admission_supported": self.runtime_admission_supported,
        }


@dataclass(frozen=True, slots=True)
class RoleDeployment:
    graph_digest: str
    roles: tuple[RolePlacement, ...]

    def __post_init__(self) -> None:
        _digest(self.graph_digest, "graph_digest")
        if type(self.roles) is not tuple or not 1 <= len(self.roles) <= 16:
            raise ConfigurationError("role deployment requires 1 to 16 roles")
        if any(type(role) is not RolePlacement for role in self.roles):
            raise ConfigurationError("role deployment requires immutable RolePlacement records")
        ordered = tuple(sorted(self.roles, key=lambda role: role.role_id))
        if len({role.role_id for role in ordered}) != len(ordered):
            raise ConfigurationError("role deployment IDs must be unique")
        object.__setattr__(self, "roles", ordered)

    def to_spec(self) -> dict[str, object]:
        return {
            "schema": ROLE_DEPLOYMENT_SCHEMA,
            "graph_digest": self.graph_digest,
            "roles": [role.to_spec() for role in self.roles],
        }

    @classmethod
    def from_spec(cls, spec: object) -> RoleDeployment:
        value = _record(spec, {"schema", "graph_digest", "roles"})
        if value["schema"] != ROLE_DEPLOYMENT_SCHEMA or type(value["roles"]) is not list:
            raise ConfigurationError("unsupported role deployment schema")
        return cls(
            graph_digest=value["graph_digest"],
            roles=tuple(RolePlacement.from_spec(role) for role in value["roles"]),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> RoleDeployment:
        file = Path(path)

        def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ConfigurationError("duplicate role deployment field")
                result[key] = value
            return result

        def reject_constant(_: str) -> None:
            raise ConfigurationError("non-finite role deployment JSON")

        try:
            with file.open("rb") as source:
                document = source.read(65_537)
            if len(document) > 65_536:
                raise ConfigurationError("role deployment document exceeds 64 KiB")
            spec = json.loads(
                document.decode("utf-8"), object_pairs_hook=unique_pairs,
                parse_constant=reject_constant,
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ConfigurationError("invalid role deployment JSON") from exc
        return cls.from_spec(spec)

    def canonical_bytes(self) -> bytes:
        return json.dumps(self.to_spec(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()

    def digest(self) -> str:
        return hashlib.sha256(b"pllm.role_deployment.v1\0" + self.canonical_bytes()).hexdigest()

    def assess(self, graph: RoleGraph) -> PlacementAssessment:
        if type(graph) is not RoleGraph or self.graph_digest != graph.digest():
            raise ConfigurationError("role deployment graph digest does not match")
        operators = {role.role_id: role.operator for role in self.roles}
        violations = set(graph.separation_violations(operators))
        origins = {role.role_id: role.origin for role in self.roles}
        for left, right in graph.separate_operators:
            first, second = origins[left], origins[right]
            if first and second:
                first_host, second_host = urlsplit(first).hostname, urlsplit(second).hostname
                if first_host == second_host or (first_host in _LOOPBACK and second_host in _LOOPBACK):
                    violations.add((left, right))
        return PlacementAssessment(
            declared_separation_violations=tuple(sorted(violations)),
            roles_requiring_unverified_attestation=tuple(
                role.role_id for role in self.roles if role.hardware == "tee"
            ),
        )


__all__ = [
    "AttestationPolicy", "PlacementAssessment", "ROLE_DEPLOYMENT_SCHEMA",
    "RoleDeployment", "RolePlacement",
]
