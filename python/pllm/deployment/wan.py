"""Immutable consumer access-link scenarios, in decimal megabits per second."""
from __future__ import annotations

from dataclasses import dataclass

from .network import PublicRecord, NetworkError, LinkConditions, finite, identity


def _rate(value, name):
    finite(value, name, minimum=0.001)
    if value > 1_000_000:
        raise NetworkError(f"{name} exceeds 1000000 Mbps")


@dataclass(frozen=True, slots=True)
class PartyAccess(PublicRecord):
    SCHEMA = "pllm.party_access.v1"
    party_id: str
    download_mbps: float = 100
    upload_mbps: float = 40

    def __post_init__(self):
        identity(self.party_id, "party_id")
        _rate(self.download_mbps, "download_mbps")
        _rate(self.upload_mbps, "upload_mbps")
        object.__setattr__(self, "download_mbps", float(self.download_mbps))
        object.__setattr__(self, "upload_mbps", float(self.upload_mbps))

    @classmethod
    def from_spec(cls, value):
        return cls(**cls._fields(value))


@dataclass(frozen=True, slots=True)
class WanConditions(PublicRecord):
    """Full-duplex access capacities shared across each party's concurrent links.

    Every role is a separate hypothetical WAN party unless explicitly mapped.
    Co-located roles mapped to the same party incur no WAN transfer between them.
    This public scenario is declarative. Explicit local benchmark emulation or
    build_roles(wan=...) enforces the capacities; it does not establish independence.
    """

    SCHEMA = "pllm.wan_conditions.v1"
    download_mbps: float = 100
    upload_mbps: float = 40
    parties: tuple[PartyAccess, ...] = ()
    role_parties: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        _rate(self.download_mbps, "download_mbps")
        _rate(self.upload_mbps, "upload_mbps")
        object.__setattr__(self, "download_mbps", float(self.download_mbps))
        object.__setattr__(self, "upload_mbps", float(self.upload_mbps))
        if (type(self.parties) is not tuple or len(self.parties) > 128
                or any(type(p) is not PartyAccess for p in self.parties)
                or len({p.party_id for p in self.parties}) != len(self.parties)):
            raise NetworkError("parties must be a bounded tuple of unique PartyAccess records")
        if type(self.role_parties) is not tuple or len(self.role_parties) > 128:
            raise NetworkError("role_parties must be a bounded tuple")
        for pair in self.role_parties:
            if type(pair) is not tuple or len(pair) != 2:
                raise NetworkError("role_parties requires role/party pairs")
            identity(pair[0], "role")
            identity(pair[1], "party")
        if len({pair[0] for pair in self.role_parties}) != len(self.role_parties):
            raise NetworkError("duplicate role mapping")

    def party_for(self, role: str) -> str:
        identity(role, "role")
        return dict(self.role_parties).get(role, role)

    def access(self, party_id: str) -> PartyAccess:
        return next((p for p in self.parties if p.party_id == party_id),
                    PartyAccess(party_id, self.download_mbps, self.upload_mbps))

    def link_conditions(self, source: str, destination: str, *, latency_ms: float = 0):
        """Single-flow rate for the existing shaper; not shared-capacity emulation."""
        a, b = self.party_for(source), self.party_for(destination)
        if a == b:
            raise NetworkError("an intra-party edge has no WAN link")
        rate = min(self.access(a).upload_mbps, self.access(b).download_mbps)
        return LinkConditions(latency_ms=latency_ms, bytes_per_second=int(rate * 1_000_000 / 8))

    @classmethod
    def from_spec(cls, value):
        data = cls._fields(value)
        if type(data["parties"]) is not list or type(data["role_parties"]) is not list:
            raise NetworkError("parties and role_parties must be arrays")
        data["parties"] = tuple(PartyAccess.from_spec(p) for p in data["parties"])
        if any(type(p) is not list for p in data["role_parties"]):
            raise NetworkError("role_parties must contain arrays")
        data["role_parties"] = tuple(tuple(p) for p in data["role_parties"])
        return cls(**data)
