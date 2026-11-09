"""Bounded observations, not process ownership or permission to reap.

The raw working-set estimate covers topology/dashboard/report views, worker
encoding and controller decoding/canonical copies. It is not measured or hard
RSS, and ordinary caller-owned BenchmarkRun dictionaries are not leased here.
"""
import json
from dataclasses import dataclass, replace


RAW_PEAK_BYTES = 1 << 20
MAX_ROLES = 16
MAX_OWNERS = 32
MAX_ASSOCIATIONS = 64
MAX_ROLE_BYTES = 64
MAX_SUMMARY_BYTES = 32 << 10
SCOPE = "configured_topology_owned_providers"
MEMORY_SCOPE = "conservative raw working-set estimate; not measured or hard RSS; client only"
UNAVAILABLE = b'{"available":false,"scope":"configured_topology_owned_providers","observation_complete":false,"topology_close_outcome":"incomplete"}'


def _integer(value, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("retirement integer outside bound")
    return value


def _role(value):
    # Check character count before allocating bounded UTF-8 bytes.
    if type(value) is not str or not 1 <= len(value) <= MAX_ROLE_BYTES:
        raise ValueError("retirement role outside bound")
    if len(value.encode("utf-8")) > MAX_ROLE_BYTES:
        raise ValueError("retirement role encoding outside bound")
    return value


def _encoded(value):
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_SUMMARY_BYTES:
        raise ValueError("retirement summary outside bound")
    return encoded


@dataclass(frozen=True, slots=True)
class _Owner:
    ordinal: int
    roles: tuple[str, ...]
    initialized: bool
    pid: int | None
    returncode: int | None = None
    observation_kind: str | None = None


class Observation:
    """Ephemeral reconciliation of scalar identities only; never retains owners."""
    __slots__ = ("roles", "owners", "failed")

    def __init__(self, roles, published, pending, ready):
        if (type(roles) not in (tuple, list) or type(published) is not dict
                or type(pending) is not dict or type(ready) not in (set, frozenset)):
            raise ValueError("invalid retirement collection")
        if (len(roles) > MAX_ROLES or len(published) > MAX_OWNERS
                or len(pending) > MAX_OWNERS
                or len(published) + len(pending) > MAX_ASSOCIATIONS or len(ready) > MAX_ROLES):
            raise ValueError("retirement collection outside bound")
        self.roles = tuple(_role(role) for role in roles)
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("duplicate configured role")
        if any(_role(role) not in self.roles for role in ready):
            raise ValueError("unconfigured initialized role")
        self.owners = {}
        self.failed = False
        associations = 0

        def add(role, process, initialized):
            nonlocal associations
            role = _role(role)
            if role not in self.roles or type(initialized) is not bool:
                raise ValueError("unconfigured retirement owner")
            identity = id(process)
            owner = self.owners.get(identity)
            if owner is None:
                if len(self.owners) >= MAX_OWNERS:
                    raise ValueError("too many retirement owners")
                pid = object.__getattribute__(process, "__dict__").get("pid") if initialized else None
                if pid is not None:
                    _integer(pid, 0, (1 << 64) - 1)
                owner = _Owner(len(self.owners), (), initialized, pid)
            if role not in owner.roles:
                associations += 1
                if associations > MAX_ASSOCIATIONS:
                    raise ValueError("too many retirement associations")
                owner = replace(owner, roles=(*owner.roles, role))
            # Unready construction remains unknown even if also published.
            owner = replace(owner, initialized=owner.initialized and initialized)
            self.owners[identity] = owner

        for role, process in published.items():
            add(role, process, True)
        for role, owner in pending.items():
            if owner.role != role:
                raise ValueError("retirement owner role mismatch")
            add(role, owner.process, role in ready)

    def record(self, identity, returncode, kind):
        if returncode is None:
            return
        _integer(returncode, -(1 << 63), (1 << 63) - 1)
        owner = self.owners[identity]
        if not owner.initialized or kind not in ("poll", "wait"):
            raise ValueError("invalid retirement observation")
        self.owners[identity] = replace(owner, returncode=returncode, observation_kind=kind)

    def finish(self, outcome):
        if self.failed or outcome not in ("incomplete", "error", "completed"):
            return UNAVAILABLE
        owners = [{"ordinal": owner.ordinal, "roles": list(owner.roles),
                   "initialized": owner.initialized, "pid": owner.pid,
                   "state": "retired" if owner.returncode is not None else "unknown",
                   "returncode": owner.returncode, "observation_kind": owner.observation_kind}
                  for owner in self.owners.values()]
        complete = (all(owner.initialized and owner.returncode is not None for owner in self.owners.values())
                    and set(self.roles) == {role for owner in self.owners.values() for role in owner.roles})
        return _encoded({"available": True, "scope": SCOPE, "configured_roles": list(self.roles),
                         "owners": owners, "observation_complete": complete,
                         "topology_close_outcome": outcome})


def report(snapshot, dashboard_outcome="incomplete", *, topology_error=False):
    """Transfer only bounded scalars. Malformed/unavailable diagnostics fail closed."""
    try:
        if type(snapshot) is not bytes or len(snapshot) > MAX_SUMMARY_BYTES:
            raise ValueError("invalid retirement snapshot")
        value = json.loads(snapshot)
        if value.get("available") is not True:
            raise ValueError("retirement snapshot unavailable")
        roles, owners = value["configured_roles"], value["owners"]
        if (type(roles) is not list or len(roles) > MAX_ROLES
                or type(owners) is not list or len(owners) > MAX_OWNERS):
            raise ValueError("invalid retirement counts")
        roles = [_role(role) for role in roles]
        if len(set(roles)) != len(roles) or value["scope"] != SCOPE:
            raise ValueError("invalid configured retirement roles")
        associations, ordinals, covered = 0, set(), set()
        for owner in owners:
            ordinal = _integer(owner["ordinal"], 0, (1 << 64) - 1)
            if ordinal in ordinals or type(owner["initialized"]) is not bool:
                raise ValueError("invalid retirement owner identity")
            ordinals.add(ordinal)
            associated = owner["roles"]
            if type(associated) is not list or not associated or len(associated) > MAX_ROLES:
                raise ValueError("invalid retirement owner roles")
            associations += len(associated)
            if associations > MAX_ASSOCIATIONS:
                raise ValueError("invalid retirement association count")
            associated = [_role(role) for role in associated]
            if len(set(associated)) != len(associated) or any(role not in roles for role in associated):
                raise ValueError("invalid retirement role association")
            covered.update(associated)
            if owner["pid"] is not None:
                _integer(owner["pid"], 0, (1 << 64) - 1)
            if owner["returncode"] is not None:
                _integer(owner["returncode"], -(1 << 63), (1 << 63) - 1)
                if (not owner["initialized"] or owner["state"] != "retired"
                        or owner["observation_kind"] not in ("poll", "wait")):
                    raise ValueError("invalid terminal retirement observation")
            elif owner["state"] != "unknown" or owner["observation_kind"] is not None:
                raise ValueError("invalid unknown retirement observation")
        complete = covered == set(roles) and all(owner["initialized"] and owner["returncode"] is not None
                                               for owner in owners)
        if type(value["observation_complete"]) is not bool or value["observation_complete"] != complete:
            raise ValueError("invalid retirement completeness")
        if value["topology_close_outcome"] not in ("incomplete", "error", "completed"):
            raise ValueError("invalid topology close outcome")
        if dashboard_outcome not in ("incomplete", "error", "completed"):
            raise ValueError("invalid dashboard close outcome")
        if topology_error:
            value["topology_close_outcome"] = "error"
        value["dashboard_topology_close_outcome"] = dashboard_outcome
        # Strip unexpected fields as well as validating ordinary scalar fields.
        value = {key: value[key] for key in ("available", "scope", "configured_roles", "owners",
                 "observation_complete", "topology_close_outcome", "dashboard_topology_close_outcome")}
        value["owners"] = [{key: owner[key] for key in ("ordinal", "roles", "initialized", "pid",
                           "state", "returncode", "observation_kind")} for owner in owners]
        _encoded(value)
        return value
    except BaseException:
        return {"available": False, "scope": SCOPE, "observation_complete": False,
                "topology_close_outcome": "error" if topology_error else "incomplete",
                "dashboard_topology_close_outcome": dashboard_outcome
                if dashboard_outcome in ("incomplete", "error", "completed") else "incomplete"}
