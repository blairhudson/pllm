"""Bandwidth-only WAN floors from non-overlapping directed application bodies."""
from __future__ import annotations

from collections.abc import Mapping
import math

from pllm.deployment.wan import WanConditions


def _placement(record, conditions):
    audit = record.get("network_audit") if isinstance(record, Mapping) else None
    if audit is None:
        return {}
    if not isinstance(audit, Mapping) or type(audit.get("placement")) is not list or len(audit["placement"]) > 128:
        raise ValueError("invalid WAN network placement")
    result = {}
    declared = dict(conditions.role_parties)
    for row in audit["placement"]:
        if not isinstance(row, Mapping) or set(row) != {"role_id", "party_id"}:
            raise ValueError("invalid WAN network placement row")
        role, party = row["role_id"], row["party_id"]
        conditions.party_for(role)
        conditions.access(party)
        if role in result or role in declared and declared[role] != party:
            raise ValueError("WAN role mapping conflicts with executed network placement")
        result[role] = party
    return result


def _window(ledgers, conditions, *, online=False, placements=None):
    if not ledgers or any(not isinstance(x, Mapping)
            or x.get("tracked_body_counter_set_present") is not True
            or type(x.get("body_bytes_by_edge")) is not list for x in ledgers):
        return None
    edges = {}
    placements = placements if placements is not None else [{} for _ in ledgers]
    if len(placements) != len(ledgers):
        return None
    for ledger, placement in zip(ledgers, placements, strict=True):
        rows = ledger["body_bytes_by_edge"]
        if len(rows) > 512:
            raise ValueError("WAN edge count exceeds bound")
        total, online_total = 0, 0
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {
                "source", "destination", "phase", "serialized_body_bytes"
            }:
                raise ValueError("invalid directed WAN body record")
            size = row["serialized_body_bytes"]
            if type(size) is not int or not 0 <= size <= 2**63 - 1:
                raise ValueError("WAN byte count must be a bounded nonnegative integer")
            total += size
            online_total += size if row["phase"] == "online" else 0
            if row["phase"] not in {"cold", "offline", "online", "bundle", "setup", "teardown"}:
                raise ValueError("invalid WAN ledger phase")
            a = placement.get(row["source"], conditions.party_for(row["source"]))
            b = placement.get(row["destination"], conditions.party_for(row["destination"]))
            if not online or row["phase"] == "online":
                edges[a, b] = edges.get((a, b), 0) + size
        if (total != ledger.get("all_link_serialized_body_bytes")
                or online_total != ledger.get("online_all_link_serialized_body_bytes")):
            return None  # Never make a partial ledger look like a network floor.
    parties = {}
    local = 0
    for (a, b), size in sorted(edges.items()):
        if a == b:
            local += size
            continue
        for p in (a, b):
            parties.setdefault(p, {"upload_bytes": 0, "download_bytes": 0})
        parties[a]["upload_bytes"] += size
        parties[b]["download_bytes"] += size
    bottleneck, floor = None, 0.0
    for party, record in sorted(parties.items()):
        access = conditions.access(party)
        for direction in ("upload", "download"):
            rate = getattr(access, f"{direction}_mbps")
            seconds = record[f"{direction}_bytes"] * 8 / (rate * 1_000_000)
            record[f"{direction}_mbps"] = rate
            record[f"{direction}_seconds"] = seconds
            if seconds > floor:
                floor, bottleneck = seconds, {"party_id": party, "direction": direction}
    return {"minimum_transfer_seconds": floor, "bottleneck": bottleneck,
            "parties": parties, "wan_body_bytes": sum(edges.values()) - local,
            "intra_party_body_bytes": local}


def _measured_emulation(report, conditions):
    configuration = report.get("configuration", {})
    if configuration.get("wan_emulation") is not True:
        return None
    if configuration.get("wan_emulation_digest") != conditions.digest:
        # Re-price an existing report without relabeling its timings as a run
        # under the newly hypothesized capacities.
        return None
    roles = configuration.get("roles", [])
    if not roles or "client" not in roles:
        raise ValueError("WAN measurement lacks its executed roles")
    assignments = {role: conditions.party_for(role) for role in roles}
    expected = set(assignments.values())
    samples = (report.get("docker_accounting") or {}).get("samples")
    records = report.get("runs", [])
    if (not isinstance(samples, Mapping) or not isinstance(samples.get("runs"), list)
            or not isinstance(samples.get("warmups"), list)
            or len(samples["runs"]) != len(records)
            or len(samples["warmups"]) != len(report.get("warmup_runs", []))):
        raise ValueError("WAN measurement lacks matched kernel sampling windows")

    def check(value):
        if roles == ["client"]:
            return
        data = value.get("wan_emulation", {}) if isinstance(value, Mapping) else {}
        if (data.get("enforced") is not True or data.get("conditions_digest") != conditions.digest
                or data.get("role_parties") != assignments or set(data.get("parties", {})) != expected):
            raise ValueError("WAN kernel sampling does not cover the executed parties")
        from pllm.deployment import LinkConditions
        link = data.get("link_conditions")
        link_digest = LinkConditions.from_spec(link).digest if link is not None else None
        if link_digest != configuration.get("link_conditions_digest"):
            raise ValueError("WAN kernel delay/loss profile does not match the report")
        for party, caps in data["parties"].items():
            for direction in ("upload", "download"):
                queue = caps.get(direction, {})
                rate = int(getattr(conditions.access(party), f"{direction}_mbps") * 1_000_000 / 8)
                roots = [q for q in queue.get("qdiscs", []) if q.get("root")]
                if (queue.get("verified") is not True or queue.get("bytes_per_second") != rate
                        or len(roots) != 1 or roots[0].get("kind") != "tbf"
                        or roots[0].get("options", {}).get("rate") != rate):
                    raise ValueError("WAN kernel rate differs from the selected party capacity")
                _check_link_qdisc(queue.get("qdiscs", []), link if direction == "upload" else None)
    check(samples.get("startup"))
    for window in [*samples["warmups"], *samples["runs"]]:
        check(window.get("before"))
        check(window.get("after"))

    def duration(record, name):
        value = record.get("durations", {}).get(name)
        return value if type(value) in (int, float) and math.isfinite(value) and value > 0 else None

    rows = []
    for record in records:
        token = record.get("tokens", {})
        n = token.get("output_tokens")
        if (record.get("status") != "completed" or token.get("authoritative") is not True
                or type(n) is not int or n < 1):
            n = None
        full, online, decode = [duration(record, name) for name in
                                ("full_seconds", "online_seconds", "generation_seconds")]
        after_first = n - 1 if n is not None and n > 1 else None
        rows.append({"generated_output_tokens": n, "decode_output_tokens": after_first,
                     "full_seconds": full, "online_seconds": online, "decode_seconds": decode,
                     "end_to_end_tokens_per_second": n / full if n and full else None,
                     "online_tokens_per_second": n / online if n and online else None,
                     "decode_tokens_per_second": after_first / decode if after_first and decode else None})
    summary = {}
    for label, tokens, seconds in (("end_to_end", "generated_output_tokens", "full_seconds"),
                                  ("online", "generated_output_tokens", "online_seconds"),
                                  ("decode", "decode_output_tokens", "decode_seconds")):
        valid = rows and all(row[tokens] is not None and row[seconds] is not None for row in rows)
        summary[f"{label}_tokens_per_second"] = (
            sum(row[tokens] for row in rows) / sum(row[seconds] for row in rows) if valid else None)
    return {"schema": "pllm.measured_wan_throughput.v1",
            "backend": "client-only-no-wan-links" if roles == ["client"] else "linux-tbf-routed-party-ports",
            "scope": "measured local inference under enforced shared party access rates",
            "kernel_rate_snapshots_checked": roles != ["client"], "conditions_digest": conditions.digest,
            "inter_party_links_present": len(expected) > 1,
            "request_scope": "demand preparation and request execution; excludes provider startup and warmups",
            "decode_scope": "first-to-last output interval with N-1 authoritative outputs",
            "internet_measurement": False, "runs": rows, "summary": summary}


def _check_link_qdisc(queues, conditions):
    """Check netem JSON against an optional immutable egress-delay/loss profile."""
    from pllm.deployment import LinkConditions
    netem = [row for row in queues if row.get("kind") == "netem"]
    if conditions is None:
        if netem:
            raise ValueError("WAN kernel has unselected delay/loss")
        return
    shape = LinkConditions.from_spec(conditions)
    if shape.bytes_per_second is not None or len(netem) != 1 or netem[0].get("parent") != "10:1":
        raise ValueError("WAN kernel delay/loss queue differs from its profile")
    options = netem[0].get("options", {})
    delay = options.get("delay", {})
    loss = options.get("loss-random", {})
    if (not math.isclose(delay.get("delay", 0), shape.latency_ms / 1000, abs_tol=1e-6)
            or delay.get("jitter", 0) != 0 or delay.get("correlation", 0) != 0
            or not math.isclose(loss.get("loss", 0), shape.loss_fraction, abs_tol=1e-8)
            or loss.get("correlation", 0) != 0
            or shape.loss_fraction and options.get("seed") != shape.seed
            or any(key in options for key in ("rate", "loss-state", "loss-gemodel"))):
        raise ValueError("WAN kernel delay/loss differs from its profile")


def wan_readiness(report, conditions: WanConditions | None = None) -> dict:
    """Estimate capacity floors, never label a loopback timing as WAN latency.

    A shared access port is charged for all concurrent peers together. Full-duplex
    up/down capacities are separate. Missing counters remain unknown; startup
    and warmups are charged once. RTT, packet overhead, loss/retries, dependency
    rounds, server compute and checkpoint distribution are outside this floor.
    """
    conditions = conditions if conditions is not None else WanConditions()
    if type(conditions) is not WanConditions or not isinstance(report, Mapping):
        raise TypeError("WAN analysis requires a report and WanConditions")
    if report.get("schema_version") != "pllm.loopback_benchmark.v1":
        raise ValueError("WAN readiness requires a canonical loopback report")
    records, warmups = report.get("runs", []), report.get("warmup_runs", [])
    if type(records) is not list or type(warmups) is not list or len(records) + len(warmups) > 1024:
        raise ValueError("invalid bounded WAN run windows")
    accounting = report.get("topology_accounting") or {}
    if not isinstance(accounting, Mapping):
        raise ValueError("invalid WAN accounting")
    ledgers, warm_ledgers = accounting.get("runs", []), accounting.get("warmups", [])
    if type(ledgers) is not list or type(warm_ledgers) is not list:
        raise ValueError("invalid WAN accounting windows")
    aligned = len(ledgers) == len(records) and len(warm_ledgers) == len(warmups)
    tokens = []
    results = []
    placements = [_placement(r, conditions) for r in records]
    warm_placements = [_placement(r, conditions) for r in warmups]
    for i, record in enumerate(records):
        if not isinstance(record, Mapping) or not isinstance(record.get("tokens", {}), Mapping):
            raise ValueError("invalid WAN run record")
        value = record.get("tokens", {})
        n = value.get("output_tokens")
        if record.get("status") != "completed" or value.get("authoritative") is not True or type(n) is not int or n < 1:
            n = None
        tokens.append(n)
        online = _window([ledgers[i]], conditions, online=True, placements=[placements[i]]) if aligned else None
        results.append({"covered": _window([ledgers[i]], conditions, placements=[placements[i]]) if aligned else None,
                        "online": online, "generated_output_tokens": n, "executed_role_parties": placements[i] or None})
    all_tokens = sum(tokens) if tokens and all(n is not None for n in tokens) else None
    online = _window(ledgers, conditions, online=True, placements=placements) if aligned else None
    startup_placement = next(iter(warm_placements or placements), {})
    setup = _window([accounting.get("startup"), *warm_ledgers, *ledgers], conditions,
                    placements=[startup_placement, *warm_placements, *placements]) if aligned else None
    def ceiling(window):
        floor = window["minimum_transfer_seconds"] if window else None
        return all_tokens / floor if all_tokens and floor else None
    return {"schema": "pllm.wan_readiness.v1", "conditions": conditions.to_spec(),
            "conditions_digest": conditions.digest, "rate_unit": "Mbps = 1000000 bits/second; 8 bits/byte",
            "scope": "analytic access-capacity floor on covered application bodies; not measured WAN latency",
            "assumptions": ["independent full-duplex upload/download ports", "each party's peers share its access port"],
            "unmeasured": ["RTT and dependent protocol rounds", "headers/TLS/ACKs, loss and retries", "checkpoint distribution", "WAN compute/queueing latency"],
             "measured_wan_latency_seconds": None, "internet_ready": None,
             "emulation": _measured_emulation(report, conditions),
            "runs": results, "summary": {"online": online, "setup_inclusive": setup,
                "online_generated_tokens_per_second_ceiling": ceiling(online),
                "setup_inclusive_generated_tokens_per_second_ceiling": ceiling(setup)}}
