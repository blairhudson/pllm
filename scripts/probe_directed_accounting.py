"""Conserve Docker service IP counters against recorded application bodies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def check(path):
    report = json.loads(Path(path).read_text())
    snapshots = report["docker_accounting"]["samples"]
    final = (snapshots["runs"] or snapshots["warmups"])[-1]["after"]
    ip = final["directed_ip"]
    bodies = report["summary"]["total_accounted_benchmark_body_bytes"]
    assert ip["bytes"] == sum(row["bytes"] for row in ip["links"].values())
    assert ip["bytes"] >= bodies, "service IP counters undercount conserved application bodies"
    return {"report": Path(path).name, "application_bodies": bodies,
            "service_ip_hook_bytes": ip["bytes"], "directed_links": ip["links"],
            "ip_above_body_bytes": ip["bytes"] - bodies,
            "scope": "non-overlapping IPv4/TCP kernel hooks; not complete physical wire",
            "measurement_helpers": final["measurement_helpers"], "conservation_passed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = [check(path) for path in arguments.reports]
    Path(arguments.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
