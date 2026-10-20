"""Strict future native exchange checker. Does not start a native producer."""

import argparse

from freeze import expected, verify_lock
from oracle import HERE, read_json, strict_equal


def check(path):
    verify_lock()
    inputs = read_json(HERE / "inputs.json")
    regenerated = expected(inputs)
    strict_equal(regenerated, read_json(HERE / "expected.json"), "frozen-expected")
    strict_equal(regenerated, read_json(path), "producer-output")
    return len(regenerated["results"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="producer JSON batch (not a replay claim)")
    args = parser.parse_args()
    print(f"{check(args.output)} exact D13 decoder exchanges match; producer provenance not established by this check")
