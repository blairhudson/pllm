"""Strict whole-batch semantic-execution comparison, never subset selection."""
import argparse

from freeze_semantic import verify_lock
from oracle import HERE, read_json, strict_equal
from semantic_reference import expected


def check(path):
    verify_lock()
    regenerated = expected()
    strict_equal(regenerated, read_json(HERE / "semantic-expected.json"), "frozen-semantic-reference")
    strict_equal(regenerated, read_json(path), "native-semantic-output")
    return len(regenerated["results"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args = parser.parse_args()
    print(f"{check(args.output)} strict semantic execution exchanges match; original 5125-tensor exchange not claimed; producer provenance separate")
