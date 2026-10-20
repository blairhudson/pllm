"""Freeze/check distinct strict semantic schema; never write original payloads."""
import argparse
from hashlib import sha256

from oracle import HERE, ROOT, canonical, read_json, strict_equal
from semantic_reference import expected

LOCAL = ["semantic_reference.py", "freeze_semantic.py", "check_semantic.py", "test_semantic.py",
         "SEMANTIC_EXCHANGE.md", "SEMANTIC_VALIDATION.md", "semantic-graphs.json", "semantic-expected.json", "inputs.json",
         "expected.json", "oracle.py", "source-lock.json"]


def original_unchanged():
    baseline = read_json(HERE / "source-lock.json")["sha256"]
    for name in ("inputs.json", "expected.json", "oracle.py"):
        path = HERE / name
        if sha256(path.read_bytes()).hexdigest() != baseline[str(path.relative_to(ROOT))]:
            raise ValueError("original frozen payload/arithmetic changed: " + name)


def lock():
    files = [HERE / name for name in LOCAL]
    # Freeze the independent oracle, not the implementation under test.
    return {"schema": "pllm.fusefss.semantic-source-lock.v1", "native_replay_or_build_identity_claimed": False,
            "sha256": {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest() for path in files}}


def verify_lock():
    from freeze import verify_lock as original_lock
    original_unchanged()
    original_lock()
    strict_equal(lock(), read_json(HERE / "semantic-source-lock.json"), "semantic-source-lock")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    original_unchanged()
    result = expected()
    if args.write:
        (HERE / "semantic-expected.json").write_bytes(canonical(result))
        (HERE / "semantic-source-lock.json").write_bytes(canonical(lock()))
    else:
        verify_lock()
        strict_equal(result, read_json(HERE / "semantic-expected.json"), "semantic-expected")
    print(f"{len(result['results'])} fixed semantic graphs; {sum(len(r['nodes']) for r in result['results'])} complete node observations; no native replay")


if __name__ == "__main__":
    main()
