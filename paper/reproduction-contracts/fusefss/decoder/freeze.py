"""Explicit bounded fixture freeze. --check never writes. No native execution."""

import argparse
from hashlib import sha256
from pathlib import Path

from generate import fixtures
from oracle import HERE, ROOT, canonical, read_json, run, strict_equal

SOURCES = [
    "paper/methods/DECISIONS.md",
    "paper/reproduction-contracts/shared-profile-resolution.md",
    "paper/reproduction-contracts/fusefss/gate-set/reference.py",
    "paper/reproduction-contracts/fusefss/gate-set/fixtures.json",
    "paper/reproduction-contracts/fusefss/gelu-table.json",
    "python/pllm/runtime/semantic_attention.py",
    "python/pllm/runtime/semantic_numeric.py",
    "python/pllm/runtime/tiny_llama.py",
]
LOCAL = ["oracle.py", "generate.py", "freeze.py", "check_outputs.py", "test_decoder.py",
         "README.md", "NUMERIC_DECISIONS.md", "EXCHANGE.md", "VALIDATION.md", "inputs.json", "expected.json",
         "fetch_gpt2_source.py", "gpt2-source/README.md", "gpt2-source/source-lock.json",
         "gpt2-source/repository-metadata.json", "gpt2-source/config.json",
         "gpt2-source/tokenizer_config.json", "gpt2-source/generation_config.json"]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def lock():
    paths = SOURCES + [str((HERE / name).relative_to(ROOT)) for name in LOCAL]
    return {"schema": "pllm.fusefss.decoder-source-lock.v1", "claim": "bounded-clear-d13-arithmetic-only",
            "sha256": {name: digest(ROOT / name) for name in paths}}


def verify_lock():
    strict_equal(lock(), read_json(HERE / "source-lock.json"), "source-lock")


def inputs():
    return {"schema": "pllm.fusefss.decoder-inputs.v1", "cases": fixtures()}


def expected(data):
    return {"schema": "pllm.fusefss.decoder-batch.v1", "results": [run(c) for c in data["cases"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--refresh-source-lock", action="store_true",
                      help="verify frozen payloads without writing them; refresh source hashes only")
    args = parser.parse_args()
    data = inputs()
    result = expected(data)
    if args.write:
        (HERE / "inputs.json").write_bytes(canonical(data))
        (HERE / "expected.json").write_bytes(canonical(result))
        (HERE / "source-lock.json").write_bytes(canonical(lock()))
    elif args.refresh_source_lock:
        before = {name: digest(HERE / name) for name in ("inputs.json", "expected.json", "oracle.py")}
        strict_equal(data, read_json(HERE / "inputs.json"), "inputs")
        strict_equal(result, read_json(HERE / "expected.json"), "expected")
        (HERE / "source-lock.json").write_bytes(canonical(lock()))
        after = {name: digest(HERE / name) for name in before}
        strict_equal(before, after, "unchanged-frozen-payloads")
    else:
        verify_lock()
        strict_equal(data, read_json(HERE / "inputs.json"), "inputs")
        strict_equal(result, read_json(HERE / "expected.json"), "expected")
    print(f"{len(data['cases'])} bounded D13 cases; {sum(len(r['tensors']) for r in result['results'])} captured tensors; no native replay")


if __name__ == "__main__":
    main()
