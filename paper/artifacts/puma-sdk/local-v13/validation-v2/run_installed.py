"""Bound-intersection follow-up; original failed wrapper retained separately."""
import hashlib
import json
from pathlib import Path

base = Path("target/puma-trie-norm-validation-v1/run_installed.py")
source = base.read_bytes()
out = Path(__file__).resolve().parent
tests = [
    "tests/test_replicated_public_weights.py",
    "tests/test_replicated_policy_sdk.py",
    "tests/test_replicated_trie_sdk.py",
    "tests/test_replicated_worker.py",
    "tests/test_quality_outcomes_sdk.py",
]
text = source.decode().replace("puma-trie-norm-build-v1", "puma-trie-norm-build-v2")
start = text.index("tests = [")
end = text.index("\ninputs = {", start)
text = text[:start] + "tests = " + repr(tests) + "\n" + text[end:]
with (out / "base-harness.json").open("x") as stream:
    json.dump({"path": str(base), "sha256": hashlib.sha256(source).hexdigest(),
               "build": "target/puma-trie-norm-build-v2", "tests": tests}, stream, indent=2)
    stream.write("\n")
exec(compile(text, str(base), "exec"), {"__file__": __file__})
