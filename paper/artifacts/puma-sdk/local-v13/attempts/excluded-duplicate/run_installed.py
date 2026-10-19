"""Bound-intersection follow-up using the frozen full-suite driver."""
import ast
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
tree = ast.parse(text)
assignments = [node for node in tree.body if (
    isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "tests"
                                      for target in node.targets)
    or isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "tests"
)]
assert len(assignments) == 2
lines = text.splitlines(keepends=True)
for node in reversed(assignments):
    lines[node.lineno - 1:node.end_lineno] = ["tests = " + repr(tests) + "\n"] if node is assignments[0] else []
text = "".join(lines)
ast.parse(text)
with (out / "base-harness.json").open("x") as stream:
    json.dump({"path": str(base), "sha256": hashlib.sha256(source).hexdigest(),
               "build": "target/puma-trie-norm-build-v2", "tests": tests,
               "executed_harness_sha256": hashlib.sha256(text.encode()).hexdigest()}, stream, indent=2)
    stream.write("\n")
exec(compile(text, str(base), "exec"), {"__file__": __file__})
