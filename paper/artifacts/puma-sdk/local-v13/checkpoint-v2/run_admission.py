"""Pinned source publication through the ordinary SDK; no role execution."""
import hashlib
import json
from pathlib import Path

base = Path("paper/artifacts/puma-sdk/local-v10/validation/run_validation.py")
source = base.read_bytes()
out = Path(__file__).resolve().parent
replacements = {
    "puma-attention-range-build-v1": "puma-trie-norm-build-v2",
    'experiment = pllm.Experiment.from_spec(saved["experiment"])':
        'experiment = pllm.Experiment.from_spec(saved["experiment"]).with_params('
        'pipeline__components__linear__range_policy="checked", '
        'pipeline__components__linear__max_abs_activation=256, '
        'pipeline__components__linear__lookup="trie", '
        'pipeline__components__linear__selection="tiled", '
        'pipeline__components__linear__peer_transport="streams")',
    '"memory_preflight": report, "wheel_sha256": sha(wheel),':
        '"memory_preflight": report, "wheel_sha256": sha(wheel), '
        '"execution_policy": {"range_policy":"checked", "max_abs_activation":256, '
        '"lookup":"trie", "selection":"tiled", "peer_transport":"streams"},',
}
with (out / "base-harness.json").open("x") as stream:
    json.dump({"path": str(base), "sha256": hashlib.sha256(source).hexdigest(),
               "replacements": replacements, "scope": "checkpoint_admission only"},
              stream, indent=2)
    stream.write("\n")
text = source.decode().split("tests = [", 1)[0]
for before, after in replacements.items():
    assert before in text, before
    text = text.replace(before, after)
text += '\ntry:\n    checkpoint_admission()\nfinally:\n    installed("after")\n'
exec(compile(text, str(base), "exec"), {"__file__": __file__})
