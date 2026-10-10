"""Internal native instruction ownership, preceding a complete native client.

An explicit trusted artifact digest is required. Bundle validation, all tensor
kernels, transport, Python scheduling and dependency imports remain charged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pllm import _native


def execution_build_digest():
    digest = hashlib.sha256(b"pllm.execution_capsule_build.v1\0")
    with Path(_native.__file__).open("rb") as source:
        digest.update(hashlib.file_digest(source, "sha256").digest())
    root = Path(__file__).resolve().parents[1]
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        with path.open("rb") as source:
            digest.update(hashlib.file_digest(source, "sha256").digest())
    return digest.hexdigest()


def publish_execution_capsule(compiled):
    compiled.validate()
    return _native.ExecutionCapsule.compile(compiled._plan.canonical_bytes(),
        compiled._canonical_composition, execution_build_digest(), compiled.digest)


class CapsuleExecution:
    def __init__(self, runtime, owner):
        metadata = json.loads(owner.metadata())
        self.owner, self.facts = owner, metadata["facts"]
        for phase in ("prefill", "decode"):
            self.facts[phase]["native_phase"] = hashlib.sha256(json.dumps({
                "source_plan": runtime._source_plan.digest, "graph": runtime._graphs[phase],
                "schedule": runtime._schedule[phase],
            }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        runtime._graphs, runtime._schedule = metadata["graphs"], metadata["schedule"]

    def frames(self, phase):
        index = 0 if phase == "prefill" else 1
        for step in range(self.owner.step_count(index)):
            yield json.loads(self.owner.step(index, step))


def capsule_runtime(compiled, remote, payload, expected_digest):
    """Verify a trusted offline capsule before selecting native metadata ownership."""
    owner = _native.ExecutionCapsule(payload, expected_digest, execution_build_digest(),
        compiled.digest, compiled._plan.digest, _composition_digest(compiled))
    runtime = compiled.runtime(remote)
    if (runtime._retained is not None or runtime._mlp_tiling is not None or runtime._tiered is not None
            or runtime._verification_bits or runtime.nonlinear_evaluator is not None
            or runtime._window_contracts or runtime._hybrid_contracts or runtime._text_only_tokens
            or runtime._generated_prefix_contract is not None):
        runtime.reset()
        raise ValueError("execution capsule requires plain dense full-KV execution")
    runtime._capsule = CapsuleExecution(runtime, owner)
    return runtime


def _composition_digest(compiled):
    from pllm.configuration import Pipeline
    return Pipeline.from_spec(json.loads(compiled._canonical_composition)).digest()
