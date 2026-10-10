import copy
import itertools

import numpy as np
import pytest

from pllm.metrics import BuildRef
from pllm.runtime.stage_kernels import compile_stage_kernels, stage_groups, StageKernelExecutor
from test_batched_greedy_verification import fixture
from test_decoder_continuation import runtime_for


def profile_for(compiled, engine):
    return {"schema": "pllm.stage_kernel_profile.v1", "model_plan_digest": compiled.model_plan_digest,
        "schedule_digest": compiled.runtime_schedule_digest, "binding_digest": compiled.digest,
        "build_digest": BuildRef.capture().expected_digest, "prefill_rows": 4, "decode_steps": 3,
        "profiles": [{"weights": list(weights), "cpu_prefill_ns": 1000 + index * 93, "cpu_decode_ns": 100,
                      "metal_prefill_ns": 200 + index * 17, "metal_decode_ns": 150}
                     for index, weights in enumerate(stage_groups(compiled, engine))]}


def test_native_frontier_matches_exhaustive_joint_budget(fixture):
    engine, compiled, _ = fixture
    groups, profile = stage_groups(compiled, engine), profile_for(compiled, engine)
    sizes = [stage.spec.in_features * stage.spec.out_features + 32 * (stage.spec.in_features + stage.spec.out_features)
             for _, stage in groups.values()]
    budget = sum(sorted(sizes)[-2:])
    best = None
    assert len(sizes) <= 16
    for selections in itertools.product((0, 1), repeat=len(sizes)):
        memory = sum(size * choice for size, choice in zip(sizes, selections, strict=True))
        cost = sum((row["metal_prefill_ns"] if choice else row["cpu_prefill_ns"]) + row["cpu_decode_ns"] * 3
                   for row, choice in zip(profile["profiles"], selections, strict=True))
        if memory <= budget and (best is None or (cost, memory) < best):
            best = cost, memory
    plan = compile_stage_kernels(compiled, profile, memory_bytes=budget)
    assert (plan["estimated_kernel_ns"], plan["device_bytes"]) == best
    assert all(row["decode"] == "cpu" for row in plan["choices"])
    zero = compile_stage_kernels(compiled, profile, memory_bytes=0)
    assert zero["device_bytes"] == 0 and all(row["prefill"] == "cpu" for row in zero["choices"])
    for change in ({"build_digest": "0" * 64}, {"binding_digest": "0" * 64},
                   {"profiles": profile["profiles"][:-1]}, {"prefill_rows": 10000}):
        with pytest.raises(ValueError):
            compile_stage_kernels(compiled, profile | change, memory_bytes=budget)
    forged = copy.deepcopy(profile)
    forged["profiles"][0]["cpu_prefill_ns"] = 0
    with pytest.raises(ValueError, match="timing"):
        compile_stage_kernels(compiled, forged, memory_bytes=budget)


def test_mixed_stage_devices_preserve_complete_decoder_and_burn_failure(fixture):
    pytest.importorskip("mlx.core")
    engine, compiled, _ = fixture
    reference, _ = runtime_for(compiled, engine)
    executor = StageKernelExecutor(compiled, engine, profile_for(compiled, engine), memory_bytes=64 << 10)
    candidate = compiled.runtime(executor)
    try:
        _, expected, _ = reference.prepare_ids([1, 2, 4, 7])
        _, actual, _ = candidate.prepare_ids([1, 2, 4, 7])
        for _ in range(3):
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            for a, b in zip(candidate.caches, reference.caches, strict=True):
                np.testing.assert_array_equal(a.key[:a.length].view(np.uint32), b.key[:b.length].view(np.uint32))
                np.testing.assert_array_equal(a.value[:a.length].view(np.uint32), b.value[:b.length].view(np.uint32))
            token = int(np.argmax(expected))
            actual, expected = candidate.forward_ids([token])[-1], reference.forward_ids([token])[-1]
        assert executor.counts["prefill_metal_calls"] > 0
        assert executor.counts["prefill_cpu_calls"] > 0
        assert executor.counts["decode_cpu_calls"] > 0
        with pytest.raises(ValueError, match="geometry"):
            stage_id = next(iter(executor.stages))
            executor(stage_id, np.zeros((2, 1), np.float32))
        assert executor.closed and not executor.gpu
    finally:
        executor.close()
        candidate.reset()
        reference.reset()
