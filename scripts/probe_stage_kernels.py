"""Independent profile then fresh whole-decoder stage/device confirmation.

Public synthetic inputs profile pinned checkpoint matrices; a separate complete
local-clear Qwen prefill/decode checks identical logits and KV. No provider/WAN
performance follows from this reference. CPU and GPU snapshots coexist.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import statistics
import sys
import time


def process_cpu():
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime


class FixedThresholdControl:
    """Ordinary 64-row CPU/Metal policy, independently dispatched for comparison."""

    def __init__(self, compiled, engine, *, memory_bytes, rows=150):
        from pllm.runtime.metal import MetalGEMM
        from pllm.runtime.stage_kernels import stage_groups

        self.stages = {stage_id: stage for stage_id, stage in stage_groups(compiled, engine).values()}
        self.counts = Counter()
        self.gpu = {}
        self.closed = False
        device_bytes = sum(stage.spec.in_features * stage.spec.out_features
                           + 8 * rows * (stage.spec.in_features + stage.spec.out_features)
                           for stage in self.stages.values())
        if not device_bytes <= memory_bytes <= 2 << 30:
            raise ValueError("fixed Metal control exceeds weight/workspace admission")
        self.plan = {"schema": "pllm.stage_kernel_uniform_control.v1", "min_rows": 64,
                     "prefill_rows": rows, "device_bytes": device_bytes,
                     "budget_bytes": memory_bytes, "binding_digest": compiled.digest}
        try:
            self.gpu = MetalGEMM().bind_stages({key: stage.compiled_weight.weight_view()
                                               for key, stage in self.stages.items()})
        except BaseException:
            self.close()
            raise

    def __call__(self, stage_id, activation):
        import numpy as np
        from pllm.runtime.quantization import quantize_activation_per_row, dequantize_matmul

        try:
            if self.closed:
                raise ValueError("fixed Metal control is closed")
            stage = self.stages[stage_id]
            if (activation.ndim != 2 or activation.dtype != np.float32
                    or activation.shape[0] not in {1, self.plan["prefill_rows"]}
                    or activation.shape[1] != stage.spec.in_features
                    or not np.all(np.isfinite(activation))):
                raise ValueError("fixed Metal control exceeds public geometry")
            backend = "metal" if activation.shape[0] >= self.plan["min_rows"] else "cpu"
            matrix = self.gpu[stage_id] if backend == "metal" else stage.compiled_weight
            quantized = quantize_activation_per_row(activation, bits=stage.spec.activation_bits)
            integer = matrix.clear(quantized.values)
            result = dequantize_matmul(integer, quantized.scales, stage.weight.scales,
                                      output_shape=quantized.original_shape[:-1] + (stage.spec.out_features,))
            if stage.bias is not None:
                result = result + stage.bias
            phase = "decode" if activation.shape[0] == 1 else "prefill"
            self.counts[f"{phase}_{backend}_calls"] += 1
            return np.ascontiguousarray(result, dtype=np.float32)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.closed = True
        self.gpu.clear()
        self.stages.clear()


def worker(spec):
    import numpy as np
    from pllm.metrics import BuildRef
    from pllm.runtime.benchmark_memory import process_memory
    from pllm.runtime.metal import MetalGEMM
    from pllm.runtime.stage_kernels import stage_groups, StageKernelExecutor
    from decoder_probe_support import DecoderFixture

    fixture = DecoderFixture(inputs=150, outputs=8)
    if spec["mode"] == "profile":
        metal = MetalGEMM()
        start, cpu = time.perf_counter(), process_cpu()
        rows, checked = [], 0
        rng = np.random.default_rng(7143)
        for weights, (_, stage) in stage_groups(fixture.compiled, fixture.engine).items():
            gpu = metal.compile(stage.compiled_weight.weight_view())
            timings = {}
            for phase, count in (("prefill", 150), ("decode", 1)):
                x = rng.integers(-128, 128, (count, stage.spec.in_features), dtype=np.int8)
                expected = stage.compiled_weight.clear(x)
                for device, kernel in (("cpu", stage.compiled_weight), ("metal", gpu)):
                    samples = []
                    for _ in range(4):
                        before = time.perf_counter_ns()
                        result = kernel.clear(x)
                        samples.append(time.perf_counter_ns() - before)
                        if not np.array_equal(result, expected):
                            raise AssertionError("profile changed exact integer output")
                        checked += result.size
                    timings[f"{device}_{phase}_ns"] = round(statistics.median(samples[1:]))
            rows.append({"weights": list(weights), **timings})
            del gpu
        return {"profile": {"schema": "pllm.stage_kernel_profile.v1",
            "model_plan_digest": fixture.plan.digest, "schedule_digest": fixture.compiled.runtime_schedule_digest,
            "binding_digest": fixture.compiled.digest, "build_digest": BuildRef.capture().expected_digest,
            "prefill_rows": 150, "decode_steps": 7, "profiles": rows},
            "source": fixture.lock(), "checked_values": checked, "memory": process_memory(),
            "wall_seconds": time.perf_counter() - start, "cpu_seconds": process_cpu() - cpu}
    start, cpu = time.perf_counter(), process_cpu()
    executor = (FixedThresholdControl(fixture.compiled, fixture.engine, memory_bytes=spec["budget"])
                if spec["mode"] == "fixed" else
                StageKernelExecutor(fixture.compiled, fixture.engine, spec["profile"], memory_bytes=spec["budget"]))
    runtime = fixture.compiled.runtime(executor)
    construction = {"seconds": time.perf_counter() - start, "cpu_seconds": process_cpu() - cpu}
    start, cpu = time.perf_counter(), process_cpu()
    _, logits, _ = runtime.prepare_ids([(i * 17 + 11) % 10000 for i in range(150)])
    prefill = time.perf_counter() - start
    scores = hashlib.sha256(logits.tobytes())
    for _ in range(7):
        logits = runtime.forward_ids([int(np.argmax(logits))])[-1]
        scores.update(logits.tobytes())
    report = {"mode": spec["mode"], "budget_bytes": spec["budget"], "source": fixture.lock(),
        "plan": executor.plan, "calls": dict(executor.counts), "construction": construction,
        "logits": scores.hexdigest(), "execution_seconds": time.perf_counter() - start,
        "prefill_seconds": prefill, "execution_cpu_seconds": process_cpu() - cpu,
        "cold_process_cpu_seconds": process_cpu(), "memory": process_memory(),
        "gpu_weights_bytes": sum(matrix.weight_bytes for matrix in executor.gpu.values())}
    state = hashlib.sha256()
    for cache in runtime.caches:
        state.update(cache.key[:cache.length].tobytes())
        state.update(cache.value[:cache.length].tobytes())
    report["kv"] = state.hexdigest()
    runtime.reset()
    executor.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--processes", type=int, choices=range(1, 4), default=3)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    if args.output is None or args.output.exists():
        parser.error("a new --output path is required")
    from frontier_probe_support import fresh_workers
    samples = []
    with fresh_workers(__file__, memory_bytes=5 << 30, timeout=900) as (run, guard, host):
        measured = run({"mode": "profile"})
        choices = (("cpu", 0), ("limited", 64 << 20), ("full", 2 << 30), ("fixed", 2 << 30))
        for repeat in range(args.processes):
            pair = {}
            for mode, budget in (choices if repeat % 2 == 0 else tuple(reversed(choices))):
                pair[mode] = run({"mode": mode, "budget": budget, "profile": measured["profile"]})
            for key in ("source", "logits", "kv"):
                if any(row[key] != pair["cpu"][key] for row in pair.values()):
                    raise AssertionError(f"stage kernels change {key}")
            samples.append(pair)
    report = {"schema": "pllm.stage_kernel_probe.v1", "scope": __doc__, "profile": measured,
        "samples": samples, "host": asdict(host), "build": run.build,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes, "exact_parity": True}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"groups": len(samples), "parity": True}))


if __name__ == "__main__":
    main()
