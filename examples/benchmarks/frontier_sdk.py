"""Replay installed frontier targets with ordinary immutable benchmark suites.

python examples/benchmarks/frontier_sdk.py --output frontier-sdk.json
"""
from __future__ import annotations

import argparse

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, Sampling, benchmark_suite
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow


def suite(*, processes: int = 1) -> BenchmarkSuite:
    experiment = Experiment("frontier-sdk", MaskedLinearCpu(Model.tiny()),
        Deployment.local(root="local://frontier-sdk"), ExecutionBudget(1, 128, 4))
    # Public synthetic geometry lowered by the ordinary hybrid semantic adapter.
    hybrid = dict(model_type="qwen3_5_text", hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        vocab_size=256, tie_word_embeddings=True, attention_bias=False, attention_dropout=0.0,
        max_position_embeddings=256, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        full_attention_interval=4, hidden_act="silu", rms_norm_eps=1e-6,
        attn_output_gate=True, mamba_ssm_dtype="float32", mlp_only_layers=[], use_cache=True,
        mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=False,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters=dict(rope_type="default", rope_theta=10000000,
                             partial_rotary_factor=0.25, mrope_interleaved=True,
                             mrope_section=[2, 1, 1]))
    recurrence = BenchmarkTarget.recurrence(experiment,
        config={"model_type": "qwen3_5", "text_config": hybrid, "tie_word_embeddings": True,
                "architectures": ["Qwen3_5ForConditionalGeneration"]}, tokens=32)
    preparation = BenchmarkTarget.preparation_batch(experiment, rows=8, inventories=4,
                                                    batch_size=4, window_rows=32)
    serial = preparation.with_params(batch={**preparation.to_spec()["parameters"]["batch"], "batch_size": 1})
    checkpoint = BenchmarkTarget.checkpoint(experiment.with_params(pipeline=MaskedLinearCpu(
        Model.tiny(), quantization=SymmetricPerRow(causal_reduction="prefix_f32"))),
        tokens=8, steps=2, page_rows=4)
    return BenchmarkSuite((
        BenchmarkCase("recurrence-copy", recurrence.with_params(owner="copy_reference")),
        BenchmarkCase("recurrence-owned", recurrence),
        BenchmarkCase("preparation-serial", serial),
        BenchmarkCase("preparation-batched", preparation),
        BenchmarkCase("checkpoint-pages", checkpoint.with_params(storage="layer_pages")),
        BenchmarkCase("checkpoint-graph", checkpoint),
    ), Sampling(processes=processes, memory_bytes=1 << 30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args()
    report = benchmark_suite(suite(processes=args.processes), output=args.output)
    statuses = [result.to_dict()["status"] for result in report.results]
    print({"output": args.output, "statuses": statuses})
    if any(status != "completed" for status in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
