"""Matched ordinary benchmark sequence; archives digests/counters, never text or IDs."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.kernels import Cpu
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport
from pllm.quantization import SymmetricPerRow
from pllm.roles import ClientLinearRoles
from pllm.state import ClientPrefixReuse
from pllm.runtime.benchmark_cli import run_loopback_benchmark, accounted_benchmark_body_totals


def contexts():
    base = "Public context: count every role's bytes and compute. Keep private state at the client. " * 8
    return (base + "Explain preparation.", base + "Explain preparation.",
            base + "Explain preparation. Then explain online inference.",
            base + "Explain preparation. Then explain online inference. Compare their costs.",
            "Unrelated public task: list three common fruits briefly.")


def run(*, tiny=False, output_tokens=8, docker=False, repetitions=1, _source=None, candidate=None, checkpoint=None):
    if tiny and _source is None:
        from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
        with TemporaryDirectory(prefix="pllm-conversation-") as directory:
            root = create_tiny_llama_checkpoint(Path(directory) / "model", hidden_size=32,
                intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=8, seed=812)
            return run(tiny=True, output_tokens=output_tokens, docker=docker, repetitions=repetitions, candidate=candidate, checkpoint=checkpoint,
                       _source=Model.path(str(root), model_id="conversation-control"))
    source = _source or Model.hf("Qwen/Qwen2.5-0.5B-Instruct",
                               revision="7ae557604adf67be50417f59c2c2f167def9a775")
    sequence = contexts() if not tiny else (
        "shared public context " * 5 + "one", "shared public context " * 5 + "one",
        "shared public context " * 5 + "one two", "shared public context " * 5 + "one two three", "other")
    rows = []
    bound = 192 if tiny else 256
    for placement in (False, True):
        for reuse in (False, True):
            if candidate is not None and (2 * int(placement) + int(reuse)) != candidate:
                continue
            experiment = Experiment(f"attention-{placement}-reuse-{reuse}", MaskedLinearCpu(
                source, kernels=Cpu(threads=1), quantization=SymmetricPerRow(causal_reduction="prefix_f32"),
                inventory=PreparedInventory("request-sized", rows=1),
                delivery=ClientBundleTransport("artifacts"),
                placement=ClientLinearRoles(["qkv_projection", "attention_output"]) if placement else None,
                cache=ClientPrefixReuse(max_bytes=64 << 20, fixed_input_tokens=bound) if reuse else None),
                Deployment.local(root="local://conversation-control"),
                ExecutionBudget(requests=len(sequence) * repetitions, max_input_tokens=bound, max_new_tokens=32))
            print(f"Running {experiment.name}", flush=True)
            report = run_loopback_benchmark(model=source.source, model_id=experiment.resolve().model,
                tiny=False, prompt=sequence[0], prompt_sequence=sequence, max_output_tokens=output_tokens,
                warmups=0, repetitions=repetitions, timeout_seconds=300, experiment=experiment,
                _cohort_salt=b"pllm.public.conversation-control.v1", temperature=0,
                capture_output_digest=True, docker=docker)
            rows.append({"experiment": experiment.to_spec(), "report": report,
                         "bodies": accounted_benchmark_body_totals(report.get("topology_accounting") or {})})
            if checkpoint is not None:
                checkpoint({"schema": "pllm.conversation_reuse_partial.v1", "complete": False, "candidates": rows})
    golden = [(item["generation"]["output_text_digest"], item["tokens"]) for item in rows[0]["report"]["runs"]]
    matched = all([(item["generation"]["output_text_digest"], item["tokens"])
                   for item in row["report"]["runs"]] == golden for row in rows)
    if not matched:
        raise ValueError("cached/placement output digest or usage differs from same-mode control")
    return {"schema": "pllm.conversation_reuse_evidence.v1", "scope": "ordered public prompt growth/repeat/negative-control; separate SDK clients",
            "output_and_usage_parity": matched if candidate is None else None,
            "full_wire_bytes": None, "independent_provider_privacy": False, "candidates": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--docker", action="store_true")
    parser.add_argument("--output-tokens", type=int, choices=(8, 32), default=8)
    parser.add_argument("--repetitions", type=int, choices=(1, 2, 20), default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=int, choices=range(4))
    args = parser.parse_args()
    def checkpoint(result):
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    result = run(tiny=args.tiny, output_tokens=args.output_tokens, docker=args.docker, repetitions=args.repetitions, candidate=args.candidate, checkpoint=checkpoint)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps([{ "name": row["experiment"]["name"], "bodies": row["bodies"]}
                      for row in result["candidates"]], indent=2))
