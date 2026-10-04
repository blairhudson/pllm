"""Header/schedule-only admission and resource estimates for the cached cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.benchmarks.qwen3_4b import SOURCE, baseline, lean, placed, offset
from pllm import lower_model
from pllm.model_loader import resolve_model
from pllm.runtime.preparation_window import WINDOW_BYTES
from pllm.runtime.tools import tool_policy
from pllm.runtime.semantic_stages import client_owns_linear, _scheduled_stage_specs
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hi")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resolved = resolve_model(SOURCE)
    config = json.loads((resolved.path / "config.json").read_text())
    descriptor = MaskedTransformerEngine._load_tokenizer_descriptor(resolved.path, resolved.manifest, config)
    view = SimpleNamespace(tokenizer_descriptor=descriptor, manifest=resolved.manifest.metadata, cfg=config)
    messages = [{"role": "user", "content": args.prompt}]
    policy = tool_policy({"input": args.prompt})
    if policy.prompt_instruction():
        messages.insert(0, {"role": "system", "content": policy.prompt_instruction()})
    rendered = ClientBundle.render_prompt(view, messages)
    tokens = ClientBundle.tokenizer(view).encode(rendered, add_bos=bool(descriptor.get("add_bos_token", True)))
    plan = lower_model(config, batch=1, max_input_tokens=64, max_new_tokens=8)
    rows = len(tokens) + 7
    results = []
    for experiment in (baseline, lean, placed, offset):
        profile = experiment.resolve()
        schedule, specs = _scheduled_stage_specs(plan, experiment.pipeline)
        if not schedule.complete:
            raise RuntimeError(f"incomplete schedule: {experiment.name}")
        body = [stage for stage in specs if stage.role not in {"token_lookup", "lm_head"}]
        local = [stage for stage in body if client_owns_linear(stage,
            client_prefix_layers=profile.client_prefix_layers, client_linear_roles=profile.client_linear_roles)]
        total_weights = sum(stage.in_features * stage.out_features for stage in body)
        client_weights = sum(stage.in_features * stage.out_features for stage in local)
        max_work = max(4096 + rows * (16 * stage.in_features + 24 * stage.out_features)
                       for stage in body if stage not in local)
        inventory = experiment.pipeline.components.get("inventory")
        if inventory and inventory.params.get("stage_window", 1) > 1 and max_work > WINDOW_BYTES:
            raise RuntimeError(f"{experiment.name}: {max_work} declared preparation bytes exceeds {WINDOW_BYTES}")
        results.append({"name": experiment.name, "schedule_complete": schedule.complete,
            "body_stages": len(body), "client_body_stages": len(local), "body_i8_weight_bytes": total_weights,
            "extra_client_i8_weight_bytes": client_weights, "extra_client_native_snapshot_bytes": client_weights,
            "remote_body_linear_mac_fraction": (total_weights - client_weights) / total_weights,
            "largest_declared_preparation_work_bytes": max_work})
    report = {"schema": "pllm.cached_qwen3_4b_preflight.v1", "model": SOURCE.to_spec(),
        "source_lock": resolved.manifest.metadata["source_lock"], "input_tokens": len(tokens),
        "max_output_tokens": 8, "required_stage_rows": rows,
        "config_sha256": hashlib.sha256((resolved.path / "config.json").read_bytes()).hexdigest(),
        "client_tied_token_weight_bytes": config["vocab_size"] * config["hidden_size"],
        "candidates": results, "scope": "source hashing, tokenizer rendering and schedule geometry; no tensor execution or peak memory"}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report | {"source_lock": "retained in report"}, indent=2))


if __name__ == "__main__":
    main()
