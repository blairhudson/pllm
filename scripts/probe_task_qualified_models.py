"""Cache-only, fixed-task W8A8 semantic clear-kernel screen and body projection.

Only scalar metrics and digests of generated content are persisted. Optional
canonical examples validate prepared role-body accounting, not full wire costs.
Run both cohorts together so review tasks cannot be selected after the screen.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Must precede NumPy/runtime imports, including in canonical child processes.
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "PLLM_NATIVE_THREADS",
    "PLLM_ENGINE_THREADS",
    "RAYON_NUM_THREADS",
)


def configure_environment() -> None:
    """Set CLI-only controls before model imports; importing graders is inert."""
    for name in THREAD_ENV:
        os.environ[name] = "1"
    os.environ["PLLM_KERNEL_BACKEND"] = "rust"
    os.environ["PLLM_REQUIRE_RUST"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "examples/benchmarks/task_qualified_tasks.json"
MODELS = {
    "qwen": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "smol": ("HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac"),
}
COST_KEYS = (
    "projected_online_arithmetic_bytes",
    "projected_preparation_correction_arithmetic_bytes",
    "projected_all_link_arithmetic_bytes",
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def expected_answer(grader: dict) -> str:
    kind = grader["kind"]
    if kind == "exact":
        return grader["expected"]
    if kind == "arithmetic":
        a, b = grader["operands"]
        operations = {"add": a + b, "subtract": a - b, "multiply": a * b}
        return str(operations[grader["operation"]])
    if kind == "parity":
        return "EVEN" if grader["number"] % 2 == 0 else "ODD"
    if kind == "sign":
        number = grader["number"]
        return "ZERO" if number == 0 else "POSITIVE" if number > 0 else "NEGATIVE"
    if kind == "case":
        return getattr(grader["value"], grader["operation"])()
    if kind == "sorted_csv":
        return ",".join(str(value) for value in sorted(grader["values"]))
    raise ValueError(f"unknown objective grader: {kind}")


def grade(task: dict, text: str, *, eos: bool) -> dict:
    answer_correct = text.strip() == expected_answer(task["grader"])
    return {
        "answer_correct": answer_correct,
        "success": answer_correct and eos,
        "failure": None if answer_correct and eos else "truncation" if not eos else "wrong_answer",
        "empty_eos_failure": eos and not text.strip(),
    }


def load_tasks(path: Path = TASKS) -> dict:
    document = json.loads(path.read_text())
    tasks = document["tasks"]
    policy = document["policy"]
    if (
        document["schema"] != "pllm.task_qualified_tasks.v1"
        or not 12 <= len(tasks) <= 24
        or len({row["id"] for row in tasks}) != len(tasks)
        or len({row["prompt"] for row in tasks}) != len(tasks)
        or {row["cohort"] for row in tasks} != {"screen", "review"}
        or not 1 <= policy["max_input_tokens"] <= 512
        or not 1 <= policy["max_output_tokens"] <= 64
        or policy["smol_retries"] != 0
        or policy["require_eos"] is not True
        or policy["sampling"] != "greedy"
        or not 0 < policy["minimum_success_fraction"] <= 1
        or not 0 < policy["maximum_cost_ratio"] < 1
        or not set(policy["canonical_task_ids"]) <= {row["id"] for row in tasks}
    ):
        raise ValueError("invalid fixed, bounded, disjoint task contract")
    for task in tasks:
        expected_answer(task["grader"])
    return document


def arithmetic_cost(
    stages: list[dict], consumed_rows: int, *, preparation_rows: int | None = None
) -> dict:
    """Prepared seeded rings: client->inference input, return output, prep correction.

    No two-worker multiplier: prepared has one inference worker plus preparation.
    Excludes seed requests, typed envelopes, setup, client bundle, HTTP and TLS.
    """
    online = sum(
        consumed_rows * (s["in_features"] + s["out_features"]) * s["wire_bits"] // 8 for s in stages
    )
    prepared = consumed_rows if preparation_rows is None else preparation_rows
    if not 0 < consumed_rows <= prepared:
        raise ValueError("prepared inventory must cover all consumed rows")
    correction = sum(prepared * s["out_features"] * s["wire_bits"] // 8 for s in stages)
    return dict(zip(COST_KEYS, (online, correction, online + correction), strict=True))


def aggregate(rows: list[dict]) -> dict:
    successes = sum(row["success"] for row in rows)
    totals = {key: sum(row[key] for row in rows) for key in COST_KEYS}
    return {
        "tasks": len(rows),
        "successes": successes,
        "success_fraction": successes / len(rows),
        "failures": dict(Counter(row["failure"] for row in rows if not row["success"])),
        "empty_eos_failures": sum(row.get("empty_eos_failure", False) for row in rows),
        "attempt_truncations": sum(
            row.get("attempt_truncations", int(row["failure"] == "truncation")) for row in rows
        ),
        "attempt_wrong_answers": sum(
            row.get("attempt_wrong_answers", int(row["failure"] == "wrong_answer")) for row in rows
        ),
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens_excluding_eos": sum(row["output_tokens"] for row in rows),
        "selected_tokens_including_eos": sum(row["selected_tokens"] for row in rows),
        "eos_count": sum(row["eos"] for row in rows),
        "consumed_rows_per_stage": sum(row["consumed_rows_per_stage"] for row in rows),
        "prepared_rows_per_stage": sum(row["prepared_rows_per_stage"] for row in rows),
        "unused_prepared_rows_per_stage": sum(
            row["prepared_rows_per_stage"] - row["consumed_rows_per_stage"] for row in rows
        ),
        "clear_execution_cpu_seconds": sum(row["cpu_seconds"] for row in rows),
        **totals,
        "projected_bytes_per_successful_task": {
            key: value / successes if successes else None for key, value in totals.items()
        },
    }


def compare(qwen: list[dict], smol: list[dict], policy: dict) -> dict:
    """Offline objective oracle; Qwen rows reused, never double-charged in standalone."""
    q_by_id = {row["task_id"]: row for row in qwen}
    if set(q_by_id) != {row["task_id"] for row in smol}:
        raise ValueError("fallback requires identical task sets")
    combined = []
    fallback_ids = []
    for small in smol:
        row = dict(small)
        if not small["success"]:
            fallback_ids.append(small["task_id"])
            large = q_by_id[small["task_id"]]
            for key in COST_KEYS + (
                "input_tokens",
                "output_tokens",
                "selected_tokens",
                "consumed_rows_per_stage",
                "prepared_rows_per_stage",
                "cpu_seconds",
            ):
                row[key] = small[key] + large[key]
            row.update(
                success=large["success"],
                failure=large["failure"],
                eos=int(small["eos"]) + int(large["eos"]),
                empty_eos_failure=int(small["empty_eos_failure"]) + int(large["empty_eos_failure"]),
                attempt_truncations=int(small["failure"] == "truncation")
                + int(large["failure"] == "truncation"),
                attempt_wrong_answers=int(small["failure"] == "wrong_answer")
                + int(large["failure"] == "wrong_answer"),
            )
        combined.append(row)
    summaries = {
        "qwen": aggregate(qwen),
        "smol": aggregate(smol),
        "smol_then_qwen": aggregate(combined),
    }
    baseline = summaries["qwen"]
    decisions = {}
    for candidate in ("smol", "smol_then_qwen"):
        result = summaries[candidate]
        key = "projected_all_link_arithmetic_bytes"
        baseline_bps = baseline["projected_bytes_per_successful_task"][key]
        candidate_bps = result["projected_bytes_per_successful_task"][key]
        ratio = candidate_bps / baseline_bps if baseline_bps and candidate_bps else None
        qualified = (
            result["successes"] >= baseline["successes"]
            and result["success_fraction"] >= policy["minimum_success_fraction"]
        )
        decisions[candidate] = {
            "quality_qualified": qualified,
            "projected_all_link_bytes_per_success_ratio_to_qwen": ratio,
            "promising_under_fixed_gate": qualified
            and ratio is not None
            and ratio <= policy["maximum_cost_ratio"],
        }
    return {
        "policies": summaries,
        "fallback_attempts": len(fallback_ids),
        "fallback_task_ids": fallback_ids,
        "total_attempts_with_fallback": len(smol) + len(fallback_ids),
        "decisions": decisions,
    }


def native_prompt(bundle, source: Path, prompt: str) -> str:
    """Check imported source template and render explicitly: no fallback accepted."""
    from jinja2 import StrictUndefined
    from jinja2.sandbox import SandboxedEnvironment

    config = json.loads((source / "tokenizer_config.json").read_text())
    template = config["chat_template"]
    if template != bundle.tokenizer_descriptor["chat_template"]:
        raise ValueError("bundle template differs from checkpoint-native template")
    environment = SandboxedEnvironment(
        undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True
    )

    def raise_exception(message):
        raise ValueError(message)

    environment.globals["raise_exception"] = raise_exception
    messages = [{"role": "user", "content": prompt}]
    rendered = environment.from_string(template).render(
        messages=messages,
        add_generation_prompt=True,
        bos_token=config.get("bos_token") or "",
        eos_token=config.get("eos_token") or "",
        tools=None,
    )
    if rendered != bundle.render_prompt(messages):
        raise ValueError("source-native prompt rendering differs from compiled bundle")
    return rendered


def run_model(name: str, document: dict) -> dict:
    # Imported only after thread/offline policy is installed. Models loaded serially.
    import numpy as np

    from pllm import Model, lower_model
    from pllm.model_loader import resolve_model
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime._native_support import capabilities
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    model_id, revision = MODELS[name]
    choice = Model.hf(model_id, revision=revision, local_files_only=True)
    source = resolve_model(choice)
    if source.path is None or source.path.name != revision:
        raise ValueError("pinned cached snapshot required")
    policy = document["policy"]
    start = time.process_time()
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=1)
    if not engine.kernel.native or engine.kernel.threads != 1:
        raise RuntimeError("requires compiled native kernel with one thread")
    asyncio.run(engine.load(source.manifest))
    model = engine.models[source.manifest.id]
    bundle = ClientBundle.unpack(engine.client_bundle(source.manifest.id))
    composition = MaskedLinearCpu(
        choice, quantization=SymmetricPerRow(weight_bits=8, activation_bits=8)
    )
    body = {
        key: stage
        for key, stage in model.stages.items()
        if stage.spec.role not in {"token_lookup", "lm_head"}
    }
    stages = [
        {
            "id": key,
            "role": stage.spec.role,
            "in_features": stage.spec.in_features,
            "out_features": stage.spec.out_features,
            **stage.seeded_profile.to_dict(),
            "weight_digest": stage.weight_digest,
        }
        for key, stage in sorted(body.items())
    ]
    if any(
        stage.spec.weight_bits != 8
        or stage.spec.activation_bits != 8
        or stage.compiled_weight is None
        for stage in body.values()
    ):
        raise RuntimeError("stage is not compiled W8A8")
    load_cpu = time.process_time() - start
    records = []
    for task in document["tasks"]:
        # Exact input-sized plans; never clip/pad a prompt to historical token counts.
        rendered = native_prompt(bundle, source.path, task["prompt"])
        tokenizer = bundle.tokenizer()
        ids = tokenizer.encode(
            rendered, add_bos=bool(bundle.tokenizer_descriptor.get("add_bos_token", True))
        )
        if not 1 <= len(ids) <= policy["max_input_tokens"]:
            raise ValueError("source-native input exceeds fixed token budget")
        plan = lower_model(
            (source.path / "config.json").read_bytes(),
            batch=1,
            max_input_tokens=len(ids),
            max_new_tokens=policy["max_output_tokens"],
        )
        compiled = compile_runtime_model(plan, bundle, composition=composition)
        counts = Counter()
        calls = Counter()
        by_role = defaultdict(Counter)

        def remote(key, activation):
            stage = body[key]
            quantized = quantize_activation_per_row(activation, bits=8)
            integer = stage.compiled_weight.clear(quantized.values)
            rows = quantized.rows
            counts[key] += rows
            calls[key] += 1
            output = dequantize_matmul(
                integer,
                quantized.scales,
                stage.weight.scales,
                output_shape=quantized.original_shape[:-1] + (stage.spec.out_features,),
            )
            if stage.bias is not None:
                output = output + stage.bias
            return np.ascontiguousarray(output, dtype=np.float32)

        runtime = compiled.runtime(remote)
        if runtime.encode_prompt(rendered) != ids:
            raise ValueError("compiled tokenizer disagrees with native encoding")
        generated = []
        selected = []
        eos = False
        start_cpu, start_wall = time.process_time(), time.perf_counter()
        _, logits, _ = runtime.prepare_ids(ids)
        for index in range(policy["max_output_tokens"]):
            token = int(np.argmax(logits))
            selected.append(token)
            if token == int(runtime.cfg["eos_token_id"]):
                eos = True
                break
            generated.append(token)
            if index + 1 < policy["max_output_tokens"]:
                logits = runtime.forward_ids([token])[-1]
        cpu, wall = time.process_time() - start_cpu, time.perf_counter() - start_wall
        rows = len(ids) + len(selected) - 1
        if set(counts) != set(body) or set(counts.values()) != {rows}:
            raise AssertionError("executed stage rows differ from actual token trajectory")
        # Ordinary request-sized preparation reserves cap rows before EOS is known.
        prepared_rows = len(ids) + policy["max_output_tokens"] - 1
        costs = arithmetic_cost(stages, rows, preparation_rows=prepared_rows)
        for role in {stage["role"] for stage in stages}:
            by_role[role].update(
                arithmetic_cost(
                    [stage for stage in stages if stage["role"] == role],
                    rows,
                    preparation_rows=prepared_rows,
                )
            )
        if any(sum(value[key] for value in by_role.values()) != costs[key] for key in COST_KEYS):
            raise AssertionError("role arithmetic projections do not reconcile")
        text = tokenizer.decode(generated)
        records.append(
            {
                "task_id": task["id"],
                "cohort": task["cohort"],
                "category": task["category"],
                **grade(task, text, eos=eos),
                "eos": eos,
                "input_tokens": len(ids),
                "output_tokens": len(generated),
                "selected_tokens": len(selected),
                "input_token_digest": digest(ids),
                "rendered_prompt_digest": digest(rendered),
                "output_token_digest": digest(generated),
                "output_text_digest": digest(text),
                "plan_digest": plan.digest,
                "schedule_digest": compiled.runtime_schedule_digest,
                "binding_digest": compiled.digest,
                "consumed_rows_per_stage": rows,
                "prepared_rows_per_stage": prepared_rows,
                "stage_calls": sum(calls.values()),
                "stage_rows": sum(counts.values()),
                "cpu_seconds": cpu,
                "wall_seconds": wall,
                **costs,
                "arithmetic_projection_by_role": dict(by_role),
            }
        )
    # Function return releases the model before the next checkpoint loads.
    return {
        "model": model_id,
        "revision": revision,
        "checkpoint_digest": source.checkpoint_digest,
        "source_lock_digest": source.source_lock_digest,
        "body_fingerprint": model.manifest.metadata["body_fingerprint"],
        "native_capabilities": capabilities(),
        "native_threads": engine.kernel.threads,
        "blas_configuration": np.__config__.CONFIG.get("Build Dependencies", {}).get("blas"),
        "weight_bits": engine.weight_bits,
        "activation_bits": engine.activation_bits,
        "load_cpu_seconds_cached_source_and_compiled_cache": load_cpu,
        "stages": stages,
        "stage_inventory_digest": digest(stages),
        "records": records,
    }


def canonical_example(name: str, task: dict, policy: dict, clear: dict) -> dict:
    """Ordinary prepared loopback, fresh processes, request-sized inventory, no warmup."""
    from pllm import Deployment, ExecutionBudget, Experiment, Model
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.benchmark_cli import run_loopback_benchmark

    model_id, revision = MODELS[name]
    experiment = Experiment(
        name=f"task-qualified-{name}-{task['id']}",
        pipeline=MaskedLinearCpu(
            Model.hf(model_id, revision=revision, local_files_only=True),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
        ),
        deployment=Deployment.local(root=f"local://task-qualified-{name}"),
        budget=ExecutionBudget(
            requests=1,
            max_input_tokens=policy["max_input_tokens"],
            max_new_tokens=policy["max_output_tokens"],
        ),
    )
    report = run_loopback_benchmark(
        model=model_id,
        model_id=None,
        tiny=False,
        prompt=task["prompt"],
        max_output_tokens=policy["max_output_tokens"],
        warmups=0,
        repetitions=1,
        timeout_seconds=600,
        experiment=experiment,
        inventory_policy="request-sized",
        temperature=0.0,
        capture_output_digest=True,
    )
    record = report["runs"][0]
    ledger = report["topology_accounting"]["runs"][0]
    stage = report["topology_accounting"]["stages"]["runs"][0]
    generation = record.get("generation", {})
    eos = generation.get("response_status") == "completed"
    selected_tokens = record["tokens"]["output_tokens"] + int(eos)
    checks = {
        "completed": record["status"] == "completed",
        "public_output_digest_capture_enabled": report["configuration"].get("capture_output_digest")
        is True,
        "same_body": record["model_fingerprint"] == clear["body_fingerprint"],
        "same_input_tokens": record["tokens"]["input_tokens"] == clear["record"]["input_tokens"],
        "same_output_tokens": record["tokens"]["output_tokens"] == clear["record"]["output_tokens"],
        "explicit_greedy_sampling": record.get("sampling")
        == {
            "requested_temperature": 0.0,
            "effective_temperature": 0.0,
            "mode": "greedy",
            "top_p": None,
        },
        "same_output_text_digest": generation.get("output_text_digest")
        == clear["record"]["output_text_digest"],
        "same_eos_termination": eos == clear["record"]["eos"]
        and generation.get("response_status") in {"completed", "incomplete"},
        "same_selected_tokens": selected_tokens == clear["record"]["selected_tokens"],
        "same_consumed_inventory_rows": record["inventory"]["consumed"]
        == clear["record"]["consumed_rows_per_stage"],
        "same_prepared_inventory_rows": record["inventory"]["required"]
        == clear["record"]["prepared_rows_per_stage"],
        "same_burned_inventory_rows": record["inventory"]["burned"]
        == clear["record"]["prepared_rows_per_stage"] - clear["record"]["consumed_rows_per_stage"],
        "reconciled_stage_bodies": stage["reconciled_with_protocol_bodies"],
        "no_plaintext_prompt": record["privacy"]["plaintext_prompt_bytes_sent"] == 0,
        "no_plaintext_token_ids": record["privacy"]["plaintext_token_ids_sent"] == 0,
    }
    if not all(checks.values()):
        return {
            "model": name,
            "task_id": task["id"],
            "status": "cohort_mismatch",
            "checks": checks,
            "observed_input_tokens": record["tokens"]["input_tokens"],
            "observed_output_tokens": record["tokens"]["output_tokens"],
            "clear_input_tokens": clear["record"]["input_tokens"],
            "clear_output_tokens": clear["record"]["output_tokens"],
            "sampling": record.get("sampling"),
            "generation": generation,
            "inventory": record["inventory"],
            "full_wire_bytes": None,
        }
    measured = stage["body_bytes_by_stage_and_edge"]
    expected_ids = {row["id"] for row in clear["stages"]}
    if set(measured) != expected_ids:
        raise RuntimeError("canonical semantic stage set differs from clear execution")
    projection_by_edge = Counter()
    serialized_by_role = defaultdict(Counter)
    consumed = clear["record"]["consumed_rows_per_stage"]
    prepared = clear["record"]["prepared_rows_per_stage"]
    for row in clear["stages"]:
        word_bytes = row["wire_bits"] // 8
        projected = {
            "client->inference": consumed * row["in_features"] * word_bytes,
            "inference->client": consumed * row["out_features"] * word_bytes,
            "preparation->inference": prepared * row["out_features"] * word_bytes,
        }
        for edge, amount in projected.items():
            if measured[row["id"]].get(edge, 0) < amount:
                return {
                    "model": name,
                    "task_id": task["id"],
                    "status": "projection_mismatch",
                    "stage_id": row["id"],
                    "edge": edge,
                    "projected_arithmetic_bytes": amount,
                    "observed_serialized_bytes": measured[row["id"]].get(edge, 0),
                    "full_wire_bytes": None,
                }
        projection_by_edge.update(projected)
        serialized_by_role[row["role"]].update(measured[row["id"]])
    stage_bytes = sum(sum(edges.values()) for edges in measured.values())
    return {
        "model": name,
        "task_id": task["id"],
        "status": "validated",
        "input_tokens": record["tokens"]["input_tokens"],
        "output_tokens": record["tokens"]["output_tokens"],
        "body_fingerprint": record["model_fingerprint"],
        "stage_body_bytes_reconciled": True,
        "checks": checks,
        "clear_output_content_parity_measured": True,
        "output_text_digest": generation["output_text_digest"],
        "sampling": record["sampling"],
        "eos": eos,
        "selected_tokens_including_eos": selected_tokens,
        "observed_consumed_rows_per_stage": record["inventory"]["consumed"],
        "observed_prepared_rows_per_stage": record["inventory"]["required"],
        "observed_burned_rows_per_stage": record["inventory"]["burned"],
        "consumed_rows_per_stage": consumed,
        "prepared_rows_per_stage": prepared,
        "arithmetic_projection_by_edge": dict(projection_by_edge),
        "serialized_stage_body_bytes_by_role_and_edge": dict(serialized_by_role),
        "projection_payloads_bounded_by_matching_stage_bodies": True,
        "covered_online_serialized_role_body_bytes": ledger[
            "online_all_link_serialized_body_bytes"
        ],
        "covered_all_link_serialized_role_body_bytes": ledger["all_link_serialized_body_bytes"],
        "serialized_stage_body_bytes": stage_bytes,
        "projected_all_link_arithmetic_bytes": clear["record"][
            "projected_all_link_arithmetic_bytes"
        ],
        "stage_body_bytes_beyond_arithmetic_projection": stage_bytes
        - clear["record"]["projected_all_link_arithmetic_bytes"],
        "other_setup_control_bundle_body_bytes": ledger["all_link_serialized_body_bytes"]
        - stage_bytes,
        "full_wire_bytes": None,
    }


def markdown(report: dict) -> str:
    comparison = report["comparison"]
    lines = [
        f"# Task-qualified models: {report['cohort']} (2026-10-01)",
        "",
        "Fixed checkpoints, no training; source-native prompts; compiled native W8A8 semantic clear-kernel execution. One CPU/native/BLAS compute thread; separate model loads.",
        "",
        "## Objective results",
        "",
        "| Policy | Success | Attempt EOS / truncation | Input / output tokens | Projected all-link arithmetic MB | Projected MB / success |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in comparison["policies"].items():
        bps = row["projected_bytes_per_successful_task"]["projected_all_link_arithmetic_bytes"]
        bps_text = f"{bps / 1e6:.3f}" if bps is not None else "undefined (0 successes)"
        lines.append(
            f"| {name} | {row['successes']}/{row['tasks']} | {row['eos_count']} / {row['attempt_truncations']} | {row['input_tokens']} / {row['output_tokens_excluding_eos']} | {row['projected_all_link_arithmetic_bytes'] / 1e6:.3f} | {bps_text} |"
        )
    lines += [
        "",
        f"Fallback: {comparison['fallback_attempts']} additional Qwen attempts; {comparison['total_attempts_with_fallback']} total attempts. All failed Smol attempts charged. No retries. Objective grader is offline oracle, not a deployable general-purpose router.",
        "",
        "## Fixed decision gate",
        "",
        "Each cohort independently requires >=75% task success, at least Qwen's successes, and <=0.8 times Qwen's projected all-link arithmetic bytes per successful task. Gate and canonical examples fixed before either cohort executes.",
        "",
    ]
    for name, decision in comparison["decisions"].items():
        lines.append(
            f"- {name}: quality-qualified={decision['quality_qualified']}; promising={decision['promising_under_fixed_gate']}; cost ratio={decision['projected_all_link_bytes_per_success_ratio_to_qwen']}."
        )
    lines += [
        "",
        "## Success by task family",
        "",
        "| Family | Qwen | Smol |",
        "|---|---:|---:|",
    ]
    for category, models in report["task_family_successes"].items():
        lines.append(
            f"| {category} | {models['qwen']['successes']}/{models['qwen']['tasks']} | {models['smol']['successes']}/{models['smol']['tasks']} |"
        )
    lines += [
        "",
        "## Accounting and limitations",
        "",
        *[f"- {line}" for line in report["limitations"]],
        "",
        "Prepared arithmetic projection uses actual checkpoint seeded rings: online = consumed rows * (input width + output width) * ring bits / 8; preparation correction = prepared rows * output width * ring bits / 8. Fixed request-sized preparation reserves input tokens + output cap - 1 rows before EOS is known; unused prepared rows charged on every attempt, including fallback. Sum across compiled remote body stages. Token lookup and LM head remain client-local. No historical 39+32 body figure substituted.",
        "",
        "EOS token not returned as text; counted separately among selected tokens. Rows per stage = input tokens + selected tokens - 1, including final EOS prediction. Output-cap exhaustion fails even if answer prefix is correct. Only outer whitespace stripped; case, internal spacing, extra prose graded strictly.",
        "",
        "## Canonical prespecified examples",
        "",
        "Controlled ordinary benchmark API uses temperature=0.0. Validation checks output-text digest parity, EOS/selected-token counts, consumed/prepared/burned inventory rows, body fingerprint, full stage set, protocol-body reconciliation, and arithmetic payload bounds per stage/edge. Setup/client-bundle bodies shown separately; client cache is not isolated.",
        "",
        "```json",
        json.dumps(report["canonical_examples"], indent=2, sort_keys=True),
        "```",
        "",
        "## Sampling correction and historical provenance",
        "",
        "Prior canonical dashboard requests omitted temperature; SDK effective default was 0.8, whereas clear execution used greedy argmax. Optional temperature now flows through ordinary benchmark API/config and dashboard request into SDK only when explicit. None preserves historical default. Effective sampling included in report and comparison cohort keys; unreported sampling cannot rank. Internal Python API and dashboard JSON request support this control; no new CLI flag.",
        "",
        "```json",
        json.dumps(report["sampling_correction"], indent=2, sort_keys=True),
        "```",
        "",
        "## Public output-digest capture policy",
        "",
        "Public task canonical calls explicitly enable capture_output_digest=True. Ordinary benchmark API/config defaults to False: generation.response_status remains available, output fingerprints are omitted. Opt-in flag requires exact bool; sampling defaults and cohort matching remain independent. Benchmark archive remains text/token-ID/digest-free. No new CLI flag.",
        "",
        "```json",
        json.dumps(report["output_digest_capture_policy"], indent=2, sort_keys=True),
        "```",
        "",
        "## Reproduce",
        "",
        "```sh",
        ".venv/bin/python scripts/probe_task_qualified_models.py --canonical",
        ".venv/bin/python -m pytest tests/test_benchmark_sampling.py tests/test_benchmark_cli.py tests/test_dashboard.py tests/test_dashboard_history.py tests/test_task_qualified_models.py -m 'not integration'",
        ".venv/bin/python -m ruff check python/pllm/runtime/dashboard.py python/pllm/runtime/benchmark_cli.py scripts/probe_task_qualified_models.py tests/test_benchmark_sampling.py tests/test_task_qualified_models.py",
        ".venv/bin/python -m ruff format --check python/pllm/runtime/dashboard.py python/pllm/runtime/benchmark_cli.py scripts/probe_task_qualified_models.py tests/test_benchmark_sampling.py tests/test_task_qualified_models.py",
        "```",
        "",
        "Task contract: `examples/benchmarks/task_qualified_tasks.json`. JSON evidence stores scalar counts, source locks, and digests; no generated text or token IDs.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical", action="store_true", help="run four prespecified prepared examples"
    )
    args = parser.parse_args()
    document = load_tasks()
    historical_examples = {}
    for cohort in ("screen", "review"):
        path = ROOT / f"docs/evidence/task-qualified-models-{cohort}-2026-10-01.json"
        if path.is_file():
            previous = json.loads(path.read_text())
            historical_examples[cohort] = previous.get("sampling_correction", {}).get(
                "historical_canonical_examples", previous.get("canonical_examples", [])
            )
    print(f"fixed_task_contract_sha256={digest(document)}", flush=True)
    results = {}
    for name in MODELS:
        print(f"running cache-only {name} W8A8; one compute thread", flush=True)
        results[name] = run_model(name, document)
        gc.collect()
    canonical = []
    if args.canonical:
        for task in document["tasks"]:
            if task["id"] not in document["policy"]["canonical_task_ids"]:
                continue
            for name in MODELS:
                clear = {
                    "body_fingerprint": results[name]["body_fingerprint"],
                    "stages": results[name]["stages"],
                    "record": next(
                        row for row in results[name]["records"] if row["task_id"] == task["id"]
                    ),
                }
                print(f"canonical prepared {name} {task['id']}", flush=True)
                try:
                    canonical.append(canonical_example(name, task, document["policy"], clear))
                except Exception as exc:
                    # No exception messages: external errors may contain model text.
                    canonical.append(
                        {
                            "model": name,
                            "task_id": task["id"],
                            "status": "unavailable_or_mismatch",
                            "exception_type": type(exc).__name__,
                        }
                    )
    limitations = [
        "16 fixed new public tasks total, 8 screen + 8 disjoint review; same families, no tuning between cohorts. One greedy sample per task/model; too small for broad quality claims.",
        "Success means exact objective answer AND EOS within fixed 32-token output cap. Inputs bounded at 512; actual counts recorded, never cropped. Longer outputs and open-ended generation untested.",
        "Clear-kernel task results are measured; cohort traffic is numeric arithmetic projection only. Canonical examples validate covered serialized role bodies only where status=validated; not full-wire or full-cohort network measurement.",
        "Projection excludes seed requests, envelopes, setup/control, client bundle, HTTP/TLS, checkpoint distribution, local token lookup/head, and upstream source download. Full-wire bytes unknown.",
        "Source snapshots and compiled cache already available. Cached-source load CPU is not cold provisioning; client bundle cache not isolated. Cold download/network/CPU and accelerator energy/memory unmeasured.",
        "Fallback uses known public objective grader as offline oracle. Router implementation/latency and failures of a real-world quality detector unmeasured; model+fallback promise applies only where such a grader exists.",
        "Compute threads capped at one; process may retain housekeeping threads. Co-located CPU timings are descriptive, not WAN latency, independent operators, or cryptographic privacy evidence.",
        "Output fingerprints are restricted to this explicit public-task diagnostic; ordinary benchmark capture_output_digest defaults to False and keeps response status only. Benchmark history never stores output digests, text, or token IDs.",
    ]
    if any(row["status"] != "validated" for row in canonical):
        limitations.insert(
            3,
            "One or more controlled canonical examples failed validation; inspect scalar checks and counts below. No parity or accounting claim for failed examples.",
        )
    for cohort in ("screen", "review"):
        model_results = {
            name: {
                **result,
                "records": [row for row in result["records"] if row["cohort"] == cohort],
            }
            for name, result in results.items()
        }
        report = {
            "schema": "pllm.task_qualified_models.v1",
            "date": "2026-10-01",
            "cohort": cohort,
            "task_contract_digest": digest(document),
            "cohort_digest": digest(
                [task for task in document["tasks"] if task["cohort"] == cohort]
            ),
            "policy": document["policy"],
            "models": model_results,
            "comparison": compare(
                model_results["qwen"]["records"],
                model_results["smol"]["records"],
                document["policy"],
            ),
            "task_family_successes": {
                category: {
                    name: {
                        "tasks": sum(row["category"] == category for row in result["records"]),
                        "successes": sum(
                            row["success"]
                            for row in result["records"]
                            if row["category"] == category
                        ),
                    }
                    for name, result in model_results.items()
                }
                for category in sorted({task["category"] for task in document["tasks"]})
            },
            "canonical_examples": [row for row in canonical if row["task_id"].startswith(cohort)],
            "canonical_requested": args.canonical,
            "sampling_correction": {
                "root_cause": "historical dashboard omitted temperature; client None default was 0.8, clear execution was greedy",
                "client_default_source": "python/pllm/runtime/client.py::_sampling_temperature",
                "historical_effective_temperature": 0.8,
                "historical_sampling_source": "inferred from omitted request and unchanged SDK default; historical records lacked sampling field",
                "controlled_canonical_temperature": 0.0,
                "task_contract_unchanged": True,
                "historical_canonical_examples": historical_examples.get(cohort, []),
                "historical_examples_are_greedy_parity_evidence": False,
            },
            "output_digest_capture_policy": {
                "capture_output_digest": True,
                "ordinary_benchmark_default": False,
                "scope": "public fixed-task diagnostic only; no raw text or token IDs persisted",
                "canonical_capture_provenance": "explicit public opt-in"
                if args.canonical
                else "no canonical execution",
                "canonical_examples_rerun_for_opt_in_policy": args.canonical,
            },
            "limitations": limitations,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "threads": {name: os.environ[name] for name in THREAD_ENV},
                "offline": True,
            },
            "full_wire_bytes": None,
        }
        target = ROOT / f"docs/evidence/task-qualified-models-{cohort}-2026-10-01"
        if not target.parent.is_dir():
            raise ValueError("evidence directory must already exist")
        # Generated evidence contains metrics/digests only, never model output.
        for suffix, content in (
            (".json", json.dumps(report, indent=2, sort_keys=True) + "\n"),
            (".md", markdown(report)),
        ):
            path = Path(str(target) + suffix)
            path.write_text(content)
        print(
            json.dumps({"cohort": cohort, "comparison": report["comparison"]}, sort_keys=True),
            flush=True,
        )


if __name__ == "__main__":
    configure_environment()
    main()
