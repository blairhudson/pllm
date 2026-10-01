"""Bounded existing semantic placements: archived-body forecasts, not traffic.

Default replays retained public geometry and ledgers without a checkpoint.
--collect-source reads only the pinned local cache, quantizing one stage at a
time with one thread. --verify-tiny executes matched native W8A8 logits on a
generated checkpoint; its clear remote callback is not a protocol measurement.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

if __name__ == "__main__":
    for _key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[_key] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np

from pllm import Deployment, ExecutionBudget, Experiment, Model, lower_model
from pllm.kernels import Cpu
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu, resolve_runtime_composition
from pllm.roles import ClientLinearRoles, ClientPrefixLayers
from pllm.runtime.semantic_stages import (
    client_owns_linear,
    scheduled_stage_specs,
    semantic_stage_role,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/evidence/placement-frontier-screen-2026-10-01.json"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SOURCE_LOCK = "880f80ec61274d3c80e2a0c1336394b9e955d53ff98436a426a680a460a592e7"
CHECKPOINT_SOURCE_LOCK = "383f5ed6947ba56ceed6083a9503d6302f09559e1e57ee614409900b36084c29"
CHECKPOINT_DIGEST = "f69322057253c4fb853c10e73da5d84111262e0fb1edba33933b9e5b8caf59fd"
BODY = "5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974"
ROLES = ("attention_output", "mlp_down", "mlp_gate_up", "qkv_projection")
BASELINE = "prepared-request-sized"
ATTENTION = "client-attention"
EDGES = (
    "client->preparation",
    "preparation->client",
    "preparation->inference",
    "client->inference",
    "inference->client",
)
RAW_LOCKS = {
    "qwen-incremental-32-warm.json": "871d6d7c2aa3d5cc0f337e2aa73e73ebff10e7b82510816b59e99d2ae82665d8",
    "qwen-incremental-32-cold.json": "aa1c05efa40628c3b458ff39a36d35a641639dcd4e54b04d8edfb006960b8011",
}
ARCHIVE_LOCK = "fad309506f2cc6d537b0778ca7d29c3ef4fb7a9a70ab8df66b99d114da586082"
# Filled from the retained full-schedule source/ledger collection; replay fails closed.
RETAINED_LOCK = "d44f3111edce0d462bbffbd4860a64f42c8823045dc675d111a5b7a250e473bd"
BUDGETS = (
    ("small", 0.95, 64 << 20),
    ("moderate", 0.90, 128 << 20),
    ("remote80", 0.80, 160 << 20),
    ("remote70", 0.70, 256 << 20),
    ("remote60", 0.60, 320 << 20),
)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def locked_json(path: Path, lock: str) -> dict:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != lock:
        raise ValueError(f"source lock mismatch: {path.name}")
    return json.loads(data)


def candidates() -> list[dict]:
    result = [{"name": BASELINE, "roles": [], "prefix_layers": 0}]
    for mask in range(1, 16):
        roles = [role for i, role in enumerate(ROLES) if mask & (1 << i)]
        name = (
            ATTENTION
            if roles == ["attention_output", "qkv_projection"]
            else "roles-" + "+".join(roles)
        )
        result.append({"name": name, "roles": roles, "prefix_layers": 0})
    result.extend(
        {"name": f"client-prefix-{n}", "roles": [], "prefix_layers": n} for n in range(1, 9)
    )
    return result


def composition(candidate: dict, source: Model) -> MaskedLinearCpu:
    placement = (
        ClientLinearRoles(candidate["roles"])
        if candidate["roles"]
        else ClientPrefixLayers(candidate["prefix_layers"])
        if candidate["prefix_layers"]
        else None
    )
    return MaskedLinearCpu(source, kernels=Cpu(threads=1), placement=placement)


def owns(stage: dict, candidate: dict) -> bool:
    return client_owns_linear(
        {"op": "linear", "role": stage["role"], "layer_index": stage["layer"]},
        client_prefix_layers=candidate["prefix_layers"],
        client_linear_roles=tuple(candidate["roles"]),
    )


def verify_schedule(config: dict, candidate: dict, source: Model) -> dict:
    selected = composition(candidate, source)
    experiment = Experiment(
        candidate["name"],
        selected,
        Deployment.local(root="local://placement-frontier"),
        ExecutionBudget(requests=3, max_input_tokens=39, max_new_tokens=32),
    )
    resolved = experiment.resolve()
    numeric = resolve_runtime_composition(selected)
    if numeric is None or (numeric.weight_bits, numeric.activation_bits) != (8, 8):
        raise ValueError("candidate must resolve to W8A8 with one thread")
    if selected.kernels.params["threads"] != 1:
        raise ValueError("candidate must resolve to W8A8 with one thread")
    if tuple(resolved.client_linear_roles) != tuple(candidate["roles"]):
        raise ValueError("candidate role resolution differs")
    if resolved.client_prefix_layers != candidate["prefix_layers"]:
        raise ValueError("candidate prefix resolution differs")
    plan = lower_model(config, batch=1, max_input_tokens=39, max_new_tokens=32)
    schedule = plan.runtime_schedule(selected)
    if not schedule.complete or schedule.protected_execution:
        raise ValueError("schedule is not complete existing public execution")
    phases = {}
    head_rows = {}
    for phase in ("prefill", "decode"):
        operations = {op["id"]: op for op in getattr(plan, phase)["operations"]}
        head = [op for op in operations.values() if op["operator"] == "output_head"]
        if len(head) != 1 or tuple(head[0]["output_shape"]) != (1, config["vocab_size"]):
            raise ValueError("expected one client output-head logit row per phase")
        head_rows[phase] = 1
        counts = Counter()
        for step in schedule.to_dict()[phase]["steps"]:
            if step["layer"] is None or not step["weight_ids"]:
                continue
            role = semantic_stage_role(step, operations)
            expected = (
                "client_linear"
                if owns({"role": role, "layer": step["layer"]}, candidate)
                else "remote_stage"
            )
            if step["executor"] != expected:
                raise ValueError("native executor differs from semantic ownership")
            counts[expected] += 1
        phases[phase] = dict(sorted(counts.items()))
    if phases["prefill"] != phases["decode"]:
        raise ValueError("prefill/decode ownership differs")
    return {
        "resolved": True,
        "plan_digest": plan.digest,
        "schedule_digest": schedule.digest,
        "phase_body_stage_counts": phases,
        "phase_output_head_rows": head_rows,
    }


def extract_archives(archive_dir: Path) -> dict:
    """Retain sanitized per-stage bodies; bind to original raw and public summary."""
    summary = locked_json(
        ROOT / "docs/evidence/incremental-network-qwen25-2026-10-01.json", ARCHIVE_LOCK
    )
    result = {}
    for phase in ("warm", "cold"):
        filename = f"qwen-incremental-32-{phase}.json"
        raw = locked_json(archive_dir / filename, RAW_LOCKS[filename])
        cohort = next(c for c in summary["cohorts"] if c["file"] == filename)
        result[phase] = {}
        for entry in raw["candidates"]:
            if entry["name"] not in (BASELINE, ATTENTION, "client-prefix-one", "client-prefix-two"):
                continue
            name = {
                "client-prefix-one": "client-prefix-1",
                "client-prefix-two": "client-prefix-2",
            }.get(entry["name"], entry["name"])
            report = entry["report"]
            record = report["runs"][0]
            stage = report["topology_accounting"]["stages"]["runs"][0]
            ledger = report["topology_accounting"]["runs"][0]
            control = next(c for c in cohort["candidates"] if c["name"] == entry["name"])
            if (
                report["configuration"]["source_lock_digest"] != SOURCE_LOCK
                or record["model_fingerprint"] != BODY
                or (record["tokens"]["input_tokens"], record["tokens"]["output_tokens"]) != (39, 32)
                or record["cold"] != (phase == "cold")
                or not all(report["checks"].values())
                or not stage["reconciled_with_protocol_bodies"]
                or record["inventory"]
                != {"burned": 0, "consumed": 70, "generated": 70, "required": 70, "reused": 0}
                or report["configuration"]["prompt_digest"]
                != cohort["comparison_key"]["prompt_digest"]
                or ledger["all_link_serialized_body_bytes"]
                != control["summary"]["median_covered_all_link_serialized_body_bytes"]
            ):
                raise ValueError("archive source/workload/ledger mismatch")
            result[phase][name] = {
                "prompt_digest": report["configuration"]["prompt_digest"],
                "edges": ledger["body_bytes_by_edge"],
                "stages": stage["body_bytes_by_stage_and_edge"],
                "total_body_bytes": ledger["all_link_serialized_body_bytes"],
                "online_body_bytes": ledger["online_all_link_serialized_body_bytes"],
                "placement": report["client_body_placement"],
                "cold_cpu": report["process_cpu_accounting"],
                "run_cpu_seconds_by_role": ledger["run_window_cpu_seconds_by_role"],
            }
    return result


def collect_source(archive_dir: Path) -> dict:
    from pllm.model_loader import resolve_model
    from pllm.runtime.preparation_protocol import seeded_ring_profile
    from pllm.runtime.safetensors_store import SafeTensorStore
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    # Environment is offline before resolution. No fallback to downloading.
    source = resolve_model(Model.hf(MODEL, revision=REVISION))
    if (
        source.source_lock_digest != CHECKPOINT_SOURCE_LOCK
        or source.checkpoint_digest != CHECKPOINT_DIGEST
        or source.path is None
    ):
        raise ValueError("cached checkpoint source lock differs")
    config = json.loads((source.path / "config.json").read_text())
    plan = lower_model(config, batch=1, max_input_tokens=39, max_new_tokens=32)
    specs = scheduled_stage_specs(
        plan, composition(candidates()[0], Model.hf(MODEL, revision=REVISION))
    )
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=1)
    store = SafeTensorStore(source.path)
    rows = []
    for original in specs:
        if original.layer_index is None:
            continue
        spec = replace(original, weight_bits=8, activation_bits=8)
        sources = engine._resolve_stage_sources(store, spec, source.manifest)
        weight = engine._quantize_sources(store, spec, sources)
        bound = int(np.abs(weight.values.astype(np.int16)).sum(axis=1, dtype=np.int64).max()) * 127
        if weight.values.shape != (spec.out_features, spec.in_features):
            raise ValueError("source orientation differs from compiled shape")
        rows.append(
            {
                "id": spec.id,
                "role": spec.role,
                "layer": spec.layer_index,
                "in_features": spec.in_features,
                "out_features": spec.out_features,
                "i8_weight_bytes": weight.values.nbytes,
                "f32_scale_bytes": weight.scales.nbytes,
                "weight_sha256": hashlib.sha256(weight.values.tobytes()).hexdigest(),
                "scales_sha256": hashlib.sha256(weight.scales.tobytes()).hexdigest(),
                "signed_output_bound": bound,
                "wire_bits": seeded_ring_profile(bound).wire_bits,
            }
        )
        store.clear_cache()
        del weight
        gc.collect()
    print(
        f"Collected {len(rows)} public W8A8 stages, one resident stage, one thread.",
        file=sys.stderr,
    )
    return {
        "source": {
            "model": MODEL,
            "revision": REVISION,
            "source_lock_digest": source.source_lock_digest,
            "checkpoint_digest": source.checkpoint_digest,
            "archived_body_fingerprint": BODY,
            "archived_execution_source_lock_digest": SOURCE_LOCK,
        },
        "config": config,
        "stages": rows,
        "archives": extract_archives(archive_dir),
    }


def validate_retained(retained: dict) -> None:
    if digest(retained) != RETAINED_LOCK:
        raise ValueError("retained geometry/directed ledger source lock mismatch")
    stages = retained["stages"]
    if len(stages) != 96 or len({s["id"] for s in stages}) != 96:
        raise ValueError("expected complete 96-stage body")
    for stage in stages:
        if stage["role"] not in ROLES or stage["wire_bits"] not in (16, 24, 32):
            raise ValueError("unsupported source stage/ring")
        if (
            stage["i8_weight_bytes"] != stage["in_features"] * stage["out_features"]
            or stage["f32_scale_bytes"] != 4 * stage["out_features"]
        ):
            raise ValueError("invalid measured matrix/scale shape")
    for phase, archive in retained["archives"].items():
        for item in archive.values():
            edge_totals = {
                edge: sum(row.get(edge, 0) for row in item["stages"].values()) for edge in EDGES
            }
            stage_total = sum(edge_totals.values())
            if any(
                type(v) is not int or v < 0 for row in item["stages"].values() for v in row.values()
            ):
                raise ValueError("invalid archived directed stage body")
            if sum(e["serialized_body_bytes"] for e in item["edges"]) != item["total_body_bytes"]:
                raise ValueError("directed ledger sum differs")
            if stage_total > item["total_body_bytes"]:
                raise ValueError("stage bodies exceed ledger")
            if phase == "warm" and any(
                edge_totals[f"{e['source']}->{e['destination']}"] > e["serialized_body_bytes"]
                for e in item["edges"]
                if e["phase"] != "cold"
            ):
                raise ValueError("stage edge exceeds ledger")
    if (
        retained["archives"]["warm"][BASELINE]["total_body_bytes"] != 178970558
        or retained["archives"]["warm"][ATTENTION]["total_body_bytes"] != 148297986
    ):
        raise ValueError("archived controls differ")


def forecast(retained: dict, candidate: dict) -> dict:
    local = [s for s in retained["stages"] if owns(s, candidate)]
    remote = [s for s in retained["stages"] if not owns(s, candidate)]
    control = retained["archives"]["warm"][BASELINE]
    archive = retained["archives"]["warm"].get(candidate["name"])
    # Surviving baseline stage bodies are a forecast for unchanged remote stages.
    # Existing placements retain their actual setup remainder for calibration;
    # new placements use the baseline setup allowance, not a bound or measurement.
    setup_source = archive or control
    setup = {
        edge: sum(
            e["serialized_body_bytes"]
            for e in setup_source["edges"]
            if f"{e['source']}->{e['destination']}" == edge and e["phase"] != "cold"
        )
        - sum(row.get(edge, 0) for row in setup_source["stages"].values())
        for edge in EDGES
    }
    edges = {
        edge: sum(control["stages"][s["id"]].get(edge, 0) for s in remote) + setup[edge]
        for edge in EDGES
    }
    online = edges["client->inference"] + edges["inference->client"]
    weight = sum(s["i8_weight_bytes"] for s in local)
    scales = sum(s["f32_scale_bytes"] for s in local)
    body = sum(s["i8_weight_bytes"] for s in retained["stages"])
    base_cold = retained["archives"]["cold"][BASELINE]
    baseline_bundle = base_cold["total_body_bytes"] - control["total_body_bytes"]
    bundle_payload = baseline_bundle + weight + scales
    tensor_edges = {
        "client->preparation": 32 * len(remote),
        "preparation->client": 0,
        "preparation->inference": sum(
            70 * s["out_features"] * (s["wire_bits"] // 8) for s in remote
        ),
        "client->inference": sum(70 * s["in_features"] * (s["wire_bits"] // 8) for s in remote),
        "inference->client": sum(70 * s["out_features"] * (s["wire_bits"] // 8) for s in remote),
    }
    # Output head remains client-owned. Show both one-head-row-per-body-row proxy
    # and emitted-logit-row workload count; CPU is a separate unmeasured cap.
    head = retained["config"]["hidden_size"] * retained["config"]["vocab_size"]
    row = {
        **candidate,
        "local_stage_count": len(local),
        "remote_stage_count": len(remote),
        "body_client_macs_per_row": weight,
        "body_remote_macs_per_row": body - weight,
        "body_remote_mac_fraction": (body - weight) / body,
        "body_client_macs_70_rows": weight * 70,
        "remote_preparation_macs_70_rows": (body - weight) * 70,
        "client_head_macs_per_logit_row": head,
        "all_linear_remote_fraction_one_head_row_per_body_row": (body - weight) / (body + head),
        "all_linear_remote_fraction_70_body_32_head_rows": 70
        * (body - weight)
        / (70 * body + 32 * head),
        "all_linear_workload_scope": "native schedule verifies one head row per prefill/decode: 1+31=32 client head rows, 39+31=70 body rows; token lookup is not a dense MAC",
        "whole_cpu_remote_fraction": None,
        "whole_cpu_cap_admitted": False,
        "client_body_i8_weight_bytes": weight,
        "client_body_scale_bytes": scales,
        "native_body_i8_snapshot_bytes_forecast": weight,
        "client_body_resident_array_bytes_floor": 2 * weight + scales,
        "client_mask_uint32_array_bytes": 70
        * 4
        * sum(s["in_features"] + s["out_features"] for s in remote),
        "client_body_plus_masks_array_bytes_floor": 2 * weight
        + scales
        + 70 * 4 * sum(s["in_features"] + s["out_features"] for s in remote),
        "client_total_peak_ram_bytes": None,
        "client_checkpoint_disk_bytes": None,
        "directed_warm_body_bytes_forecast": edges,
        "setup_allowance_bytes": sum(setup.values()),
        "setup_allowance_source": "archived same placement"
        if archive
        else "baseline allowance; not proven bound",
        "warm_all_link_body_bytes_forecast": sum(edges.values()),
        "online_all_link_body_bytes_forecast": online,
        "directed_tensor_and_root_payload_bytes_70_rows": tensor_edges,
        "cold_inference_to_client_bundle_body_bytes_payload_forecast": bundle_payload,
        "cold_all_link_body_bytes_payload_forecast": sum(edges.values()) + bundle_payload,
        "cold_bundle_metadata_delta_bytes": None,
        "forecast_status": "archived stage subtraction + setup allowance + cold payload; not measured candidate traffic",
        "accounted_body_forecast_for_cached_model_response_counts": {
            str(n): bundle_payload + n * sum(edges.values()) for n in (1, 2, 8, 32)
        },
    }
    if archive:
        cold = retained["archives"]["cold"][candidate["name"]]
        row["archive_calibration"] = {
            "warm_measured_body_bytes": archive["total_body_bytes"],
            "warm_forecast_minus_archive_bytes": sum(edges.values()) - archive["total_body_bytes"],
            "online_measured_body_bytes": archive["online_body_bytes"],
            "online_forecast_minus_archive_bytes": online - archive["online_body_bytes"],
            "cold_measured_body_bytes": cold["total_body_bytes"],
            "cold_payload_forecast_minus_archive_bytes": row[
                "cold_all_link_body_bytes_payload_forecast"
            ]
            - cold["total_body_bytes"],
            "native_snapshot_measured_bytes": archive["placement"]["native_body_i8_snapshot_bytes"],
            "whole_cpu_remote_fraction": None,
        }
    saved = control["total_body_bytes"] - sum(edges.values())
    row["warm_body_saving_fraction"] = saved / control["total_body_bytes"]
    row["amortization_first_response_count_beating_baseline_payload_forecast"] = (
        (weight + scales) // saved + 1 if saved > 0 else None
    )
    return row


def dominates(a: dict, b: dict, keys: tuple[str, ...]) -> bool:
    return all(a[k] <= b[k] for k in keys) and any(a[k] < b[k] for k in keys)


def frontier(rows: list[dict], keys: tuple[str, ...]) -> list[str]:
    return [b["name"] for b in rows if not any(dominates(a, b, keys) for a in rows)]


def public_experiment(name: str) -> Experiment:
    candidate = next(c for c in candidates() if c["name"] == name)
    pipeline = composition(candidate, Model.hf(MODEL, revision=REVISION))
    pipeline = pipeline.with_params(inventory=PreparedInventory("request-sized"))
    return Experiment(
        name,
        pipeline,
        Deployment.local(root="local://placement-frontier"),
        ExecutionBudget(requests=8, max_input_tokens=512, max_new_tokens=32),
    )


# Ordinary Experiment targets using existing components; import performs no model load.
public_baseline = public_experiment(BASELINE)
public_qkv = public_experiment("roles-qkv_projection")
public_output = public_experiment("roles-attention_output")
public_attention = public_experiment(ATTENTION)


def review(result: dict) -> dict:
    names = {b["lowest_warm_body_forecast_candidate"] for b in result["budgets"]}
    names.add(BASELINE)
    return {
        "schema": "pllm.placement_frontier_review.v1",
        "date": result["date"],
        "status": "bounded placement improvement, not a 10x network method",
        "retained_inputs_sha256": result["retained_inputs_sha256"],
        "source": result["source"],
        "workload": result["workload"],
        "budgets": result["budgets"],
        "budget_winners_and_baseline": [r for r in result["candidates"] if r["name"] in names],
        "archive_calibrations": {
            r["name"]: r["archive_calibration"]
            for r in result["candidates"]
            if "archive_calibration" in r
        },
        "valid_placements": len(result["candidates"]),
        "rejected_candidates": result["rejected_candidates"],
        "tiny_parity_valid_placements": len(result["tiny_numerical_parity"]["candidates"]),
        "tiny_parity_all_exact": result["tiny_numerical_parity"]["all_exact"],
        "source_ring_counts": result["source_ring_counts"],
        "next_gate": "matched pinned no-download W8A8 baseline, QKV-only, output-only and attention 39+32 cold/warm loopback; reconcile every directed stage body, logit parity, head MACs, full process CPU and complete client peak RAM/storage before admission",
        "omissions": result["omissions"],
    }


def build(retained: dict, tiny: dict | None = None) -> dict:
    validate_retained(retained)
    accepted, rejected = [], []
    source = Model.hf(MODEL, revision=REVISION)
    for candidate in candidates():
        try:
            schedule = verify_schedule(retained["config"], candidate, source)
        except (ValueError, RuntimeError) as error:
            rejected.append({**candidate, "reason": str(error)})
            continue
        row = forecast(retained, candidate)
        row["schedule_verification"] = schedule
        accepted.append(row)
    keys = (
        "warm_all_link_body_bytes_forecast",
        "body_client_macs_per_row",
        "client_body_resident_array_bytes_floor",
    )
    budgets = []
    for name, remote_fraction, storage in BUDGETS:
        eligible = [
            r
            for r in accepted
            if r["body_remote_mac_fraction"] >= remote_fraction
            and r["client_body_resident_array_bytes_floor"] <= storage
        ]
        best = min(eligible, key=lambda r: r["warm_all_link_body_bytes_forecast"])
        budgets.append(
            {
                "name": name,
                "minimum_body_remote_mac_fraction": remote_fraction,
                "maximum_client_body_resident_array_bytes_floor": storage,
                "eligible_candidates": [r["name"] for r in eligible],
                "warm_pareto": frontier(eligible, keys),
                "cold_pareto": frontier(
                    eligible,
                    (
                        "cold_all_link_body_bytes_payload_forecast",
                        "body_client_macs_per_row",
                        "client_body_resident_array_bytes_floor",
                    ),
                ),
                "lowest_warm_body_forecast_candidate": best["name"],
                "admission": "body MAC + body array floors only; whole CPU, peak RAM and full storage unadmitted",
            }
        )
    eligible_all_linear = [
        r
        for r in accepted
        if r["all_linear_remote_fraction_70_body_32_head_rows"] >= 0.80
        and r["client_body_resident_array_bytes_floor"] <= 160 << 20
    ]
    budgets.append(
        {
            "name": "all-linear-remote80",
            "minimum_all_linear_remote_mac_fraction": 0.80,
            "maximum_client_body_resident_array_bytes_floor": 160 << 20,
            "eligible_candidates": [r["name"] for r in eligible_all_linear],
            "warm_pareto": frontier(eligible_all_linear, keys),
            "lowest_warm_body_forecast_candidate": min(
                eligible_all_linear, key=lambda r: r["warm_all_link_body_bytes_forecast"]
            )["name"],
            "admission": "body+head MAC cap; whole CPU and complete storage remain unadmitted",
        }
    )
    return {
        "schema": "pllm.bounded_semantic_placement_frontier.v1",
        "date": "2026-10-01",
        "source": retained["source"],
        "workload": {"input_tokens": 39, "output_tokens": 32, "fresh_rows_per_stage": 70},
        "retained_inputs_sha256": digest(retained),
        "retained_inputs": retained,
        "archive_sha256": RAW_LOCKS,
        "public_archive_sha256": ARCHIVE_LOCK,
        "resource_limits": {
            "one_stage_at_a_time": True,
            "threads": 1,
            "no_download": True,
            "no_training": True,
        },
        "source_ring_counts": dict(
            sorted(Counter(s["wire_bits"] for s in retained["stages"]).items())
        ),
        "candidates": accepted,
        "rejected_candidates": rejected,
        "warm_pareto": frontier(accepted, keys),
        "budgets": budgets,
        "tiny_numerical_parity": tiny,
        "omissions": [
            "Forecasts are not measured candidate protocol traffic; baseline setup allowance is not an upper bound.",
            "Cold payload forecast omits placement-dependent metadata and layout changes; native snapshots are resident copies, not a second weight transfer.",
            "Checkpoint distribution, full persistent storage, peak RAM, embedding/head/nonlinear arrays, import temporaries and allocator overhead are unmeasured.",
            "Covered five stage directions plus cold inference->client bundle exclude some forwarding/ack/status/control bodies and HTTP/TLS/framing/retries; missing bytes are unknown, not zero.",
            "Body MAC cap excludes client head, attention/nonlinear work, masks, serialization, preparation CPU and startup; all-linear ratios do not establish whole CPU admission.",
            "Tiny clear native callback checks arithmetic/placement parity, not cryptographic traffic, public-model quality or operator independence.",
            "No combined prefix+role placement, arbitrary layer subset, q/k/v split or compiler-contract change is introduced.",
        ],
    }


def verify_tiny() -> dict:
    from pllm.model_loader import resolve_model
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    # Same 24-layer ownership proportions as public source; small deterministic weights.
    with tempfile.TemporaryDirectory(prefix="pllm-placement-") as temp:
        root = create_tiny_llama_checkpoint(Path(temp) / "tiny", num_hidden_layers=24)
        source_model = Model.path(str(root), model_id="placement-tiny")
        source = resolve_model(source_model)
        config = json.loads((root / "config.json").read_text())
        plan = lower_model(config, batch=1, max_input_tokens=39, max_new_tokens=3)
        references = None
        observations = []
        for candidate in candidates():
            selected = composition(candidate, source_model)
            try:
                verify_schedule(config, candidate, source_model)
            except (ValueError, RuntimeError):
                continue
            engine = MaskedTransformerEngine(
                threads=1,
                weight_bits=8,
                activation_bits=8,
                client_linear_roles=tuple(candidate["roles"]),
                client_prefix_layers=candidate["prefix_layers"],
            )
            asyncio.run(engine.load(deepcopy(source.manifest)))
            try:
                model = engine.models[source.manifest.id]
                bundle = ClientBundle.unpack(engine.client_bundle(source.manifest.id))
                compiled = compile_runtime_model(plan, bundle, composition=selected)
                calls = []

                def remote(stage_id, activation):
                    stage = model.stages[stage_id]
                    if owns({"role": stage.spec.role, "layer": stage.spec.layer_index}, candidate):
                        raise ValueError("client-owned stage reached remote callback")
                    values = quantize_activation_per_row(activation, bits=8)
                    output = dequantize_matmul(
                        stage.compiled_weight.clear(values.values),
                        values.scales,
                        stage.weight.scales,
                        output_shape=values.original_shape[:-1] + (stage.spec.out_features,),
                    )
                    if stage.bias is not None:
                        output += stage.bias
                    calls.append(stage_id)
                    return np.ascontiguousarray(output, dtype=np.float32)

                observed = []
                for text in ("Public parity.", "Seven plus nine?", "A river flows."):
                    runtime = compiled.runtime(remote)
                    ids = runtime.encode_prompt(text)
                    _, logits, _ = runtime.prepare_ids(ids)
                    observed.append(logits.copy())
                    for _ in range(2):
                        logits = runtime.forward_ids([int(np.argmax(logits))])[-1]
                        observed.append(logits.copy())
                if references is None:
                    references = observed
                for reference, actual in zip(references, observed, strict=True):
                    np.testing.assert_array_equal(reference, actual)
                local = [
                    s
                    for s in bundle.stages.values()
                    if s.layer_index is not None and s.client_weight is not None
                ]
                observations.append(
                    {
                        "name": candidate["name"],
                        "exact_prefill_decode_logits": True,
                        "compared_logit_vectors": len(observed),
                        "max_abs_logit_error": max(
                            float(np.abs(a - b).max())
                            for a, b in zip(references, observed, strict=True)
                        ),
                        "logits_sha256": hashlib.sha256(
                            b"".join(a.tobytes() for a in observed)
                        ).hexdigest(),
                        "remote_stage_calls": len(calls),
                        "body_fingerprint": model.manifest.metadata["body_fingerprint"],
                        "bundle_body_bytes": len(engine.client_bundle(source.manifest.id)),
                        "client_body_weight_bytes": sum(s.client_weight.nbytes for s in local),
                        "client_body_scale_bytes": sum(
                            s.client_weight_scales.nbytes for s in local
                        ),
                        "native_body_snapshot_bytes": sum(
                            bundle.stages[key].client_weight.nbytes
                            for key in bundle._local_matrices
                            if bundle.stages[key].layer_index is not None
                        ),
                        "native_head_snapshot_bytes": sum(
                            bundle.stages[key].client_weight.nbytes
                            for key in bundle._local_matrices
                            if bundle.stages[key].role == "lm_head"
                        ),
                    }
                )
            finally:
                asyncio.run(engine.unload(source.manifest.id))
                del engine
                gc.collect()
        if len({o["body_fingerprint"] for o in observations}) != 1:
            raise ValueError("tiny source fingerprint differs by placement")
        return {
            "scope": "generated seed23 Qwen2 W8A8, 24 layers, hidden32, intermediate64, one thread; native clear callback, no network",
            "prompts": 3,
            "decode_steps_per_prompt": 2,
            "candidates": observations,
            "all_exact": all(o["exact_prefill_decode_logits"] for o in observations),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-source", action="store_true")
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("/var/folders/3d/16hqtng12ng_bfz1n99jg04w0000gn/T/opencode"),
    )
    parser.add_argument("--verify-tiny", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--section",
        choices=(
            "budgets",
            "candidates",
            "rejected_candidates",
            "tiny_numerical_parity",
            "retained_inputs_sha256",
            "review",
        ),
    )
    args = parser.parse_args()
    previous = json.loads(ARTIFACT.read_text()) if ARTIFACT.exists() else {}
    retained = (
        collect_source(args.archive_dir) if args.collect_source else previous["retained_inputs"]
    )
    tiny = verify_tiny() if args.verify_tiny else previous.get("tiny_numerical_parity")
    result = build(retained, tiny)
    selected = (
        review(result)
        if args.section == "review"
        else result[args.section]
        if args.section
        else result
    )
    text = json.dumps(selected, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
