"""Bounded Paper 1 audit; installed wheel only, generated data, no downloads/builds.

Run with an already installed wheel's interpreter. --reference-python names an
existing interpreter with Torch and Transformers; it never imports PLLM. Output
must be a new directory. This is functionality evidence, not a performance run.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PREFILL = [0, 2, 3] * 11
DECODE = [5, 7, 9]
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def dependency_lock(name):
    dist = metadata.distribution(name)
    files = {}
    for entry in dist.files or ():
        path = dist.locate_file(entry)
        if path.is_file() and path.suffix in {".py", ".so", ".dylib"}:
            files[str(entry)] = sha(path)
    return {"version": dist.version, "code_files": files}


def reference(model_dir, output):
    """Upstream FP32 oracle in a separate interpreter; no PLLM numerical helper."""
    import numpy as np
    import torch
    import transformers

    torch.set_num_threads(1)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager",
    ).eval()
    arrays = {}
    past = None
    with torch.inference_mode():
        for phase, ids in enumerate([PREFILL, *[[token] for token in DECODE]]):
            result = model(input_ids=torch.tensor([ids]), past_key_values=past, use_cache=True)
            arrays[f"logits_{phase}"] = result.logits[0].numpy().copy()
            past = result.past_key_values
            for layer, state in enumerate(past.layers):
                for name, value in (("key", state.keys), ("value", state.values)):
                    arrays[f"{name}_{phase}_{layer}"] = value[0].transpose(0, 1).numpy().copy()
    assert not any(name == "pllm" or name.startswith("pllm.") for name in sys.modules)
    np.savez(output, **arrays)
    write_json(str(output) + ".environment.json", {
        "python": sys.version, "python_binary_sha256": sha(sys.executable),
        "torch": dependency_lock("torch"), "transformers": dependency_lock("transformers"),
        "pllm_imported": False, "attention": "eager", "dtype": "float32",
        "cache": "upstream cached prefill followed by three one-token decodes",
    })


def command(argv, log, *, timeout=300):
    start = time.monotonic()
    env = {**os.environ, **THREAD_ENV}
    env.pop("PYTHONPATH", None)
    with Path(log).open("x") as stream:
        result = subprocess.run(argv, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=timeout, check=False)
    record = {"argv": list(map(str, argv)), "cwd": str(ROOT),
              "exit_code": result.returncode, "elapsed_seconds": time.monotonic() - start,
              "log": Path(log).name, "log_sha256": sha(log)}
    write_json(str(log) + ".command.json", record)
    if result.returncode:
        raise RuntimeError(f"command exit {result.returncode}; see {log}")
    return record


def provenance(output, wheel):
    import pllm
    from pllm._benchmark_build import inspect

    package = Path(pllm.__file__).resolve().parent
    if package == ROOT / "python/pllm":
        raise RuntimeError("Editable/source import rejected; use an existing installed wheel")
    direct_url = json.loads(metadata.distribution("pllm.run").read_text("direct_url.json") or "{}")
    if direct_url.get("dir_info", {}).get("editable"):
        raise RuntimeError("Editable installation rejected")
    installed_files = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not name.startswith("pllm/") or name.endswith("/"):
                continue
            path = package.parent / name
            expected = hashlib.sha256(archive.read(name)).hexdigest()
            actual = sha(path)
            if expected != actual:
                raise RuntimeError(f"Installed file differs from supplied wheel: {name}")
            installed_files[name] = actual
    build = inspect()
    write_json(output / "installed-build.json", build)
    differences = []
    for relative, digest in build["attributes"]["sdk_sources"].items():
        current = ROOT / "python/pllm" / relative
        current_sha = sha(current) if current.is_file() else None
        if current_sha != digest:
            differences.append({"path": "python/pllm/" + relative,
                                "installed_sha256": digest, "checkout_sha256": current_sha})
    installed_names = set(build["attributes"]["sdk_sources"])
    for path in sorted((ROOT / "python/pllm").rglob("*")):
        if path.is_file() and path.suffix in {".py", ".json"}:
            relative = path.relative_to(ROOT / "python/pllm").as_posix()
            if relative not in installed_names:
                differences.append({"path": "python/pllm/" + relative,
                                    "installed_sha256": None, "checkout_sha256": sha(path)})
    record = {
        "checkout_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "checkout_status_at_start": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "python": sys.executable, "package_path": str(package),
        "version": metadata.version("pllm.run"), "direct_url": direct_url,
        "wheel": str(wheel), "wheel_sha256": sha(wheel),
        "wheel_package_files": installed_files, "build_digest": build["digest"],
        "native_sha256": build["attributes"]["native_sha256"],
        "native_source_commit": None,
        "native_source_commit_note": "Wheel has no authenticated source-commit attestation; byte identity is verified, source revision is not inferred.",
        "checkout_sdk_differences": differences,
        "checkout_native_sha256": sha(ROOT / "python/pllm/_native.abi3.so"),
        "torch": dependency_lock("torch"),
        "installed_distributions": {d.metadata["Name"]: d.version for d in metadata.distributions()},
    }
    write_json(output / "provenance.json", record)
    return {"build_digest": build["digest"], "wheel_sha256": sha(wheel),
            "different_sdk_files": len(differences), "import_scope": "installed wheel"}


def generated_conformance(output, reference_python):
    import numpy as np
    import torch
    from pllm import Model, lower_model
    from pllm.profiles import ClientOnlyCpu
    from pllm.schedulers import MlpPrefillTiling
    from pllm.runtime.loaders import load_hf_directory
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    torch.set_num_threads(1)
    rows = []
    for family in ("qwen2", "qwen3", "llama"):
        root = create_tiny_llama_checkpoint(
            output / "fixtures" / family, seed=23, model_type=family,
            num_hidden_layers=2, with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3",
        )
        reference_path = output / f"reference-{family}.npz"
        command([reference_python, "-I", str(Path(__file__).resolve()),
                 "--reference-model", str(root), "--reference-output", str(reference_path)],
                output / f"reference-{family}.log")
        config = json.loads((root / "config.json").read_text())
        model_id = f"generated-paper-{family}"
        engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
        asyncio.run(engine.load(load_hf_directory(root, model_id=model_id)))
        bundle = ClientBundle.unpack(engine.client_bundle(model_id, placement="client"))
        plan = lower_model(config, batch=1, max_input_tokens=len(PREFILL), max_new_tokens=4)
        trajectories, bindings = [], []
        for tile in (None, 7):
            composition = ClientOnlyCpu(Model(model_id))
            if tile is not None:
                composition = composition.with_params(scheduler=MlpPrefillTiling(tile_rows=tile))
            compiled = compile_runtime_model(plan, bundle, composition=composition)
            runtime = compiled.runtime(compiled.client_linear_executor(engine))
            bindings.append(compiled.to_spec())
            arrays = {}
            for phase, ids in enumerate([PREFILL, *[[token] for token in DECODE]]):
                # forward_ids intentionally evaluates multiple IDs tokenwise. Use
                # the batch phase entry here so this check exercises prefill tiles.
                arrays[f"logits_{phase}"] = runtime._forward(np.asarray(ids, dtype=np.int64)).copy()
                assert runtime.position == len(PREFILL) + phase
                for layer, cache in enumerate(runtime.caches):
                    arrays[f"key_{phase}_{layer}"] = cache.key[:cache.length].copy()
                    arrays[f"value_{phase}_{layer}"] = cache.value[:cache.length].copy()
            if tile is not None:
                assert runtime._mlp_tiling.calls == 2
                assert runtime._mlp_tiling.tiles == 10
                assert runtime._mlp_tiling.peak_tile_rows == 7
            trajectories.append(arrays)
        assert bindings[0]["runtime_schedule_digest"] != bindings[1]["runtime_schedule_digest"]
        errors, checked = {}, 0
        with np.load(reference_path) as upstream:
            for key, values in trajectories[0].items():
                candidate = trajectories[1][key]
                assert values.tobytes() == candidate.tobytes(), key
                expected = upstream[key]
                assert values.shape == expected.shape, (key, values.shape, expected.shape)
                error = float(np.max(np.abs(values - expected)))
                errors[key] = error
                # Predeclared generated-fixture sanity tolerance, not exact W8A8/FP32 identity.
                assert error <= 0.1, (family, key, error)
                checked += values.size
        np.savez(output / f"candidate-{family}.npz", **trajectories[1])
        rows.append({"family": family, "generated": True,
                     "fixture_sha256": {p.name: sha(p) for p in sorted(root.iterdir()) if p.is_file()},
                     "prefill_ids": PREFILL, "same_token_decode_ids": DECODE,
                     "tile_rows": 7, "exact_tiled_vs_untiled_values": checked,
                     "tiled_vs_untiled_bit_exact": True, "upstream_fp32_atol": 0.1,
                     "upstream_max_absolute_errors": errors, "bindings": bindings})
    write_json(output / "generated-conformance.json", rows)
    return {"families": len(rows), "bit_exact_values": sum(r["exact_tiled_vs_untiled_values"] for r in rows)}


def pytest_checks(output):
    groups = {
        "tiling": ["tests/test_mlp_tiling.py", "-k", "component or selected_scheduler"],
        "binding": [
            "tests/test_runtime_model_binding.py::test_compiled_binding_rejects_another_model_source",
            "tests/test_runtime_model_binding.py::test_plan_mismatch_rejected",
            "tests/test_runtime_model_binding.py::test_bundle_tamper_rejected",
            "tests/test_runtime_model_binding.py::test_runtime_rejects_post_bind_mutation",
            "tests/test_runtime_model_binding.py::test_stage_spec_drift_rejected",
            "tests/test_runtime_model_binding.py::test_fused_weight_order_is_bound_to_semantic_outputs",
            "tests/test_live_compiled_execution.py::test_provider_checks_semantic_plan_before_reserving_material",
        ],
        "operators": [
            "tests/test_wavelet_gated.py::test_inputs_and_invalid_configurations_reject_before_any_material",
            "tests/test_wavelet_gated.py::test_native_boundary_range_frame_and_commitment_validation",
            "tests/test_wavelet_gated.py::test_saved_cost_and_quality_keep_source_numeric_and_execution_scopes",
            "tests/test_logrow_product.py::test_invalid_inputs_reject_before_any_garbling",
            "tests/test_logrow_product.py::test_native_frame_and_range_validation",
            "tests/test_logrow_product.py::test_retained_cost_cohort_conserves_roles_and_existing_quality_scope",
            "tests/test_compact_product.py::test_input_and_calibration_admission_precedes_native_material",
            "tests/test_compact_product.py::test_native_malformed_frames_and_resource_limits",
            "tests/test_compact_product.py::test_retained_evidence_binds_frozen_profiles_complete_costs_and_quality_scope",
        ],
    }
    results = {}
    for name, tests in groups.items():
        # Override repository pythonpath=python: current TEST sources, installed implementation.
        snippet = "import pllm,pytest,sys; print('IMPLEMENTATION', pllm.__file__, flush=True); sys.exit(pytest.main(sys.argv[1:]))"
        results[name] = command([sys.executable, "-I", "-c", snippet,
            "-o", "pythonpath=", "-o", "addopts=", "-p", "no:cacheprovider", "-vv",
            f"--junitxml={output / (name + '.xml')}", *tests], output / f"{name}.log")
    return results


def published_requests(output, work):
    import pllm
    from starlette.testclient import TestClient

    example = ROOT / "examples/prepared_sdk.py"
    cli = command([sys.executable, "-I", str(example), "--root", str(work / "example"),
                   "--prompt", "Public generated fixture"], output / "prepared-sdk.log")
    factory = runpy.run_path(str(example))["build_experiment"]
    experiment = factory(root=work / "gateway")
    write_json(output / "published-generated-experiment.json", experiment.to_spec())
    with pllm.serve_local(experiment) as roles:
        with roles.client(bundle_cache_dir=work / "client-artifacts") as client:
            result = client.responses.create(input="Public generated fixture", max_output_tokens=8, temperature=0)
            audit = client.privacy_audit.to_dict()
            assert audit["online_steps"] > 0
            assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0
            text = result.output_text
        with TestClient(roles.gateway_app(local_api_key="paper-generated-fixture")) as gateway:
            reply = gateway.post("/v1/responses", headers={"authorization": "Bearer paper-generated-fixture"},
                json={"model": result.model, "input": "Public generated fixture",
                      "max_output_tokens": 8, "temperature": 0})
            assert reply.status_code == 200, reply.text
            body = reply.json()
            gateway_text = "".join(c.get("text", "") for item in body["output"] for c in item.get("content", []))
            assert gateway_text == text
            assert body["usage"]["output_tokens"] > 0
    record = {"published_source": "examples/prepared_sdk.py", "source_sha256": sha(example),
              "standalone_command": cli, "generated_fixture": True,
              "listing_1_real_checkpoint_executed": False, "sdk_audit": audit,
              "gateway_status": reply.status_code, "gateway_usage": body["usage"],
              "sdk_gateway_text_equal": True,
              "output_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "transport": "In-process HTTP ASGI gateway; real local provider child processes/TCP"}
    write_json(output / "published-requests.json", record)
    return {"standalone_example": "completed", "gateway_status": reply.status_code}


def operators(output):
    from pllm.metrics import benchmark_operator, replay_operator_benchmark
    from pllm.nonlinear import CompactGatedMultiplyQ12, CurlWaveletGatedMultiplyQ9, LogRowGatedMultiplyQ9

    reports = []
    for component in (CurlWaveletGatedMultiplyQ9(), CurlWaveletGatedMultiplyQ9(levels=4),
                      LogRowGatedMultiplyQ9(), CompactGatedMultiplyQ12(max_pieces=4, degree=2)):
        report = benchmark_operator(component, lanes=8, samples=1)
        replay = replay_operator_benchmark(report.configuration)
        assert report.resources == replay.resources
        assert report.quality == replay.quality
        assert report.quality.checked_outputs == report.quality.exact_integer_outputs == 8
        reports.append(report.to_dict())
    assert reports[0]["numeric_contract_digest"] == reports[2]["numeric_contract_digest"]
    write_json(output / "operator-replay.json", {
        "scope": "Four bounded whole gated operators plus fresh replay; same process, not a new matched performance cohort",
        "lanes_per_operator": 8, "samples": 1, "replay_resources_and_quality_equal": True,
        "reports": reports,
    })
    return {"operators": 4, "checked_outputs": 32, "additional_replay_checked_outputs": 32}


def ordinary_benchmark(output, work):
    from pllm import Deployment, ExecutionBudget, Experiment, Model
    from pllm.kernels import Cpu
    from pllm.metrics import (
        BenchmarkCase, BenchmarkSuite, BenchmarkTarget, BuildRef, Sampling, Scenario, benchmark_suite,
    )
    from pllm.preparation import PreparedInventory
    from pllm.profiles import MaskedLinearCpu
    from pllm.schedulers import MlpPrefillTiling

    experiment = Experiment("paper-generated-tiling", MaskedLinearCpu(
        Model.tiny(), kernels=Cpu(threads=1), scheduler=MlpPrefillTiling(tile_rows=7),
        inventory=PreparedInventory(rows=1, refill="on-demand")),
        Deployment.local(root=str(work / "benchmark")),
        ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=2))
    digest = json.loads((output / "installed-build.json").read_text())["digest"]
    suite = BenchmarkSuite((BenchmarkCase("generated-tiling",
        BenchmarkTarget.response(experiment, scenario=Scenario.cold(max_output_tokens=2)),
        build=BuildRef(sys.executable, digest)),),
        Sampling(processes=1, repetitions=1, warmups=0, threads=1,
                 memory_bytes=1 << 30, timeout_seconds=90), name="paper-generated-conformance")
    run = benchmark_suite(suite, inputs={"generated-tiling": {"prompt": "Hi"}},
                          output=output / "ordinary-benchmark.json")
    statuses = [r.to_dict()["status"] for r in run.results]
    assert statuses == ["completed"], statuses
    return {"statuses": statuses, "processes": 1, "scope": "generated functionality, not a performance claim"}


def evidence_locks(output):
    inventory = json.loads((ROOT / "docs/data/research/implementations.json").read_text())["papers"]
    paths = {"scripts/audit_paper_conformance.py", "examples/prepared_sdk.py", "tests/test_mlp_tiling.py",
             "tests/test_prepared_row_residues.py", "tests/conftest.py", "tests/test_live_compiled_execution.py",
             "tests/test_runtime_model_binding.py", "python/pllm/schedulers/__init__.py",
             "python/pllm/runtime/mlp_tiling.py", "python/pllm/runtime/model_binding.py",
             "python/pllm/runtime/semantic_executor.py", "python/pllm/runtime/tiny_llama.py",
             "crates/pllm-compiler/src/mlp_tiling.rs", "docs/data/research/implementations.json",
             "docs/data/research/paper-library.json", "Cargo.lock", "uv.lock", "pyproject.toml"}
    records = {}
    for method in ("curl", "logrow", "compact"):
        paths.update(inventory[method]["code"])
        base = {"curl": "wavelet-gated", "logrow": "logrow-product", "compact": "compact-product"}[method]
        paths.update({f"docs/research/{base}-reference.md", f"tests/test_{base.replace('-', '_')}.py"})
        for path in (f"docs/evidence/research-{base}-cost.json",
                     f"docs/evidence/research-{base}-quality-qwen25.json"):
            full = ROOT / path
            if not full.is_file():
                continue
            paths.add(path)
            raw = json.loads(full.read_text())
            source_records = raw.get("source_sha256", {})
            source_map = source_records if isinstance(source_records, dict) else {}
            records[path] = {
                "sha256": sha(full), "schema": raw.get("schema"),
                "fresh_processes": raw.get("fresh_processes"),
                "checked_outputs": raw.get("checked_outputs"),
                "protected_checkpoint_execution": raw.get("protected_checkpoint_execution"),
                "recorded_native_binary_sha256": raw.get("native_sha256"),
                "recorded_source_sha256": source_records,
                "recorded_source_matches_current_checkout": {
                    p: {"recorded_sha256": h, "checkout_sha256": sha(ROOT / p) if (ROOT / p).is_file() else None,
                        "matches": (ROOT / p).is_file() and sha(ROOT / p) == h}
                    for p, h in source_map.items()
                },
            }
            paths.update(source_map)
    for path in ("docs/evidence/research-mixed-gated-cost.json", "docs/evidence/research-mixed-gated-quality-qwen25.json",
                 "docs/research/mixed-gated-reference.md", "python/pllm/metrics/wavelet_gated.py",
                 "python/pllm/metrics/logrow_product.py", "python/pllm/metrics/compact_product.py"):
        paths.add(path)
    missing = sorted(p for p in paths if not (ROOT / p).is_file())
    if missing:
        raise RuntimeError(f"Source lock paths missing: {missing}")
    library = json.loads((ROOT / "docs/data/research/paper-library.json").read_text())
    paper_records = []
    for row in library["papers"]:
        if row["id"] not in {"compact", "curl", "logrow"}:
            continue
        path = ROOT / row["file"]
        observed = sha(path) if path.is_file() else None
        if observed is not None and observed != row["sha256"]:
            raise RuntimeError(f"Local paper differs from library record: {row['id']}")
        paper_records.append({"library_record": row, "local_pdf_sha256": observed})
    write_json(output / "source-evidence-lock.json", {
        "files": {p: sha(ROOT / p) for p in sorted(paths)}, "historical_records": records,
        "paper_records": paper_records,
        "note": "Historical bytes are frozen, not rerun. Source records and archived cohorts keep their own identities.",
    })
    return {"locked_files": len(paths), "historical_records": len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--reference-python")
    parser.add_argument("--reference-model", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--reference-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ.update(THREAD_ENV)
    if args.reference_model:
        reference(args.reference_model, args.reference_output)
        return
    if not (args.output and args.wheel and args.reference_python):
        parser.error("--output, --wheel and --reference-python are required")
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    (output / "driver.py").write_bytes(Path(__file__).read_bytes())
    temporary_root = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "opencode"
    if not temporary_root.is_dir():
        temporary_root = Path(tempfile.gettempdir())
    manifest = {"schema": "pllm.paper_conformance.v1", "argv": sys.argv,
                "scope": "Generated functionality and bounded operator checks; no real checkpoint or resource-tier qualification",
                "thread_environment": THREAD_ENV, "cases": []}
    with tempfile.TemporaryDirectory(prefix="paper-conformance-", dir=temporary_root) as directory:
        work = Path(directory)
        os.environ["XDG_CACHE_HOME"] = str(work / "cache")
        cases = [("provenance", lambda: provenance(output, args.wheel.absolute())),
                 ("source-evidence-lock", lambda: evidence_locks(output)),
                 ("generated-independent-conformance", lambda: generated_conformance(output, args.reference_python)),
                 ("focused-current-test-sources", lambda: pytest_checks(output)),
                 ("published-sdk-gateway", lambda: published_requests(output, work)),
                 ("ordinary-generated-benchmark", lambda: ordinary_benchmark(output, work)),
                 ("bounded-complete-operators", lambda: operators(output))]
        for name, run in cases:
            print(name, flush=True)
            start = time.monotonic()
            try:
                result = run()
                item = {"name": name, "status": "passed", "result": result}
            except Exception as exc:
                error = output / f"{name}.error.txt"
                error.write_text(traceback.format_exc())
                item = {"name": name, "status": "failed", "error": str(exc), "traceback": error.name}
            item["elapsed_seconds"] = time.monotonic() - start
            manifest["cases"].append(item)
            # A provenance failure must not fall through into unchecked execution.
            if name == "provenance" and item["status"] != "passed":
                break
    manifest["artifacts"] = {p.relative_to(output).as_posix(): sha(p)
                             for p in sorted(output.rglob("*")) if p.is_file()}
    manifest["status"] = "passed" if all(c["status"] == "passed" for c in manifest["cases"]) else "failed"
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "output": str(output)}), flush=True)
    if manifest["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
