"""Serial, bounded installed-wheel replay for the five remaining Paper 1 families.

Uses existing dependencies and native admission; no build, installation, downloads,
checkpoint cohort or performance claim. Output directory must be new.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "paper/artifacts/conformance"
OLD = ARCHIVE / "2026-10-16-installed-v4"
BUILD = ARCHIVE / "2026-10-16-local-build-v1"
WHEEL = ROOT / "target/certified-head-sdk-build-v3/wheels/pllm_run-0.1.0a2-cp311-abi3-macosx_11_0_arm64.whl"
ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
}
TESTS = {
    "slalom": ["tests/test_native_freivalds.py",
               "tests/test_live_compiled_execution.py::test_verified_compiled_decoder_uses_prepared_roles_and_gateway"],
    "equalization": ["tests/test_public_equalization.py"],
    "masked-wire": [
        "tests/test_masked_wire_sdk.py::test_fused_complete_block_preserves_numerics_and_prices_all_work[1]",
        "tests/test_masked_wire_sdk.py::test_fail_closed_if_binding_disagrees_and_fresh_session_per_sample",
        "tests/test_masked_wire_sdk.py::test_replay_rejects_changed_or_extra_contract_fields",
    ],
    "embedding": [
        "tests/test_packed_embedding.py::test_conversion_corrects_cross_slot_borrows_and_sign_extends_before_resharing",
        "tests/test_packed_embedding.py::test_public_geometry_and_input_bounds_reject_before_model_or_material_allocation",
        "tests/test_packed_embedding.py::test_bounded_seal_codec_rejects_length_and_padding_before_backend_load",
        "tests/test_packed_embedding.py::test_native_geometry_and_code_bounds_reject",
        "tests/test_packed_embedding.py::test_operator_configuration_cannot_activate_unimplemented_decoder_placement",
        "tests/test_repacked_embedding.py::test_party_input_type_and_foreign_context_burn_before_conversion",
        "tests/test_repacked_embedding.py::test_repacking_admission_precedes_material",
    ],
    "encrypted-softmax": [
        "tests/test_encrypted_softmax.py::test_invalid_private_range_and_geometry_reject_before_material",
    ],
}
EVIDENCE = [
    "research-slalom-qwen25.json", "research-slalom-admission.json",
    "slalom-prepared-topologies-cold-cpu-2026-09-26.json",
    "research-smoothquant-calibration-qwen25.json", "research-smoothquant-quality-qwen25.json",
    "research-smoothquant-qwen25.json", "research-smoothquant-controls-qwen25.json",
    "research-masked-wire-reference-qwen25.json", "research-masked-wire-fusion-sdk.json",
    "research-packed-embedding-cost.json", "research-repacked-embedding-cost.json",
    "research-packed-embedding-quality-qwen25.json", "research-encrypted-softmax-cost.json",
    "research-encrypted-attention-range-qwen25.json",
]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def child_operator(name, output):
    from pllm.metrics import benchmark_operator, replay_operator_benchmark
    from pllm.nonlinear import MaskedWireGatedMultiplyQ12, PackedEncryptedSoftmax
    from pllm.protocols import PrivateEmbeddingLookup

    if name.startswith("masked-"):
        component = MaskedWireGatedMultiplyQ12(layout=name.removeprefix("masked-"))
        kwargs = {"lanes": 8, "samples": 1}
        expected, ceiling = 8, 64 << 20
    elif name.startswith("embedding-"):
        active = name.startswith("embedding-active-")
        mode = name.removeprefix("embedding-active-" if active else "embedding-")
        component = PrivateEmbeddingLookup(vocabulary_size=4096 if active else 17, precision="mixed_4_3_3",
                                          repacking=mode, conversion="party_local")
        kwargs = {"lanes": 7, "samples": 1}
        expected, ceiling = 7, 128 << 20
    else:
        component = PackedEncryptedSoftmax(method=name.removeprefix("softmax-"))
        kwargs = {"lanes": 1, "samples": 1, "geometry": {"width": 2}}
        expected, ceiling = 2, 2 << 30
    report = benchmark_operator(component, **kwargs)
    assert report.resources.protocol_allocation_estimate_bytes <= ceiling
    save(output / f"{name}-initial.json", report.to_dict())
    replay = replay_operator_benchmark(report.configuration)
    save(output / f"{name}-replay.json", replay.to_dict())
    assert replay.resources == report.resources
    assert replay.numeric_contract_digest == report.numeric_contract_digest
    for item in (report, replay):
        assert item.quality.checked_outputs == expected
        if name.startswith("softmax-"):
            assert item.quality.outputs_within_tolerance == expected
            assert item.quality.exact_integer_outputs == 0
            assert item.quality.maximum_absolute_error <= 0.0002
            assert item.quality.maximum_profile_absolute_error <= 0.0002
            assert item.quality.maximum_normalization_error <= 0.0002
            assert item.samples[0].minimum_remaining_depth == 0
        else:
            assert item.quality.exact_integer_outputs == expected
            assert replay.quality == report.quality
        assert item.resources.peer_frame_bytes == sum(
            sum(phase) for phase in item.samples[0].peer_frame_bytes_by_phase)
        if name.startswith("embedding-active-") and mode != "none":
            assert any(label == "repacking-galois-key-copies" and a > 0 and b > 0
                       for label, a, b in item.resources.key_payload_bytes_by_operation)
            assert item.samples[0].peer_frame_bytes_by_phase[0] == (69_632, 69_632)
    if name.startswith("softmax-"):
        from pllm import Deployment, ExecutionBudget, Experiment, Model, Pipeline
        from pllm.configuration import ConfigurationError
        from pllm.profiles import MaskedLinearCpu
        base = MaskedLinearCpu(Model.tiny())
        try:
            Experiment(name="bounded-softmax-only",
                       pipeline=Pipeline(model=base.model, components={**base.components, "nonlinear": component}),
                       deployment=Deployment.local(root="local://bounded-softmax-only"),
                       budget=ExecutionBudget(requests=1, max_input_tokens=2, max_new_tokens=1)).resolve()
        except ConfigurationError as exc:
            save(output / f"{name}-decoder-rejection.json", {"exception": type(exc).__name__, "reason": str(exc)})
        else:
            raise AssertionError("bounded encrypted softmax unexpectedly entered a decoder")
    print(json.dumps({"case": name, "initial_checked": expected, "fresh_replay_checked": expected,
                      "resources_equal": True, "numeric_identity_equal": True}))


def child_equalization(output):
    import numpy as np
    from pllm.runtime.public_equalization import equalize_activation, equalize_weight_chunk, fit_input_scale

    activation_max = np.array([0, 1e-4, 0.125, 1, 32, 1e4], dtype=np.float32)
    weight_max = np.array([2, 0, 8, 1, 0.25, 1e-3], dtype=np.float32)
    raw = [max(float(a), 1e-6) ** 0.75 / max(float(w), 1e-6) ** 0.25
           for a, w in zip(activation_max, weight_max, strict=True)]
    geometric_mean = math.exp(math.fsum(math.log(x) for x in raw) / len(raw))
    expected = np.array([min(64, max(1 / 64, x / geometric_mean)) for x in raw], dtype=np.float32)
    actual = fit_input_scale(activation_max, weight_max)
    np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=0)
    x = np.arange(18, dtype=np.float32).reshape(3, 6) / 16 - 0.5
    w = np.arange(24, dtype=np.float32).reshape(4, 6) / 32 - 0.25
    scaled_x, scaled_w = equalize_activation(x, actual), equalize_weight_chunk(w, actual)
    np.testing.assert_allclose(scaled_x, x.astype(np.float64) / actual, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(scaled_w, w.astype(np.float64) * actual, rtol=1e-7, atol=1e-7)
    baseline = x.astype(np.float64) @ w.astype(np.float64).T
    equalized = scaled_x.astype(np.float64) @ scaled_w.astype(np.float64).T
    np.testing.assert_allclose(equalized, baseline, rtol=1e-6, atol=1e-6)
    save(output / "equalization-independent-algebra.json", {
        "public_activation_maxima": activation_max.tolist(), "public_weight_maxima": weight_max.tolist(),
        "expected_scales": expected.tolist(), "observed_scales": actual.tolist(),
        "scale_relative_tolerance": 3e-7, "matrix_output_absolute_tolerance": 1e-6,
        "maximum_scale_absolute_error": float(np.max(np.abs(actual - expected))),
        "maximum_matrix_output_absolute_error": float(np.max(np.abs(equalized - baseline))),
        "checked_scales": 6, "checked_matrix_outputs": 12,
        "oracle": "Python scalar powers/logs and float64 matrix multiplication; no native quantizer oracle",
        "scope": "unquantized redistribution algebra only; quantized model quality is separate",
    })


def child_framing(output):
    import numpy as np
    from pllm.runtime.stage_protocol import PreparedStageBatchResponse

    rows = []
    for server_ns in (65_535, 65_536):
        packet = PreparedStageBatchResponse("0" * 32, np.zeros((33, 32), dtype=np.uint32), 16, server_ns)
        encoded = packet.pack()
        path = output / f"public-framing-{server_ns}.bin"
        path.write_bytes(encoded)
        rows.append({"server_ns": server_ns, "frame_bytes": len(encoded), "sha256": sha(path)})
    assert rows[1]["frame_bytes"] - rows[0]["frame_bytes"] == 2
    save(output / "equalization-framing-mechanism.json", {
        "cases": rows, "fixed_geometry": [33, 32], "fixed_wire_bits": 16,
        "only_changed_field": "server_ns", "extra_encoded_bytes": 2,
        "scope": "deterministic installed framing diagnostic; not captured frames from the earlier failed test",
        "accounting": "PreparedRemoteLinear sums complete payload lengths, including variable-width MessagePack server_ns",
        "original_test_status": "failed; original payloads were not retained; this mechanism does not retroactively prove their exact differing fields",
    })


def run_command(argv, output, name):
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory

    host = host_memory()
    preflight = asdict(host) | {"reserve_bytes": host.reserve}
    if host.available - host.reserve < 4 * GiB:
        save(output / f"{name}.blocked.json", {"host": preflight, "required_headroom_bytes": 4 * GiB})
        raise RuntimeError("fewer than 4 GiB available beyond existing host reserve")
    log = output / f"{name}.log"
    env = {**os.environ, **ENV}
    env.pop("PYTHONPATH", None)
    reason = []
    start = time.monotonic()
    with log.open("x") as stream:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)

        def abort(message):
            reason.append(message)
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        guard = MemoryWatchdog({"host": preflight}, abort)
        try:
            guard.start()
            code = process.wait(timeout=600)
            guard.check()
        except Exception as exc:
            abort(f"{type(exc).__name__}: {exc}")
            try:
                code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=15)
        finally:
            guard.close()
    record = {
        "argv": argv, "cwd": str(ROOT), "environment_overrides": ENV,
        "exit_code": code, "elapsed_seconds": time.monotonic() - start,
        "log_sha256": sha(log), "host_before": preflight, "host_after": asdict(host_memory()),
        "headroom_required_bytes": 4 * GiB, "watchdog_error": guard.error, "abort_reasons": reason,
        "minimum_host_available_bytes": guard.minimum_available_bytes,
        "maximum_host_swap_growth_bytes": guard.maximum_swap_growth_bytes,
    }
    save(output / f"{name}.command.json", record)
    if code != 0 or reason or guard.error:
        raise RuntimeError(f"{name} failed; see frozen log and command record")
    return record


def main(output, selected):
    output.mkdir(parents=True, exist_ok=False)
    (output / "driver.py").write_bytes(Path(__file__).read_bytes())
    (output / "prior-archive-index.json").write_bytes((ARCHIVE / "index.json").read_bytes())
    helpers = runpy.run_path(str(ROOT / "scripts/audit_paper_conformance.py"))
    helpers["provenance"](output, WHEEL)
    provenance = json.loads((output / "provenance.json").read_text())
    original = json.loads((OLD / "provenance.json").read_text())
    for field in ("wheel_sha256", "native_sha256", "build_digest", "wheel_package_files"):
        if provenance[field] != original[field]:
            raise RuntimeError(f"execution identity changed: {field}")
    import _sealapi_cpp
    save(output / "crypto-dependency.json", {
        "tenseal_version": metadata.version("tenseal"),
        "seal_extension": _sealapi_cpp.__file__, "seal_extension_sha256": sha(_sealapi_cpp.__file__),
        "tenseal_code": helpers["dependency_lock"]("tenseal"),
    })
    prior_sources = json.loads((BUILD / "sources-before.json").read_text())["files"]
    source_files = {p: sha(ROOT / p) for p in prior_sources}
    if source_files != prior_sources:
        raise RuntimeError("build receipt source files changed")
    files = set(source_files)
    files.update(node.split("::")[0] for nodes in TESTS.values() for node in nodes)
    files.update(f"docs/evidence/{name}" for name in EVIDENCE)
    files.update(["tests/conftest.py", "paper/PRIOR-ART-REVIEW.md", "docs/data/research/paper-library.json",
                  "docs/data/research/implementations.json"])
    library = json.loads((ROOT / "docs/data/research/paper-library.json").read_text())
    citations = {}
    for row in library["papers"]:
        if row["id"] in {"slalom", "smoothquant", "sigma", "fusefss", "fastquery", "bumblebee", "nexus", "thor", "logrow"}:
            assert sha(ROOT / row["file"]) == row["sha256"]
            files.add(row["file"])
            citations[row["id"]] = row
    locks = {p: sha(ROOT / p) for p in sorted(files)}
    save(output / "source-evidence-lock.json", {"files": locks, "primary_source_records": citations,
        "prior_build_receipt_sha256": sha(BUILD / "receipt.json"),
        "prior_execution_manifest_sha256": sha(OLD / "manifest.json")})
    cases = []
    for name, nodes in TESTS.items():
        if selected and f"tests-{name}" not in selected:
            continue
        print(f"tests-{name}", flush=True)
        snippet = "import pllm,pytest,sys; print('IMPLEMENTATION',pllm.__file__,flush=True); sys.exit(pytest.main(sys.argv[1:]))"
        argv = [sys.executable, "-I", "-c", snippet, "-q", "-o", "pythonpath=",
                f"--junitxml={output / (name + '.xml')}", *nodes]
        try:
            run_command(argv, output, f"tests-{name}")
            rows = [{"class": n.get("classname"), "name": n.get("name"),
                     "skipped": n.find("skipped") is not None, "failed": n.find("failure") is not None}
                    for n in ET.parse(output / f"{name}.xml").iter("testcase")]
            if any(r["skipped"] or r["failed"] for r in rows):
                raise RuntimeError("selected cases skipped or failed")
            cases.append({"name": f"tests-{name}", "status": "passed", "tests": rows})
        except Exception:
            cases.append({"name": f"tests-{name}", "status": "failed", "traceback": traceback.format_exc()})
    names = ["equalization-algebra", "equalization-framing", "masked-unfused", "masked-fused", "embedding-none",
             "embedding-interleave", "embedding-pack_lwes", "embedding-active-none", "embedding-active-interleave",
             "embedding-active-pack_lwes", "softmax-repeated_squaring", "softmax-square_normalize"]
    for name in names:
        if selected and name not in selected:
            continue
        print(name, flush=True)
        try:
            run_command([sys.executable, "-I", str(Path(__file__).resolve()), "--worker", name,
                         "--output", str(output)], output, name)
            cases.append({"name": name, "status": "passed"})
        except Exception:
            cases.append({"name": name, "status": "failed", "traceback": traceback.format_exc()})
    after = {p: sha(ROOT / p) for p in locks}
    save(output / "sources-after.json", after)
    stable = locks == after
    passed = stable and all(c["status"] == "passed" for c in cases)
    manifest = {
        "schema": "pllm.paper_family_conformance.v1", "status": "passed" if passed else "failed",
        "argv": [sys.executable, "-I", *sys.argv], "cases": cases, "sources_unchanged": stable,
        "wheel_sha256": provenance["wheel_sha256"], "native_sha256": provenance["native_sha256"],
        "build_digest": provenance["build_digest"], "original_execution_manifest_sha256": sha(OLD / "manifest.json"),
        "local_build_receipt_sha256": sha(BUILD / "receipt.json"),
        "scope": "tiny current installed-wheel functionality and independent oracles; serial; no system/performance reproduction",
        "artifacts": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
    }
    save(output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "output": str(output)}))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    for key, value in ENV.items():
        os.environ[key] = value
    if args.worker == "equalization-algebra":
        child_equalization(args.output)
    elif args.worker == "equalization-framing":
        child_framing(args.output)
    elif args.worker:
        child_operator(args.worker, args.output)
    else:
        raise SystemExit(main(args.output.resolve(), args.only))
