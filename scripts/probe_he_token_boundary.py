"""Cost-gate a compiler-derived Qwen layer against public-key CKKS depth.

Requires only pinned config.json in the shared Hugging Face cache, not model
weights or private prompts. Synthetic encrypted arithmetic samples test public
backend capability; no whole layer or protected inference is executed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download

from pllm import Model, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.he_layer_feasibility import token_boundary_he_layer_gate

_ROOT = Path(__file__).resolve().parents[1]
_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
_CONTROL = _ROOT / "docs/evidence/latent-response-network-qwen25-2026-09-28.json"
_HUB_CACHE = os.environ.get("HF_HUB_CACHE") or str(
    Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
)
_PARAMETERS = (
    (8192, (60, 40, 40, 60)),
    (16384, (60, 40, 40, 40, 40, 60)),
)


def probe_backend() -> dict[str, Any]:
    """Measure actual secret-free public keys, serialized ciphertext and depth.

    No model-size ciphertext operation is attempted after a depth veto: the
    compulsory private nonlinear chain cannot execute under these parameters.
    """
    import tenseal as ts

    bootstrap_methods = sorted(
        name
        for object_type in (ts.CKKSVector, ts.Context)
        for name in dir(object_type)
        if "bootstrap" in name.lower()
    )
    contexts: list[dict[str, Any]] = []
    for degree, chain in _PARAMETERS:
        started = time.process_time()
        client = ts.context(
            ts.SCHEME_TYPE.CKKS, degree, coeff_mod_bit_sizes=list(chain), n_threads=1
        )
        client.global_scale = 2**40
        base = client.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_galois_keys=False,
            save_relin_keys=False,
        )
        client.generate_galois_keys()
        public = client.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_galois_keys=True,
            save_relin_keys=True,
        )
        provider = ts.context_from(public, n_threads=1)
        setup_cpu = time.process_time() - started
        if (
            provider.has_secret_key()
            or not provider.has_relin_keys()
            or not provider.has_galois_keys()
        ):
            raise RuntimeError("provider CKKS context contains a secret or lacks public eval keys")
        values = [0.75] * 896
        request = ts.ckks_vector(client, values).serialize()
        ciphertext = ts.ckks_vector_from(provider, request)
        source = ts.ckks_vector_from(provider, request)
        depth_samples: list[dict[str, Any]] = []
        depth_failure: dict[str, str | int] | None = None
        for depth in range(1, 9):
            started = time.process_time()
            try:
                ciphertext = ciphertext * source
            except ValueError as exc:
                depth_failure = {
                    "attempted_depth": depth,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                break
            body = ciphertext.serialize()
            decrypted = np.asarray(ts.ckks_vector_from(client, body).decrypt(), dtype=np.float64)
            if decrypted.shape != (len(values),) or not np.all(np.isfinite(decrypted)):
                raise RuntimeError("bounded CKKS ciphertext fails to decrypt to finite values")
            error = float(np.max(np.abs(decrypted - 0.75 ** (depth + 1))))
            if error > 1e-3:
                raise RuntimeError("sampled CKKS arithmetic misses bounded plaintext oracle")
            depth_samples.append(
                {
                    "serial_ciphertext_product_depth": depth,
                    "response_ciphertext_body_bytes": len(body),
                    "sampled_max_absolute_error": error,
                    "process_cpu_seconds": time.process_time() - started,
                }
            )
        if depth_failure is None or depth_failure["message"] != "scale out of bounds":
            raise RuntimeError("CKKS depth limit did not fail closed at scale exhaustion")
        contexts.append(
            {
                "poly_modulus_degree": degree,
                "slots": degree // 2,
                "coeff_modulus_bits": list(chain),
                "global_scale_bits": 40,
                "provider_has_secret_key": False,
                "public_context_without_eval_keys_body_bytes": len(base),
                "public_context_with_rotation_and_relin_keys_body_bytes": len(public),
                "one_ciphertext_request_body_bytes": len(request),
                "setup_and_public_context_cpu_seconds": setup_cpu,
                "bounded_depth_samples": depth_samples,
                "first_failed_depth": depth_failure,
                "confirmed_serial_ciphertext_product_depth": len(depth_samples),
            }
        )
        del client, provider, source, ciphertext, public, base
        gc.collect()
    return {
        "tenseal_version": ts.__version__,
        "python_bootstrap_methods": bootstrap_methods,
        "contexts": contexts,
        "sample_input": "896 encrypted public synthetic values of 0.75, no checkpoint or user data",
        "complete_layer_executed": False,
    }


def run(*, sample_backend: bool = True) -> dict[str, Any]:
    config_path = Path(
        hf_hub_download(
            _MODEL, "config.json", revision=_REVISION, local_files_only=True, cache_dir=_HUB_CACHE
        )
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    control = json.loads(_CONTROL.read_text(encoding="utf-8"))
    composition = MaskedLinearCpu(
        Model.hf(_MODEL, revision=_REVISION),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    reports: dict[str, dict[str, Any]] = {}
    for output_tokens in (8, 32):
        cohort = next(row for row in control["cohorts"] if row["output_tokens"] == output_tokens)
        baseline = cohort["prepared_control"]
        plan = lower_model(config, batch=1, max_input_tokens=39, max_new_tokens=output_tokens)
        if plan.digest != cohort["official_plan_digest"]:
            raise ValueError("pinned CKKS model plan differs from matched prepared control")
        report = token_boundary_he_layer_gate(
            plan,
            composition,
            response_new_tokens=output_tokens,
            covered_all_link_body_budget_bytes=baseline["tenfold_covered_budget_bytes"],
            covered_online_body_budget_bytes=baseline["tenfold_online_budget_bytes"],
        )
        if (
            report["schedule_digest"] != cohort["official_schedule_digest"]
            or report["composition_digest"] != control["source"]["pipeline_digest"]
        ):
            raise ValueError("pinned CKKS schedule/composition differs from prepared control")
        reports[str(output_tokens)] = report
    result: dict[str, Any] = {
        "schema": "pllm.token_boundary_he_feasibility_evidence.v1",
        "source": {
            "checkpoint": f"{_MODEL}@{_REVISION}",
            "matched_body_fingerprint": control["source"]["body_fingerprint"],
            "precision": "W8A8",
        },
        "cohorts": reports,
        "backend": probe_backend() if sample_backend else None,
        "whole_layer_executable": False,
        "numeric_fidelity_validated": False,
        "full_wire_measured": False,
    }
    if result["backend"] is not None:
        backend = result["backend"]
        for report in reports.values():
            required = report["semantic_phase_contracts"]["prefill"][
                "optimistic_serial_ciphertext_product_depth"
            ]
            samples = []
            for sample in backend["contexts"]:
                uploaded = (
                    math.ceil(report["input_tokens"] * report["hidden_width"] / sample["slots"])
                    + report["output_tokens"]
                    - 1
                )
                downloaded = report["output_tokens"]
                illustrative = uploaded * sample["one_ciphertext_request_body_bytes"] + (
                    downloaded
                    * sample["bounded_depth_samples"][-1]["response_ciphertext_body_bytes"]
                )
                samples.append(
                    {
                        "poly_modulus_degree": sample["poly_modulus_degree"],
                        "idealized_client_to_provider_ciphertexts_at_sampled_slots": uploaded,
                        "idealized_provider_to_client_ciphertexts": downloaded,
                        "confirmed_serial_ciphertext_product_depth": sample[
                            "confirmed_serial_ciphertext_product_depth"
                        ],
                        "depth_shortfall_even_ignoring_norms_and_softmax": required
                        - sample["confirmed_serial_ciphertext_product_depth"],
                        "illustrative_online_boundary_bytes_at_sampled_request_and_last_depth": illustrative,
                        "sampled_boundary_exceeds_online_tenfold_budget": illustrative
                        > report["covered_online_body_budget_bytes"],
                        "one_client_key_public_eval_context_exceeds_all_link_tenfold_budget": sample[
                            "public_context_with_rotation_and_relin_keys_body_bytes"
                        ]
                        > report["covered_all_link_body_budget_bytes"],
                    }
                )
            report["sampled_contexts"] = samples
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-only", action="store_true", help="sample CKKS without a model")
    parser.add_argument("--no-backend", action="store_true", help="skip bounded TenSEAL sampling")
    parser.add_argument("--summary", action="store_true", help="hash per-layer semantic records")
    args = parser.parse_args()
    if args.backend_only:
        if args.no_backend:
            parser.error("--backend-only conflicts with --no-backend")
        print(json.dumps(probe_backend(), indent=2, sort_keys=True))
        return
    result = run(sample_backend=not args.no_backend)
    if args.summary:
        for report in result["cohorts"].values():
            for phase in report["semantic_phase_contracts"].values():
                layers = phase.pop("layers")
                phase["layer_provenance_sha256"] = hashlib.sha256(
                    json.dumps(layers, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                phase["first_layer_contract"] = layers[0]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
