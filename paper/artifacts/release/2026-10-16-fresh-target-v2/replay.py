"""Fresh installed-wheel checks with generated NumPy data and frozen test sources.

No upstream model runtime, model download, service, or performance benchmark is
used. The generated trajectory check compares two schedules of PLLM itself; it
does not establish agreement with an independent model implementation.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import hashlib
import importlib.metadata as metadata
import importlib.util
import io
import json
from pathlib import Path
import runpy
import site
import sys
import traceback
import zipfile


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def inventory():
    """Hash installed distribution payloads and independently validate RECORD."""
    distributions, errors = {}, []
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        files = {}
        record = dist.read_text("RECORD")
        if record is None:
            raise RuntimeError(f"Missing RECORD: {name}")
        checked = 0
        for entry, recorded, size in csv.reader(io.StringIO(record)):
            path = Path(dist.locate_file(entry))
            if path.suffix == ".pyc":
                continue
            if not path.is_file():
                errors.append(f"{name}: missing {entry}")
                continue
            actual = sha(path)
            files[entry] = actual
            if recorded:
                algorithm, expected = recorded.split("=", 1)
                if algorithm != "sha256":
                    raise RuntimeError(f"Unexpected RECORD hash: {algorithm}")
                encoded = base64.urlsafe_b64encode(bytes.fromhex(actual)).decode().rstrip("=")
                if encoded != expected or int(size) != path.stat().st_size:
                    errors.append(f"{name}: RECORD mismatch {entry}")
                checked += 1
        distributions[name] = {"version": dist.version, "files": files,
                               "digest": digest(files), "record_hashes_checked": checked,
                               "direct_url": json.loads(dist.read_text("direct_url.json") or "{}")}
    return {"distributions": distributions, "digest": digest(distributions), "errors": errors,
            "scope": "All distribution RECORD files except generated pyc; RECORD self-entry hashed but not self-authenticated"}


def installed_provenance(wheel, environment):
    import pllm
    from pllm import _native
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    environment = environment.resolve()
    package = Path(pllm.__file__).resolve().parent
    native = Path(_native.__file__).resolve()
    assert Path(sys.prefix).resolve() == environment
    assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
    assert package.is_relative_to(environment) and native.is_relative_to(package)
    distributions = {canonicalize_name(d.metadata["Name"]): d for d in metadata.distributions()}
    for banned in ("torch", "transformers", "mlx", "tenseal"):
        assert banned not in distributions and importlib.util.find_spec(banned) is None
    errors, requirements_checked = [], 0
    for dist in distributions.values():
        for text in dist.requires or ():
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            installed = distributions.get(canonicalize_name(requirement.name))
            if installed is None or not requirement.specifier.contains(installed.version, prereleases=True):
                errors.append(f"{dist.metadata['Name']}: {text}")
            requirements_checked += 1
    assert not errors, errors
    expected = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("pllm/") and not name.endswith("/"):
                expected[name] = hashlib.sha256(archive.read(name)).hexdigest()
    actual = {"pllm/" + p.relative_to(package).as_posix(): sha(p)
              for p in sorted(package.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    assert actual == expected, "Installed package differs from wheel payload"
    direct_url = distributions["pllm-run"].read_text("direct_url.json")
    assert not json.loads(direct_url).get("dir_info", {}).get("editable")
    return {"python": sys.executable, "python_sha256": sha(sys.executable), "python_version": sys.version,
            "prefix": sys.prefix, "base_prefix": sys.base_prefix, "sys_path": sys.path,
            "user_site_enabled": site.ENABLE_USER_SITE, "isolated_flag": sys.flags.isolated,
            "package_path": str(package), "native_path": str(native), "native_sha256": sha(native),
            "wheel_sha256": sha(wheel), "package_files": actual, "package_digest": digest(actual),
            "direct_url": json.loads(direct_url), "active_requirements_checked": requirements_checked,
            "dependency_errors": errors, "upstream_runtime_distributions_absent": True}


def generated_checkpoint(path, family):
    """New NumPy PCG64 fixture; deliberately not the historical Torch fixture."""
    import numpy as np
    from safetensors.numpy import save_file

    path.mkdir(parents=True)
    rng = np.random.default_rng(23)
    config = {
        "architectures": [{"qwen2": "Qwen2ForCausalLM", "qwen3": "Qwen3ForCausalLM", "llama": "LlamaForCausalLM"}[family]],
        "model_type": family, "name_or_path": f"release-numpy-{family}",
        "vocab_size": 258, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 8, "max_position_embeddings": 256, "sliding_window": None,
        "hidden_act": "silu", "rms_norm_eps": 1e-6, "tie_word_embeddings": True,
        "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 1, "rope_theta": 10000.0,
        "attention_bias": family == "qwen2", "attention_dropout": 0.0, "rope_scaling": None,
        "use_cache": True, "use_sliding_window": False, "max_window_layers": 2,
        "layer_types": ["full_attention"] * 2, "pllm_test_tokenizer": "byte",
    }
    write(path / "config.json", config)
    write(path / "tokenizer_config.json", {"bos_token": "<bos>", "eos_token": "<eos>", "model_max_length": 256})

    def matrix(rows, cols, scale=0.08):
        return (rng.standard_normal((rows, cols)) * scale).astype(np.float32)

    tensors = {"model.embed_tokens.weight": matrix(258, 32, 0.12),
               "model.norm.weight": np.ones(32, np.float32)}
    for index in range(2):
        prefix = f"model.layers.{index}"
        for name, rows, cols in (("self_attn.q_proj", 32, 32), ("self_attn.k_proj", 16, 32),
                                 ("self_attn.v_proj", 16, 32), ("self_attn.o_proj", 32, 32),
                                 ("mlp.gate_proj", 64, 32), ("mlp.up_proj", 64, 32), ("mlp.down_proj", 32, 64)):
            tensors[f"{prefix}.{name}.weight"] = matrix(rows, cols)
        for name in ("input_layernorm", "post_attention_layernorm"):
            tensors[f"{prefix}.{name}.weight"] = np.ones(32, np.float32)
        if family == "qwen2":
            for name, width in (("q", 32), ("k", 16), ("v", 16)):
                tensors[f"{prefix}.self_attn.{name}_proj.bias"] = (rng.standard_normal(width) * 0.02).astype(np.float32)
        if family == "qwen3":
            for name in ("q", "k"):
                tensors[f"{prefix}.self_attn.{name}_norm.weight"] = np.ones(8, np.float32)
    save_file(tensors, path / "model.safetensors", metadata={"format": "pt"})
    return config


def trajectory_checks(output):
    import numpy as np
    import pytest
    from pllm import Model, lower_model
    from pllm.profiles import ClientOnlyCpu
    from pllm.schedulers import MlpPrefillTiling
    from pllm.runtime.loaders import load_hf_directory
    from pllm.runtime.model_binding import compile_runtime_model, RuntimeBindingError
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    rows = []
    for family in ("qwen2", "qwen3", "llama"):
        fixture = output / "fixtures" / family
        config = generated_checkpoint(fixture, family)
        model_id = f"release-numpy-{family}"
        engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
        asyncio.run(engine.load(load_hf_directory(fixture, model_id=model_id)))
        bundle = ClientBundle.unpack(engine.client_bundle(model_id, placement="client"))
        plan = lower_model(config, batch=1, max_input_tokens=33, max_new_tokens=4)
        composition = ClientOnlyCpu(Model(model_id))
        arrays_by_tile, bindings = {}, []
        for tile in (None, 1, 7, 16):
            selected = composition if tile is None else composition.with_params(scheduler=MlpPrefillTiling(tile_rows=tile))
            compiled = compile_runtime_model(plan, bundle, composition=selected)
            runtime = compiled.runtime(compiled.client_linear_executor(engine))
            bindings.append(compiled.runtime_schedule_digest)
            arrays = {}
            for phase, ids in enumerate(([0, 2, 3] * 11, [5], [7], [9])):
                arrays[f"logits_{phase}"] = runtime._forward(np.asarray(ids, dtype=np.int64)).copy()
                assert runtime.position == 33 + phase
                for layer, cache in enumerate(runtime.caches):
                    arrays[f"key_{phase}_{layer}"] = cache.key[:cache.length].copy()
                    arrays[f"value_{phase}_{layer}"] = cache.value[:cache.length].copy()
            if tile is not None:
                assert runtime._mlp_tiling.calls == 2
                assert runtime._mlp_tiling.tiles == 2 * ((33 + tile - 1) // tile)
                assert runtime._mlp_tiling.peak_tile_rows == tile
            arrays_by_tile[tile] = arrays
            np.savez(output / f"trajectory-{family}-{tile}.npz", **arrays)
        assert len(set(bindings)) == 4
        baseline = arrays_by_tile[None]
        for tile in (1, 7, 16):
            for name, expected in baseline.items():
                actual = arrays_by_tile[tile][name]
                assert np.isfinite(actual).all()
                assert expected.shape == actual.shape and expected.dtype == actual.dtype
                assert expected.tobytes() == actual.tobytes(), (family, tile, name)
        with pytest.raises(RuntimeBindingError):
            compile_runtime_model(plan, bundle, composition=ClientOnlyCpu(Model("wrong-model")))
        bound = compile_runtime_model(plan, bundle, composition=composition)
        executor = bound.client_linear_executor(engine)
        bundle.arrays["model.layers.0.input_layernorm.weight"][0] += 1.0
        with pytest.raises(RuntimeBindingError):
            bound.runtime(executor)
        rows.append({"family": family, "fixture_rng": "NumPy PCG64 seed 23; new generated fixture",
                     "fixture_files": {p.name: sha(p) for p in sorted(fixture.iterdir())},
                     "tile_rows": [1, 7, 16], "array_pairs_compared": 3 * len(baseline),
                     "bit_exact_values_compared": 3 * sum(a.size for a in baseline.values()),
                     "prefill_ids": [0, 2, 3] * 11, "decode_ids": [5, 7, 9],
                     "schedule_digests": bindings, "wrong_model_rejected": True,
                     "post_binding_norm_mutation_rejected": True})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--frozen-boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = installed_provenance(args.wheel, args.environment)
    write(args.output / "provenance.json", provenance)
    before = inventory()
    write(args.output / "dependencies-before.json", before)
    assert not before["errors"], before["errors"]
    cases = []
    try:
        rows = trajectory_checks(args.output)
        write(args.output / "generated-trajectories.json", rows)
        cases.append({"name": "generated-tiled-trajectories", "status": "passed",
                      "families": 3, "tiled_comparisons": 9, "binding_rejections": 6})
        frozen = runpy.run_path(str(args.frozen_boundary))
        for check in frozen["CHECKS"]:
            result = check()
            cases.append({"name": check.__name__, "status": "passed", "result": result})
    except Exception:
        (args.output / "failure.txt").write_text(traceback.format_exc())
        cases.append({"name": "replay", "status": "failed", "traceback": "failure.txt"})
    after = inventory()
    write(args.output / "dependencies-after.json", after)
    imported = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
                if (name == "pllm" or name.startswith("pllm.")) and getattr(module, "__file__", None)}
    package = Path(provenance["package_path"])
    assert imported and all(Path(p).is_relative_to(package) for p in imported.values())
    assert not any(name.split(".")[0] in {"torch", "transformers", "mlx", "tenseal"} for name in sys.modules)
    stable = before == after
    result = {"schema": "pllm.fresh_release_replay.v1", "cases": cases,
              "status": "passed" if stable and all(c["status"] == "passed" for c in cases) else "failed",
              "dependency_files_unchanged": stable, "imported_pllm_modules": imported,
              "no_upstream_runtime_imported": True, "frozen_boundary_sha256": sha(args.frozen_boundary),
              "scope": "Generated intra-implementation schedule parity, binding rejection, frozen bounded algebra/native checks; no independent model oracle or serving/performance cohort"}
    write(args.output / "result.json", result)
    print(json.dumps({"status": result["status"], "case_groups": len(cases), "imported_pllm_modules": len(imported)}))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
