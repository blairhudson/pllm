"""Queued original cached Qwen3 config -> ordinary native semantic lowering.

One config-only regression, not a weight import, compiler-build job, model request,
source lock, host/resource admission, numeric fidelity or measured memory result.
Existing Rust fixture tests already cover explicit head_dim. This test closes only
the original pinned cache -> public Python lower_model(1,64,8) integration seam.
Run standalone after Parent grants a native validation lane, with a matching
noneditable wheel and no pytest source overlay. No approved83 compatibility claim:
current lowering source files differ from that build.
"""

import importlib.abc
import json
import os
from pathlib import Path
import sys

import pytest


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("PLLM_TEST_QWEN3_CONFIG_LOWERING") != "1",
    reason="queued: cached config native lowering requires Parent lane grant")
def test_original_cached_qwen3_config_lowers_1_64_8(monkeypatch):
    forbidden = ("torch", "transformers", "safetensors", "tokenizers")
    assert not any(name == prefix or name.startswith(prefix + ".")
                   for name in sys.modules for prefix in forbidden), "run config lowering standalone"

    class NoTensorImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in forbidden):
                raise AssertionError("config-only phase forbids import: " + fullname)

    active, shard_targets = [True], set()

    def no_payload_or_jobs(event, args):
        if not active[0]:
            return
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0]))
            if (path.suffix in {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
                    or path.resolve() in shard_targets):
                raise AssertionError("config-only phase forbids shard/blob payload open")
        if event in {"mmap.__new__", "socket.connect", "subprocess.Popen", "os.system",
                     "os.fork", "os.posix_spawn", "os.exec", "os.spawn"}:
            raise AssertionError("config-only phase forbids mapping/network/child jobs: " + event)

    guard = NoTensorImports()
    sys.meta_path.insert(0, guard)
    sys.addaudithook(no_payload_or_jobs)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    try:
        import pllm
        from pllm import lower_model
        from pllm.profiles import resolve_runtime_composition
        from pllm.runtime.hf_hub import resolve_huggingface_source
        from pllm.runtime.semantic_tensors import required_checkpoint_tensors

        prefix = Path(sys.prefix).resolve()
        assert sys.flags.isolated and sys.dont_write_bytecode
        assert Path(pllm.__file__).resolve().is_relative_to(prefix), "matching installed wheel required"
        # The published factory is loaded from clean source cwd; pllm stays from
        # installed wheel (-I, --import-mode=importlib, -o pythonpath=).
        root = Path(__file__).resolve().parents[1]
        monkeypatch.syspath_prepend(str(root))
        from examples.papers.low_memory_qwen3 import NAMES, PREPARED_ROWS, SOURCE, recipes, validate_pair

        assert SOURCE.local_files_only is True
        source = resolve_huggingface_source(SOURCE.source, revision=SOURCE.revision,
            local_files_only=True, token=False)
        assert source.repo_id == "Qwen/Qwen3-4B"
        assert source.revision == source.path.name == "1cfa9a7208912126459214e8b04321603b3df60c"
        config = json.loads((source.path / "config.json").read_text())
        index = json.loads((source.path / "model.safetensors.index.json").read_text())
        shard_targets.update((source.path / name).resolve() for name in set(index["weight_map"].values()))
        assert {key: config[key] for key in ("hidden_size", "num_attention_heads",
            "num_key_value_heads", "head_dim", "num_hidden_layers")} == {
                "hidden_size": 2560, "num_attention_heads": 32,
                "num_key_value_heads": 8, "head_dim": 128, "num_hidden_layers": 36}
        assert config["head_dim"] != config["hidden_size"] // config["num_attention_heads"]
        arms = recipes()
        validate_pair(*(arms[name] for name in NAMES))
        assert PREPARED_ROWS == 71
        for arm in arms.values():
            assert (arm.budget.requests, arm.budget.max_input_tokens, arm.budget.max_new_tokens) == (1, 64, 8)
            assert arm.pipeline.model == SOURCE
            options = resolve_runtime_composition(arm.pipeline)
            assert options.weight_bits == options.activation_bits == 8
            assert options.inventory_policy == "prewarm" and options.prepared_inventory_rows == 71
            assert arm.pipeline.components["quantization"].params["causal_reduction"] == "prefix_f32"
            assert not {"state", "cache", "placement", "boundary"} & set(arm.pipeline.components)
        print(json.dumps({"scope": "original_cached_config_only", "repo_id": source.repo_id,
            "revision": source.revision, "batch": 1, "input_tokens": 64, "output_tokens": 8,
            "tensor_payload_bytes_read": 0, "tensor_hashing_or_issuance": False}, sort_keys=True))
        # No resolve_model/load_hf_directory: they subsequently hash whole shards.
        # Leave original lowerer exceptions/cap refusals intact in ordinary pytest
        # output. Never pad dimensions, drop head_dim, alter limits or xfail them.
        from pllm import _native
        native_path = Path(_native.__file__).resolve()
        assert native_path.is_relative_to(prefix) and native_path.suffix in {".so", ".pyd"}
        plan = lower_model(config, batch=1, max_input_tokens=64, max_new_tokens=8)
        document = plan.to_dict()
        assert (document["model_family"], document["adapter"]) == ("qwen3", "pllm.qwen3.v1")
        assert document["transformations"] == []  # Unmodified full-KV semantic plan.
        required = required_checkpoint_tensors(plan)
        assert set(required) == set(index["weight_map"]) and len(required) == 398
        for phase, sequence in (("prefill", 64), ("decode", 1)):
            graph = document[phase]
            operations = {row["id"]: row for row in graph["operations"]}
            assert graph["query_sequence"] == sequence
            assert graph["maximum_key_sequence"] == (64 if phase == "prefill" else 71)
            assert len(graph["state_outputs"]) == 72
            assert all(row["shape"] == [1, 8, 71, 128] and row["maximum_sequence"] == 71
                       for row in graph["state_outputs"])
            for layer in range(36):
                op, tensor = f"layer.{layer}", f"model.layers.{layer}.self_attn"
                for projection, width in (("q", 4096), ("k", 1024), ("v", 1024)):
                    linear = operations[f"{op}.{projection}_linear"]
                    key = f"{tensor}.{projection}_proj.weight"
                    assert linear["attributes"]["weight"] == key
                    assert linear["output_shape"] == [1, sequence, width]
                    assert required[key] == (width, 2560)
                for projection, heads in (("q", 32), ("k", 8)):
                    norm = operations[f"{op}.{projection}_norm"]
                    key = f"{tensor}.{projection}_norm.weight"
                    assert norm["operator"] == "rms_norm" and norm["inputs"] == [f"{op}.{projection}_heads"]
                    assert norm["attributes"]["weight"] == key and required[key] == (128,)
                    assert norm["output_shape"] == [1, heads, sequence, 128]
                    assert operations[f"{op}.rope_{projection}"]["inputs"][0] == norm["id"]
                output = operations[f"{op}.o_proj"]
                assert output["inputs"] == [f"{op}.attention_hidden"]
                assert operations[output["inputs"][0]]["output_shape"] == [1, sequence, 4096]
                assert output["output_shape"] == [1, sequence, 2560]
                assert output["attributes"]["weight"] == f"{tensor}.o_proj.weight"
                assert required[output["attributes"]["weight"]] == (2560, 4096)
        # No estimate_plan_memory here: it constructs a native Matrix capability
        # before geometry pricing. This strict config-only regression claims no
        # allocation, source-header, host, compiler-schedule or execution admission.
        assert "torch" not in sys.modules and "transformers" not in sys.modules
    finally:
        active[0] = False
        sys.meta_path.remove(guard)
