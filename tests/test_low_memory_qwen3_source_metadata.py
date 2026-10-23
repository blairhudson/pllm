"""Original cached metadata qualification; NOT source/compiler/RSS admission.

Run alone with PLLM_TEST_QWEN3_SOURCE_METADATA=1, HF_HUB_OFFLINE=1,
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, PYTHONPATH=python:. and ordinary pytest
--noconftest -q -s tests/test_low_memory_qwen3_source_metadata.py.

The existing offline resolver, metadata digest helper and bounded header parser
execute. Safetensors headers establish declared shapes/dtypes/offsets, NOT values.
Its total_size is serialized checkpoint metadata, not an allocation/RSS estimate.
Declared W8A8 components do not establish source numerical fidelity or native
support. No full source lock or compiled body/state identity is manufactured.

The next ordinary calls, NOT executed by this test, are:
* resolve_model(SOURCE): load_hf_directory calls native lower_model; _source_lock
  subsequently calls _file_digest on ALL source files, including weight shards.
* lower_model(original_config, batch=1, max_input_tokens=64, max_new_tokens=8):
  native lowering, then plan.coverage(pipeline) / plan.runtime_schedule(pipeline).
* benchmark_memory(experiment, memory_budget_bytes=explicit_allowance): native
  macOS host telemetry, source resolution, header preflight and native plan cost.
  Keep ordinary 25% safety margin, inventory 71 and all existing admission caps.
* Only after those gates: the committed cli_suite factory, unchanged exact64/8
  checks, full state audit and owner-aware client/helper lifetime measurements.
None of these follow-up gates is replaced by this metadata test or a new planner.
"""
from __future__ import annotations

import importlib.abc
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.skipif(os.environ.get("PLLM_TEST_QWEN3_SOURCE_METADATA") != "1",
                               reason="explicit cached metadata check; never fetch or execute a model")

ASSETS = {
    "config.json": "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "model.safetensors.index.json": "6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2",
}


@pytest.fixture(scope="module")
def header_open_permission():
    return [None, None]


@pytest.fixture(scope="module")
def metadata(header_open_permission):
    """Trap execution and shard opens except explicitly instrumented header I/O."""
    forbidden = ("pllm._native", "numpy", "torch", "tokenizers", "safetensors")
    assert not any(name == prefix or name.startswith(prefix + ".")
                   for name in sys.modules for prefix in forbidden), "run metadata check alone"

    class NoExecution(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in forbidden):
                raise AssertionError("metadata-only check forbids import: " + fullname)

    active = [True]
    shard_paths = set()

    def no_weight_reads(event, args):
        if active[0] and event == "open" and isinstance(args[0], int):
            assert args[0] != header_open_permission[1], "shard descriptor buffering forbidden"
        if active[0] and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0]))
            if path.suffix in {".safetensors", ".bin", ".pt", ".pth", ".gguf"} or path.resolve() in shard_paths:
                # os.open only: no buffered read-ahead, writes, or tensor mapping.
                assert path.resolve() == header_open_permission[0] and args[1] is None, "shard open forbidden"
                assert args[2] & os.O_ACCMODE == os.O_RDONLY
                assert not args[2] & (os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        if active[0] and event == "mmap.__new__":
            raise AssertionError("metadata-only check forbids tensor mapping")

    guard = NoExecution()
    sys.meta_path.insert(0, guard)
    sys.addaudithook(no_weight_reads)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("HF_HUB_OFFLINE", "1")
            patch.setenv("TRANSFORMERS_OFFLINE", "1")
            from examples.papers.low_memory_qwen3 import MODEL_ID, REVISION, SOURCE
            from pllm.model_loader import _file_digest
            from pllm.runtime.hf_hub import HuggingFaceSourceError, resolve_huggingface_source

            assert SOURCE.local_files_only is True
            try:
                source = resolve_huggingface_source(SOURCE.source, revision=SOURCE.revision,
                                                   local_files_only=True, token=False)
            except HuggingFaceSourceError as exc:
                pytest.skip(f"unqualified: original pinned offline cache unavailable: {exc}")
            assert source.repo_id == MODEL_ID == "Qwen/Qwen3-4B"
            assert source.revision == source.path.name == REVISION == "1cfa9a7208912126459214e8b04321603b3df60c"
            assert source.path.parent.parent.name == "models--Qwen--Qwen3-4B"
            before = {name: _file_digest(source.path / name) for name in ASSETS}
            assert before == ASSETS
            config = json.loads((source.path / "config.json").read_text())
            tokenizer = json.loads((source.path / "tokenizer_config.json").read_text())
            index = json.loads((source.path / "model.safetensors.index.json").read_text())
            shards = sorted(set(index["weight_map"].values()))
            assert all(re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name) for name in shards)
            shard_paths.update((source.path / name).resolve() for name in shards)
            # The metadata fixture itself still never opens a shard/blob alias.
            sizes = {name: (source.path / name).stat().st_size for name in shards}
            yield source, config, tokenizer, index, sizes
            assert {name: _file_digest(source.path / name) for name in ASSETS} == before
            assert source == resolve_huggingface_source(SOURCE.source, revision=SOURCE.revision,
                                                       local_files_only=True, token=False)
            assert "pllm._native" not in sys.modules
    finally:
        active[0] = False
        sys.meta_path.remove(guard)


def test_original_config_tokenizer_and_index_geometry(metadata):
    source, config, tokenizer, index, sizes = metadata
    geometry = {name: config[name] for name in (
        "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings")}
    assert geometry == {"hidden_size": 2560, "intermediate_size": 9728, "num_hidden_layers": 36,
        "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        "vocab_size": 151936, "max_position_embeddings": 40960}
    assert config["architectures"] == ["Qwen3ForCausalLM"] and config["model_type"] == "qwen3"
    assert config["tie_word_embeddings"] is True and config["torch_dtype"] == "bfloat16"
    assert tokenizer["add_bos_token"] is False and tokenizer["chat_template"]
    weights = index["weight_map"]
    assert len(weights) == 398 and len(sizes) == 3
    assert index["metadata"]["total_size"] == 8044936192
    assert "model.embed_tokens.weight" in weights and "model.norm.weight" in weights
    assert "lm_head.weight" not in weights  # Declaration only; no coupled-boundary adoption.
    indexed_layers = {int(match[1]) for name in weights
                      if (match := re.match(r"model\.layers\.(\d+)\.", name))}
    assert indexed_layers == set(range(config["num_hidden_layers"]))
    print(json.dumps({"scope": "original_cached_metadata_only", "repo_id": source.repo_id,
        "revision": source.revision, "asset_sha256": ASSETS, "geometry": geometry,
        "indexed_tensor_names": len(weights), "index_declared_tensor_bytes": index["metadata"]["total_size"],
        "shard_file_sizes_only": sizes, "tensor_shapes_dtypes_values_verified": False}, sort_keys=True))


def test_common_declared_body_state_and_71_row_envelope(metadata):
    from examples.papers.low_memory_qwen3 import BUDGET, NAMES, PREPARED_ROWS, SOURCE, recipes, validate_pair
    from pllm.profiles import resolve_runtime_composition

    _, config, _, _, _ = metadata
    arms = recipes()
    validate_pair(*(arms[name] for name in NAMES))
    control, candidate = (arms[name] for name in NAMES)
    left, right = control.pipeline.components, candidate.pipeline.components
    assert set(right) - set(left) == {"delivery"}
    assert all(component == right[name] for name, component in left.items())
    assert all(arm.pipeline.model == SOURCE and arm.budget == BUDGET for arm in arms.values())
    assert BUDGET.max_input_tokens == 64 and BUDGET.max_new_tokens == 8 and BUDGET.requests == 1
    assert BUDGET.max_input_tokens + BUDGET.max_new_tokens <= config["max_position_embeddings"]
    assert PREPARED_ROWS == 71
    for arm in arms.values():
        options = resolve_runtime_composition(arm.pipeline)
        assert options.weight_bits == options.activation_bits == 8
        assert options.requires_preparation and options.prepared_inventory_rows == 71
        assert options.inventory_policy == "prewarm" and options.inventory_refill == "idle"
        assert options.background_inventory_refill is True
        assert not options.decode and not options.private_boundary
        assert arm.pipeline.components["quantization"].params["causal_reduction"] == "prefix_f32"
    assert not {"state", "cache", "placement", "boundary"} & set(right)


@pytest.fixture(scope="module")
def headers(metadata, header_open_permission):
    """Real cached I/O, existing parser; this adapter is only a two-read trap.

    Deliberately do not instantiate the probe's whole-checkpoint/host admission,
    tensor decoder or hashing machinery. This is NOT its numerical profile gate.
    """
    from scripts import probe_gkm_checkpoint_bounds as reader
    from examples.papers.low_memory_qwen3 import NAMES, recipes
    from pllm.runtime.hf_hub import HuggingFaceSourceError, resolve_huggingface_source

    original, _, _, index, _ = metadata
    inventories, traces, identities = {}, {}, {}
    for arm_name, arm in recipes().items():
        model = arm.pipeline.model
        try:
            source = resolve_huggingface_source(model.source, revision=model.revision,
                                               local_files_only=model.local_files_only, token=False)
        except HuggingFaceSourceError as exc:
            pytest.skip(f"unqualified: original pinned offline headers unavailable: {exc}")
        assert source == original
        tensors, locations, observations, reads = {}, {}, {}, []
        for filename in sorted(set(index["weight_map"].values())):
            path = source.path / filename
            if not path.is_file():
                pytest.skip(f"unqualified: cached shard header missing: {path}")
            header_open_permission[0] = path.resolve()
            fd = None
            try:
                fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
                header_open_permission[1] = fd
                before = reader.file_identity(path, fd)
                size = os.fstat(fd).st_size
                spans, header_length = [], [None]
                original_pread = os.pread

                def pread(descriptor, length, offset):
                    expected = (0, 8) if not spans else (8, header_length[0])
                    assert descriptor == fd and len(spans) < 2, "prefix/header-only read trap"
                    assert (offset, length) == expected and 0 <= offset <= size - length, "prefix/header-only read trap"
                    # Unbuffered range read: bytes outside these two spans never enter Python.
                    raw = original_pread(descriptor, length, offset)
                    assert len(raw) == length, "unqualified: truncated cached header"
                    if not spans:
                        header_length[0] = int.from_bytes(raw, "little")
                        assert 2 <= header_length[0] <= reader.MAX_HEADER
                        assert header_length[0] + 8 <= size
                    spans.append((offset, length))
                    return raw

                def read(offset, length, purpose):
                    assert purpose == "headers", "prefix/header-only read trap"
                    return os.pread(fd, length, offset)

                def block_streaming(original_read):
                    def checked(descriptor, *args, **kwargs):
                        assert descriptor != fd, "shard streaming/payload read forbidden"
                        return original_read(descriptor, *args, **kwargs)
                    return checked

                # Negative controls reach the same I/O gate, not a replacement parser.
                with pytest.MonkeyPatch.context() as io:
                    io.setattr(os, "pread", pread)
                    for name in ("read", "readv", "preadv"):
                        if hasattr(os, name):
                            io.setattr(os, name, block_streaming(getattr(os, name)))
                    with pytest.raises(AssertionError, match="read trap"):
                        read(0, 9, "headers")
                    rows, observation = reader.parse_header(SimpleNamespace(size=size, read=read))
                    with pytest.raises(AssertionError, match="read trap"):
                        read(8 + observation["header_bytes"], 1, "headers")
                    with pytest.raises(AssertionError, match="streaming/payload"):
                        os.read(fd, 1)
                    with pytest.raises(AssertionError, match="shard descriptor buffering"):
                        open(fd, "rb", closefd=False)
                    with pytest.raises(AssertionError, match="shard open forbidden"):
                        path.open("rb")  # Full hashing/buffering forbidden even here.
                    with pytest.raises(AssertionError, match="shard open forbidden"):
                        path.resolve().open("rb")
                assert spans == [(0, 8), (8, observation["header_bytes"])]
                assert before == reader.file_identity(path, fd)
                assert not set(tensors) & set(rows), "duplicate shard tensor ownership"
                tensors.update(rows)
                locations.update(dict.fromkeys(rows, filename))
                observations[filename] = observation
                identities.setdefault(arm_name, {})[filename] = before
                reads.extend((filename, offset, length) for offset, length in spans)
            finally:
                if fd is not None:
                    os.close(fd)
                header_open_permission[:] = [None, None]
        assert locations == index["weight_map"]
        assert sum(row["payload_bytes"] for row in observations.values()) == index["metadata"]["total_size"]
        inventories[arm_name] = tensors
        traces[arm_name] = reads
        payload_read = sum(max(0, offset + length - max(offset, 8 + observations[name]["header_bytes"]))
                           for name, offset, length in reads)
        assert payload_read == 0
        print(json.dumps({"scope": "cached_headers_only_not_byte_pin_authority", "arm": arm_name,
            "headers": observations, "pread_spans": reads, "payload_bytes_read": payload_read}, sort_keys=True))
    assert inventories[NAMES[0]] == inventories[NAMES[1]]
    assert identities[NAMES[0]] == identities[NAMES[1]]
    assert traces[NAMES[0]] == traces[NAMES[1]]
    return inventories[NAMES[0]], traces


def test_all_398_cached_header_shapes_match_original_qwen3_config(metadata, headers):
    _, config, _, index, _ = metadata
    tensors, _ = headers
    hidden, intermediate = config["hidden_size"], config["intermediate_size"]
    head, q_heads, kv_heads = config["head_dim"], config["num_attention_heads"], config["num_key_value_heads"]
    assert (hidden, q_heads * head, kv_heads * head) == (2560, 4096, 1024)
    # Independent checkpoint/config assertions, NOT a model plan or compiler shape authority.
    expected = {"model.embed_tokens.weight": [config["vocab_size"], hidden], "model.norm.weight": [hidden]}
    layer_shapes = {
        "input_layernorm.weight": [hidden], "post_attention_layernorm.weight": [hidden],
        "self_attn.q_proj.weight": [q_heads * head, hidden],
        "self_attn.k_proj.weight": [kv_heads * head, hidden],
        "self_attn.v_proj.weight": [kv_heads * head, hidden],
        "self_attn.o_proj.weight": [hidden, q_heads * head],
        "self_attn.q_norm.weight": [head], "self_attn.k_norm.weight": [head],
        "mlp.gate_proj.weight": [intermediate, hidden], "mlp.up_proj.weight": [intermediate, hidden],
        "mlp.down_proj.weight": [hidden, intermediate],
    }
    for layer in range(config["num_hidden_layers"]):
        expected.update({f"model.layers.{layer}.{name}": shape for name, shape in layer_shapes.items()})
    assert {name: row["shape"] for name, row in tensors.items()} == expected
    assert set(tensors) == set(index["weight_map"]) and len(tensors) == 398
    assert {row["dtype"] for row in tensors.values()} == {"BF16"}
    assert config["tie_word_embeddings"] is True and "lm_head.weight" not in tensors
    assert tensors["model.embed_tokens.weight"]["shape"] == [151936, 2560]


def test_header_trace_and_actual_source_declared_body_state_identity(metadata, headers):
    from examples.papers.low_memory_qwen3 import NAMES, recipes, validate_pair

    tensors, traces = headers
    assert all(len(reads) == 6 for reads in traces.values())
    assert all(sum(length for _, _, length in reads) < 3 * (1 << 20) + 24 for reads in traces.values())
    arms = recipes()
    validate_pair(*(arms[name] for name in NAMES))
    # Real files/headers were independently re-read for each normal pinned arm.
    # Only declared body/state identity is checked: compiled/state-audit identities remain unrun.
    left, right = (arms[name].pipeline for name in NAMES)
    assert left.model.to_spec() == right.model.to_spec()
    assert all(component == right.components[name] for name, component in left.components.items())
    assert len(tensors) == 398


def test_declaration_limits_are_not_resource_or_compiler_admission(metadata):
    from examples.papers.low_memory_qwen3 import matched_suite
    from pllm.metrics import BuildRef

    # Synthetic build/cap fixture only, NOT an approved build or chosen run budget.
    case = matched_suite(build=BuildRef(expected_digest="a" * 64), memory_bytes=16 << 30, processes=1)
    assert case.sampling.memory_bytes == 16 << 30
    with pytest.raises(ValueError, match="memory_bytes"):
        matched_suite(build=BuildRef(expected_digest="a" * 64), memory_bytes=(16 << 30) + 1)
    for arm in case.cases:
        parameters = arm.target.to_spec()["parameters"]
        assert parameters["expected_input_tokens"] == 64 and parameters["expected_output_tokens"] == 8
        assert parameters["state_audit"] is True
    print("NOT RUN: full source lock; native lower_model/coverage/runtime_schedule; "
          "benchmark_memory host/cost admission; accepted64/8; matched output/state; client/helper peak RSS; 10x.")
