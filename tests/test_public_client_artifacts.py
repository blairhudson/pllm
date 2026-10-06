"""Trusted public artifact admission, local queries, and role-backed composition."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

import pllm
from pllm.preparation import ModelAwareCorrections, PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport, MaskedLinear
from pllm.quantization import SymmetricPerRow
from pllm.state import ClientPrefixReuse, PublicPrefixCapsule
from pllm.tokenization import IndexedTokenizer
from pllm.runtime.indexed_tokenizer import IndexedClientTokenizer
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.prefill_cache import ExactPrefillCache
from pllm.runtime.public_prefix import import_capsule
from pllm.runtime.semantic_source import semantic_source_config
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def bpe(path):
    reference = Tokenizer(models.BPE())
    reference.normalizer = normalizers.NFC()
    reference.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    reference.decoder = decoders.ByteLevel()
    reference.train_from_iterator(["hello café cafe\u0301 中文 🙂  spaces\n\t12345", "Public service policy"] * 3,
        trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[AddedToken("<bos>", special=True, normalized=False),
                            AddedToken("<eos>", special=True, normalized=False)]))
    reference.add_tokens([AddedToken("<added>", normalized=False)])
    reference.save(str(path))
    return reference


def test_index_exact_random_unicode_and_private_snapshot(tmp_path):
    source = tmp_path / "tokenizer.json"
    reference = bpe(source)
    selection = IndexedTokenizer.compile(source, tmp_path / "index")
    bundle = SimpleNamespace(tokenizer_descriptor={"kind": "tokenizer_json", "model": source.read_bytes()},
                             cfg={"bos_token_id": 0, "eos_token_id": 1})
    indexed = IndexedClientTokenizer(bundle, selection)
    owned = Path(indexed._temporary.name)
    cases = ["", "hello hello", "café cafe\u0301", "中文 🙂🦀", "\x00\x7f", "<bos><added><eos>", "x" * 512]
    rng = np.random.default_rng(827)
    alphabet = list("abcxyz0123 \n\t'!?é中🙂\u0301")
    cases += ["".join(rng.choice(alphabet, size=int(n))) for n in rng.integers(1, 128, size=200)]
    with ThreadPoolExecutor(max_workers=1) as worker:
        for text in cases:
            expected = reference.encode(text, add_special_tokens=False).ids
            assert worker.submit(indexed.encode, text).result() == expected
            assert indexed.decode(expected) == reference.decode(expected, skip_special_tokens=True)
            assert indexed.encode(text, add_bos=True) == [0] + expected
        (tmp_path / "index" / "bpe.sqlite").write_bytes(b"changed source")
        assert indexed.encode("hello") == reference.encode("hello").ids
    indexed.close()
    assert not owned.exists()
    with pytest.raises(ValueError, match="closed"):
        indexed.encode("hello")
    with pytest.raises(ValueError, match="identity mismatch"):
        IndexedClientTokenizer(bundle, selection)
    bundle.tokenizer_descriptor["model"] += b" "
    with pytest.raises(ValueError, match="source mismatch"):
        IndexedClientTokenizer(bundle, selection)


@pytest.fixture
def artifacts(tmp_path):
    root = create_tiny_llama_checkpoint(tmp_path / "model", vocab_size=512, num_hidden_layers=2, seed=179)
    cfg = json.loads((root / "config.json").read_bytes())
    cfg.pop("pllm_test_tokenizer")
    (root / "config.json").write_text(json.dumps(cfg))
    bpe(root / "tokenizer.json")
    index = IndexedTokenizer.compile(root / "tokenizer.json", tmp_path / "index")
    source = pllm.Model.path(str(root), model_id="public-artifacts")
    baseline = MaskedLinearCpu(source,
        linear=MaskedLinear(output_encoding="row_residues", request_encoding="stage_packed", prefill_pruning="terminal"),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
        inventory=PreparedInventory(refill="on-demand"),
        cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=248),
        delivery=ClientBundleTransport("artifacts", compression="zlib", storage="paged"))
    from pllm.model_loader import resolve_model
    manifest = resolve_model(source).manifest
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=1, prepared_output_encoding="row_residues")
    asyncio.run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle(manifest.id))
    plan = pllm.lower_model(semantic_source_config(bundle.cfg), batch=1, max_input_tokens=248, max_new_tokens=2)
    compiled = compile_runtime_model(plan, bundle, composition=baseline.with_params(delivery=None))
    from pllm.runtime.quantization import quantize_activation_per_row, dequantize_matmul
    def remote(sid, values):
        stage = engine.models[manifest.id].stages[sid]
        q = quantize_activation_per_row(values, bits=8)
        result = dequantize_matmul(stage.compiled_weight.clear(q.values), q.scales, stage.weight.scales,
            output_shape=q.original_shape[:-1] + (stage.spec.out_features,))
        return np.ascontiguousarray(result if stage.bias is None else result + stage.bias, dtype=np.float32)
    public = "Public service policy: explain concepts accurately. " * 3
    prompt = public + "hello café"
    ids = bundle.tokenizer().encode(bundle.render_prompt([{"role": "user", "content": prompt}]),
                                   add_bos=bool(bundle.tokenizer_descriptor.get("add_bos_token", True)))
    prefix = ids[:48]
    runtime = compiled.runtime(remote)
    _, logits, _ = runtime.prepare_ids(prefix)
    capsule = PublicPrefixCapsule.publish(compiled, public_token_ids=prefix, snapshot=runtime.snapshot(),
                                          logits=logits, path=tmp_path / "prefix.msgpack")
    asyncio.run(engine.unload(manifest.id))
    return baseline, compiled, index, capsule, prompt


def test_capsule_auth_source_geometry_and_configuration(artifacts, tmp_path):
    baseline, compiled, index, capsule, _ = artifacts
    cache = ExactPrefillCache(1 << 20, causal_reduction="prefix_f32")
    assert import_capsule(compiled, capsule, cache, "b" * 64) == capsule.params["size_bytes"]
    assert cache.entry_count == 1
    for option in (baseline.with_params(tokenizer=index), baseline.with_params(public_prefix=capsule),
                   baseline.with_params(tokenizer=index, public_prefix=capsule)):
        experiment = pllm.Experiment("artifacts", option, pllm.Deployment.local(root=str(tmp_path)), pllm.ExecutionBudget(1, 248, 2))
        experiment.resolve()
        assert pllm.Experiment.from_spec(experiment.to_spec()).configuration_digest() == experiment.configuration_digest()
    bad = PublicPrefixCapsule(capsule.params["path"], digest="0" * 64, size_bytes=capsule.params["size_bytes"])
    with pytest.raises(ValueError, match="authentication"):
        import_capsule(compiled, bad, cache, "b" * 64)
    document = __import__("msgpack").unpackb(Path(capsule.params["path"]).read_bytes(), raw=False)
    import hashlib
    import msgpack
    for key, value in (("numeric", "0" * 64), ("states", []), ("tokens", [True] * 48)):
        payload = msgpack.packb(document | {key: value}, use_bin_type=True)
        path = tmp_path / (key + ".msgpack")
        path.write_bytes(payload)
        forged = PublicPrefixCapsule(str(path), digest=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))
        with pytest.raises(ValueError):
            import_capsule(compiled, forged, cache, "b" * 64)
    for changes in ({"cache": None}, {"quantization": SymmetricPerRow()}):
        with pytest.raises(ValueError):
            pllm.Experiment("unsupported", baseline.with_params(public_prefix=capsule, **changes),
                             pllm.Deployment.local(root=str(tmp_path)), pllm.ExecutionBudget(1, 248, 2)).resolve()


def test_capsule_rechecks_changed_compiled_binding(artifacts, tmp_path):
    from pllm.runtime.client import _TransformerCryptoState

    baseline, original, _, capsule, _ = artifacts
    pipeline = baseline.with_params(public_prefix=capsule, delivery=None)
    experiment = pllm.Experiment("binding", pipeline, pllm.Deployment.local(root=str(tmp_path)),
                                  pllm.ExecutionBudget(1, 248, 2))
    compiled = compile_runtime_model(original._plan, original._bundle, composition=pipeline)
    state = _TransformerCryptoState(original._bundle, "b" * 64, "none")
    with pllm.OpenAI(experiment=experiment) as client:
        client._core._prefill_cache_candidate(state, compiled, [0] * 64, store=True)
        assert state.public_prefix_binding == compiled.digest
        plan = pllm.lower_model(semantic_source_config(original._bundle.cfg), batch=1,
                                max_input_tokens=248, max_new_tokens=1)
        changed = compile_runtime_model(plan, original._bundle, composition=pipeline)
        with pytest.raises(ValueError, match="source or numeric"):
            client._core._prefill_cache_candidate(state, changed, [0] * 64, store=True)
        assert state.public_prefix_binding == compiled.digest


@pytest.mark.integration
@pytest.mark.parametrize("asynchronous", [False, True])
def test_all_four_artifacts_through_prepared_sdk(artifacts, tmp_path, monkeypatch, asynchronous):
    baseline, _, index, capsule, prompt = artifacts
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime
    snapshots = []
    original = SemanticDecoderRuntime._forward_phase
    def capture(runtime, *args, **kwargs):
        result = original(runtime, *args, **kwargs)
        snapshots.append((result[..., -1, :].copy(),
            [(c.key[:c.length].copy(), c.value[:c.length].copy()) for c in runtime.caches]))
        return result
    monkeypatch.setattr(SemanticDecoderRuntime, "_forward_phase", capture)
    paths = []
    original_index = IndexedClientTokenizer.__init__
    def capture_index(tokenizer, *args, **kwargs):
        original_index(tokenizer, *args, **kwargs)
        paths.append(Path(tokenizer._temporary.name))
    monkeypatch.setattr(IndexedClientTokenizer, "__init__", capture_index)
    actual = []
    candidate = baseline.with_params(tokenizer=index, public_prefix=capsule,
        preparation=ModelAwareCorrections(storage="paged"), inventory=PreparedInventory(refill="on-demand", allocation="demand"))
    for name, pipeline in (("control", baseline), ("candidate", candidate)):
        experiment = pllm.Experiment(name, pipeline, pllm.Deployment.local(root=str(tmp_path / name)),
                                     pllm.ExecutionBudget(max_input_tokens=248, max_new_tokens=2, requests=2))
        with build_roles(experiment, engine_threads=1) as roles:
            if asynchronous:
                async def run():
                    async with pllm.AsyncOpenAI(experiment=experiment,
                        base_url=roles.inference_url, api_key=roles._inference_key,
                        preparation_base_url=roles.preparation_url, preparation_api_key=roles._preparation_key,
                        bundle_cache_dir=tmp_path / (name + "-cache")) as client:
                        with client.tokenizer_scope():
                            await client.prepare_response(prompt, 2)
                            response = await client.responses.create(input=prompt, max_output_tokens=2, temperature=0)
                        return response, client.privacy_audit.to_dict()
                response, audit = asyncio.run(run())
            else:
                with roles.client(bundle_cache_dir=tmp_path / (name + "-cache")) as client:
                    with client.tokenizer_scope():
                        client.prepare_response(prompt, 2)
                        response = client.responses.create(input=prompt, max_output_tokens=2, temperature=0)
                    audit = client.privacy_audit.to_dict()
            assert all(not path.exists() for path in paths)
            actual.append((response.output_text, response.usage, snapshots.copy(), audit))
            snapshots.clear()
    assert actual[0][:2] == actual[1][:2]
    for (logits, states), (other_logits, other_states) in zip(actual[0][2], actual[1][2], strict=True):
        np.testing.assert_array_equal(logits, other_logits)
        for pair, other in zip(states, other_states, strict=True):
            for a, b in zip(pair, other, strict=True):
                np.testing.assert_array_equal(a, b)
    a, b = actual[0][3], actual[1][3]
    assert b["prefill_prefix_tokens_reused"] == 48
    assert b["public_prefix_artifact_local_bytes"] == capsule.params["size_bytes"]
    assert b["tokenizer_artifact_local_bytes"] > 0
    assert b["prepared_stage_rows_issued"] < a["prepared_stage_rows_issued"]
    assert b["correction_push_bytes"] < a["correction_push_bytes"]
    assert b["masked_online_upload_bytes"] < a["masked_online_upload_bytes"]
    assert b["plaintext_prompt_bytes_sent"] == b["plaintext_token_ids_sent"] == 0
