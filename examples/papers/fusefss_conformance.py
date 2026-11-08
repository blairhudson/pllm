"""Frozen generated Qwen2 GQA control through the ordinary three-role SDK.

This is a numerical functionality fixture, not a trained model or quality claim.
Only inert public inputs are read; the independent oracle is used by tests only.
The two-layer h16 control retains its original IEEE words, dense output/input
orientation, numeric profile and three selections. RoPE comes from the ordinary
source importer using the public config; tests compare its F32 bytes to the fixture.

    uv run --no-sync pllm benchmark run \
      --experiment examples/papers/fusefss_conformance.py:experiment --factory --trust-python \
      --backend native --prompt hello --max-output-tokens 3 --timeout 300 \
      --warmups 0 --repetitions 1 --capture-output-digest \
      --output target/fusefss-conformance-qwen2.json

Generated source and role files remain in ignored target/ by default.
"""

import json
import math
from pathlib import Path
import struct

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import FuseFssSourceCpu

FIXTURE_ID = "qwen2-tied-generated"
INPUTS = (Path(__file__).resolve().parents[2]
          / "paper/reproduction-contracts/fusefss/decoder/inputs.json")


def experiment(path=Path("target/fusefss-conformance-qwen2"), *, requests=1, capacity="capsule"):
    """Materialize the frozen Qwen2 source and return a bounded public Experiment."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    case, = [c for c in json.loads(INPUTS.read_bytes())["cases"] if c["id"] == FIXTURE_ID]
    d = case["dimensions"]
    path = Path(path)
    root = path / "source"
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": case["family"], "hidden_size": d["hidden"],
        "intermediate_size": d["intermediate"], "num_hidden_layers": d["layers"],
        "num_attention_heads": d["heads"], "num_key_value_heads": d["kv_heads"],
        "head_dim": d["head_dim"], "vocab_size": d["vocab"],
        "max_position_embeddings": d["max_positions"], "hidden_act": "silu",
        "rms_norm_eps": "0.000001", "rope_theta": 10000, "attention_bias": True,
        "attention_dropout": 0, "use_sliding_window": False, "sliding_window": None,
        "rope_scaling": None, "use_cache": True, "max_window_layers": d["layers"],
        "tie_word_embeddings": case["tied_head"], "bos_token_id": 1,
        "eos_token_id": case["eos_id"],
    }
    (root / "config.json").write_text(json.dumps(config, sort_keys=True))
    # This generated fixture's explicit template renders "user: hello", without
    # an assistant suffix even with add_generation_prompt=True. WhitespaceSplit
    # preserves "user:" as one word; no injected IDs, postprocessor or BOS.
    vocab = {"zero": 0, "<unk>": 1, "user:": 2, "three": 3,
             "hello": 4, "five": 5, "six": 6}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(json.dumps({
        "add_bos_token": False, "model_max_length": d["max_positions"],
        "chat_template": (
            "{% for message in messages %}"
            "{{ '\\n' if not loop.first else '' }}"
            "{{ message['role'] }}: {{ message['content'] }}"
            "{% endfor %}"
        ),
    }, sort_keys=True))
    _write_source(root / "model.safetensors", case)
    return Experiment(
        "frozen-qwen2-fss-sdk",
        FuseFssSourceCpu(Model.path(str(root)), numeric_profile=case["profile"], capacity=capacity),
        Deployment.local(root=str(path / "roles")),
        ExecutionBudget(requests=requests, max_input_tokens=len(case["prefix"]),
                        max_new_tokens=case["generation_count"]),
    )


def decoder_experiment():
    """Same frozen arithmetic through file-backed sources and native-priced launch."""
    return experiment(Path("target/fusefss-conformance-qwen2-decoder"), capacity="decoder")


def _write_source(path, case):
    names = {"model.embed_tokens.weight": "embedding", "model.norm.weight": "final.gamma"}
    for layer in range(case["dimensions"]["layers"]):
        public, frozen = f"model.layers.{layer}.", f"layer.{layer}."
        for target, source in (("input_layernorm", "norm1"),
                               ("post_attention_layernorm", "norm2")):
            names[public + target + ".weight"] = frozen + source + ".gamma"
        for role in ("q", "k", "v", "o"):
            names[public + f"self_attn.{role}_proj.weight"] = frozen + role + ".weight"
            if role != "o":
                names[public + f"self_attn.{role}_proj.bias"] = frozen + role + ".bias"
        for role in ("gate", "up", "down"):
            names[public + f"mlp.{role}_proj.weight"] = frozen + role + ".weight"
    if set(names.values()) != set(case["tensors"]) - {"rope.cos", "rope.sin"}:
        raise ValueError("Qwen2 source inventory differs from the frozen control")
    header, payload = {}, bytearray()
    for name, source in sorted(names.items()):
        tensor = case["tensors"][source]
        width = {"f32": 4, "f16": 2, "bf16": 2}[tensor["dtype"]]
        if len(tensor["words"]) != math.prod(tensor["shape"]):
            raise ValueError("Frozen tensor shape differs from original words")
        # Pack integer bit patterns, never a float conversion or Q12 round trip.
        # Separate dense Q/K/V and gate/up/down weights already use [out, in].
        raw = b"".join(int(word, 16).to_bytes(width, "little") for word in tensor["words"])
        header[name] = {"dtype": tensor["dtype"].upper(), "shape": tensor["shape"],
                        "data_offsets": [len(payload), len(payload) + len(raw)]}
        payload.extend(raw)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
