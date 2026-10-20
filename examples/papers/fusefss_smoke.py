"""Generated two-selection FuseFSS experiment through the ordinary SDK.

The small, nonzero GPT-2-shaped source exercises full-prefix private feedback.
It is a functionality fixture, not a trained checkpoint or a quality benchmark.
Source files are reproducible and remain in ignored target/ by default.

    uv run --no-sync pllm benchmark run \
      --experiment examples/papers/fusefss_smoke.py:experiment --factory --trust-python \
      --backend native --prompt hello --max-output-tokens 2 --timeout 300 \
      --capture-output-digest --output target/fusefss-smoke.json
"""

import json
from pathlib import Path
import struct

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import FuseFssSourceCpu

NUMERIC_PROFILE = "fusefss.gpt2-complete-ring50-q12.v1"
DENSE_PROFILE = "fusefss.dense-rope-ring50-q12.v1"


def experiment(path=Path("target/fusefss-smoke"), *, requests=8):
    return _experiment(path, requests=requests, dense=False)


def dense_experiment(path=Path("target/fusefss-dense-smoke"), *, requests=8):
    """Same small independent native oracle, with RMSNorm, RoPE and gated SiLU."""
    return _experiment(path, requests=requests, dense=True)


def dense_single_smoke(path=Path("target/fusefss-dense-delivery-smoke")):
    """One bounded request; matched source/workload for material delivery tests."""
    return dense_experiment(path, requests=1)


def dense_batch_smoke(path=Path("target/fusefss-dense-delivery-smoke")):
    """Same generated D13 source and two selections with batch-v1 delivery."""
    return dense_single_smoke(path).with_params(
        name="generated-fss-dense-batch", pipeline__linear__material_delivery="batch-v1")


def dense_wavefront_smoke(path=Path("target/fusefss-dense-delivery-smoke")):
    """Same Q12 source and workload; bounded independent online waves."""
    return dense_batch_smoke(path).with_params(
        name="generated-fss-dense-wavefront", pipeline__linear__online_schedule="wavefront-v1")


def _experiment(path, *, requests, dense):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    path = Path(path)
    root = path / "source"
    root.mkdir(parents=True, exist_ok=True)
    config = {"model_type": "gpt2", "n_embd": 2, "n_head": 1, "n_layer": 1,
              "n_positions": 4, "vocab_size": 3, "n_inner": 2,
              "bos_token_id": 1, "eos_token_id": 0, "tie_word_embeddings": True}
    if dense:
        config = {"model_type": "qwen2", "hidden_size": 2, "intermediate_size": 2,
                  "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
                  "vocab_size": 3, "max_position_embeddings": 4, "hidden_act": "silu",
                  "rms_norm_eps": 1e-6, "rope_theta": 10000, "tie_word_embeddings": True,
                  "bos_token_id": 1, "eos_token_id": 0}
    (root / "config.json").write_text(json.dumps(config))
    # The complete ordinary prompt rendering is one token. EOS=0 must not
    # truncate the two private selections; expected final text is "zero zero".
    tokenizer = Tokenizer(WordLevel({"zero": 0, "user: hello": 1, "<unk>": 2}, unk_token="<unk>"))
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(json.dumps({"add_bos_token": False, "model_max_length": 4}))
    tensors = {
        "transformer.wte.weight": ([3, 2], [((r*7+c*5) % 17-8)*256 for r in range(3) for c in range(2)]),
        "transformer.wpe.weight": ([4, 2], [((r*3+c*11) % 13-6)*128 for r in range(4) for c in range(2)]),
    }
    for name, gamma, beta in (
        ("transformer.h.0.ln_1", [4096, 4160], [32, -32]),
        ("transformer.h.0.ln_2", [4032, 4096], [-32, 64]),
        ("transformer.ln_f", [4160, 4032], [0, 32]),
    ):
        tensors[name + ".weight"] = ([2], gamma)
        tensors[name + ".bias"] = ([2], beta)
    for index, name in enumerate(("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")):
        columns = 6 if index == 0 else 2
        base = "transformer.h.0." + name
        tensors[base + ".weight"] = ([2, columns], [((r*7+c*3+index*5) % 11-5)*64 for c in range(2) for r in range(columns)])
        tensors[base + ".bias"] = ([columns], [((r+index) % 5-2)*32 for r in range(columns)])
    if dense:
        tensors = {"model.embed_tokens.weight": tensors["transformer.wte.weight"]}
        for name, gamma in (("model.layers.0.input_layernorm", [4096, 4160]),
                            ("model.layers.0.post_attention_layernorm", [4032, 4096]),
                            ("model.norm", [4160, 4032])):
            tensors[name + ".weight"] = ([2], gamma)
        for name, kind, offset in (("self_attn.q_proj", 0, 0), ("self_attn.k_proj", 0, 2),
                                  ("self_attn.v_proj", 0, 4), ("self_attn.o_proj", 1, 0),
                                  ("mlp.gate_proj", 4, 0), ("mlp.up_proj", 2, 0),
                                  ("mlp.down_proj", 3, 0)):
            base = "model.layers.0." + name
            tensors[base + ".weight"] = ([2, 2], [((r*7+c*3+kind*5) % 11-5)*64
                                                for r in range(offset, offset+2) for c in range(2)])
            if kind == 0:
                tensors[base + ".bias"] = ([2], [((r+kind) % 5-2)*32 for r in range(offset, offset+2)])
    header, payload = {}, bytearray()
    for name, (shape, codes) in sorted(tensors.items()):
        raw = struct.pack("<" + "f"*len(codes), *(code/4096 for code in codes))
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [len(payload), len(payload)+len(raw)]}
        payload.extend(raw)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return Experiment("generated-fss-dense-sdk" if dense else "generated-fss-f3-sdk",
        FuseFssSourceCpu(Model.path(str(root)), numeric_profile=DENSE_PROFILE if dense else NUMERIC_PROFILE),
        Deployment.local(root=str(path / "roles")),
        ExecutionBudget(requests=requests, max_input_tokens=1, max_new_tokens=2))
