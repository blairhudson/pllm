# Model plans

Lower a model configuration, apply a compiler component, and inspect execution coverage.

[View canonical HTML](https://pllm.run/sdk/plans/)

Document ID: `pllm.docs.sdk.plans`  
Release: `0.1.0`

`pllm.lower_model(...)` turns a supported model configuration and workload limits
into a `ModelPlan`. Lowering records model operations and state; it does not load
weights or make the plan executable.

## Python SDK example

```python
from pllm import KvCacheEviction, lower_model

config = {
    "model_type": "qwen2",
    "hidden_size": 64,
    "intermediate_size": 192,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 256,
    "max_position_embeddings": 128,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
}
base = lower_model(config, batch=1, max_input_tokens=16, max_new_tokens=4)
optimized = base.apply(KvCacheEviction())
assert optimized.digest != base.digest
assert optimized.to_dict()["transformations"][-1]["component"] == "pllm/kv-cache-eviction"
```

API: [Python SDK objects and signatures](/sdk/reference/python/pllm/#objects-and-signatures)

`KvCacheEviction` is a deterministic structural compiler pass for dense Qwen2 and
Qwen3 plans. It preserves fixed-capacity Key/Value state across prefill and decode,
adds explicit static-selection index state, and rewires decode attention through
bounded per-query dynamic gathers. Gemma and incompatible cache topologies are
rejected before mutation.

Use `plan.coverage(profile)` to inspect the operators a compiler profile can and
cannot execute. A valid structural transform is not by itself evidence of protected
runtime execution, privacy, model quality, or deployment support.

See [models](/sdk/build/models/) for supported adapters and [compiler](/sdk/pipeline/compiler/)
for coverage rules.
