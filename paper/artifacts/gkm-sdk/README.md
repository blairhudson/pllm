# GKM experiment results

Reusable experiments live in [`examples/papers/`](../../../examples/papers/README.md)
and [`examples/delegated_sdk.py`](../../../examples/delegated_sdk.py). Use the
ordinary benchmark commands and normal tests; build with uv/Maturin.

- [First source-present Qwen response](qwen-response.md): one functionality sample,
  with its original [benchmark output](qwen-response-v1/benchmark.json).
- [Quality smoke](quality-smoke.json): three shared Qwen prompts through the
  standard quality command. Scope and rerun are in the experiment README.
- [Original rejected preflight](decoder-capacity-v1/qwen-smoke-v2/preflight.json)
  and [later admitted preflight](qwen-response-v1/preflight.json) retain separate
  host observations.

Numeric, ownership and execution contracts are in the
[SDK reference](../../../docs/research/gkm-sdk-reference.md) and
[Decoder-capacity reference](../../../docs/research/gkm-decoder-capacity.md).
Old source versions and build history live in Git.
