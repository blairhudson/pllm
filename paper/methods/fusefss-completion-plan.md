# FuseFSS implementation plan

Status: bounded source decoder and ordinary SDK integration implemented and
tested on generated checkpoints. Full trained-model qualification remains open.
PUMA remains parked at its separate admission gate.

## Contract

Follow [D1, D3, D5, D8, D9 and D13](DECISIONS.md), the
[shared computation contract](shared-computation-contracts.md#4-fusefss) and
[frozen scalar specification](../reproduction-contracts/fusefss/README.md).
The implementation is independent PLLM Rust/Python code.

D13 selects two explicit ring50/Q12 profiles:

- `fusefss.gpt2-complete-ring50-q12.v1`: learned positions, centered LayerNorm,
  affine GELU and Conv1D-oriented public projections.
- `fusefss.dense-rope-ring50-q12.v1`: source-table RMSNorm, gated SiLU and
  split-half public RoPE. Negative rotary coefficients are applied before each
  product floor; floors cannot move across subtraction.

Both preserve nearest-even external encoding, source-defined internal floors,
two online share holders, a separate trusted offline dealer, complete-vocabulary
lowest-index argmax, fixed-count full-prefix feedback and final client-only IDs.
The dense-RoPE profile is a named adaptation. The paper's missing evaluated
descriptor manifest still prevents claiming its exact evaluated numeric identity.

## Gates

| Gate | Current evidence | Remaining work |
| --- | --- | --- |
| F0: source and feasibility | Pinned equations, public model metadata and explicit cost estimates | Reprice complete trained-model execution after each scaling change |
| F1: scalar | Typed descriptors, shared DCF, affine lookup and consuming products; independent frozen GELU checks | Independent security qualification |
| F2: protected gates | Complete SiLU, softmax, LayerNorm/RMSNorm and floor/conversion gates; 275 frozen cases | Full-model domains and efficient complete widths |
| F3: semantic decoder | Five generated GPT-2/Qwen2/Qwen3/Llama controls; 918 complete semantic observations, private feedback and final IDs | Trained-checkpoint integer/domain admission; original 5,125 internal-tensor interface is not claimed |
| F4: ordinary SDK | Immutable components, source import, three actual role children, encrypted one-use tapes, SDK/gateway/CLI, failure/cancellation and standard benchmark paths | Larger admitted workloads, independent deployment and security review |
| F5: model qualification | No passing full trained-model result | Original-model and pinned common-Qwen capacity, exact integer parity, quality and matched resource/performance cohorts |

The complete method remains unqualified until F5 passes. Earlier gates supply
only their stated scopes; generated benchmarks cannot fill a trained-model cell.

## Implemented paths

- Shared absolute-position model lowering lives in
  `crates/pllm-models/src/absolute_position_decoder.rs`; checkpoint names and
  orientations remain source-mapping data.
- `crates/pllm-compiler/src/fss_decoder/` binds public source, numeric profile,
  semantic operations, fixed-count execution, resource bounds and one-use roles.
- `crates/pllm-garble/src/fusefss_backend/` supplies the common typed gates,
  exact kernels, compact private one-hot lookup and bounded material codecs.
  Immutable source-table sharing removes repeated table construction; fresh
  masks, keys and request state are never cached.
- `FuseFssSourceDecoder`, `TrustedDealerTwoShareRoles` and
  `FixedCountPrivatePrefix` compose through `FuseFssSourceCpu`.
- `python/pllm/runtime/fss_*` owns bounded source import, authenticated channels,
  one-use encrypted preprocessing tapes, offline-dealer retirement and draining
  cancellation. Local children establish role separation, not non-collusion.

## Reproduce

Build/install with the standard uv/Maturin commands in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md). Use ordinary tests for correctness:

```sh
cargo test --release --locked --offline -p pllm-compiler --lib fss_decoder:: -- --test-threads=1
uv run --no-sync python -m pytest -o pythonpath=. tests/test_fss_sdk.py tests/test_fss_native_transport.py
```

[`examples/papers/fusefss_smoke.py`](../../examples/papers/fusefss_smoke.py)
constructs reproducible generated sources for both profiles. Its ordinary CLI
outputs are in [`paper/artifacts/fusefss-sdk/`](../artifacts/fusefss-sdk/README.md).
Use [`examples/papers/fusefss.py`](../../examples/papers/fusefss.py) for the pinned
common-Qwen configuration; selection does not waive its admission failures.

Benchmark reports retain dealer and holder costs, directed bodies, separate tape
I/O and setup-inclusive final disclosure. Buffered output supplies neither streaming
TTFT nor model decode throughput. No receipt archive or copied build/source tree
is required.

## Next blocking slice

The bounded F3 fixtures fit native work/live-owner admission after immutable-table
sharing, but their 24,769–49,933 primitive imports exceed F4's request-long
4,096-entry replay/tape limit. Do not reset the guard or reuse material. A bounded
monotonic schedule owner needs replay, out-of-order, cancellation and partial-import
tests before it can replace that retained set; disk and cumulative work remain
separate limits.

Full GPT-2 and Qwen also exceed existing source/sequence/value bounds. Their
complete norm/table, private-product, full-vocabulary and full-prefix work must be
admitted and priced before any tensor or material launch. Raising a constant does
not supply this proof. Preserve the approved arithmetic and role graph; obtain
approval before substantial protocol, numeric, state or trust adaptations.

See the [SDK contract](../../docs/research/fusefss-sdk-reference.md) and the
[historical scalar cost inventory](fusefss-decoder-feasibility.md) for separate
current and earlier scopes.
