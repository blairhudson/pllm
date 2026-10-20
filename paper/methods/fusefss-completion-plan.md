# FuseFSS implementation plan

Status: bounded source decoder and ordinary SDK integration implemented and
tested on generated checkpoints. Full trained-model qualification remains open.
PUMA remains parked at its separate admission gate.

## Contract

Follow [D1, D3, D5, D8, D9, D13 and D14](DECISIONS.md), the
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
- `pllm_core::{public_file,fixed_file,streamed_matrix,integer_bounds}` supplies
  shared immutable storage, streamed public contractions and integer certificates.
  PUMA and FuseFSS retain their separate source encodings, rounding and proofs.
  Shared tokenizer validation and process-local resource reservations remain
  outside method-specific arithmetic.
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

Prioritize pinned Qwen benchmarking. Exact-key indexed replay owners and
framing-derived tape admission now support the larger generated Qwen SDK control,
including file-backed Decoder execution. Native tests preserve all 186 independent
Qwen observations; three-role tests preserve three selections and failure cleanup.

The static profile's earlier 1+1 checkpoint import rejects numeric admission
before workers launch. An offline Gram/Hadamard-lift diagnostic certifies the
layer-zero residual, but has neither full-layer coverage nor a bounded production
owner. Later conditional propagation does not certify earlier obligations.

The approved [D14](DECISIONS.md#d14-opt-in-checked-ranges-for-fusefss-qwen-completion)
profile now passes actual pinned-Qwen source, numeric and work admission at 4+1.
It preserves D13 arithmetic on accepted requests and rejects invalid requests only
at final aggregate acceptance. Shared private predicates/selection and their full
schedule are charged. Generated three-role tests cover late rejection without
token disclosure and fresh-request recovery. Actual checkpoint request acceptance
is still unmeasured; static proof improvements remain a separate follow-up.

Next: qualify whole-topology disk and execute the ordinary short Qwen factory,
check independent complete integer results, then quality and matched benchmarks.
The initial inline-metadata tape quote needs 71.53 GB including reserve; the host
rejects before child launch. External-AAD tape storage reduces the quote to
47.63 GB without removing authenticated metadata, material or replay claims.
Generated Decoder tests cover this format through actual role processes.
A separate small serial-HTTP benchmark attributes 97.82% of dealer preparation
to delivery callbacks; bounded delivery batching is the next execution-cost gate.
Launch uses bounded temporary source qualification, fresh exact
topology pricing and independently checked worker imports. Qualification cost,
all source/tape/replay disk, cumulative material and execution work stay charged.
The common 64+8 workload still exceeds sequence/selection bounds. Preserve the
approved arithmetic and role graph; obtain approval before substantial protocol,
numeric, state or trust adaptations.

See the [SDK contract](../../docs/research/fusefss-sdk-reference.md) and the
[historical scalar cost inventory](fusefss-decoder-feasibility.md) for separate
current and earlier scopes.
