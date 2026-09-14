# Extensible model support

## First target

Use Qwen2.5-0.5B-Instruct because it is the original reported full-model baseline, not because its small shape proves generality. The official configuration describes a dense decoder with grouped-query attention, RMSNorm, SiLU, rotary positions and tied embeddings. Pin the source revision, tokenizer and every tensor shard before execution. Metadata inspection is not full-checkpoint validation.

The original manuscript names source revision `7ae557604adf67be50417f59c2c2f167def9a775` and PLLM `277d19f`. Those are historical source-reported values; verify their accessibility and recover the numeric manifest before claiming reproduction. Do not silently replace it with a modern quantization setting.

## Adapter contract

Each adapter supplies source tensor mapping, alias/tied-weight semantics, dtype and shape validation, exact architecture feature flags, semantic graph, state transitions, tokenizer/config hashes, and conformance fixtures. Rust owns heavy file scanning, hashing, quantization and graph lowering. Optional Python integrations resolve user-friendly model identifiers and provide upstream trusted reference execution for tests; they do not define the production protected graph by implicit dynamic behavior.

Tensor stores support safe read-only row/tile access and out-of-core Safetensors layouts first. GGUF/MLX or other formats are independent format adapters with explicit packing/scaling support. Unknown custom code is disabled by default. Reject inconsistent shards, feature flags or tensor shapes; fail rather than guessing a familiar architecture from names.

## Required semantic operations

Public-weight dense maps; bias and scale; embedding gather; positional transforms; residual add; grouped/multihead attention layouts; private-private QK and probability-V products; causal/padding masks; softmax/normalization; activation/gating; quantization/rescale/cast; KV append/read; private argmax/sampling; EOS and encoded feedback.

Each operation has its own numerical graph and protected lowerings. Unsupported private selection does not become server-visible token indices. Output privacy means protecting the generated token feedback, not merely returning encrypted final strings.

## Conformance levels

1. **Format parity:** the intended tensors and aliases were read.
2. **Semantic parity:** upstream operation/state behavior matches on public fixtures.
3. **Numeric parity:** the frozen quantized graph matches its declared reference.
4. **Protected parity:** encoded execution matches that graph, including all conversions and state.
5. **Model quality:** quantify deviation from the original checkpoint and task performance.
6. **Deployment coverage:** actual separated-party full request with declared privacy topology.

A synthetic shape-compatible checkpoint can validate machinery but cannot establish official-model support. Every support badge names the level, checkpoint, context, mode and evidence file.

## New architectures

Model support is `format × semantic feature set × numeric profile × protocol family × device × workload`. Reuse existing operator implementations when only tensor naming/layout changes. New attention variants, routing, multimodal towers or recurrent state require semantic and protected implementations plus error/lifetime tests.

MoE weights are public but routing and per-expert token counts can be private; a sparse fast path must declare or protect that leakage. Sliding-window and compressed/requantized KV states invalidate naive immutable-cache assumptions. Learned normalization parameters and approximation errors must enter the compiler's range certificates. New RoPE scaling or cache behavior is not automatically supported by reading `config.json`.
