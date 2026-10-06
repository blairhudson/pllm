# HE softmax and private-pruning gates

These are source-specific feasibility checks, not reproductions of NEXUS, THOR
or CipherPrune. A numeric operator and an encrypted autoregressive service have
separate implementation and evidence requirements.

## NEXUS and THOR: bounded exponential references

`crates/pllm-core/src/polynomial_softmax_reference.rs` independently implements:

- NEXUS equation 4: repeated squaring of a shifted linear exponential estimate.
  This adaptation fixes a public shift of 16 and eight squarings.
- THOR Algorithm 3 and Appendix C: the printed degree-15 exponential polynomial,
  with public `delta1=1`, `delta2=32` and five square/normalize steps.

Both accept finite attention scores in the fixed public domain `[-16,16]`.
Exact float64 normalization is an **optimistic plaintext oracle**: it omits
encrypted reciprocal evaluation, CKKS error, bootstrapping and ciphertext costs.
No private maximum is subtracted, and no private-prompt fitting or clipping is
performed. Causal slots are removed only after checking their public positions;
a nonfinite valid score cannot masquerade as a padding slot.

Native tests check probability/order invariants, rejection, approximation error
and the repeated-squaring construction's lack of exact shift invariance. An
independent NumPy equation oracle checks native values and public causal extents.
The same-token pinned Qwen2.5 W8A8 diagnostic rejects **all eight public prompts**
for both methods: 100,676 observed finite scores per method exceed the fixed
range before a complete position finishes. This count covers the visited
operators before rejection, not every attention score in the full trajectory.

There is no completed candidate output and therefore **no quality score**.
This rejects these fixed parameters; it does not prove that either paper's
method cannot support Qwen with a different public calibration or representation.
Public calibration, protected range reduction, full reciprocal/normalization,
Qwen RMSNorm/SiLU/GQA/rotary state, ciphertext packing and private token feedback
remain separate gates. Ciphertext execution and network costs are unmeasured.

```sh
uv run python -m benchmarks.research.he_softmax_reference \
  --output docs/evidence/research-he-softmax-reference-reproduction.json
cargo test -p pllm-core polynomial_softmax_reference
uv run pytest -q tests/test_he_softmax_reference.py
```

The canonical report is `docs/evidence/research-he-softmax-reference-qwen25.json`.
It records exact paper, source, checkpoint, public-policy and token-cohort digests.

## CipherPrune: source contract before implementation

The pinned ICLR paper, arXiv:2502.16782v2, section 2 and sections 3.2–3.4,
describes a client/server private-weight, semi-honest setting. Its pruning and
polynomial-reduction thresholds are jointly learned offline. This differs from
the source-preserving public-weight prepared baseline.

The basic pruning protocol reconstructs a token mask. The additional `Pi_mask`
protocol hides original selected locations through oblivious stable compaction,
but reveals the retained count. It binds each selection bit to its token,
converts and opens the count, performs oblivious swaps and removes the binding
bit. The paper gives `O(mn)` swaps for pruning `m` of `n` tokens. Its polynomial
reduction additionally discloses a mask in compacted order. These disclosures
must be stated in any proposed PLLM leakage contract, rather than inherited as
an unspecified claim that selection is private.

No CipherPrune executor or benchmark is added by this review. Required work:

1. Independently implement one-use comparison, bit/arithmetic conversion and
   oblivious compaction, including complete material and swap costs.
2. Declare count/reduction-mask leakage or implement padded hidden alternatives
   under their own costs and contracts.
3. Bind original positions, RoPE and persistent KV under progressive pruning;
   a shortened full-KV array is not an admitted decoder state.
4. Lock the trained/calibrated model and validate held-out same-token and
   whole-generation quality before matched role-backed measurements.

The [MPCache selected-view oracle](mpcache-reference.md) establishes none of
these protected compaction or trained-model obligations.
