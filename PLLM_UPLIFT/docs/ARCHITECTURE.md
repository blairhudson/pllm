# Architecture: native execution, composable privacy

## 1. Product boundary

PLLM turns a pinned model, declared numerical graph, privacy contract, party topology, workload and device profile into a **validated, role-sliced executable plan**. It can reproduce research, compare implementations and search for a better plan without silently changing the computation or trust model.

The original manuscript implements masked linear outsourcing with the client retaining attention, nonlinearities, embeddings/output head and model state (`legacy/Pasted text.txt`, sections 2–4). That remains the archival full-model baseline. It is not the thin-client target. Subsequent single-evaluator references demonstrate components and small synthetic graphs; they do not establish an integrated private transformer or near-native throughput.

## 2. Deployment profiles

| Profile | Client | Preparation | Inference | Eligibility |
|---|---|---|---|---|
| `archive.masked_linear` | Model boundary matrices and local operations | Model-aware, offline | One masked-linear worker | Reproduce archived numeric manifest first; client-heavy |
| `target.single_evaluator` | Input encoding and output decoding | Model-aware, offline only | One encoded complete graph | Preferred; fail until all operators/state are covered |
| `compare.two_online_fss` | Scheme-specific | Scheme-specific offline dealer | Two online workers | Explicit comparison only |
| `compare.he` | Scheme-specific | Scheme-specific | Scheme-specific | Explicit opt-in, never target default |
| `compare.attested` | Verify approved execution identity | Usually absent | Attested plaintext execution | Different hardware trust cohort, not ciphertext privacy |

The benchmark coordinator launches and collects public receipts and scrubbed metrics. It must not receive all role secrets. The user-facing Responses-compatible endpoint belongs to the trusted client/gateway. Remote Inference accepts only the encoded wire protocol. TLS authenticates and protects channels; it does not prove the evaluator computed the declared function.

The same host may simulate roles for functional tests, but cannot demonstrate administrative non-collusion or isolate against a host administrator. Frozen-inventory tests disconnect Preparation after readiness. Sustainable provisioning is measured as a separate input-independent supply regime, with all preparation resources counted.

## 3. Native-first ownership

| Native module | Responsibility | Python boundary |
|---|---|---|
| `pllm-model` | Read-only tensor store, shard/alias validation, content hashing, graph lowering and streaming conversion | Model source/config; inspection results |
| `pllm-ir` / `pllm-compiler` | Four IRs; bound propagation; fusion; legal-region enumeration; conversion placement and cost search | Immutable plan/config objects |
| `pllm-types` | Numeric, protection, role, state and evidence contracts | JSON/config validation diagnostics |
| `pllm-arithmetic` | Scalar/SIMD/GPU exact arithmetic, packing, CRT, modular reduction and reference oracles | Opaque buffers; batched/region operations |
| `pllm-garble` | Reviewed Boolean/arithmetic constructions and conversions; secret-label lifecycle | Backend configuration only |
| `pllm-prepare` | Model-aware setup, correlation/label generation, resource forecasts and role-specific material | Progress and public receipts |
| `pllm-inventory` | Atomic bind-before-send, durable allocation, retries, retirement and capacities | Public counters and typed errors |
| `pllm-runtime` | Per-role DAG scheduling, compute/transfer overlap, state progression, verification | Session handle, cancellation, output events |
| `pllm-transport` | Authenticated binary framing, flow control, sequence/replay limits, byte accounting | Deployment config; not Python hot-path transport |
| `pllm-bench` | Launch agents, clocks, resource accounting, native timing, comparison/search records | Experiment creation and presentation |
| `pllm-assurance` | View-limited adversaries, trace capture policies, native state/arithmetical tests and solver bridges | Public fixture selection; evidence review |
| `pllm-python` | PyO3/maturin bindings | Ergonomic SDK and CLI, no tensor/gate loops |

Five small native crates are included as integration source; most production modules above are work packages, not shipped implementations. Reuse mature crypto/device libraries through narrow Rust/C APIs. “Rust-first” does not require rewriting CUDA kernels or AES libraries into pure Rust. Python tensor arithmetic is limited to trusted test oracles and exploratory notebooks; its timings never populate native performance comparisons.

The Rust API releases Python execution through PyO3's documented `Python::detach` when doing substantial native work. A compiled region crosses FFI once; callbacks per label, scalar, row or network frame are prohibited in an optimized backend. Large tensors use owned native buffers, memory maps or validated zero-copy views with explicit lifetime/device synchronization. Do not return secrets through `Debug`, exceptions or serde logs. Mixed project layout and binding APIs follow the official Maturin/PyO3 references in `research/tooling_sources.json`.

## 4. Four IRs

1. **Semantic IR:** public parameters, operator mathematics, state slots and input/output semantics. No cryptographic assumptions.
2. **Numeric IR:** exact integer/fixed-point definition, quantization groups, rounding, scale, clipping, approximation coefficients, interval/ellipsoid certificates and accumulator bounds.
3. **Protected IR:** representation, holders, permitted views, encoding epoch, state lifetime, conversions, freshness effects, integrity obligations and allowed leakage.
4. **Executable plan:** per-party kernels, stream dependencies, buffers, messages, preparation objects and public workload buckets. No unresolved `auto` or mutable model revision.

A protected tensor includes logical shape, numeric type, bound-certificate ID, protection family, holder/view set, secret epoch, layout, state mutability and semantic-use identity. A `MaskedRing` is not an `ArithmeticLabel`; an `ArithmeticLabel` is not a `BooleanLabel`. An explicit, eligible conversion is required. Protocol input types must also specify side information: a client-mask codec is not automatically a codec for two-party shares or garbled labels.

## 5. Compiler pipeline

Resolve and lock model → semantic conformance → freeze numeric graph → derive/check public bounds → enumerate legal regional methods → insert conversions → validate contract/coverage → forecast time, memory and one-time material → place and schedule → measure promising plans → lock selected plan.

Planning constraints include all online parties, whether Preparation may hold the model, client memory/work, HE/hardware allowances, corruption model, accepted leakage, numeric fidelity and capacity limits. A cost optimizer cannot weaken these. Every candidate carries an assurance proof-obligation graph; a successful type check only satisfies declared preconditions, not the cryptographic theorem.

Compute region alternatives rather than independently picking each fastest operator. Costs include conversion, state transitions, garbling reload, cache residency and material transfer. Use measured tables, Pareto beam search or constrained dynamic programming first. E-graphs/ILP are optional later tools. Search calibration and held-out reporting workloads are separate; cryptographic randomness is fresh for distinct executions even with deterministic public fixture seeds.

Plan keys bind source/checkpoint and tokenizer digests, semantic/numeric graph hashes, method and kernel versions, parameter profiles, compiler/ABI version, role topology, leakage contract and workload bucket. Portable logical and hardware-specific execution hashes are distinct. Private seeds and live material never enter public plan manifests.

## 6. Method versus kernel versus adapter

A **method** defines protocol equations, required representations, parties, assumptions and phase lifetimes. A **kernel** implements one operation under that contract on a device. A **model adapter** describes architecture semantics. A **preparation provider** supplies a typed correlation/garbling schema. A **transport** moves bytes without changing the protocol.

Example: a matrix operator may lower to masked-ring evaluation, arithmetic-label evaluation or public multiplication on additive shares. These are distinct methods. Their CPU tile and CUDA implementations are distinct kernels. Only the first two may be eligible for a one-online-worker profile, and only when their remaining graph and client boundary meet that profile.

Unsupported methods return `E_METHOD_NOT_INSTALLED`; no placeholder or plaintext fallback executes. An experimental method can be benchmarked only with explicit authorization and an experimental evidence label; it cannot be auto-promoted by a fast time.

## 7. Preparation and state

Material types expose (a) reusable public compilation, (b) reusable-with-bounds private verification setup, (c) one-distinct-input privacy records, (d) immutable-operand fan-out material, and (e) request/token/circuit-specific garbling. These lifetimes are not interchangeable.

Client/dealer allocation binds the exact semantic execution before exposure. Retransmission sends identical frozen application bytes, or creates a new fully prepared execution. Timeout, cancellation or restored KV state never recreates fresh randomness. An in-memory state object and ordinary WAL do not defend against snapshot rollback or cloned writers; select a supported monotonic-state mechanism or exclude that adversary explicitly.

For complete single-evaluator generation, specify private embedding lookup, attention's private/private products, norm/rescale/activation, KV writes, sampler state, EOS, private selection and encoded token feedback. A client-held head is a different profile. A per-token request/response to the client is an explicit dependency, not “minimal client work” by wording. Garbled loops need bounded unrolling or a proved reusable/composable state construction; ordinary one-time circuits cannot be reused for different tokens.

## 8. Default selection

There is no currently validated single-evaluator SOTA full-model preset in the archived work. Preserve the original masked-linear path as an archival baseline after recovering its numerical lock. Develop the target profile on complete small regions with conservative native methods and reviewed conversion/scaling gadgets; add experimental weighted paths only as explicit alternatives.

The eventual default is a measured eligible plan indexed by model, numeric semantics, privacy/topology, client/resource budgets, hardware, network and workload. Numeric precision is chosen from a valid public certificate; cryptographic parameters come from the method's reviewed security profile, not the activation's small observed range. No global fixed prime count or label width is justified by these experiments.
