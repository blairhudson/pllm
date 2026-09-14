# Privacy assurance: prove scoped properties, actively try to refute claims

## 1. What the benchmarking tool can and cannot establish

The user-facing system should expose **assurance evidence alongside performance**, not claim a benchmark universally proves privacy. It can:

- prove an explicitly modeled finite/specification property using a solver or proof assistant;
- check a computational proof's prerequisites and track its applicability to the implementation;
- find counterexamples, concrete leakage and successful adversaries against an explicit target;
- measure distinguishability/timing under a defined statistical experiment;
- confirm operational constraints visible to the harness, such as zero online Preparation contacts.

It cannot prove arbitrary computational hardness, administrative non-collusion, secure physical erasure, absence of every side channel, or universal privacy from unsuccessful attacks. An assumption can be mathematically stated yet remain unverifiable in a deployment. “No plaintext payload found” is telemetry hygiene, not semantic confidentiality.

## 2. Three different layers

| Layer | Example | Accepted evidence |
|---|---|---|
| Specification | Masked arithmetic reconstructs a bounded integer; each material ID has one semantic assignment | SMT/Lean/Kani/EasyCrypt artifact with exact model, bounds and assumptions |
| Implementation | Rust refinement matches that spec; SIMD arithmetic agrees; no state-reuse race | Unit/property/fuzz tests, code-model correspondence, concurrency model checks, independent review |
| Deployment | Parties do not collude; machine is configured as claimed; secrets stay out of host logs | Explicit operational controls, attestations where applicable, audits and residual assumptions |

The included SMT suite proves only finite bitvector/algebraic or bounded transition-model statements. It does **not** verify the supplied Rust machine code or prove the complete protocol. Formal verification tools are also part of the trusted computing base; record their versions and check proof artifacts independently where supported.

## 3. Claim and experiment records

Each `ClaimSpec` includes:

`id`, `method_version`, `protected_property`, `adversary` (semi-honest/malicious/adaptive), `corruption_sets`, `party_inputs_and_views`, `allowed_leakage`, `freshness`, `randomness_model`, `numeric_domain`, `setup_assumptions`, `hardware_assumptions`, `proof_reference`, `composition_obligations`, `parameter_profile`, `known_attacks`.

Each `AssuranceResult` uses one of:

- `proved_in_model`: checked theorem/specification, with scope and model-to-code link status;
- `refuted_in_scope`: witness/adversary violates the stated claim;
- `not_refuted`: finite campaign found no violation; not a theorem;
- `inconclusive`: resource limit, confidence too weak or missing instrumentation;
- `outside_contract`: the attempted condition intentionally violates the declared assumption;
- `unchecked`: not executed or not reviewed.

A deliberately colluding experiment recovering a prompt is an expected demonstration of the non-collusion boundary, not a refutation of a theorem that excludes collusion. The tool should nevertheless show that deployment dependence prominently.

## 4. Cryptographic game harness to implement in Rust

1. Freeze a public workload, numerical graph, security parameter, shape/length/access leakage and corruption set.
2. Challenger samples bit b and selects one of two inputs with **equal permitted leakage**. Use a deterministic public fixture seed only for inputs; draw fresh cryptographic randomness independently.
3. Roles run in separate processes. An adversary receives only the corrupted party's specified initial knowledge, its actual messages, permitted memory/state and observed timing/access metadata. The test oracle may know everything, but must not leak that knowledge into the adversary process.
4. Adversary returns a guess or a reconstruction witness. Malicious adapters may tamper with messages, labels, model references or acknowledgements within a bounded interface.
5. Record success relative to the optimal public-prior/allowed-leakage baseline, not a naive 50% baseline for an unbalanced reconstruction problem.
6. Retest on held-out inputs, unseen sessions and fresh randomness. Avoid training and evaluation on the same traces. Report failed/time-limited trials and attack budget.

For balanced binary distinguishing, report `abs(P[D(view1)=1] - P[D(view0)=1])`; `2*accuracy-1` requires balanced labels and a pre-fixed label orientation. Confidence intervals must account for request/session clustering. Repeated tokens from one stateful request are not independent trials. Predetermine multiple-testing and sequential-stopping controls. A classifier failing to distinguish finite samples does not prove the underlying distributions equal.

Trace capture with honest-party secrets is allowed only in clearly marked public-fixture debug mode. It invalidates the realistic adversary test if that trace is available to the attacker. Never put real prompts, active masks/labels or reusable secrets into normal benchmark output. Public test transcripts may be retained under a dedicated evidence policy and cannot be reused as production material.

## 5. Required adversary families

| Family | Regressions and witnesses |
|---|---|
| Mask/label algebra | Reused pads, affine-offset recovery, public low-rank nullspaces, scalar remasking finite differences, Fourier phase disclosure |
| Representation | Sparse point-and-permute support, unsafe modulus reduction, missing truncation carry, field/ring confusion, overflow, public lane reconstruction |
| State and transport | Send-before-bind, changed-input retry, late correction reinstallation, duplicate token emission, cancellation/restart, cloned state, rollback |
| Protocol integrity | Wrong matrix, wrong model/quantization lock, malformed labels, forged receipts, invalid CRT/proof parameters, selective failures |
| Metadata/side channels | EOS/length distinctions outside the leakage budget, secret-dependent fallback, variable decision paths, secret expert routing, table/cache access, allocation and timings |
| Setup assumptions | Combined dealer/evaluator view, retained seeds plus later online transcript, malicious correlations, public seeds that regenerate evaluator-prohibited masks |
| Numeric correctness | Approximation tails, decoder domain collisions, saturation mistaken for modular arithmetic, per-residue sign versus true integer sign |

A timing campaign can use dudect-style fixed-vs-random measurements, but Welch statistics are diagnostic evidence, not a proof of constant-time execution. Test every relevant compiled target, compiler option, ISA and device. Secret-index table paths need a justified cryptographic access-pattern argument; constant path length alone is insufficient.

## 6. Included executable checks

`security/formal/manifest.json` lists ten queries. Seven are expected UNSAT and three SAT. Two UNSAT queries prove **leakage identities** for intentionally broken schemes; they are not privacy successes. The finite ideal-view inverse gives a scoped algebraic basis for a uniformity argument, not a PRG assumption proof.

The Python command invokes the native Z3 C API through a small wrapper. The six public negative controls reproduce mask reuse, affine-label reuse, sparse point-and-permute leakage, missing truncation carry, early-exit metadata and a low-rank mask projection. A separate exhaustive ideal uniform control checks 512 mask/input assignments.

The native Rust smoke includes a similar ideal-view check and an in-memory assignment invariant; its tests are supplied but not executed here because the toolchain was unavailable. Full role-view attacks against current PLLM must be connected during migration.

## 7. Formal proof programme

- Start from the exact source protocol and identify which proof applies to each registered implementation.
- Express arithmetic and lifecycle properties in SMT/Kani with a clearly bounded state model.
- Link model functions to Rust implementations using refinement assertions or independent exhaustive/property tests; these are different strengths of evidence.
- Use a game-based proof assistant such as EasyCrypt for randomness replacement, party-view simulation and composition where appropriate. The package specifies obligations; it does not supply a fabricated completed cryptographic proof.
- Document key/hash assumptions, subgroup/modulus conditions, composition dependencies, security parameter selection and lifetime bounds. Parameters justified for one field are not silently carried into a ring or another label format.
- For a novel weighted-path or sparse-LUT scheme, prove the actual observable view, including graph addresses, correlations, failures and repeated execution permissions. A proof of ordinary decision diagrams does not cover the custom encryption.

Official Kani, EasyCrypt, Z3 and dudect sources are recorded in `research/tooling_sources.json`. Finite specification proofs can be valuable immediately while larger computational proofs and implementation reviews remain open; do not conflate either with the other.

## 8. Promotion and reporting

A method may be an experimental benchmark candidate with unresolved assurance obligations. A preferred default additionally needs complete functional coverage, applicable proof/assumption review, tested production cryptographic parameters, supported lifecycle/deployment controls and an independently checked implementation. Native performance success alone never grants eligibility.

Normal reports show the privacy contract first, then the proof/evidence matrix, then compatible performance comparisons. An unsafe optimization may appear only in an attack appendix, with its witness—not as a fast frontier point. Source-reported security claims, a paper's theorem, our adaptation and independent review are separate records.
