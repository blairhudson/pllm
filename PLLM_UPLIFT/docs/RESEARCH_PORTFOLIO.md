# Top 20 research reproductions

The ranking prioritizes the preferred one-online-evaluator architecture, essential dependencies and strong/adversarial comparisons. It is not a speed ranking and does not imply all twenty fit one topology. Source access and prior implementation evidence are recorded independently.

| ID | Source | Track | Native modules |
|---|---|---|---|
| R01 | [Dash: Accelerating Distributed Private Convolutional Neural Network Inference with Arithmetic Garbled Circuits](../research/cards/R01-dash.md) | single_evaluator | garble.affine_label, kernel.label_tensor, gate.projection |
| R02 | [ReDASH: Fast and efficient Scaling in Arithmetic Garbled Circuits for Secure Outsourced Inference](../research/cards/R02-redash.md) | single_evaluator | convert.rns_base_extension, gate.scale, numeric.scalequant_plus |
| R03 | [Garbling Gadgets for Boolean and Arithmetic Circuits](../research/cards/R03-garbling_gadgets.md) | single_evaluator | garble.arithmetic, gate.mixed_modulus |
| R04 | [Two Halves Make a Whole: Reducing Data Transfer in Garbled Circuits using Half Gates](../research/cards/R04-half_gates.md) | single_evaluator | garble.boolean, kernel.half_gates, kernel.free_xor |
| R05 | [Garbled Circuit Lookup Tables with Logarithmic Number of Ciphertexts](../research/cards/R05-logrow.md) | single_evaluator | gate.logrow, convert.lookup_io |
| R06 | [Duty-Free Bits: Projectivizing Garbling Schemes](../research/cards/R06-duty_free_bits.md) | single_evaluator | convert.bit_to_affine, prepare.vector_ole |
| R07 | [Slalom: Fast, Verifiable and Private Execution of Neural Networks in Trusted Hardware](../research/cards/R07-slalom.md) | client_heavy_baseline | protocol.masked_linear, verify.preprocessed_linear |
| R08 | [Slalom at the Carnival: Privacy-preserving Inference with Masks from Public Knowledge](../research/cards/R08-carnival.md) | quarantined_comparison | prepare.subset_sum, attack.carnival_projection |
| R09 | [HyCC: Compilation of Hybrid Protocols for Practical Secure Computation](../research/cards/R09-hycc.md) | compiler | compiler.region_selection, compiler.conversion_costs |
| R10 | [Oblivious decision program evaluation](../research/cards/R10-oblivious_decision_programs.md) | single_evaluator_adaptation | gate.oblivious_branching_program, convert.branching_input |
| R11 | [Factored Edge-Valued Binary Decision Diagrams](../research/cards/R11-fevbdd.md) | public_compilation | compiler.weighted_decision_diagram, compiler.variable_order |
| R12 | [Piranha: A GPU Platform for Secure Computation](../research/cards/R12-piranha.md) | mpc_comparison | kernel.modular_gpu, runtime.device_buffers |
| R13 | [SIGMA: Secure GPT Inference with Function Secret Sharing](../research/cards/R13-sigma.md) | two_online_comparison | protocol.fss, gate.fss_nonlinear, prepare.fss |
| R14 | [FuseFSS: Efficient Secure LLM Inference with Function Secret Sharing](../research/cards/R14-fusefss.md) | two_online_comparison | gate.fss_fused, compiler.fss_fusion |
| R15 | [Efficient Pseudorandom Correlation Generators over Z/p^k Z](../research/cards/R15-ring_pcg.md) | preparation_comparison | prepare.ring_pcg, correlation.ole, correlation.authenticated_triple |
| R16 | [Practical Secure Delegated Linear Algebra with Trapdoored Matrices](../research/cards/R16-trapdoored_matrices.md) | linear_comparison | prepare.trapdoored_matrix, protocol.delegated_linear |
| R17 | [Curl: Private LLMs through Wavelet-Encoded Look-Up Tables](../research/cards/R17-curl.md) | mpc_comparison | gate.wavelet_lookup, numeric.wavelet, gate.probabilistic_truncation |
| R18 | [Compact: Approximating Complex Activation Functions for Secure Computation](../research/cards/R18-compact.md) | numeric_transform | numeric.piecewise_fit, compiler.approximation_profile |
| R19 | [Private and Verifiable Outsourcing of Open-Weight LLM Inference](../research/cards/R19-open_weight_verifiable.md) | two_online_comparison | protocol.open_weight_verified, verify.model_owner_setup |
| R20 | [Oblivious Ciphertext Compression via Linear Codes](../research/cards/R20-oblivious_compression.md) | representation_specific | codec.oblivious_linear_code, codec.domain_certificate |

## Delivery waves

**Foundations:** R03/R04 arithmetic and Boolean garbling, R07 archived masked outsourcing and verification, R09 compiler comparison, R11 public decision-diagram compilation.

**Preferred target:** R01/R02 complete arithmetic-garbling/scaling, R05/R06 lookup and conversions, R10 protected path comparison. Integrate complete regions before claiming a full decoder.

**Native and broad comparison:** R12 GPU execution, R13/R14 FSS, R17/R18 alternative nonlinear representations. Their original roles/numeric semantics remain visible.

**Frontier and adversarial review:** R08 attacked preprocessing variant, R15 correlation generation, R16 trapdoored linear delegation, R19 current open-weight verification and R20 representation-specific compression. Abstract-only sources must be acquired before a faithful reproduction is possible.

## Reproduction format

Every research card separates the source contribution, proposed PLLM module, exact data flow, test/benchmark requirements, privacy obligations, existing evidence and unimplemented parts. `recipes/` specifies acquisition-to-publication gates and required artifact names. All original-artifact commands and source hashes are unset until actually acquired; inventing them would defeat reproducibility.

The portfolio includes public-mathematics/compiler work (R09/R11/R18), cryptographic primitives, full systems and comparison-only methods. A method's native implementation status and proof status are separate. An original paper's proof is not inherited by a modified arithmetic domain, conversion or role assignment.

## Source access audit

Several papers were available only through primary abstracts or author metadata. Those records are explicitly gated. We did not package full third-party papers or upstream source code without a license review. Sources remain canonical links with citation records; source snapshots and original artifact licenses are the first implementation task.

Twenty cards do not mean twenty reviewed cryptographic constructions. This package's native code is an integration foundation, not an implementation of every paper. Existing Python/C++ experiments are archived adaptations. Their particular equations, correctness counts and CPU timings do not establish a native reproduction of the cited source system.

## Source differences that must not disappear

- R08 is a quarantined attack-comparison target because the precise Carnival instantiation must be checked against the reported Maverick attack.
- R19 requires two online non-colluding servers; it cannot be reported as the preferred one-evaluator architecture.
- R20's original ciphertext-compression setting is not automatically our masked-ring codec or compatible with a no-HE profile.
- R06 supplies specific conversion directions, not a free cast between every protected representation.
- R11 compresses public functions; privacy comes from a separate protected-evaluation construction.
- R17/R18 numerical transformations require model-quality evaluation beyond exact agreement with their own approximation.

Supplementary compiler/runtime and mathematical literature from all previous rounds remains in `legacy/research_registry_v1.json`. The selected twenty are the implementation programme, not an exhaustive bibliography.
