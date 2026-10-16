# Writing rules for the PLLM papers

Use ASD-STE100 Issue 9 for `manuscript.md`, `low-memory.md`, and `whitepaper.md`.
The issue date is 15 January 2025.
The [official standard](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf)
contains the writing rules and the dictionary.
The standard remains the authority for word meanings and parts of speech.
The whitepaper is a short introduction, not a third academic manuscript.
Keep its explanations accessible without changes to its evidence limits.

## Sentence and paragraph rules

- Use a maximum of 25 words in a descriptive sentence.
- Use a maximum of 20 words in an instruction.
- Give only one instruction in each sentence, unless the actions occur together.
- Give each paragraph one topic and no more than six sentences.
- Use the active voice. In descriptions, use the passive voice only when the actor is unknown.
- Use only the verb forms and tenses permitted by Rules 3.1–3.4.
- Use infinitives, imperatives, and the simple present, past, or future tense.
- Do not use perfect or progressive tenses.
- Use an `-ing` form only as a technical noun, a modifier in a technical noun, or an approved dictionary word.
- Use verbs for actions, instead of abstract nouns.
- Use phrasal verbs only with a literal meaning or an approved dictionary meaning.
- Keep articles, subjects, and verbs. Short text must still have correct grammar.
- Use the same term for the same item.
- Give the meaning of necessary technical terms when they first occur.
- Avoid noun groups of more than three words.
- For an established technical noun with more than three words, use its full form first. Then apply Rule 2.2.
- Use a list when a sentence contains several conditions or separate items.
- Do not use semicolons in prose.
- Use American English spelling. Keep official names and quoted text unchanged.

Rules 8.4–8.7 specify word counts for lists, parentheses, units, names, and other special text.
Do not add parentheses or hyphens only to reduce a word count.
Do not replace a precise scientific term with a short word that changes its meaning.

## Technical terms

These terms have specific meanings in the papers.
They are technical nouns under the terminology rules in Section 1 of ASD-STE100.
Use a short explanation instead when the term is not necessary.

| Term | Meaning in these papers |
| --- | --- |
| activation | An intermediate value that the model calculates from its input or earlier values. |
| API gateway | The local application programming interface that connects an application to the runtime. |
| admission check | A check of the specified identities, resources, and policies before execution. |
| all-link traffic | Traffic across all measured links. Count each directed transfer once. State whether the count covers bodies, TCP streams, or physical wire. |
| arithmetic ring | A set of integers with addition and multiplication modulo a specified integer. |
| artifact | A stored file or group of files for an experiment, its build, or its audit. |
| benchmark | A specified test that measures a system or an operation. |
| boundary worker | A provider process for private token lookup and shared output-head computation. |
| causal simulator | A simulator that produces a party's view in protocol order without access to later private inputs. |
| centered reconstruction | Selection of the unique signed integer within the admitted bound from its modular representation. |
| checkpoint | A saved set of model parameters and its configuration. A conversation checkpoint is a separate saved state. |
| client | The trusted software that holds the private input and reconstructs the model output. |
| client memory tier | A maximum memory footprint for the complete client under the specified measurement contract. |
| cold-client run | A run with a fresh client and the declared initial public-cache state. Identify any artifacts already present. |
| coalition | A specified set of parties that combine their information. |
| cohort | A recorded group of runs. Comparisons within the group must obey the specified matching conditions. |
| collusion | Cooperation in which parties combine information that must stay separate under the privacy assumptions. |
| component | An implementation with a defined interface and execution requirements in PLLM. |
| composition | A specified combination of components. |
| coupled boundary | Private token lookup and the full output head at the same additional worker pair. |
| corruption | The adversary's control of a party under the specified security model. |
| decode | The phase that processes subsequent tokens after prefill. |
| digest | The result of a hash function. A digest does not authenticate its publisher by itself. |
| digit8 | An exact Preparation kernel with balanced eight-bit digits, bounded float32 products and sums, and integer ring reconstruction. |
| distributed point function (DPF) | Two keys that specify shares of a point function. Each worker evaluates only its own key. |
| embedding | A vector that the model uses in place of a token. |
| exact execution | Execution that keeps all values equal under the specified numerical and state contracts. |
| execution plan | The operations, dependencies, placement, and bounds that the compiler selects. |
| exclusive consumers | A compiler condition that prevents operations outside a selected region from use of its intermediate values. |
| frontier | Eligible configurations for which no eligible competitor is better in one metric and no worse in all others. |
| fresh private material | Private protocol material that is new for its specified use and obeys the protocol's independence and one-use rules. |
| full head | An output head that calculates a logit for every vocabulary token. |
| hash | A fixed-length value calculated from data by a specified hash function. |
| hybrid experiment | An intermediate experiment in a cryptographic proof. Each replacement has its own security assumption or exact equality. |
| inference | Use of a model to calculate an output from an input. Capitalized Inference names the prepared protocol's online role. |
| KV cache | Saved keys and values that attention uses for tokens already processed. |
| leakage | Information that the security model permits a party to receive. |
| logit | A score for one vocabulary token before the sampler selects the next token. |
| model boundary | The token lookup and output-head operations at the ends of the decoder. |
| numerical contract | Rules for data types, quantization, arithmetic order, rounding, and other numerical behavior. |
| native runtime | The compiled software that does the model and protocol operations. |
| non-collusion | The assumption that specified parties do not combine their information. |
| one-use owner | A program object that controls access to consumable protocol material. Its retirement does not guarantee secure erasure. |
| output head | The operation that calculates vocabulary scores from the decoder's hidden vector. |
| party | A participant in the protocol. A role name does not prove that its operator is independent. |
| peak RSS | The largest measured resident set size in the stated process and time window. |
| prefill | The phase that processes the input tokens before subsequent token decoding. |
| Preparation | The role that supplies fresh correction material before online prepared execution. |
| private lookup | A protocol to get a table row. Its index stays private under the specified assumptions. |
| provider | A service that does a specified part of the computation. |
| quantization | Conversion to a specified finite numerical representation. |
| residue | The remainder of an integer modulo a specified positive integer. |
| ring32 | Integer arithmetic modulo 2^32. |
| role | A specified set of duties in the protocol. |
| sampler | The client operation that selects a token from the complete logits. |
| sampling RNG | The random-number generator used for token selection. It is separate from the randomness used to protect protocol inputs. |
| secret share | One part of a protected representation. The protocol specifies how to combine the parts. |
| semi-honest party | A party that obeys the protocol but examines all information it receives. |
| source-present case | A case in which the specified checkpoint files are already available at the stated locations. |
| source lock | A record that identifies source files and their hashes. |
| stack | A specified composition of implementation mechanisms. The whitepaper lists the mechanisms in its exact prepared stack. |
| state | The cache, positions, history, logits, pending token, and sampling RNG state. Each test specifies its coverage. |
| state commitment | A digest of the recorded decoder state. The measurement contract specifies the included values and events. |
| static corruption | Corruption in which the adversary chooses the controlled parties before the protocol starts. |
| TCP-stream bytes | Bytes in measured TCP streams. This count excludes the kernel's packet headers, acknowledgements, and retransmissions. |
| token | One item in the model's vocabulary. It can represent a word, part of a word, or another text unit. |
| topology | The roles and connections used by a composition. |
| untimed view | Protocol data under the theorem's stated leakage rules. This view excludes timing and other specified observations. |
| verifier | A protocol operation with a pass or fail result. Each protocol defines the property and the conditions for that result. |
| warm-public run | A run that keeps declared public artifacts but uses fresh private material. Private-state reuse belongs to a separate comparison. |
| weight owner | A storage object that controls access to a weight matrix and its lifetime. It is not the model publisher. |
| worker | A process that does the work for a role. Several workers can belong to one operator. |
| XOR-PIR | Private information retrieval with exclusive-OR operations. |

Use complete technical nouns when modifiers change the meaning.
For example, a model owner and a memory owner are different concepts.
Prefer “the object that controls the memory” when “memory owner” is not necessary.

The following technical verbs are for computing operations:

- `bind` for data and execution identities
- `compile`, `decode`, `decrypt`, `encode`, `encrypt`, `hash`, `quantize`, and `serialize`
- `reconstruct` for a value from shares
- `XOR` for the bitwise exclusive-OR operation.

Use these verbs only when approved words cannot give the same information.
First consider an approved verb with a technical noun.
Technical verbs must also obey Section 3.

Software context alone does not justify `run` as a technical verb.
Prefer “do inference,” “start the software,” or “do the benchmark again” when these words give the same information.
Do not use `execute` or `run` as a general replacement for `do`.
Use ordinary approved verbs elsewhere, such as “calculate,” “compare,” “give,” “keep,” and “use.”

Names of models, papers, algorithms, APIs, and formats keep their official spelling.
An approved technical term does not make nearby ordinary jargon acceptable.

Rule 1.12 also permits technical verbs for mathematical and scientific processes.

- Use `prove` for a mathematical argument from stated assumptions.
- Use `assume` for a formal assumption in a mathematical or security model.
- Use `sample` for a draw from a specified probability distribution.
- Use `condition` for a probability distribution after an observation.
- Use `approximate` for a numerical approximation.
- Use `wrap` for modular arithmetic at a specified integer width.

These meanings do not apply to ordinary descriptions of features or measurements.
The same necessity rule applies to these mathematical verbs.

Prefer the following dictionary choices in ordinary prose:

| Avoid | Use when the meaning agrees |
| --- | --- |
| check, as an ordinary verb | do a check, or make sure |
| compute | calculate |
| ensure | make sure |
| establish | make sure, when the meaning agrees |
| execute | do |
| implement | do, or describe the specific software change |
| obtain | get |
| provide | give, or supply |
| reduce | decrease |
| require, or need | is necessary |
| retain | keep |
| test, as an ordinary verb | do a test |
| utilize | use |
| verify | make sure, or compare with the reference |

Before each substitution, make sure that the part of speech and the approved meaning are correct.
For example, use “do a test” instead of the ordinary verb “test.”
`Show` means to cause something to be seen, or to be in view.
Do not use it as a general replacement for `prove`.
`Follow` means to come after, not to obey a protocol or rule.
Use `obey` for that meaning.

The whitepaper keeps necessary technical nouns for its measured mechanisms.
These include packed inputs, output residues, pruned rows, demand-aware preparation, paged weights, compressed delivery, and an indexed tokenizer.
These names identify the measured stack, not ordinary-word exceptions for other contexts.
Use the same limited technical meanings as the framework's implementation records.

Explain necessary abbreviations at first use.
The established label `W8A8` means eight-bit weights and activations.
The scorecard label `SOTA` means the best measured eligible composition for a specified metric and workload.
It does not, by itself, give evidence for a field-wide state-of-the-art claim.

## Keep the scientific meaning

- Keep every measurement, unit, denominator, comparison direction, and uncertainty.
- Keep the original source, build, workload, cohort, and measurement window.
- Keep all assumptions, limitations, exclusions, and open evidence requirements.
- Keep equations, code, citation keys, identifiers, and artifact paths unchanged.
- When a table label must stay exact for an audit, give its meaning in the prose.
- Do not change “zero observed swap growth” to “zero swap.”
- Do not change a failed test into a missing test, or a missing measurement into zero.
- Do not change a conditional mathematical result into a claim about runtime privacy.
- Do not describe invalidation of a handle as secure erasure of its underlying data.
- Do not describe an archived comparison as a new experiment or a state-of-the-art result.

## Review

Compare the rewritten text with its previous version before the paper build.
Do separate checks of sentence structure, dictionary meanings, technical terms, and scientific meaning.
A sentence-length check is not a full ASD-STE100 review.

After changes to the manuscripts, do the evidence checks in `README.md`.
Build all three PDFs.
Examine their tables, captions, equations, and references.
Make sure that the whitepaper still fits its two-page limit.
