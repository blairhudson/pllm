# Public finite-language boundary: pinned checkpoint gate

## Scope

This follows the [cold grammar accounting screen](cold-grammar-boundary-reference.md)
with the actual pinned Qwen2.5-0.5B tied boundary, its original tokenizer, the
installed integer kernels and sampler, and native private page queries in separate
party interpreters. It is a boundary research executor, not a selectable Pipeline
or a complete response benchmark.

The public output language is exactly these three UTF-8 strings:

```json
{"label":"negative"}
{"label":"neutral"}
{"label":"positive"}
```

Whitespace variants, arbitrary JSON strings, private label lists and unrestricted
output are different contracts. Body activations and KV are not executed here;
the head audit uses eight public synthetic hidden vectors, including zero/tied
logits, an isolated coordinate and small/large scales. This is not a sentiment
accuracy result or a complete checkpoint-quality comparison.

## Measured result

The [frozen report](../evidence/public-grammar-boundary-qwen.json) contains one
publisher, one head auditor and twelve fresh lookup-party interpreters, with
three alternating replicates at each input length. Its SHA-256 is
`eee9b7742319c8b49adc3a2d6b8f906379cbb4f0a26cec541618b18c9ee0942a`.

* All completing tokenizations need **73 head rows**, rather than the eight-token
  union from canonical answer encodings and EOS. The compiler retains 38
  byte-prefix states. An independent full-vocabulary scan checks every state's
  outgoing token set against the original tokenizer.
* Full tied-head and scale delivery is **121,084,623 bytes** under the explicit
  bounded-zlib research framing; the selected rows, scales, global IDs and grammar
  are **63,710 bytes**, a **99.947% boundary-artifact reduction**. Raw framed
  sizes are 136,744,358 and 69,863 bytes. Tokenizer and ordinary body costs remain.
* **1,215,488 complete-head values**, **92,377,088 restricted-vocabulary values**,
  and **1,824 sampler/RNG comparisons** agree exactly. All **446,208 recovered
  embedding values** agree, and every party rejects the tested replay.
* Arbitrary JSON strings already admit at least **145,459** original tokens in
  their interior. This is not a general JSON or unrestricted-chat optimization.

| Prompt lookup rows | Both-party native bodies | Median summed evaluation CPU | Median summed setup CPU | Largest single-party peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 43,296 B | 12.583 s | 1.240 s | 308.81 MB |
| 150 | 405,900 B | 118.282 s | 1.238 s | 317.52 MB |

Every lookup sends two 388-byte queries and receives two 965-byte replies. At
150 rows, median client issuance takes 10.11 ms and native reply decoding 0.44 ms;
these exclude client startup, pipe orchestration, integer audits and serving work.
Each party evaluates the longer workload in roughly 59 seconds; that is not
concurrent-role or complete-request latency. All runs observe zero swap growth.
Paired fixture publication takes 9.789 CPU seconds and peaks at 460.26 MB; it
constructs both full and sliced controls, so it is not candidate-only setup cost.
The combined head/oracle auditor peaks at 118.72 MB, not whole-client RSS.

With locally derived worker tables, the selected artifact plus native lookup
bodies total **107,006 / 469,610 bytes** at 16/150 rows. Compared with the same
research full-boundary framing, those partial subtotals remove 120.98/120.62 MB.
Role setup, descriptors, authentication, tokenizer, body execution and transport
are still excluded. This establishes substantial **restricted-boundary byte
headroom with expensive lookup computation**, not a complete-request speedup,
traffic result or Low/Medium qualification. No task accuracy was measured.

Next gates: cheaper complete batched lookup, an explicit online lookup role graph,
authenticated source-bound delivery and compiler-bound restricted sampling/state,
then a serial cold SDK comparison charging all source-residency cases. The
118-second aggregate lookup CPU must stay visible rather than being amortized
over later requests. The existing offline-only prepared graph remains a different
contract.

## Compiler and numeric contract

`scripts/finite_language_boundary.py` compiles public finite byte strings against
the original ByteLevel BPE vocabulary and admitted added-token decoder semantics.
It includes **every completing token segmentation**, rather than taking only the
tokens returned by canonical `encode(answer)`. UTF-8 byte boundaries can lie
inside a code point. Unreachable/dead-end states are removed; all retained strings
must be reachable. EOS is the only admitted special output token and consumes a
selection. Remaining-budget pruning ensures completion including EOS fits the
declared cap. The grammar identity includes source tokenizer bytes, vocabulary,
EOS, byte prefixes, transitions, union and minimum completion lengths.

Admission bounds 32 nonempty strings, 256 UTF-8 bytes per string, 4,096 total bytes,
262,144 vocabulary entries and 65,536 transitions. The pinned screen separately
bounds the selected head to 512 rows. The language must be explicitly public;
that assertion does not prove a caller's provenance or permit private labels to
be published.

The publisher streams the source BF16 matrix through the installed per-row W8
quantizer in 512-row chunks. Head rows and lookup records retain those exact i8
codes and float32 scales. A record is 896 codes followed by its four-byte scale.
The table descriptor binds domain-separated SHA-256, row count and record width.
Actual source files, derived files, tokenizer and installed implementation have
separate digest locks.

The full paged head and sliced resident head use the installed `ClientBundle`
linear boundary. A separate bounded int64 oracle checks the full output, followed
by the original ordered float32 activation-scale and weight-scale multiplications.
For each grammar state and checked completion budget, the candidate rebuilds a
151,936-element float32 vector with disallowed positions at negative infinity.
The unchanged SDK greedy/nucleus sampler then receives the **same shape, global
IDs and bits** as the masked full-head oracle. This preserves NumPy sort/tie and
RNG behavior; a compact-array sampler cannot simply inherit that result. The
607,744-byte dense vector and sampler temporaries remain client owners.

The complete finite union is public and delivered once. Grammar state, selected
tokens and state-specific masks stay client-local. A tied generated-token row
already exists in that union, so an admitted composition would require private
lookups only for prompt rows. The older screen's dummy output lookups remain
part of its separate conservative accounting.

## Native lookup execution and ownership

The runner issues fresh native DPF query pairs for 16 or 150 **public fixture**
tokens. Each fresh worker interpreter receives only its own key half and owns
one immutable complete public table. Batches contain at most 64 queries; the
150-row case executes 64, 64 and 22. Both parties' native query and reply lengths,
table import/validation CPU, evaluation CPU/wall time and process peaks are
reported separately. Client issue/decode CPU is also counted. Every reconstructed
record is checked against the original file, and float32 embedding values agree.
Replay must reject; client decoder ownership retires on completion or failure.

The two interpreters run serially under the shared frontier measurement lock,
2 GiB post-reserve headroom admission, host-pressure/swap guard and process-group
cleanup. The publisher and head auditor are separate fresh interpreters. None
of their peaks is a complete client peak, and independent party elapsed times
cannot be added or maximized into a measured concurrent request latency.

Workers communicate over local pipes. Their native frames are measured; JSON/hex
pipe plumbing is test control and is **not** counted as provider transport. No
HTTP/TLS, lease admission or deployed authentication is implemented. Co-located
processes do not establish non-collusion, and the existing semi-honest DPF reference
has no independent cryptographic review or malicious-response verification.
Fresh process startup resets the research replay ledger; serving must bind a
proper session/epoch lifecycle before activation.

## Cold delivery accounting

Full and sliced head payloads are priced under the same explicit research artifact
framing: a 12-byte prefix, canonical source/shape/selection header and independent
at-most-1-MiB zlib blocks with eight-byte block lengths. Selected rows, scales,
global IDs and the full grammar contract are charged. These are exact counts for
that research encoding, **not an implementation of SDK bundle delivery**.
Tokenizer, body tensors, provider source distribution, role authentication,
descriptors and lifecycle controls still belong in a complete response bill.

Source residency has three distinct cases:

* **Existing full source at both lookup roles:** derive the tables locally.
  Network transfer of a second derived copy is unnecessary, but publication CPU,
  retained tables, import overlap and private storage remain costs.
* **Two new workers missing source:** delivering both raw record tables costs
  273,484,800 bytes before framing. This can erase the client-boundary saving.
* **All roles cold:** count source provisioning in both arms. When those same
  source files already include the tied matrix and records are derived locally,
  charging another full table transfer would double-count the information's
  delivery. A body-only provider artifact or a genuinely additional role still
  owes its missing boundary source. The result depends on the actual role graph.

The ordinary prepared graph has no online lookup role. Reusing Preparation for
lookup would select a different explicit role contract; it must not silently
alter baseline offline-only behavior. Adding two separate lookup services instead
must charge their complete cold setup and table distribution.

No raw head size is subtracted from historical compressed traffic. The previous
150-token control's 26.18% free-public-delivery ceiling remains an optimistic
whole-body ceiling, before lookup and setup, rather than this candidate's saving.

## Replay

```sh
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath=. tests/test_public_grammar_boundary.py
target/frontier-sdk-env/bin/python scripts/probe_public_grammar_boundary.py --output target/public-grammar-boundary-replay.json --repeats 3
```

The runner requires the already-cached immutable source revision
`7ae557604adf67be50417f59c2c2f167def9a775`; it never downloads checkpoints or builds
extensions. Timing, OS randomness and process peaks vary. Source/derived digests,
language reachability, numeric hashes and exact native frame counts are the
deterministic replay commitments. Output publication happens only after final
implementation identity checks and worker retirement.

Twenty-one focused tests pass, including exhaustive tiny token segmentations,
dead-end pruning, UTF-8 fragments, completion-budget/EOS rules, added tokens,
sampler RNG/ties, artifact mutation and native one-use lookup controls. Ruff and
the 435-page docs-content check pass. The parent independently reconciles all six
sample ledgers and medians and verifies seven implementation hashes; these
checks do not replace serving or cryptographic validation.
