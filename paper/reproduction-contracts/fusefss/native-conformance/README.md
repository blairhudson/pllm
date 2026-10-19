# Independent native conformance: track 3

**Status: independent algebra fixtures and tested read-only checker delivered.
No native implementation or cryptographic backend was run by this track.**
The subsequent coordinator integration replays all 58,400 compiler gates and
43,544 freshly issued protected GELU outputs through this unchanged checker.
Its [separate native evidence](../../../artifacts/fusefss-scalar/local-v1/README.md)
records both receipts, original outputs, source hashes and engineering reviews.
All files owned by this track are in this directory. Original scalar evidence is
unchanged. Whole-model, processes/wheels, security and source-manifest qualification
remain open. Passing the fixture generator is not native parity.

## Stable interface and commands

From repository root, Python 3 standard library only:

```sh
python3 -B -m unittest discover -s paper/reproduction-contracts/fusefss/native-conformance -p 'test_*.py' -v
python3 -B paper/reproduction-contracts/fusefss/native-conformance/check_native.py /path/to/native-output.jsonl
python3 -B paper/reproduction-contracts/fusefss/native-conformance/check_native.py --suite gelu /path/to/protected-output.jsonl
```

`fixtures.json` is the stable input, SHA-256
`64a2dfd493eab789a8aaca9fb12326affc7b12c09cc32ad3bcfbfeb8e3f7babd`,
2,542,504 bytes (including final newline). `receipt.json` is the compact receipt.
Regenerate **to stdout**, without touching original reference files:

```sh
python3 -B paper/reproduction-contracts/fusefss/native-conformance/independent.py
```

The checker takes a file or `-` for stdin; never executes it, never writes files,
never shells out, never loads models. Exit 0 plus one JSON summary means pass;
exit 1 plus JSON error on stderr means rejection. It rebuilds independent expected
values and verifies fixture canonical bytes before accepting native output.

### Parent Rust integration test/example contract

Input schema: `pllm.fusefss.independent-native-conformance.v1`.
`descriptors` entries provide `id`, full public `descriptor` object and expected
public `shape`. Descriptor canonical bytes are recursively sorted-key, compact
ASCII JSON; identity is SHA256(`FuseFSS-reference-descriptor-v1\0` || bytes).
`runs` appear in mandatory replay order. Each has `id`, `descriptor_id`, clear
dealer-test `mask`, unique `context`, original `words`, corresponding
`masked_words`, and independently calculated `expected` pairs `[A_word,[B_bits]]`.
**Native adapter must calculate outputs, not echo `expected`.** This interface is
offline dealer/test diagnostics, never an online evaluator wire/key format.

For `all`, replay all 1,552 runs. For `gelu`, filter `runs` by descriptor ID
`5f1e888e7bdabe3cbad4b6bbf1abd6749c8dff0918c6f243916db071398e79d3`, preserving
order: 8 runs, 43,544 evaluations. In either suite, finish with all 38
`external_encoding` cases, using exact numerator/denominator Q12 nearest-even.
Thus GELU-only backend need not pretend to support Boolean circuits.

Write JSONL, exactly one header, one record per selected run, one footer:

```json
{"schema":"pllm.fusefss.independent-native-conformance.v1/native-output","fixture_sha256":"64a2dfd493eab789a8aaca9fb12326affc7b12c09cc32ad3bcfbfeb8e3f7babd","suite":"all","execution":"compiler-clear","implementation":"parent-supplied native build identity"}
```

Each run record has **exactly** these keys:

| Field | Native adapter value |
| --- | --- |
| `id`, `descriptor_id`, `mask`, `context` | Echo run binding; adapter must also bind/validate its actual native instance. |
| `shape` | Object with exactly `source_intervals`, `padded_intervals`, `boolean_outputs`, `queries`, `degree`, `coefficients_per_interval`, `horner_products`; derive from actual compilation, not fixture echo. |
| `native_bytes` | Positive integer, <=16 MiB, actual fixed-width compiled diagnostic serialization byte count (same definition for every mask). Not RSS, traffic or cumulative preprocessing material. |
| `compiled` | Object with exactly `boundaries`, `ordinals`, `payloads`, `queries`. Dealer-only diagnostics. |
| `outputs` | Ordered actual `[A_word,[B_bits]]` pairs, one per `words` entry. Canonical ring words and integer bits; booleans/floats forbidden. |

`compiled.boundaries`: unsigned half-open partition including **wide N sentinel**.
`ordinals`: original source-piece index per translated/padded interval.
`payloads`: `[constant,linear]` per interval, original ring coefficients.
`queries`: `[bits,public_shift,private_threshold]` in normalized descriptor order,
two slots per nonsentinel atom; no deduplication. No Boolean VM transcript needed:
native Boolean outputs are checked against direct original-predicate truth tables.

Footer: `{"external_encoding":[<38 signed Q12 integer results in input order>]}`.
No extra records, blank lines, missing newline or duplicate JSON keys allowed.

For protected replay set `suite:"gelu"` and
`execution:"protected-local-reconstruction"`. Run actual protected lookup and
fresh shared degree-one Horner over original secret-shared x; reconstruct **only
in integration test**. Compile diagnostics may be captured at dealer, never sent
to evaluator. `native_bytes` still reports compiler diagnostic serialization;
actual party key sizes, fresh correlation counts, frames and security checks need
separate parent evidence. Never concatenate clear expected values into protected
result stream. This checker verifies functional values and reported structure;
it cannot prove an adapter actually invoked crypto or authenticated contexts.

## Independence and coverage

`independent.py` imports neither `reference.py` nor `math_oracle.py`, and never
calls their compile/evaluate functions. Source lock, vectors, README, source and
oracle were read as specification evidence. No compiled-record hash supplies
an expected native result. Descriptor metadata and identities necessarily follow
the frozen public contract; identical identities are not independent correctness
evidence on their own.

Expected arithmetic is a direct sum of interval indicators times expanded affine
expressions. Boolean bits directly interpret **original** low/signed/MSB/nested
predicates. GELU uses signed X and exact selected formula:

```text
q = floor(X/64)
j = min(abs(q),255)
T[j] = floor(2048*(j/64)*erfc((j/64)/sqrt(2)))
out = ([X >= 0]*X - T[j]) mod 2^50
```

Frozen `math_oracle.py` independently rational-certified table floors; this track
reuses its frozen table bytes **only after checking all 256 u64LE entries' hash**.
It does not regenerate approximate guesses or execute the oracle. A separate
host-erfc cross-check is supplementary, never authoritative for floors.
GELU interval geometry is solved from floor-bin inequalities and its affine
pieces are checked at both endpoints against that formula. Q0 slope multiplies
original Q12 x: no extra Q12 rescale. X=-1 returns -31, X=64 returns 33.

| Coverage | Independent expected cases |
| --- | ---: |
| Every complete partition of rings 2, 4 and 8, every mask/input | 138 partitions; 8,328 evaluations |
| Every small unsigned/signed/low threshold including sentinels; every MSB shift; Boolean combinations; every mask/input | 6,528 evaluations |
| Every GELU interval boundary and modular immediate neighbors plus signed38 limits, under seven masks | 1,539 unique words; 10,773 evaluations |
| Central signed codes -16,385 through 16,385 | 32,771 evaluations |
| Total | 184 descriptors; 1,552 runs; 58,400 gate evaluations |
| External encoding ties and adjacent quarter steps | 38 cases |

Signed38 provenance is **not** established by ring50 tests. Interpret X as signed50;
`-2^37 <= X < 2^37` identifies source-effective cases. Other cases test total-ring
extension only, including X=`2^37+64`, which must map to itself, not zero from
an incorrectly narrowed selector. Descriptor metadata retains this obligation.

Structural checker independently validates inverse images of native intervals,
both endpoints, original payloads, all natural boundaries, first-eligible padding,
M=min(m+1,N), full partitions with no extra padding, wide sentinels, exact query
slots/order and identical public shape/reported serialized size across masks.
Query thresholds follow public Protocol-3 equations; these structural checks are
**not** the independent numeric oracle. Sampled threshold zero retains its slot.

Hand-solved ring8 transcripts test checker acceptance without any compiler.
Negative tests reject missing/reordered/duplicate records and outputs, wrong
arithmetic/Boolean values/types, mask/descriptor/context/fixture bindings,
coefficients/order, wrong inverse images/padding/sentinels, dropped zero-threshold
queries, low-mask/shift mistakes, variable shape/bytes and wrong nearest-even
outputs. Descriptor rounding/Q-scale/selector mutations change identity.
These are checker and algebra mutation tests, **not native malformed-key,
replay-authentication or malicious-party security tests**.

## Budgets, results and remaining gates

Generator caps: 4,096 runs, 100,000 evaluations, 16 MiB fixture. Checker caps:
32 MiB native output, 2 MiB per JSONL record, bounded integer tokens; no unbounded
partition search. Parent must bound native subprocess time and Cargo resources
(one build job, serial tests), and enforce native allocation/material limits.
stdin can wait for a producer: checker is not a process supervisor.

Observed here: **22 unittest tests passed, 0 failures/errors/skips** (under 1 s).
Fixture regeneration and source hashing completed within 20 s command budget.
Local F and S PDF hashes matched source lock; S-CPU remains immutable hash-locked
remote evidence, not fetched/executed by this track. Fixture receipt is portable:
source availability is not numerical specification; present PDFs are hash-checked.

Native replay count **at the original fixture handoff: zero**. Subsequent parent
compiler/protected replay and one-use, allocation and malformed-frame controls
have separate linked evidence. Actual separate-role authenticated transport
remains unimplemented; local frame binding alone does not establish it.
Even successful protected scalar replay does not qualify whole-model semantics,
original evaluated descriptor identity, frontend signed38 provenance, security,
separate processes, installed wheels, SDK or performance.
