# Exact private lookup with full-vocabulary generation

## Contract

This follow-up accelerates the lookup substrate examined in the
[public grammar boundary](public-grammar-boundary-reference.md). The original
LLM contract remains: the complete vocabulary, tokenizer, output head, sampler,
generated-token feedback and active KV state are retained. Finite-label output
heads are a different application contract and supply no general-chat saving.

Qwen2.5-0.5B ties its token table and output head to the same quantized matrix.
Moving token lookup alone therefore removes **zero client weight bytes**. The
complete head and scales stay charged in each arm's byte ledger. Private lookup
adds its query/reply bodies and two worker table owners. This experiment measures that
extra operation's cost; it does not establish a cold-request traffic reduction.

## Measured result

The [frozen report](../evidence/private-lookup-acceleration-qwen.json) contains
one public publisher and **36 fresh party interpreters**. Its SHA-256 is
`b61652637395bfd78d318c9384841f44225c4b35435cd848fd3ff00d84d5d26b`.
Every cell has three alternating replicates. All **1,338,624 embedding values**
and their source record bytes match exactly; every tested replay rejects and
observed swap growth is zero.

| Prompt rows | Evaluator | Both-party evaluation CPU median | Both-party native bodies | Largest party peak RSS |
| ---: | --- | ---: | ---: | ---: |
| 16 | Original scalar | 12.4433 s | 43,296 B | 316.36 MB |
| 16 | Accelerated scalar | 1.4478 s | 43,296 B | 316.34 MB |
| 16 | Accelerated packed128 | 0.1470 s | 39,968 B | 316.60 MB |
| 150 | Original scalar | 117.8642 s | 405,900 B | 321.03 MB |
| 150 | Accelerated scalar | 13.3425 s | 405,900 B | 326.89 MB |
| 150 | Accelerated packed128 | 1.2786 s | 374,700 B | 326.48 MB |

At 150 rows, accelerated scalar is **8.83×** faster in the measured evaluation
CPU window. Packed128 reaches **92.18×**, or **98.92% less evaluation CPU**, with
**7.69% fewer native frame bytes**. Scalar query/reply sizes remain 388/965 bytes
per party; packed query/reply sizes are 284/965 bytes. This screen combines ARM
AES, encryption-only scheduling and parallel block encryption in its scalar
arm; it does not isolate those three effects from one another.

Import and source-validation CPU remain about 1.22–1.24 seconds summed across
the two parties. The median of each sample's setup-plus-evaluation total falls
**119.0851 → 2.5122 seconds** at 150 rows, a **47.40× scoped reduction**, rather
than 92×. Process startup and publisher work have separate windows. Publication
costs 6.7043 CPU seconds and peaks at 394.07 MB; packed client issuance/decode
medians are 1.38/0.432 ms. The largest packed party peak rises 1.70% versus the
original scalar arm. These are worker resources, not whole-client measurements.

The complete tied head still occupies 136,742,400 raw matrix/scale bytes and
**121,084,623 framed compressed bytes**. With locally derived worker tables,
the 150-prompt-row head-plus-lookup subtotal is **121,459,323 bytes**, versus
121,490,523 for original private lookup. Ordinary local lookup already needs
no query frames, so its corresponding head-only subtotal is 121,084,623 bytes.
The new substrate saves 31,200 bytes against the expensive lookup control; it
adds 374,700 bytes against local lookup with the same retained full head.
These partial subtotals exclude tokenizer, body execution, role setup and
transport. A future head-relocating composition also owes generated-token lookup
and full-vocabulary output-head execution; this cohort measures prompt lookup.

## Implementation

* `.cargo/config.toml` enables RustCrypto AES 0.8's `aes_armv8` configuration on
  AArch64 macOS/Linux. Runtime feature detection retains the portable fallback;
  portable wheels do not force AES CPU features. The configuration is included
  in the source distribution and checked by `scripts/check_distributions.py`.
* The private-page PRG uses `Aes128Enc`, avoiding unused decryption schedules,
  and encrypts the three expansion blocks together. Its scalar bytes remain
  compatible with the original implementation.
* The optional native `leaf_bits=128` contract terminates the DPF tree at groups
  of 128 original records. A 128-bit final correction selects one bit within the
  group. Its domain and query/reply namespaces are distinct from scalar lookup.
  Non-power-of-two tables retain only their original records; padding is never
  returned as a record.
* At 151,936 records, scalar expansion has 151,936 leaves and 151,942 internal
  nodes. Packed expansion has 1,187 leaves and 1,193 internal nodes. All original
  table rows still participate in the masked XOR; packing does not remove this
  dense work or the table itself.
* Batches still contain at most 64 independently issued queries and share one
  table traversal. Each worker owns only its party's query half. Descriptor,
  role, query ID, layout, replay and cancellation checks retain one-use ownership;
  the client decoder burns before Python response conversion. The process-local
  replay ledger is bounded at 4,096 queries.

This remains the bounded semi-honest, non-colluding research reference. Native
framing is not authenticated provider transport, an online role graph, or
independent cryptographic review. Ordinary Preparation remains offline-only.

## Correctness

Native exhaustive basis tables cover every selector coordinate at the 127/128
boundary and incomplete final leaves. Mixed-layout batch admission, malformed
frames, substituted namespaces, cancellation, replay and decoder retirement are
checked. An independent OpenSSL point evaluator checks each party's selection
bits, rather than only checking their reconstructed record.

Generated Qwen2, Qwen3 and Llama checkpoints execute ordinary full-vocabulary
prefill and three generated-token feedback steps under both default and
`prefix_f32` reduction. Replacing only token lookup preserves every compared
logit, active KV byte, sampled token and RNG state. These are generated-model
functionality controls, not a new real-checkpoint quality cohort or SDK transport.

## Measurement and replay

`scripts/probe_private_lookup_acceleration.py` publishes the pinned public
Qwen2.5-0.5B W8 boundary in bounded source chunks. It compares the archived
original scalar evaluator, accelerated scalar evaluator and accelerated packed
evaluator in separate fresh party interpreters. Three alternating replicates
cover 16 and 150 public prompt rows; 150 rows use batches of 64, 64 and 22.

The original native control is locked to source commit
`376686886a6e3e32981e10aba2454eb853eca831` and extension SHA-256
`baa69ca5d362f2327502e5fbd9ff8150d215692dc300d5554033f86c64fc22ff`.
The runner requires that archived extension and its identity record under
`target/private-lookup-control`, or an explicit `--control-dir`. It does not
download, build or install a control while measuring. Scalar queries in all
arms are issued by the current client implementation: the original arm isolates
the old server evaluator, not old end-to-end client issuance.

Each recovered record is checked byte-for-byte against the source-derived file;
all 896 dequantized embedding values must match float32 bits. Import/source
validation, evaluation, client issuance and reply decoding have separate CPU
windows. Single-party process peaks include their import overlap. Worker times
are sequential and cannot be presented as concurrent request latency. Native
frames exclude the local JSON/hex pipe harness, HTTP/TLS and physical wire.

The shared frontier lock serializes heavy measurements. Two GiB of post-reserve
headroom, the host-pressure/swap monitor, worker timeouts and process-group
retirement bound execution. Source, script and both extension identities are
rechecked before report publication. Whole-client RSS and complete-response
traffic/latency remain unmeasured.

When both lookup workers already have the public source, they can derive their
tables locally, paying setup CPU and retained storage. Otherwise the two raw
record tables add 273,484,800 bytes before transport framing. Neither case removes
the client's full tied head. A further general-LLM traffic improvement must solve
full-head ownership or a different exact interface and account for every new
role, artifact and online exchange.

```sh
target/frontier-sdk-env/bin/python -m pytest -o pythonpath= -q tests/test_private_pages.py tests/test_private_lookup_decoder_parity.py tests/test_public_grammar_boundary.py
target/frontier-sdk-env/bin/python scripts/probe_private_lookup_acceleration.py --output target/private-lookup-acceleration-qwen-r1.json --repeats 3
```

The report path must be new. The original command above produced the frozen
cohort; select a fresh output path for another run. Sixty-one installed-wheel
checks pass, as do six forced-portable-AES native lookup checks. Independent
report reconciliation verifies all nine implementation identities, eighteen
paired ledgers, medians, record counts and frame lengths.
Successful `otool -tv` disassembly finds zero ARM AES instructions in the old
control and 639 `aese` instructions in the new extension. This checks instruction
availability; the CPU cohort supplies the performance result.

The complete `pllm-core`/`pllm-garble` native suites additionally pass 341 tests;
one unrelated opt-in release measurement remains ignored. Cargo formatting,
Python lint, source/wheel archive checks, docs content and benchmark-coverage
drift checks pass. The coverage surface refresh acknowledges the existing
`benchmark_tier_objectives.py` link to `BuildRef`; it adds no measured capability.
