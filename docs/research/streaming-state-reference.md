# Streaming full-KV storage gate

`BenchmarkTarget.state(..., storage="sealed_stream")` uses the same bounded,
source/plan/layout-bound synthetic KV fixture as `sealed_local`. AES-GCM frames
are unchanged. Encryption uses a 1 MiB scratch buffer; decryption writes into
one record allocation and verifies its tag and trusted frame digest before
returning a read-only float32 view. Failed authentication clears that allocation.

Each generation publishes only after every record and manifest is complete.
Failed writes retain the previous trusted revision and remove pending files.
The caller retains its key and manifest head separately from untrusted storage.
This is not a persistent anti-rollback service. Successful NumPy allocations,
caller views and Python encryption-key objects have ordinary Python lifetimes;
this reference does not claim complete memory zeroization.

Admission retains the existing 64 MiB record and 256 MiB total payload bounds.
The attention consumer keeps complete layer records, with a conservative
three-record codec allocation bound plus bounded ciphertext scratch. Attention
workspace, initialization, caller owners and filesystem cache remain separate.

## Validation

One-shot and streaming codecs read each other's frames bit-for-bit. Exact attention
matches at short and multi-chunk lengths. Corruption, changed trusted revisions,
wrong contexts, truncated frames and interrupted publication reject. The installed
target also executes restoration in a new process, with keys on stdin and separate
child CPU/RSS accounting. This synthetic restart does not authorize decoder state.

Replay the matched, three-process-per-cell study:

```sh
python -m benchmarks.research.state_working_set --processes 3 \
  --output target/state-working-set.json
```

It binds pinned Qwen2.5 configuration-derived KV shapes at 256, 2,048 and 8,192
tokens to resident, one-shot and streaming controls. Values are public synthetic
fixtures. Publication CPU/disk, target attention/restore CPU and process peaks
remain separate. The five-process `--tokens 16 --processes 1` smoke checks the
executor; it establishes no full-decoder memory or performance improvement.

The next gate is whether bounded streaming improves the previous sealed-layout
CPU penalty enough to justify a compiler-bound layer-swapped state owner. Such an
owner still needs real prefill/decode provenance, exact logits and KV, append and
failure lifetimes, and full-response measurement. The SDK state target enables
that decision without activating a serving component.

## Matched storage cohort

Thirty-three fresh target processes use three replicates per cell, including six
additional restore children. All **302,072,832 checked storage/attention values**
pass. Most checked values are restored state words, not attention outputs.
Build: `0201f1d6947159b1c5c528ac9648a8154258f24e2cbba13d1344494f78138c0f`.

Medians below use decimal MB and seconds. Process RSS includes import and fixture
publication; attention CPU is the separate target window. Publication is charged
once before that window. State values remain synthetic.

| Tokens | Layout | Process peak MB | Publication CPU | Attention CPU | Written MB |
| --- | --- | ---: | ---: | ---: | ---: |
| 256 | Resident | 60.85 | 0.01550 | 0.00144 | 0 |
| 256 | One-shot sealed | 60.82 | 0.03791 | 0.01007 | 6.30 |
| 256 | Streaming sealed | 61.15 | 0.03970 | 0.01031 | 6.30 |
| 2,048 | Resident | 106.04 | 0.06918 | 0.00567 | 0 |
| 2,048 | One-shot sealed | 64.23 | 0.13054 | 0.05548 | 50.34 |
| 2,048 | Streaming sealed | 63.23 | 0.13067 | 0.05457 | 50.34 |
| 8,192 | Resident | 258.59 | 0.25307 | 0.01789 | 0 |
| 8,192 | One-shot sealed | 82.03 | 0.45980 | 0.22063 | 201.33 |
| 8,192 | Streaming sealed | 78.99 | 0.44013 | 0.20685 | 201.33 |

At 8,192 tokens, streaming lowers scoped process peak RSS **69.45%** versus
resident state, but attention CPU is **11.57×** higher. Relative to the already
sealed control, streaming lowers RSS only **3.71%** and attention CPU **6.24%**.
Those comparisons have different baselines and must not be multiplied.

The separate 8,192-token restart cells lower issuer peak RSS **77.63 → 69.35 MB**
and restore-child peak **56.48 → 53.43 MB**. Combined timed restore CPU is
**0.41594 → 0.40608 seconds**, excluding separately charged publication. The
complete **201.33 MB** encrypted state still has to be written and read.

**Decision:** bounded streaming improves codec ownership, but does not remove the
sealed-attention CPU penalty. Compiler-bound append/eviction, checkpoint values,
full-response costs and filesystem-cache/device memory remain open; this cohort
does not promote a layer-swapped decoder.

Evidence: `docs/evidence/research-state-working-set-cohort.json`, with its
`.plan.json` and `.frontier.json` companions.
